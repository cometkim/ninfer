// ninfer::ops - gqa_attention prompt-scale dispatcher: geometry, metadata, and
// dtype route selection. The per-dtype kernels live in
// gqa_attention_prefill_{bf16,i8,hq}.cu; this TU instantiates no fat kernels.
#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_route(const Tensor& k, const Tensor& v, const Tensor& positions,
                              CacheView cache, Metadata metadata, cudaStream_t stream) {
    if (cache.dtype == DType::U8) {
        gqa_prefill_append_hq<Geometry, CacheView, Metadata>(k, v, positions, cache, metadata,
                                                             stream);
    } else if (cache.dtype == DType::I8) {
        gqa_prefill_append_i8<Geometry, CacheView, Metadata>(k, v, positions, cache, metadata,
                                                             stream);
    } else {
        gqa_prefill_append_bf16<Geometry, CacheView, Metadata>(k, v, positions, cache, metadata,
                                                               stream);
    }
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_route(const Tensor& q, const Tensor& positions, float scale,
                                 const CacheView& cache, Metadata metadata, const Tensor& new_k,
                                 const Tensor& new_v, const Tensor& scratch_k,
                                 const Tensor& scratch_v, const Tensor& carry_acc,
                                 const Tensor& carry_m, const Tensor& carry_l,
                                 std::uint32_t visible_keys, const Tensor& partial_acc,
                                 const Tensor& partial_m, const Tensor& partial_l,
                                 std::int32_t split_count, Tensor& out, cudaStream_t stream) {
    if (cache.dtype == DType::U8) {
        gqa_prefill_attention_hq<Geometry, CacheView, Metadata>(
            q, positions, scale, cache, metadata, new_k, new_v, scratch_k, scratch_v, carry_acc,
            carry_m, carry_l, visible_keys, partial_acc, partial_m, partial_l, split_count, out,
            stream);
    } else if (cache.dtype == DType::I8) {
        gqa_prefill_attention_i8<Geometry, CacheView, Metadata>(q, positions, scale, cache,
                                                                metadata, out, partial_acc,
                                                                partial_m, partial_l, split_count,
                                                                stream);
    } else {
        gqa_prefill_attention_bf16<Geometry, CacheView, Metadata>(q, positions, scale, cache,
                                                                  metadata, out, partial_acc,
                                                                  partial_m, partial_l,
                                                                  split_count, stream);
    }
}

} // namespace

std::int32_t gqa_prefill_split_count(std::int32_t width, std::int32_t q_heads) {
    static const int sms = [] {
        int device = 0;
        int count  = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return 1; }
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            count <= 0) {
            return 1;
        }
        return count;
    }();
    const std::int64_t items = (static_cast<std::int64_t>(width) + 63) / 64 * q_heads;
    const double ratio_s1    = static_cast<double>((items + sms - 1) / sms);
    double best_ratio        = ratio_s1;
    std::int32_t best        = 1;
    for (std::int32_t s = 2; s <= 4; ++s) {
        const double ratio = static_cast<double>((items * s + sms - 1) / sms) / s;
        if (ratio < best_ratio) {
            best_ratio = ratio;
            best       = s;
        }
    }
    if (best_ratio > 0.9 * ratio_s1) { return 1; }
    return best;
}

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, const Tensor& scratch_k,
                                           const Tensor& scratch_v, const Tensor& carry_acc,
                                           const Tensor& carry_m, const Tensor& carry_l,
                                           std::uint32_t visible_keys,
                                           const Tensor& partial_acc, const Tensor& partial_m,
                                           const Tensor& partial_l, std::int32_t split_count,
                                           Tensor& out, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_prefill_attention_route<Gqa27Geometry>(q, positions, scale, cache, metadata, Tensor{},
                                                   Tensor{}, scratch_k, scratch_v, carry_acc,
                                                   carry_m, carry_l, visible_keys, partial_acc,
                                                   partial_m, partial_l, split_count, out, stream);
        return;
    }
    gqa_prefill_attention_route<Gqa35Geometry>(q, positions, scale, cache, metadata, Tensor{},
                                               Tensor{}, scratch_k, scratch_v, carry_acc,
                                               carry_m, carry_l, visible_keys, partial_acc,
                                               partial_m, partial_l, split_count, out, stream);
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == Gqa27Geometry::KVHeads) {
        gqa_prefill_append_route<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    gqa_prefill_append_route<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 const Tensor& scratch_k, const Tensor& scratch_v,
                                 const Tensor& carry_acc, const Tensor& carry_m,
                                 const Tensor& carry_l, std::uint32_t visible_keys,
                                 const Tensor& partial_acc, const Tensor& partial_m,
                                 const Tensor& partial_l, std::int32_t split_count, Tensor& out,
                                 cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == Gqa27Geometry::QHeads) {
            gqa_prefill_append_route<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            gqa_prefill_attention_route<Gqa27Geometry>(q, positions, scale, cache, metadata, k, v,
                                                       scratch_k, scratch_v, carry_acc, carry_m,
                                                       carry_l, visible_keys, partial_acc,
                                                       partial_m, partial_l, split_count, out,
                                                       stream);
            return;
        }
        gqa_prefill_append_route<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        gqa_prefill_attention_route<Gqa35Geometry>(q, positions, scale, cache, metadata, k, v,
                                                   scratch_k, scratch_v, carry_acc, carry_m,
                                                   carry_l, visible_keys, partial_acc, partial_m,
                                                   partial_l, split_count, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
