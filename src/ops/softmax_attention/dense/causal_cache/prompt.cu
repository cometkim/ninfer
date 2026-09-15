// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "prompt_routes.h"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

// Wave-fill split-count policy (the fork's WI-K1a): pick S in {1..4} only when the predicted
// wave-balance ratio improves by >=10% over the single-pass sweep; S=1 keeps the single-pass
// path bit-identical.
std::int32_t causal_prompt_i8_split_count(std::int32_t width, std::int32_t q_heads) {
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
    // A single-pass sweep that already fits one wave gains nothing from key-splitting; the
    // ceil-based ratio metric below degenerates for such tiny grids (S=4 would be chosen for a
    // 12-token prompt), so short-circuit before it runs. This keeps short prefills bit-identical
    // to the pre-split path.
    if (items <= sms) { return 1; }
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


namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, std::int32_t split_count,
                                                  cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Int8Group64) {
        causal_attention_prompt_i8_launch_for<Geometry>(
            q, positions, scale, cache, metadata, out, partial_acc, partial_m, partial_l,
            split_count, stream);
    } else {
        causal_attention_prompt_bf16_launch_for<Geometry>(
            q, positions, scale, cache, metadata, out, partial_acc, partial_m, partial_l,
            split_count, stream);
    }
}

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              Tensor& partial_acc, Tensor& partial_m,
                                              Tensor& partial_l, std::int32_t split_count,
                                              const Tensor& scratch_k, const Tensor& scratch_v,
                                              const Tensor& carry_acc, const Tensor& carry_m,
                                              const Tensor& carry_l, std::uint32_t visible_keys,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::HqE8Rice2B) {
        causal_attention_prompt_hq_attention_launch(q, positions, scale, cache, scratch_k,
                                                    scratch_v, carry_acc, carry_m, carry_l,
                                                    visible_keys, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, partial_acc,
                                                                       partial_m, partial_l,
                                                                       split_count, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, partial_acc,
                                                                   partial_m, partial_l,
                                                                   split_count, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, Tensor& partial_acc,
                                    Tensor& partial_m, Tensor& partial_l,
                                    std::int32_t split_count, const Tensor& scratch_k,
                                    const Tensor& scratch_v, const Tensor& carry_acc,
                                    const Tensor& carry_m, const Tensor& carry_l,
                                    std::uint32_t visible_keys, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::HqE8Rice2B) {
        causal_attention_prompt_hq_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                          cache, scratch_k, scratch_v, carry_acc, carry_m,
                                          carry_l, visible_keys, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, partial_acc, partial_m, partial_l,
                split_count, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, partial_acc,
                                                                       partial_m, partial_l,
                                                                       split_count, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
