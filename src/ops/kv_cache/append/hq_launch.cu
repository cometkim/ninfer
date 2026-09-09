// ninfer::ops::detail - hq-e8-2b append launch ownership.
#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/hq_kernel.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = kKVCacheHqFillWarps * 32;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_hq_for(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                   Metadata metadata, cudaStream_t stream) {
    const auto tokens             = static_cast<std::int32_t>(k.ne[2]);
    const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads * 2;
    const int grid =
        static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kKVCacheHqFillWarps)));
    kv_cache_append_full_hq_kernel<Geometry, Metadata>
        <<<grid, kBlock, kKVCacheHqFillSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<std::uint8_t*>(cache.v_scale_pages.data),
            static_cast<__nv_bfloat16*>(cache.residual_k.data),
            static_cast<__nv_bfloat16*>(cache.residual_v.data),
            static_cast<std::uint32_t*>(cache.side_words.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void dispatch_hq(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                 Metadata metadata, cudaStream_t stream) {
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_hq_for<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_hq_for<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

} // namespace

void kv_cache_append_hq_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                               PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data), cache.slot};
    dispatch_hq(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_hq_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& valid_columns, const Tensor& table_rows,
                                     PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        dispatch_hq(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
