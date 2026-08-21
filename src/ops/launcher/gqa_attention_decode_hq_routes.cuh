#pragma once

// ninfer::ops::detail - hq-e8-2b small-T partial-kernel launch route, shared by
// the per-geometry instantiation TUs gqa_attention_decode_hq_{27,35}.cu. The
// codec-heavy hq decode kernel dominates this route's compile time, so each
// geometry is explicitly instantiated in its own translation unit.

#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_decode_hq.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheInput>
void gqa_small_t_partial_hq(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k      = cache.k_pages;
    Tensor& cache_v      = cache.v_pages;
    Tensor& cache_k_meta = cache.k_scale_pages;
    Tensor& cache_v_meta = cache.v_scale_pages;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    constexpr std::size_t smem = gqa_hq_decode_smem_bytes<Geometry>();
    static const bool attr_set = [] {
        const cudaError_t e = cudaFuncSetAttribute(
            gqa_attention_decode_hq_kernel<Geometry, CacheInput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        CUDA_CHECK(e);
        return true;
    }();
    (void)attr_set;
    gqa_attention_decode_hq_kernel<Geometry, CacheInput>
        <<<grid, kGqaHqDecodeThreads, smem, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data), invocation.width,
            static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
            static_cast<std::uint8_t*>(cache_k_meta.data),
            static_cast<std::uint8_t*>(cache_v_meta.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
            logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
