#pragma once

// ninfer::ops::detail - shared nvfp4 small-T launch helper (the build-speed TU split). The
// dispatcher TU references these instantiations through extern-template declarations; only the
// instantiation TU expands the kernels.

// ninfer::ops::detail - group-16 NVFP4 split-KV small-T launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cuh"

#include <cstdint>
#include <stdexcept>


namespace ninfer::ops::detail {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_nvfp4_partial(const Tensor& q, CacheInput input, const Tensor& positions, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                          Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Warps                = RowTiles == 3 ? 12 : 8;
    constexpr int KeyBlock             = 32;
    constexpr int MinBlocks            = RowTiles <= 2 ? 2 : 1;
    constexpr std::size_t DynamicBytes = (RowTiles <= 2 ? 3u : 5u) * KeyBlock * kCausalHeadDim;
    using KernelInput                  = CacheInput;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    const auto launch = [&]() {
        auto kernel = causal_attention_small_t_nvfp4_tiled_kernel<
            Geometry, TokenTile, Warps, MinBlocks, KeyBlock, true, MultiBatch, Masked, KernelInput>;
        static const cudaError_t attr = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DynamicBytes));
        CUDA_CHECK(attr);

        const auto q_ptr         = static_cast<const __nv_bfloat16*>(q.data);
        const auto positions_ptr = static_cast<const std::int32_t*>(positions.data);
        const auto cache_k_ptr   = static_cast<std::uint8_t*>(cache.k_pages.data);
        const auto cache_v_ptr   = static_cast<std::uint8_t*>(cache.v_pages.data);
        const auto k_scale_ptr   = static_cast<std::uint8_t*>(cache.k_scale_pages.data);
        const auto v_scale_ptr   = static_cast<std::uint8_t*>(cache.v_scale_pages.data);
        const auto tables_ptr    = static_cast<const std::int32_t*>(cache.block_tables.data);
        const auto valid_ptr =
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data);
        const auto rows_ptr   = invocation.table_rows == nullptr
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(invocation.table_rows->data);
        auto* partial_acc_ptr = static_cast<float*>(partial_acc.data);
        auto* partial_m_ptr   = static_cast<float*>(partial_m.data);
        auto* partial_l_ptr   = static_cast<float*>(partial_l.data);
        kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
            q_ptr, input, positions_ptr, cache_k_ptr, cache_v_ptr, k_scale_ptr, v_scale_ptr,
            tables_ptr, valid_ptr, rows_ptr, cache.block_tables.ne[0], invocation.full_width,
            invocation.column_begin, logical_capacity, scale, partial_acc_ptr, partial_m_ptr,
            partial_l_ptr);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
}

extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
