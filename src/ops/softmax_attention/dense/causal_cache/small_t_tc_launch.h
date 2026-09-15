#pragma once

// ninfer::ops::detail - shared per-dtype launch helpers for the split-KV causal small-T
// route. The template definitions live here so the aggregate dispatcher (small_t.cu) and the
// per-dtype instantiation TUs (small_t_tc_i8.cu, small_t_tc_bf16.cu) see one spelling; the
// dispatcher uses extern-template declarations so only the instantiation TUs carry the heavy
// kernel expansions (the build-speed TU split, re-derived on the restructured tree).

#include "core/pdl.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging).
    CUDA_CHECK(pdl::launch_dependent(
        {grid, dim3(kBlock), 0, stream},
        causal_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta,
                                                        MultiBatch, Masked, CacheInput>,
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
            static_cast<__half*>(cache_v.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.width, invocation.full_width,
            invocation.column_begin, logical_capacity, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data)));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                         MinBlocksPerSm, KeyBlock, DynamicArena,
                                                         MultiBatch, Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        CUDA_CHECK(pdl::launch_dependent(
            {grid, dim3(WarpsPerCta * 32), kDynamicBytes, stream},
            causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, CacheInput>,
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data)));
    };
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

// The dispatcher references these instantiations without re-expanding them.
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 1, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 3, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 5, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 6, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 7, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H24Kv4, 8, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 1, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 3, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 5, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
extern template void launch_tc_partial_i8<CausalD256H16Kv2, 6, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail