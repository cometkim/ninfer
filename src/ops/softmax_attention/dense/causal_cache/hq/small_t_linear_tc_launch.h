#pragma once

// ninfer::ops::detail::hq - fork kernel-perf small-T routes for the linear KV codecs on the
// rebased base: the fork's tensor-core split-KV partial kernels (fused append; INT8 s8 QK, FP8 and
// FP8-K E4M3 QK, NVFP4 FP16 QK) and the module's shared reducer, for the profiles where they
// measured faster than the grouped routes (the int8, fp8, nvfp4 and k8v4 plans select them). The
// kernels instantiate in the small_t_<codec>_h24.cu / small_t_<codec>_h16.cu units only.
//
// Launch forms follow the grouped routes they replace, so graph tiers that switch between the two
// keep an update-compatible topology: the INT8 partial kernel is a PDL dependent (as the INT8
// grouped kernel), the FP8/NVFP4/K8V4 partial kernels are ordinary launches (as theirs), and the
// reducer is a PDL dependent (as the natural merge).
#include "ops/softmax_attention/dense/causal_cache/hq/internal.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/hq/small_t_fp8.cuh"
#include "ops/softmax_attention/dense/causal_cache/hq/small_t_i8.cuh"
#include "ops/softmax_attention/dense/causal_cache/hq/small_t_k8v4.cuh"
#include "ops/softmax_attention/dense/causal_cache/hq/small_t_nvfp4.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::hq {

enum class SmallTCodec { Int8, Fp8, Nvfp4, K8v4 };

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
    // The fork's CTA shapes per token tile and envelope (graph calls pass their replay tier, so
    // each tier keeps one kernel; the node count is the same across tiers).
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key loop iterations;
        // dynamic smem avoids penalizing the long-context path.
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
            // Three Q row tiles for the 35B group of eight. The 24/12-warp routes retain
            // eight/four consumer warps per tile; the 6-warp route is reserved for long windows
            // where CTA residency wins.
            if (implementation_window <= 1029) {
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

// FP8, NVFP4 and FP8-K/NVFP4-V partial kernels: one CTA shape per token tile (the fork's
// per-codec key block, residency and dynamic arena).
template <SmallTCodec Codec, typename Geometry, int TokenTile, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_quantized(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                                 PagedKVBatchLayerView cache,
                                 const CausalSmallTInvocation& invocation,
                                 std::int32_t logical_capacity, std::int32_t splits,
                                 Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                 cudaStream_t stream) {
    static_assert(Codec != SmallTCodec::Int8);
    constexpr bool kNvfp4   = Codec == SmallTCodec::Nvfp4;
    constexpr int RowTiles  = (TokenTile * Geometry::GroupSize + 15) / 16;
    constexpr int Warps     = RowTiles == 3 ? 12 : 8;
    constexpr int KeyBlock  = kNvfp4 || TokenTile == 1 ? 32 : 64;
    constexpr int MinBlocks = kNvfp4 ? (RowTiles <= 2 ? 2 : 1) : (TokenTile == 1 ? 2 : 1);
    constexpr std::size_t DynamicBytes =
        Codec == SmallTCodec::Fp8    ? 4u * KeyBlock * kCausalHeadDim
        : Codec == SmallTCodec::K8v4 ? 7u * KeyBlock * kCausalHeadDim / 2u
                                     : (RowTiles <= 2 ? 3u : 5u) * KeyBlock * kCausalHeadDim;
    using KeyScale    = std::conditional_t<kNvfp4, std::uint8_t, __half>;
    using ValueScale  = std::conditional_t<Codec == SmallTCodec::Fp8, __half, std::uint8_t>;
    const auto kernel = [] {
        if constexpr (Codec == SmallTCodec::Fp8) {
            return causal_attention_small_t_fp8_tiled_kernel<Geometry, TokenTile, Warps, MinBlocks,
                                                             KeyBlock, true, MultiBatch, Masked,
                                                             CacheInput>;
        } else if constexpr (Codec == SmallTCodec::K8v4) {
            return causal_attention_small_t_k8v4_tiled_kernel<Geometry, TokenTile, Warps, MinBlocks,
                                                              KeyBlock, true, MultiBatch, Masked,
                                                              CacheInput>;
        } else {
            return causal_attention_small_t_nvfp4_tiled_kernel<Geometry, TokenTile, Warps,
                                                               MinBlocks, KeyBlock, true,
                                                               MultiBatch, Masked, CacheInput>;
        }
    }();
    static const cudaError_t attr = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DynamicBytes));
    CUDA_CHECK(attr);
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache.k_pages.data),
        static_cast<std::uint8_t*>(cache.v_pages.data),
        static_cast<KeyScale*>(cache.k_scale_pages.data),
        static_cast<ValueScale*>(cache.v_scale_pages.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.full_width, invocation.column_begin, logical_capacity,
        scale, static_cast<float*>(partial_acc.data), static_cast<float*>(partial_m.data),
        static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <SmallTCodec Codec, typename Geometry, typename CacheInput>
void causal_attention_small_t_linear_launch_for(
    const Tensor& q, CacheInput input, const Tensor& pos, float scale, PagedKVBatchLayerView cache,
    const CausalSmallTInvocation& invocation, CausalAttentionExecutionEnvelope envelope,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits                = causal_attention_split_capacity_linear(
        cache.storage, Geometry::QHeads, invocation.width, envelope, invocation.batch_size,
        invocation.multiprocessor_count);
    const bool masked = invocation.valid_columns != nullptr;
    // Each launch profile (batch rows, column mask) is a separate instantiation; cached calls
    // are single-row and unmasked, so only that profile is instantiated for them.
    const auto for_profile = [&](auto&& launch_profile) {
        if constexpr (CacheInput::writes_cache) {
            if (invocation.batch_size == 1) {
                if (masked)
                    launch_profile.template operator()<false, true>();
                else
                    launch_profile.template operator()<false, false>();
            } else if (masked) {
                launch_profile.template operator()<true, true>();
            } else {
                launch_profile.template operator()<true, false>();
            }
        } else {
            if (invocation.batch_size != 1 || masked) {
                throw std::invalid_argument(
                    "causal_attention_small_t_linear_launch: cached calls are single-row");
            }
            launch_profile.template operator()<false, false>();
        }
    };
    const auto dispatch = [&]<int Tokens>() {
        for_profile([&]<bool MultiBatch, bool Masked>() {
            if constexpr (Codec == SmallTCodec::Int8) {
                launch_tc_partial_i8<Geometry, Tokens, MultiBatch, Masked>(
                    q, input, pos, scale, cache, invocation, logical_capacity,
                    implementation_window, splits, partial_acc, partial_m, partial_l, stream);
            } else {
                launch_tc_partial_quantized<Codec, Geometry, Tokens, MultiBatch, Masked>(
                    q, input, pos, scale, cache, invocation, logical_capacity, splits, partial_acc,
                    partial_m, partial_l, stream);
            }
        });
    };
    switch (invocation.width) {
    case 1:
        dispatch.template operator()<1>();
        break;
    case 2:
        dispatch.template operator()<2>();
        break;
    case 3:
        dispatch.template operator()<3>();
        break;
    case 4:
        dispatch.template operator()<4>();
        break;
    case 5:
        dispatch.template operator()<5>();
        break;
    case 6:
        dispatch.template operator()<6>();
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            dispatch.template operator()<7>();
            break;
        }
        throw std::invalid_argument("causal_attention_small_t_linear_launch: unsupported T");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            dispatch.template operator()<8>();
            break;
        }
        throw std::invalid_argument("causal_attention_small_t_linear_launch: unsupported T");
    default:
        throw std::invalid_argument("causal_attention_small_t_linear_launch: unsupported T");
    }

    // NVFP4 and FP8-K/NVFP4-V store V in the rotated frame: their reducer normalizes whole rows
    // and rotates them back. The others reduce D chunks in place.
    constexpr bool kInverseRotation = Codec == SmallTCodec::Nvfp4 || Codec == SmallTCodec::K8v4;
    constexpr int kReduceBlock      = 256;
    constexpr int kDChunk = kInverseRotation || Geometry::QHeads == 24 ? kCausalHeadDim : 64;
    constexpr auto kSplitPolicy =
        Codec == SmallTCodec::Int8 ? SmallTSplitPolicy::Int8 : SmallTSplitPolicy::Quantized;
    const auto launch_reduce = [&]<bool MultiBatch, bool Masked, bool Offset>() {
        const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, kDChunk),
                        invocation.width * invocation.batch_size);
        CUDA_CHECK(pdl::launch_dependent(
            {grid, dim3(kReduceBlock), 0, stream},
            causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, kSplitPolicy,
                                                          MultiBatch, Masked, Offset, false,
                                                          kInverseRotation>,
            static_cast<const float*>(partial_acc.data), static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data), static_cast<const std::int32_t*>(pos.data),
            invocation.valid_columns
                ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                : nullptr,
            invocation.gate != nullptr ? static_cast<const __nv_bfloat16*>(invocation.gate->data)
                                       : nullptr,
            invocation.width, invocation.full_width, invocation.column_begin, invocation.batch_size,
            splits, static_cast<__nv_bfloat16*>(out.data)));
    };
    for_profile([&]<bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0)
            launch_reduce.template operator()<MultiBatch, Masked, false>();
        else
            launch_reduce.template operator()<MultiBatch, Masked, true>();
    });
    CUDA_CHECK(cudaGetLastError());
}

#define NINFER_SMALL_T_LINEAR_INSTANCE(PREFIX, CODEC, GEOMETRY, INPUT)                             \
    PREFIX template void                                                                           \
    causal_attention_small_t_linear_launch_for<SmallTCodec::CODEC, GEOMETRY, INPUT>(               \
        const Tensor&, INPUT, const Tensor&, float, PagedKVBatchLayerView,                         \
        const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&,         \
        Tensor&, Tensor&, cudaStream_t)
#define NINFER_SMALL_T_LINEAR_INSTANCES(PREFIX, CODEC, GEOMETRY)                                   \
    NINFER_SMALL_T_LINEAR_INSTANCE(PREFIX, CODEC, GEOMETRY, CausalAppendInput);                    \
    NINFER_SMALL_T_LINEAR_INSTANCE(PREFIX, CODEC, GEOMETRY, CausalCachedInput)

NINFER_SMALL_T_LINEAR_INSTANCES(extern, Int8, CausalD256H24Kv4);
NINFER_SMALL_T_LINEAR_INSTANCES(extern, Int8, CausalD256H16Kv2);
NINFER_SMALL_T_LINEAR_INSTANCES(extern, Fp8, CausalD256H24Kv4);
NINFER_SMALL_T_LINEAR_INSTANCES(extern, Fp8, CausalD256H16Kv2);
NINFER_SMALL_T_LINEAR_INSTANCES(extern, Nvfp4, CausalD256H24Kv4);
NINFER_SMALL_T_LINEAR_INSTANCES(extern, Nvfp4, CausalD256H16Kv2);
NINFER_SMALL_T_LINEAR_INSTANCES(extern, K8v4, CausalD256H24Kv4);
NINFER_SMALL_T_LINEAR_INSTANCES(extern, K8v4, CausalD256H16Kv2);

} // namespace ninfer::ops::detail::hq
