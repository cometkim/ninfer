#pragma once

// ninfer::ops::detail - hq-e8-2b small-T launch ownership. One runtime-width tensor-core
// partial instantiation per geometry covers every legal token width, so the codec-heavy decode
// body compiles once per geometry instead of per width.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kv_cache/append/hq_kernel.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_hq.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {


template <typename Geometry>
constexpr int hq_token_tile() {
    return Geometry::QHeads == 24 ? 8 : 6;
}

template <typename Geometry, typename CacheInput, bool MultiBatch, bool Masked>
void launch_hq_partial(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                       PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                       std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                       Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int Warps  = 4;
    constexpr int Tokens = hq_token_tile<Geometry>();
    const dim3 grid(static_cast<unsigned>(Geometry::KVHeads), static_cast<unsigned>(splits),
                    MultiBatch ? static_cast<unsigned>(invocation.batch_size) : 1u);
    causal_attention_small_t_tc_partial_hq_kernel<Geometry, Tokens, Warps, MultiBatch, Masked,
                                                   CacheInput>
        <<<grid, Warps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<std::uint8_t*>(cache.k_pages.data),
        static_cast<std::uint8_t*>(cache.v_pages.data),
        static_cast<std::uint8_t*>(cache.k_scale_pages.data),
        static_cast<std::uint8_t*>(cache.v_scale_pages.data),
        static_cast<__nv_bfloat16*>(cache.residual_k.data),
        static_cast<__nv_bfloat16*>(cache.residual_v.data),
        static_cast<std::uint32_t*>(cache.side_words.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                                 : nullptr,
        invocation.table_rows ? static_cast<const std::int32_t*>(invocation.table_rows->data)
                              : nullptr,
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
            logical_capacity, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_hq_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                            float scale, PagedKVBatchLayerView cache,
                                            const CausalSmallTInvocation& invocation,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits           = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);

    const auto launch_partial = [&]<bool MultiBatch, bool Masked>() {
        launch_hq_partial<Geometry, CacheInput, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, splits, partial_acc,
            partial_m, partial_l, stream);
    };
    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_partial.template operator()<false, true>();
        } else {
            launch_partial.template operator()<false, false>();
        }
    } else if (masked) {
        launch_partial.template operator()<true, true>();
    } else {
        launch_partial.template operator()<true, false>();
    }

    // Partials are un-rotated into the original frame inside the partial kernel, so the shared
    // BF16-profile reducer combines them exactly like the other cache dtypes.
    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = Geometry::QHeads == 24 ? 256 : 64;
    const auto launch_reduce   = [&]<bool MultiBatch, bool Masked, bool Offset>() {
        const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, kDChunk),
                        invocation.width * invocation.batch_size);
        causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, false, MultiBatch,
                                                       Masked, Offset>
            <<<grid, kReduceBlock, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns
                    ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                    : nullptr,
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    };
    const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0)
            launch_reduce.template operator()<MultiBatch, Masked, false>();
        else
            launch_reduce.template operator()<MultiBatch, Masked, true>();
    };
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_profile.template operator()<false, true>();
        } else {
            launch_profile.template operator()<false, false>();
        }
    } else if (masked) {
        launch_profile.template operator()<true, true>();
    } else {
        launch_profile.template operator()<true, false>();
    }
}

extern template void causal_attention_small_t_hq_launch_for<CausalD256H24Kv4, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

extern template void causal_attention_small_t_hq_launch_for<CausalD256H24Kv4, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

extern template void causal_attention_small_t_hq_launch_for<CausalD256H16Kv2, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

extern template void causal_attention_small_t_hq_launch_for<CausalD256H16Kv2, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
