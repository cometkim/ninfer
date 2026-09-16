#include "prompt_routes.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "ops/common/math.h"
#include "core/device.h"

namespace ninfer::ops::detail {
template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_i8_launch_for(const Tensor& q, const Tensor& positions, float scale,
                                           const CacheView& cache, Metadata metadata, Tensor& out,
                                           Tensor& partial_acc, Tensor& partial_m,
                                           Tensor& partial_l, std::int32_t split_count,
                                           cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    CUDA_CHECK(attr_i8);
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                              static_cast<unsigned>(Geometry::QHeads),
                              static_cast<unsigned>(split_count));
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    causal_attention_prompt_i8_kernel<Geometry, Metadata>
        <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int8_t*>(cache_k.data),
            static_cast<const std::int8_t*>(cache_v.data),
            static_cast<const __half*>(cache_k_scale.data),
            static_cast<const __half*>(cache_v_scale.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data), split_count,
            tokens);
    if (split_count > 1) {
        const dim3 reduce_grid(static_cast<unsigned>(tokens),
                               static_cast<unsigned>(Geometry::QHeads));
        causal_attention_prompt_reduce_kernel<Geometry>
            <<<reduce_grid, kCausalPromptHeadDim, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data), scale, tokens, split_count,
                static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

template void
causal_attention_prompt_i8_launch_for<CausalD256H24Kv4, PagedKVLayerView, PagedKVDirectMetadata>(
    const Tensor&, const Tensor&, float, const PagedKVLayerView&, PagedKVDirectMetadata, Tensor&,
    Tensor&, Tensor&, Tensor&, std::int32_t, cudaStream_t);
template void causal_attention_prompt_i8_launch_for<CausalD256H24Kv4, PagedKVBatchLayerView,
                                                    PagedKVBatchMetadata<false>>(
    const Tensor&, const Tensor&, float, const PagedKVBatchLayerView&, PagedKVBatchMetadata<false>,
    Tensor&, Tensor&, Tensor&, Tensor&, std::int32_t, cudaStream_t);
template void causal_attention_prompt_i8_launch_for<CausalD256H24Kv4, PagedKVBatchLayerView,
                                                    PagedKVBatchMetadata<true>>(
    const Tensor&, const Tensor&, float, const PagedKVBatchLayerView&, PagedKVBatchMetadata<true>,
    Tensor&, Tensor&, Tensor&, Tensor&, std::int32_t, cudaStream_t);
template void
causal_attention_prompt_i8_launch_for<CausalD256H16Kv2, PagedKVLayerView, PagedKVDirectMetadata>(
    const Tensor&, const Tensor&, float, const PagedKVLayerView&, PagedKVDirectMetadata, Tensor&,
    Tensor&, Tensor&, Tensor&, std::int32_t, cudaStream_t);
template void causal_attention_prompt_i8_launch_for<CausalD256H16Kv2, PagedKVBatchLayerView,
                                                    PagedKVBatchMetadata<false>>(
    const Tensor&, const Tensor&, float, const PagedKVBatchLayerView&, PagedKVBatchMetadata<false>,
    Tensor&, Tensor&, Tensor&, Tensor&, std::int32_t, cudaStream_t);
template void causal_attention_prompt_i8_launch_for<CausalD256H16Kv2, PagedKVBatchLayerView,
                                                    PagedKVBatchMetadata<true>>(
    const Tensor&, const Tensor&, float, const PagedKVBatchLayerView&, PagedKVBatchMetadata<true>,
    Tensor&, Tensor&, Tensor&, Tensor&, std::int32_t, cudaStream_t);
} // namespace ninfer::ops::detail
