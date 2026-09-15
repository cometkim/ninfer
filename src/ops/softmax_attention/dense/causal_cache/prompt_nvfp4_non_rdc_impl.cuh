// Non-RDC ownership for the warp-specialized NVFP4 causal prompt kernel.
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4.cuh"

#include <cstdint>


namespace ninfer::ops::detail {


template <typename Geometry, typename CacheView, typename Metadata>
void prompt_nvfp4_non_rdc_geometry_launch(const Tensor& q, const Tensor& positions, float scale, const CacheView& cache,
                Metadata metadata, Tensor& out, cudaStream_t stream) {
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_nvfp4_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptNvfp4SmemBytes);
    CUDA_CHECK(attr);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(div_up(tokens, kCausalPromptNvfp4Br)),
                    static_cast<unsigned>(Geometry::QHeads), 1U);
    causal_attention_prompt_nvfp4_kernel<Geometry, Metadata>
        <<<grid, kCausalPromptNvfp4Threads, kCausalPromptNvfp4SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
