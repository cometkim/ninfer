#include "ops/launcher/dflash2_draft.h"

#include "ops/kernel/dflash2_draft.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_dynamic_conv_launch(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                                 std::int32_t side, std::int32_t block_size, Tensor& out,
                                 cudaStream_t stream) {
    const std::int64_t elements = static_cast<std::int64_t>(hidden.ne[0]) * hidden.ne[1];
    const int threads           = 256;
    const int blocks            = static_cast<int>(
        (elements + threads - 1) / threads > 65535 ? 65535 : (elements + threads - 1) / threads);
    dflash2_dynamic_conv_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(dynamic.data),
        static_cast<const __nv_bfloat16*>(base.data), static_cast<std::int32_t>(hidden.ne[0]),
        static_cast<std::int32_t>(hidden.ne[1]), block_size, side,
        static_cast<__nv_bfloat16*>(out.data));
}

void dflash2_selector_scores_launch(const Tensor& candidates, const Tensor& predecessor_ids,
                                    const Tensor& unary, const Tensor& hidden_proj,
                                    const Tensor& successor_rows, const Tensor& predecessor_rows,
                                    Tensor& out, cudaStream_t stream) {
    const std::int64_t scores = static_cast<std::int64_t>(out.ne[0]) * out.ne[1] * out.ne[2] *
                                out.ne[3];
    dflash2_selector_scores_kernel<<<static_cast<int>(scores), kDflash2SelectorThreads, 0,
                                     stream>>>(
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const std::int32_t*>(predecessor_ids.data),
        static_cast<const float*>(unary.data), static_cast<const float*>(hidden_proj.data),
        static_cast<const __nv_bfloat16*>(successor_rows.data),
        static_cast<const __nv_bfloat16*>(predecessor_rows.data),
        static_cast<std::int32_t>(hidden_proj.ne[0]), static_cast<std::int32_t>(candidates.ne[0]),
        static_cast<std::int32_t>(candidates.ne[1]), static_cast<std::int32_t>(candidates.ne[2]),
        static_cast<float*>(out.data));
}

} // namespace ninfer::ops::detail
