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

void dflash2_topk_launch(const Tensor& logits, std::int32_t k, Tensor& ids, Tensor& values,
                         cudaStream_t stream) {
    const int warps_per_block = 4;
    const int threads         = warps_per_block * 32;
    const int blocks          = static_cast<int>((logits.ne[1] + warps_per_block - 1) / warps_per_block);
    dflash2_topk_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t>(logits.ne[0]),
        static_cast<std::int32_t>(logits.ne[1]), k, static_cast<std::int32_t*>(ids.data),
        static_cast<__nv_bfloat16*>(values.data));
}

void dflash2_selector_walk_launch(const Tensor& scores, const Tensor& candidates, Tensor& out,
                                  cudaStream_t stream) {
    const int threads = 64;
    const int blocks =
        static_cast<int>((scores.ne[3] + threads - 1) / threads);
    dflash2_selector_walk_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(scores.data),
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<std::int32_t>(scores.ne[0]), static_cast<std::int32_t>(scores.ne[2]),
        static_cast<std::int32_t>(scores.ne[3]), static_cast<std::int32_t*>(out.data));
}

void dflash2_selector_scores_launch(const Tensor& candidates, const Tensor& predecessor_ids,
                                    const Tensor& unary, const Tensor& hidden_proj,
                                    const Weight& successor_rows, const Weight& predecessor_rows,
                                    Tensor& out, cudaStream_t stream) {
    const std::int64_t scores = static_cast<std::int64_t>(out.ne[0]) * out.ne[1] * out.ne[2] *
                                out.ne[3];
    dflash2_selector_scores_kernel<<<static_cast<int>(scores), kDflash2SelectorThreads, 0,
                                     stream>>>(
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const std::int32_t*>(predecessor_ids.data),
        static_cast<const float*>(unary.data), static_cast<const float*>(hidden_proj.data),
        static_cast<const std::uint8_t*>(successor_rows.qdata),
        static_cast<const std::uint8_t*>(successor_rows.scales),
        1.0F / successor_rows.weight_scale_divisor,
        static_cast<const std::uint8_t*>(predecessor_rows.qdata),
        static_cast<const std::uint8_t*>(predecessor_rows.scales),
        1.0F / predecessor_rows.weight_scale_divisor,
        static_cast<std::int32_t>(hidden_proj.ne[0]), static_cast<std::int32_t>(candidates.ne[0]),
        static_cast<std::int32_t>(candidates.ne[1]), static_cast<std::int32_t>(candidates.ne[2]),
        static_cast<float*>(out.data));
}

void dflash2_selector_predecessors_launch(const Tensor& candidates, const Tensor& anchors,
                                          Tensor& out, cudaStream_t stream) {
    const std::int64_t elements =
        static_cast<std::int64_t>(out.ne[0]) * out.ne[1] * out.ne[2] * out.ne[3];
    const int threads = 256;
    const int blocks  = static_cast<int>((elements + threads - 1) / threads);
    dflash2_selector_predecessors_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const std::int32_t*>(candidates.data),
        static_cast<const std::int32_t*>(anchors.data), static_cast<std::int32_t>(out.ne[0]),
        static_cast<std::int32_t>(out.ne[1]), static_cast<std::int32_t>(out.ne[2]),
        static_cast<std::int32_t*>(out.data));
}

} // namespace ninfer::ops::detail
