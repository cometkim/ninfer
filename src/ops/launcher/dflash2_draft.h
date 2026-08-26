#pragma once

// ninfer::ops::detail - launcher surface for the DFlash2 draft primitives.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_dynamic_conv_launch(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                                 std::int32_t side, std::int32_t block_size, Tensor& out,
                                 cudaStream_t stream);

void dflash2_selector_scores_launch(const Tensor& candidates, const Tensor& predecessor_ids,
                                    const Tensor& unary, const Tensor& hidden_proj,
                                    const Tensor& successor_rows, const Tensor& predecessor_rows,
                                    Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
