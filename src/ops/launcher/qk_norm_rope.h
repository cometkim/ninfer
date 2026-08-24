#pragma once

// ninfer::ops::detail - private launch prototype for qk_norm_rope. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.

#include "core/tensor.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void qk_norm_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_weight,
                         const Tensor& k_weight, float eps, const Tensor& positions,
                         const RopeFrequencies& frequencies, Tensor& q_out, Tensor& k_out,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
