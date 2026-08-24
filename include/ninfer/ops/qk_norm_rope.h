#pragma once

#include "core/tensor.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Fused Text attention-path Q/K preparation. For every (head, t) row of q `[256,Hq,T]` and
 * k `[256,Hkv,T]`:
 *
 *   normed = rmsnorm(x, weight, eps) rounded to BF16   (offset weights: weight + 1)
 *   out    = rope(Text 1-D, positions, rotary_dim=64)(normed)
 *
 * with q rows carrying the rope q-side temperature (frequencies.attention_factor squared) and
 * k rows rotating unscaled, exactly as ops::rope applies them. The BF16 rounding between the
 * norm and the rotation is the sequential chain's global-memory boundary, so the fused kernel's
 * outputs are bit-identical to `rmsnorm(q) -> rmsnorm(k) -> rope(qn, kn)` over the same inputs.
 *
 * Registered domain: head_dim 256, rotary_dim 64, Text positions I32 [T] (1-D) or [T,3]
 * (MRoPE; pair i uses axis i%3, matching ops::rope's Text modes), and the text head geometries
 * (Hq,Hkv) = (24,4) or (16,2); other profiles throw. q/k/q_out/k_out are contiguous BF16
 * `[256,Hq|Hkv,T]`, weights are contiguous BF16 [256], positions is contiguous sequential I32.
 * Inputs are read-only; q_out/k_out receive every element of both sides.
 */
void qk_norm_rope(const Tensor& q, const Tensor& k, const Tensor& q_weight,
                  const Tensor& k_weight, float eps, const Tensor& positions,
                  const RopeFrequencies& frequencies, Tensor& q_out, Tensor& k_out,
                  cudaStream_t stream);

} // namespace ninfer::ops
