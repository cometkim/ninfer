#pragma once

#include "core/tensor.h"

#include "ninfer/ops/rope.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Fused Text attention-path Q/K preparation. For every (head, t) row x of q `[256,Hq,T]` and
 * k `[256,Hkv,T]`, the complete mathematical operation is
 *
 *   inv    = 1 / sqrt(sum_d x[d]^2 / 256 + eps)
 *   n[d]   = x[d] * inv * (weight[d] + 1)                  (offset weights)
 *   out    = rope(Text, positions, rotary_dim=64)(n)
 *
 * with the supplied pair-frequency table and ops::rope's angle profiles. The q side carries
 * `frequencies.attention_factor` squared on the rotated dims while k stays factor-free
 * (ops::rope's contract). Normalization and rotation form one semantic operation: there is no
 * observable BF16 materialization of n, so each output element is rounded to BF16 once. The
 * unrotated dims [64,256) therefore equal `rmsnorm(x)` exactly, while the rotated dims [0,64) may
 * differ from the sequential `rmsnorm -> rope` chain, which rounds n to BF16 before rotating, by
 * about one BF16 ulp of the rotated pair's magnitude. The independent oracle evaluates the
 * formula naively in FP64 from the represented BF16 inputs.
 *
 * Registered domain: head_dim 256, rotary_dim 64, Text positions I32 [T] (1-D) or [T,3]
 * (MRoPE; pair i uses axis i%3, matching ops::rope's Text modes), and the text head geometries
 * (Hq,Hkv) = (24,4) or (16,2); other profiles throw. q/k/q_out/k_out are contiguous BF16
 * `[256,Hq|Hkv,T]`, weights are contiguous BF16 [256], positions is contiguous sequential I32.
 * Inputs are read-only; q_out/k_out receive every element of both sides. The Op owns no
 * workspace or persistent state.
 */
void qk_norm_rope(const Tensor& q, const Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
                  float eps, const Tensor& positions, const RopeFrequencies& frequencies,
                  Tensor& q_out, Tensor& k_out, cudaStream_t stream);

} // namespace ninfer::ops
