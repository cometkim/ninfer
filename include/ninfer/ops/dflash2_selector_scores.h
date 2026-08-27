#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * DFlash2 candidate-selector pairwise transition scores.
 *
 * For top-k candidates per draft position, rank R, P draft positions and L
 * lanes, with successor candidate ids `candidates[i, s, l]`, predecessor ids
 * `predecessor_ids[j, s, l]` (the caller places the anchor token id in every
 * j slot at s = 0), per-candidate unary logits `unary[i, s, l]`, per-position
 * projected hidden `hidden_proj[r, s, l]` (selector_hidden * h), and the
 * per-token codebook rows `successor_rows[id, r]` / `predecessor_rows[id, r]`:
 *
 *   score[i, j, s, l] =
 *       sum_r successor_rows[candidates[i,s,l], r]
 *                  * predecessor_rows[predecessor_ids[j,s,l], r]
 *                  * hidden_proj[r, s, l]
 *       + unary[i, s, l]
 *
 * `candidates`, `predecessor_ids` are contiguous I32 `[K, P, L]` with ids in
 * `[0, vocab)`; `unary` and `out` are contiguous FP32 `[K, P, L]` and
 * `[K, K, P, L]`; `hidden_proj` is contiguous FP32 `[R, P, L]`. The codebooks
 * are NVFP4 `Weight`s with logical shape `[vocab, R]` and the registered
 * BlockScaleK16M128x4 layout: gathered row values decode exactly as
 * `E2M1(code) * E4M3FN(group scale) / weight_scale_divisor`. R is a positive
 * multiple of 64 and vocab a positive multiple of 128. K is in [1, 32],
 * R in [16, 512], P in [1, 15], L in [1, 8]. Inputs and output must not
 * overlap. The oracle evaluates the formula naively in FP64 from the
 * represented inputs; codebook rows are decoded from the stored codes and
 * scales exactly. Accumulation order and precision are implementation
 * choices. There is no workspace or persistent state effect.
 */
void dflash2_selector_scores(const Tensor& candidates, const Tensor& predecessor_ids,
                             const Tensor& unary, const Tensor& hidden_proj,
                             const Weight& successor_rows, const Weight& predecessor_rows,
                             Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
