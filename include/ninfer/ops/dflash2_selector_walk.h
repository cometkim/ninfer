#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * DFlash2 selector path walk over the transition-score lattice.
 *
 * With candidate ids `candidates [K, P, L]` (I32, same shape as the scores Op) and the
 * pairwise transition scores `scores [K, K, P, L]` (score[i, j, s, l] is the
 * score of successor candidate i at draft slot s given predecessor candidate j
 * from slot s-1), one lane l walks greedily:
 *
 *   pred = 0
 *   for s in [0, P):
 *     i*   = argmax_i scores[i, pred, s, l]   (ties toward the smaller i)
 *     out[s, l] = candidates[i*, s, l]
 *     pred = i*
 *
 * `out` is I32 [P, L] — the selected draft token ids, one per draft slot.
 * The anchor id selects the predecessor lattice column at slot 0 through
 * `candidates[j, 0, l] == anchors[l]` by construction (the caller places the
 * anchor id there); the Op itself only consumes `scores`. K is in [1, 32],
 * P in [1, 15], L in [1, 8]. Inputs and output must not overlap. This is an
 * exact selection Op on the represented FP32 scores. There is no workspace or
 * persistent state effect.
 */
void dflash2_selector_walk(const Tensor& scores, const Tensor& candidates, Tensor& out,
                           cudaStream_t stream);

} // namespace ninfer::ops
