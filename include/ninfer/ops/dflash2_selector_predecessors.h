#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * DFlash2 selector predecessor assembly.
 *
 * Builds the predecessor-id lattice consumed by dflash2_selector_scores: with
 * top-k candidate ids `candidates [K, P, L]` (identical layout to that Op) and
 * anchor token ids `anchors [L]`,
 *
 *   predecessor_ids[j, s, l] = anchors[l]                when s == 0
 *                             = candidates[j, s-1, l]    when s > 0
 *
 * so every predecessor slot at draft position 0 reads the anchor's selector
 * codebook row while later positions pair against the previous position's
 * candidate set. `candidates`, `anchors`, and `out` are contiguous I32
 * `[K, P, L]`, `[L]`, and `[K, P, L]`; K is in [1, 32], P in [1, 15], L in
 * [1, 8]. Inputs and output must not overlap. The Op is an exact copy: every
 * output element is a represented input value, deterministically placed.
 * There is no workspace or persistent state effect.
 */
void dflash2_selector_predecessors(const Tensor& candidates, const Tensor& anchors, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops
