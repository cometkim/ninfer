#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/// Fixed by the DFlash2 checkpoint contract: dynamic-conv group size.
inline constexpr std::int32_t kDflash2ConvGroupSize = 16;

/**
 * DFlash2 two-tap dynamic depthwise convolution over one draft block.
 *
 * For hidden size H, group size G=16, taps K=2, block size B (columns per lane),
 * and L lanes (hidden columns C = B*L), per lane l, block position p in [0,B),
 * and channel c:
 *
 *   g      = c / G
 *   w[k]   = dynamic[g + (H/G)*k + 2*(H/G)*side, p + B*l] + base[side, k, c]
 *   x[k]   = hidden[c, p - k + B*l]   (zero when p - k < 0)
 *   y[c, p + B*l] = w[0]*x[0] + w[1]*x[1]
 *
 * The dynamic coefficients are the `conv_proj` output rows (group-major, then
 * tap, then side); the static base is the checkpoint `conv_base` tensor
 * `[2, 2, H]` indexed `[side, tap, channel]`. The convolution is block-local:
 * taps never read across a lane boundary. `hidden`, `dynamic`, and `out` are
 * contiguous BF16 `[H, C]`, `[2*2*H/G, C]`, and `[H, C]`; `base` is contiguous
 * BF16 `[2, 2, H]`; side is 0 or 1; B is in [1, 64]; C is B*L with L in
 * [1, 8]. Inputs and output must not overlap. The oracle evaluates the formula
 * naively in FP64 from the represented inputs and the BF16 output is promoted
 * and compared directly; weight-sum and accumulation precision are
 * implementation choices. There is no workspace or persistent state effect.
 */
void dflash2_dynamic_conv(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                          std::int32_t side, std::int32_t block_size, Tensor& out,
                          cudaStream_t stream);

} // namespace ninfer::ops
