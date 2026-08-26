#pragma once

// ninfer::ops - DFlash2 draft primitives: the two-tap dynamic depthwise
// convolution and the candidate-selector transition scores. Both are tiny
// block-local computations; one elementwise kernel and one warp-reduction
// kernel cover them without staging.

#include <cuda_bf16.h>

#include <cstdint>

#include "ninfer/ops/dflash2_dynamic_conv.h"

namespace ninfer::ops {

// One thread per (channel, column): y[c, t] = sum_k w[k] * x[c, t - k],
// zero-padded at the lane start. w[k] = dynamic[g + (H/G)*k + 2*(H/G)*side, t]
// + base[side, k, c] with G = kDflash2ConvGroupSize.

__global__ void dflash2_dynamic_conv_kernel(
    const __nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ dynamic,
    const __nv_bfloat16* __restrict__ base, std::int32_t hidden_size, std::int32_t columns,
    std::int32_t block_size, std::int32_t side, __nv_bfloat16* __restrict__ out) {
    const int groups = hidden_size / kDflash2ConvGroupSize;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < hidden_size * columns;
         idx += blockDim.x * gridDim.x) {
        const int c   = idx % hidden_size;
        const int t   = idx / hidden_size;
        const int p   = t % block_size;
        const int g   = c / kDflash2ConvGroupSize;
        float acc     = 0.0F;
        const int dy  = groups * 2 * side;
        // Fork layout: [hidden, columns] tensors store channel-fastest, so
        // element (c, t) lives at c + H*t; dynamic rows stride the same way.
        const std::int64_t dynamic_rows = static_cast<std::int64_t>(groups) * 4;
        for (int k = 0; k < 2; ++k) {
            const float w =
                __bfloat162float(dynamic[static_cast<std::int64_t>(dy + groups * k + g) +
                                            dynamic_rows * t]) +
                __bfloat162float(base[(static_cast<std::int64_t>(side) * 2 + k) * hidden_size + c]);
            float x = 0.0F;
            if (p - k >= 0) {
                x = __bfloat162float(
                    hidden[static_cast<std::int64_t>(c) + static_cast<std::int64_t>(hidden_size) *
                                                                 (t - k)]);
            }
            acc += w * x;
        }
        out[idx] = __float2bfloat16(acc);
    }
}

// One block per (i, j, s, l) score: warp threads stride the rank, one
// warp-level reduction finishes the dot; the unary term is added by thread 0.
inline constexpr int kDflash2SelectorThreads = 256;

__global__ void dflash2_selector_scores_kernel(
    const std::int32_t* __restrict__ candidates, const std::int32_t* __restrict__ pred_ids,
    const float* __restrict__ unary, const float* __restrict__ hidden_proj,
    const __nv_bfloat16* __restrict__ successor_rows,
    const __nv_bfloat16* __restrict__ predecessor_rows, std::int32_t rank, std::int32_t top_k,
    std::int32_t positions, std::int32_t lanes, float* __restrict__ out) {
    const int idx = blockIdx.x;  // one block per (i, j, s, l)
    const int l   = idx % lanes;
    const int s   = (idx / lanes) % positions;
    const int j   = (idx / (lanes * positions)) % top_k;
    const int i   = idx / (lanes * positions * top_k);

    const std::int32_t succ_id = candidates[(static_cast<std::int64_t>(i) * positions + s) * lanes + l];
    const std::int32_t pred_id = pred_ids[(static_cast<std::int64_t>(j) * positions + s) * lanes + l];
    const __nv_bfloat16* succ  = successor_rows + static_cast<std::int64_t>(succ_id) * rank;
    const __nv_bfloat16* pred  = predecessor_rows + static_cast<std::int64_t>(pred_id) * rank;
    const float* h             = hidden_proj + (static_cast<std::int64_t>(s) * lanes + l) * rank;

    float local = 0.0F;
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        local += __bfloat162float(succ[r]) * __bfloat162float(pred[r]) * h[r];
    }
    __shared__ float warp_sums[kDflash2SelectorThreads / 32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    if (lane == 0) { warp_sums[warp] = local; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0F;
        for (int w = 0; w < blockDim.x / 32; ++w) {
            total += warp_sums[w];
        }
        out[(((static_cast<std::int64_t>(i) * top_k + j) * positions + s) * lanes + l)] =
            total + unary[(static_cast<std::int64_t>(i) * positions + s) * lanes + l];
    }
}

} // namespace ninfer::ops

// Per-column top-k: one warp owns one column; each lane keeps a private
// running top-k (insertion over registers), then k warp-reduction rounds
// consume the best remaining head.
struct TopkEntry {
    float value;
    int row;
};

__device__ __forceinline__ bool topk_less(const TopkEntry& a, const TopkEntry& b) {
    return a.value > b.value || (a.value == b.value && a.row < b.row);
}

__global__ void dflash2_topk_kernel(const __nv_bfloat16* __restrict__ logits,
                                    std::int32_t rows, std::int32_t columns,
                                    std::int32_t k, std::int32_t* __restrict__ ids,
                                    __nv_bfloat16* __restrict__ values) {
    const int column = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (column >= columns) { return; }
    const int lane = threadIdx.x % 32;
    TopkEntry list[64];
    int count      = 0;
    for (int row = lane; row < rows; row += 32) {
        TopkEntry e{
            __bfloat162float(logits[static_cast<std::int64_t>(row) * columns + column]), row};
        if (count < k) {
            int pos = count++;
            while (pos > 0 && topk_less(e, list[pos - 1])) {
                list[pos] = list[pos - 1];
                --pos;
            }
            list[pos] = e;
        } else if (topk_less(e, list[k - 1])) {
            int pos = k - 1;
            while (pos > 0 && topk_less(e, list[pos - 1])) {
                list[pos] = list[pos - 1];
                --pos;
            }
            list[pos] = e;
        }
    }
    int cursor = 0;
    for (int slot = 0; slot < k; ++slot) {
        float my_v = -INFINITY;
        int my_r   = 0x7fffffff;
        if (cursor < count) {
            my_v = list[cursor].value;
            my_r = list[cursor].row;
        }
        // warp argmax on (value, smaller row); ties between equal (value,row)
        // cannot occur because rows are unique per column.
        int my_lane = lane;
        for (int offset = 16; offset > 0; offset /= 2) {
            const float ov = __shfl_down_sync(0xffffffffu, my_v, offset);
            const int or_  = __shfl_down_sync(0xffffffffu, my_r, offset);
            const int ol   = __shfl_down_sync(0xffffffffu, my_lane, offset);
            if (ov > my_v || (ov == my_v && or_ < my_r)) {
                my_v    = ov;
                my_r    = or_;
                my_lane = ol;
            }
        }
        my_v    = __shfl_sync(0xffffffffu, my_v, 0);
        my_r    = __shfl_sync(0xffffffffu, my_r, 0);
        my_lane = __shfl_sync(0xffffffffu, my_lane, 0);
        if (lane == 0) {
            ids[static_cast<std::int64_t>(slot) * columns + column] = my_r;
            values[static_cast<std::int64_t>(slot) * columns + column] = __float2bfloat16(my_v);
        }
        if (lane == my_lane) { ++cursor; }
    }
}

// One thread per lane walks the greedy argmax chain over the lattice and
// gathers the winning candidate token ids.
__global__ void dflash2_selector_walk_kernel(const float* __restrict__ scores,
                                             const std::int32_t* __restrict__ candidates,
                                             std::int32_t top_k, std::int32_t positions,
                                             std::int32_t lanes, std::int32_t* __restrict__ out) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= lanes) { return; }
    int pred = 0;
    for (int s = 0; s < positions; ++s) {
        int best_i   = 0;
        float best_v = -INFINITY;
        for (int i = 0; i < top_k; ++i) {
            const float v =
                scores[((static_cast<std::int64_t>(i) * top_k + pred) * positions + s) * lanes +
                       l];
            if (v > best_v) {
                best_v = v;
                best_i = i;
            }
        }
        out[static_cast<std::int64_t>(s) * lanes + l] =
            candidates[(static_cast<std::int64_t>(best_i) * positions + s) * lanes + l];
        pred = best_i;
    }
}
