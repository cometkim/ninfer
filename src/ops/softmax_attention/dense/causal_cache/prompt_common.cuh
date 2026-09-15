#pragma once

// Shared causal-attention dimensions and leaf PTX helpers used by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include <math_constants.h>
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptHeadDim = 256;

inline constexpr int kCausalPromptBr        = 64;
inline constexpr int kCausalPromptBc        = 64;
inline constexpr int kCausalPromptThreads   = 128;
inline constexpr int kCausalPromptSmemBytes = (kCausalPromptBr + 2 * kCausalPromptBc) *
                                              kCausalPromptHeadDim *
                                              static_cast<int>(sizeof(__nv_bfloat16));

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_prompt_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kCausalPromptHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ void causal_prompt_zero_output_rows(__nv_bfloat16* out, int q_head,
                                                               int row_begin, int row_end, int tid,
                                                               int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * kCausalPromptHeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / kCausalPromptHeadDim;
        const int d   = element - (row - row_begin) * kCausalPromptHeadDim;
        out[causal_prompt_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int causal_prompt_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_prompt_store_byte_swizzled(Byte* tile, int row, int d,
                                                                  Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off = (row * (kCausalPromptHeadDim / 2) + causal_prompt_swz(row, col_b16)) * 2 + byte;
    tile[off]     = code;
}

template <int Columns>
__device__ __forceinline__ int causal_prompt_p_swz(int row, int col) {
    if constexpr (Columns == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return causal_prompt_swz(row, col);
}

__device__ __forceinline__ unsigned causal_prompt_swz_addr(unsigned lane_base, unsigned ck,
                                                           unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}


// Key-split partial layout (the fork's WI-K1a): the i8 prompt kernel with split_count > 1
// stores per-(head, token, split) online-softmax state in the same tensor shapes the small-T
// route allocates ({kCausalPromptHeadDim, q_heads, width, splits} FP32 acc and
// {q_heads, width, splits} FP32 m/l). The formulas mirror the decode-side
// causal_partial_{acc,stat}_index; they are kept local so prefill TUs do not pull the decode
// kernel header.
template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_prompt_partial_acc_index(int q_head, int d, int token,
                                                                        int split, int width) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalPromptHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(width) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_prompt_partial_stat_index(int q_head, int token,
                                                                         int split, int width) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(width) * split);
}

// Merges the fixed-S key-split partials back into normalized bf16 rows. m = -inf marks a
// causally-empty split (or a dead output row); such rows emit zeros without reading acc.
template <typename Geometry>
__global__ __launch_bounds__(256) void causal_attention_prompt_reduce_kernel(
    const float* __restrict__ partial_acc, const float* __restrict__ partial_m,
    const float* __restrict__ partial_l, float scale, std::int32_t width,
    std::int32_t split_count, __nv_bfloat16* __restrict__ out) {
    constexpr float kLog2E = 1.4426950408889634074f;
    const int token        = static_cast<int>(blockIdx.x);
    const int q_head       = static_cast<int>(blockIdx.y);
    if (token >= width || q_head >= Geometry::QHeads) { return; }
    const int d = static_cast<int>(threadIdx.x);

    float m = -CUDART_INF_F;
    for (int split = 0; split < split_count; ++split) {
        m = fmaxf(m, partial_m[causal_prompt_partial_stat_index<Geometry>(q_head, token, split,
                                                                         width)]);
    }
    if (m == -CUDART_INF_F) {
        out[causal_prompt_q_index<Geometry>(q_head, d, token)] = __float2bfloat16(0.0f);
        return;
    }
    const float scale_l2 = scale * kLog2E;
    // l_s * exp2((m_s - m*) * scale * log2e): the split's share of the final normalization mass
    // (its partial acc is stored pre-normalized by its own l_s, so the merge re-weights by
    // exactly this).
    float l = 0.0f;
    float weights[4];
    for (int split = 0; split < split_count; ++split) {
        const float ms =
            partial_m[causal_prompt_partial_stat_index<Geometry>(q_head, token, split, width)];
        const float ls =
            partial_l[causal_prompt_partial_stat_index<Geometry>(q_head, token, split, width)];
        const float share = ms == -CUDART_INF_F ? 0.0f : ls * exp2f((ms - m) * scale_l2);
        weights[split]    = share;
        l += share;
    }
    float acc = 0.0f;
    for (int split = 0; split < split_count; ++split) {
        acc += partial_acc[causal_prompt_partial_acc_index<Geometry>(q_head, d, token, split,
                                                                     width)] *
               weights[split];
    }
    out[causal_prompt_q_index<Geometry>(q_head, d, token)] =
        __float2bfloat16(l > 0.0f ? acc / l : 0.0f);
}

} // namespace ninfer::ops
