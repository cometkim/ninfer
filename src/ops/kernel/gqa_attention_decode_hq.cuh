#pragma once

// ninfer::ops - hq-e8-2b small-T split-KV partial kernel (decode/verify).
// Correctness-first slice mirroring the bf16/int8 partials' observable
// contract: identical partial tensor layout, identical device-side active
// split policy (the BF16 tiers), neutral partials for inactive splits, and
// the fused append of the current round's K/V rows by the owning split.
// Partials are written in the ORIGINAL frame (each row un-rotated once), so
// the shared reducer combines them unchanged.
//
// Deliberately NOT templated on width/batch/mask: the token tile, multi-batch
// and masking are runtime inputs. One instantiation per (Geometry,
// CacheInput) keeps the CUDA front-end's memory footprint bounded; the phased
// loops are width-agnostic.
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaHqDecodeThreads = 256;
inline constexpr int kGqaHqDecodeKeys    = 32;
// Shared-memory row capacity: the largest token tile (6) times the largest
// registered group size (8).
inline constexpr int kGqaHqDecodeMaxRows = 6 * 8;

template <typename Geometry, typename CacheInput>
__global__ void gqa_attention_decode_hq_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int32_t tokens,
    std::uint8_t* codes_k, std::uint8_t* codes_v, std::uint8_t* meta_k, std::uint8_t* meta_v,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    constexpr int kKeys = kGqaHqDecodeKeys;
    extern __shared__ float smem[];
    __nv_bfloat16* q_rot = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* kv_smem =
        q_rot + kGqaHqDecodeMaxRows * kGqaHeadDim;
    float* p_smem = reinterpret_cast<float*>(kv_smem + kKeys * kGqaHeadDim);
    float* m_smem = p_smem + kGqaHqDecodeMaxRows * kKeys;
    float* l_smem = m_smem + kGqaHqDecodeMaxRows;
    float* corr_smem = l_smem + kGqaHqDecodeMaxRows;
    float* acc = corr_smem + kGqaHqDecodeMaxRows;
    std::int8_t* signs = reinterpret_cast<std::int8_t*>(acc + kGqaHqDecodeMaxRows * kGqaHeadDim);

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int batch   = static_cast<int>(blockIdx.z);
    const int rows    = tokens * Geometry::GroupSize;

    // Batch/column addressing mirrors the bf16 small-T kernel: q, pos, and the
    // fused-append sources live in the invocation's full-width frame, and the
    // partial tensors are batch-major.
    const std::int64_t column_base =
        column_begin + static_cast<std::int64_t>(batch) * full_width;
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int split_count = static_cast<int>(gridDim.y);
    partial_acc +=
        static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens * split_count;
    partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;

    const std::int32_t row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(row) * table_stride;

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    // Out-of-range positions would index the block table out of bounds;
    // write neutral partials for every split, like the bf16 small-T kernel.
    const bool positions_out_of_range =
        first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity;
    const std::int32_t window    = last_pos + 1;
    std::int32_t columns = tokens;
    if (valid_columns != nullptr) {
        const std::int32_t remaining = valid_columns[batch] - column_begin;
        columns = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int active_splits =
        gqa_small_t_active_splits<Geometry, false>(window, gridDim.y, tokens);

    // Neutral partials for inactive splits (and empty batches).
    if (positions_out_of_range || split >= active_splits || columns <= 0) {
        for (int t = 0; t < tokens; ++t) {
            for (int g = 0; g < Geometry::GroupSize; ++g) {
                const int q_head = kv_head * Geometry::GroupSize + g;
                if (!gqa_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
                if (tid == 0) {
                    const std::int64_t si =
                        gqa_partial_stat_index<Geometry>(q_head, t, split, tokens);
                    partial_m[si] = -INFINITY;
                    partial_l[si] = 0.0f;
                }
                for (int d = tid; d < kGqaHeadDim; d += kGqaHqDecodeThreads) {
                    partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, t, split, tokens)] =
                        __float2bfloat16(0.0f);
                }
            }
        }
        return;
    }

    hq_engine_signs_fill(signs);
    for (int i = tid; i < rows * kGqaHeadDim; i += kGqaHqDecodeThreads) { acc[i] = 0.0f; }
    for (int r = 0; r < rows; ++r) {
        if (tid == 0) {
            m_smem[r] = -INFINITY;
            l_smem[r] = 0.0f;
        }
    }
    __syncthreads();

    // Rotate all q rows of this kv-head group into the codec frame.
    for (int r = warp; r < rows; r += kGqaHqDecodeThreads / 32) {
        const int token = r / Geometry::GroupSize;
        const int g     = r - token * Geometry::GroupSize;
        const int q_head = kv_head * Geometry::GroupSize + g;
        float reg[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            reg[s] = __bfloat162float(q[gqa_q_index<Geometry>(q_head, s * 32 + lane, token)]);
        }
        hq_fwht256_sign(reg, signs, 0, lane);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            q_rot[r * kGqaHeadDim + s * 32 + lane] = __float2bfloat16(reg[s]);
        }
    }
    __syncthreads();

    // Key range owned by this split (absolute cache positions over the whole
    // visible window, like the bf16 small-T kernel). The launch grid is sized
    // for the captured envelope upper bound; only the active splits own keys.
    const std::int32_t kps    = div_up(window, active_splits);
    const std::int32_t key_lo = split * kps;
    const std::int32_t key_hi = min(key_lo + kps, window);

    // Fused append: every split quantizes the new K/V rows whose positions
    // fall in its own key range before any block reads them (each key row
    // belongs to exactly one split, so the writer block is also the only
    // reader of those rows). (token, role) units are flattened across all
    // eight warps — one encode per unit, no duplicate work — and the per-warp
    // encode scratch lives in kv_smem, which is unused before the chunk loop
    // and is exactly 8 x (256 floats + 256 u32) = its full 16 KB.
    if constexpr (CacheInput::writes_cache) {
        float* append_scratch = reinterpret_cast<float*>(kv_smem);
        for (int unit = warp; unit < columns * 2;
             unit += kGqaHqDecodeThreads / 32) {
            const int t = unit >> 1;
            const bool role_v = (unit & 1) != 0;
            const std::int32_t p = pos[t];
            if (p < key_lo || p >= key_hi) { continue; }
            float* u =
                append_scratch + warp * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
            std::uint32_t* syms =
                reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
            const std::int64_t base =
                static_cast<std::int64_t>(t) * Geometry::KVHeads * kGqaHeadDim;
            const __nv_bfloat16* src = (role_v ? input.v : input.k) + base +
                                       gqa_kv_new_index<Geometry>(kv_head, 0, 0);
            hq_encode_row_warp(src, signs, 0, u, syms,
                               hq_row_codes_mut<Geometry>(role_v ? codes_v : codes_k,
                                                          block_table, kv_head, p),
                               hq_row_meta_mut<Geometry>(role_v ? meta_v : meta_k,
                                                         block_table, kv_head, p));
        }
        // One barrier orders every owned encode's cache writes (and the
        // scratch region's reuse as the decoded-key tile) before the chunk
        // loop reads them; the branch is block-uniform because columns,
        // key_lo, and key_hi are.
        __syncthreads();
    }

    for (std::int32_t k0 = key_lo; k0 < key_hi; k0 += kKeys) {
        const int chunk = min(kKeys, key_hi - k0);
        // Eight lanes per K row (one 64-bit Rice window each): the boundary
        // fixup chain replaces the 64-symbol serial segment walk, and all
        // 256 threads decode the chunk's rows in a single wave.
        {
            const int drow = tid >> 3;
            const int dseg = tid & 7;
            if (drow < chunk) {
                hq_decode_row_group(
                    hq_row_codes<Geometry>(codes_k, block_table, kv_head, k0 + drow),
                    hq_row_meta<Geometry>(meta_k, block_table, kv_head, k0 + drow),
                    kv_smem + static_cast<std::size_t>(drow) * kGqaHeadDim, dseg);
            }
        }
        __syncthreads();
        for (int i = tid; i < rows * kKeys; i += kGqaHqDecodeThreads) {
            const int r  = i / kKeys;
            const int kk = i - r * kKeys;
            const int token = r / Geometry::GroupSize;
            float s = 0.0f;
            if (kk < chunk && k0 + kk <= pos[token]) {
                const __nv_bfloat16* qr = q_rot + r * kGqaHeadDim;
                const __nv_bfloat16* kr = kv_smem + kk * kGqaHeadDim;
                for (int d = 0; d < kGqaHeadDim; ++d) {
                    s += __bfloat162float(qr[d]) * __bfloat162float(kr[d]);
                }
                s *= scale;
            } else {
                s = -INFINITY;
            }
            p_smem[i] = s;
        }
        __syncthreads();
        for (int r = tid; r < rows; r += kGqaHqDecodeThreads) {
            float m = m_smem[r];
            for (int kk = 0; kk < kKeys; ++kk) { m = fmaxf(m, p_smem[r * kKeys + kk]); }
            const float correction = (m_smem[r] == -INFINITY) ? 0.0f : __expf(m_smem[r] - m);
            float l = 0.0f;
            for (int kk = 0; kk < kKeys; ++kk) {
                const float e = (p_smem[r * kKeys + kk] == -INFINITY)
                                    ? 0.0f
                                    : __expf(p_smem[r * kKeys + kk] - m);
                p_smem[r * kKeys + kk] = e;
                l += e;
            }
            m_smem[r]    = m;
            l_smem[r]    = l_smem[r] * correction + l;
            corr_smem[r] = correction;
        }
        __syncthreads();
        for (int i = tid; i < rows * kGqaHeadDim; i += kGqaHqDecodeThreads) {
            const int r = i / kGqaHeadDim;
            acc[i] *= corr_smem[r];
        }
        __syncthreads();
        {
            const int drow = tid >> 3;
            const int dseg = tid & 7;
            if (drow < chunk) {
                hq_decode_row_group(
                    hq_row_codes<Geometry>(codes_v, block_table, kv_head, k0 + drow),
                    hq_row_meta<Geometry>(meta_v, block_table, kv_head, k0 + drow),
                    kv_smem + static_cast<std::size_t>(drow) * kGqaHeadDim, dseg);
            }
        }
        __syncthreads();
        for (int i = tid; i < rows * kGqaHeadDim; i += kGqaHqDecodeThreads) {
            const int r = i / kGqaHeadDim;
            const int d = i - r * kGqaHeadDim;
            float a = acc[r * kGqaHeadDim + d];
            const float* pr = p_smem + r * kKeys;
            for (int kk = 0; kk < chunk; ++kk) {
                a += pr[kk] * __bfloat162float(kv_smem[kk * kGqaHeadDim + d]);
            }
            acc[r * kGqaHeadDim + d] = a;
        }
        __syncthreads();
    }

    // Un-rotate each row once and write ORIGINAL-frame partials.
    for (int r = warp; r < rows; r += kGqaHqDecodeThreads / 32) {
        const int token = r / Geometry::GroupSize;
        const int g     = r - token * Geometry::GroupSize;
        const int q_head = kv_head * Geometry::GroupSize + g;
        if (!gqa_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
        float reg[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) { reg[s] = acc[r * kGqaHeadDim + s * 32 + lane]; }
        hq_ifwht256_sign(reg, signs, 0, lane);
        if (lane == 0) {
            const std::int64_t si = gqa_partial_stat_index<Geometry>(q_head, token, split, tokens);
            partial_m[si] = m_smem[r];
            partial_l[si] = l_smem[r];
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            partial_acc[gqa_partial_acc_index<Geometry>(q_head, s * 32 + lane, token, split,
                                                        tokens)] = __float2bfloat16(reg[s]);
        }
    }
}

template <typename Geometry>
inline constexpr std::size_t gqa_hq_decode_smem_bytes() {
    return (kGqaHqDecodeMaxRows + kGqaHqDecodeKeys) * kGqaHeadDim * sizeof(__nv_bfloat16) +
           kGqaHqDecodeMaxRows * kGqaHqDecodeKeys * sizeof(float) +
           3 * kGqaHqDecodeMaxRows * sizeof(float) +
           kGqaHqDecodeMaxRows * kGqaHeadDim * sizeof(float) + kHqHeadDim;
}

} // namespace ninfer::ops
