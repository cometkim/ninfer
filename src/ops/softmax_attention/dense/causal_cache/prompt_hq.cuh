#pragma once

// hq-e8-2b prompt-scale kernels. The fill pass quantizes the current chunk's K/V rows into the
// fixed-budget E8+Rice pages (kv_cache_append_full_hq_kernel, shared with the append family).
// The scratch pass materializes the call's entire visible history once per attention call into
// contiguous rotated-frame BF16 scratch planes; the prompt attention itself is the shared FA2
// structure over that scratch, with query rows FWHT-rotated after staging and output rows
// un-rotated once in FP32 before the bf16 stores. Numerical contract: identical ideal attention
// oracle as the other cache dtypes, judged through the hq-e8-2b compute profile (rotated-frame
// decode, FP32 online softmax with the hq single-rounding boundaries).

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/append/hq_kernel.cuh"
#include "ops/kv_cache/hq_e8_rice_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

namespace ninfer::ops {

inline constexpr int kCausalPromptHqScratchThreads = 256;

// Decode the prompt call's entire visible history [0, positions[0] + valid) for every kv head
// and both roles into contiguous bf16 scratch rows in the ROTATED frame:
// scratch[role][kv_head][position][256]. EIGHT LANES per row (the cooperative group decoder);
// every query tile of the FA2 prompt kernel then reads decoded bf16 rows instead of each
// re-walking the serial Rice stream over its causal prefix. Sink and recent-ring rows come EXACT
// from the residual side planes when the feature is on (one 16 B copy per lane quarter-row);
// cleared ring slots fall back to the codec path. Rows of the CURRENT chunk are staged exact by
// causal_attention_prompt_hq_fresh_rotate_kernel (launched before this kernel on the same
// stream), so every prefill query sees its full in-chunk recent window exact and the ring serves
// the W keys before the chunk. `span` is the scratch row count per head (the execution envelope's
// key bound); the grid is sized for that bound and threads beyond the device-computed history
// length return immediately.
template <typename Geometry, typename Metadata>
__global__ void causal_attention_prompt_hq_scratch_kernel(
    const std::uint8_t* codes_k, const std::uint8_t* codes_v, const std::uint8_t* meta_k,
    const std::uint8_t* meta_v, Metadata metadata, const std::int32_t* __restrict__ positions,
    std::int32_t width, std::int32_t span, __nv_bfloat16* __restrict__ scratch_k,
    __nv_bfloat16* __restrict__ scratch_v, std::int32_t key_begin, std::int32_t band_rows,
    const __nv_bfloat16* residual_k, const __nv_bfloat16* residual_v,
    const std::uint32_t* side_words, bool has_fresh) {
    // Unary symbols are temporary decoder state, not observable scratch output.
    // Stage them on chip so only reconstructed BF16 rows reach device memory.
    __shared__ __align__(16)
        std::uint16_t symbols[(kCausalPromptHqScratchThreads / 8) * kHqHeadDim];
    const std::int32_t tid =
        static_cast<std::int32_t>(blockIdx.x) * static_cast<std::int32_t>(blockDim.x) +
        static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t keys_total = positions[0] + metadata.valid_tokens(width);
    // Banded decode: rows [0, count) of the scratch hold absolute keys
    // [key_begin, key_begin + count); count clamps to the visible window.
    const std::int32_t band_end = min(min(keys_total, key_begin + band_rows), span + key_begin);
    const std::int32_t count    = max(0, band_end - key_begin);
    const std::int32_t units    = count * Geometry::KVHeads * 2;
    const std::int32_t unit     = tid >> 3;
    if (unit >= units) { return; }
    const std::int32_t lane8 = tid & 7;
    const std::int32_t pos   = key_begin + unit / (Geometry::KVHeads * 2);
    const std::int32_t rem   = unit - (unit / (Geometry::KVHeads * 2)) * (Geometry::KVHeads * 2);
    const std::int32_t head  = rem >> 1;
    const bool role_v        = (rem & 1) != 0;
    const std::int32_t* table = metadata.block_table();
    __nv_bfloat16* dst        = (role_v ? scratch_v : scratch_k) +
                         (static_cast<std::int64_t>(head) * span + (pos - key_begin)) * kHqHeadDim;
    if (has_fresh && pos >= positions[0]) {
        // Fresh-chunk rows were staged exact by the fresh-rotate pass.
        return;
    }
    const std::int32_t slot = metadata.residual_slot();
    // With fresh coverage the ring serves the W keys BEFORE the chunk instead of the window tail.
    const std::int32_t ring_from =
        has_fresh ? positions[0] - static_cast<std::int32_t>(kCausalHqRecentKeys)
                  : keys_total - static_cast<std::int32_t>(kCausalHqRecentKeys);
    const std::uint32_t* words_row =
        residual_k != nullptr ? side_words + static_cast<std::int64_t>(slot) * kHqSideWords
                              : nullptr;
    const bool side_row =
        residual_k != nullptr &&
        ((pos < static_cast<std::int32_t>(kCausalHqSinkKeys) && hq_sink_rows_valid(words_row)) ||
         (pos >= ring_from && pos < keys_total && hq_ring_slot_valid(words_row, pos)));
    if (side_row) {
        const __nv_bfloat16* side =
            hq_residual_row<Geometry::KVHeads>(role_v ? residual_v : residual_k, slot, head, pos);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            store_vec(dst + lane8 * 32 + j * 8, load_vec<int4>(side + lane8 * 32 + j * 8));
        }
    } else {
        hq_decode_row_group<true>(
            kv_cache_hq_row_codes<Geometry>(role_v ? codes_v : codes_k, table, head, pos),
            kv_cache_hq_row_meta<Geometry>(role_v ? meta_v : meta_k, table, head, pos), dst, lane8,
            0, hq_dither_row_seed(head, pos, role_v), symbols + (threadIdx.x / 8) * kHqHeadDim);
    }
}

// Fresh-chunk rotate: prompt-phase exactness needs the chunk being prefilled, not just the ring.
// The ring holds the chunk's LAST W rows, so a query early in a wide chunk would otherwise see
// none of its own recent window exact. This companion pass rotates the CURRENT call's bf16 k/v
// rows (the tensors are in hand at the prompt route) straight into the scratch - one warp per
// (token, kv_head, role), the same single FWHT rounding as the dual-write - so every prefill
// query has its full in-chunk recent window exact, plus the ring for the W keys before the
// chunk. Warps beyond the valid chunk prefix return immediately.
template <typename Geometry, typename Metadata>
__global__ void causal_attention_prompt_hq_fresh_rotate_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata, std::int32_t width,
    std::int32_t span, __nv_bfloat16* __restrict__ scratch_k, __nv_bfloat16* __restrict__ scratch_v,
    std::int32_t key_begin, std::int32_t band_rows) {
    extern __shared__ std::int8_t signs_raw[];
    hq_engine_signs_fill(reinterpret_cast<std::int8_t*>(signs_raw));
    __syncthreads();
    const std::int32_t valid = metadata.valid_tokens(width);
    const int warp = static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const std::int64_t units = static_cast<std::int64_t>(valid) * Geometry::KVHeads * 2;
    if (warp >= units) { return; }
    const int token   = static_cast<int>(warp % valid);
    const int unit    = static_cast<int>(warp / valid);
    const int head    = unit % Geometry::KVHeads;
    const bool role_v = unit / Geometry::KVHeads != 0;
    const std::int32_t pos = positions[0] + token;
    if (pos < key_begin || pos >= key_begin + band_rows || pos >= span + key_begin) { return; }
    const __nv_bfloat16* src =
        (role_v ? v : k) + kv_cache_hq_src_index<Geometry>(head, 0, token);
    __nv_bfloat16* dst = (role_v ? scratch_v : scratch_k) +
                         (static_cast<std::int64_t>(head) * span + (pos - key_begin)) * kHqHeadDim;
    hq_store_rotated_row_warp(src, reinterpret_cast<std::int8_t*>(signs_raw), dst);
}

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous rotated scratch into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always drops) are
// zeroed so the padded scratch tail never feeds NaNs into the tensor cores. Banded staging is
// band-local: absolute key k0 addresses scratch row k0 - key_base, and rows past key_limit are
// beyond the decoded band - zero-fill them.
template <typename Geometry>
__device__ __forceinline__ void
causal_prompt_hq_stage_kv(__nv_bfloat16* dst, const __nv_bfloat16* scratch, int kv_head, int k0,
                          int max_query_abs, std::int32_t span, int tid, int key_base = 0,
                          int key_limit = 0x7fffffff) {
    constexpr int D         = kCausalPromptHeadDim;
    constexpr int Bc        = kCausalPromptBc;
    constexpr int Threads   = kCausalPromptThreads;
    constexpr int VecPerRow = D / 8; // 8 bf16 per 16B cp.async
    const int bound         = min(max_query_abs, key_limit);
    const bool full_tile    = (k0 + Bc - 1) <= bound;
    const __nv_bfloat16* scratch_block =
        scratch + (static_cast<std::int64_t>(kv_head) * span + k0 - key_base) * D;
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;
            const int d      = (chunk & 31) << 3;
            __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, d)];
            cp_async<16, Cache::cg>(p, &scratch_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;
            const int d      = (chunk & 31) << 3;
            __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, d)];
            if ((k0 + key_l) <= bound) {
                cp_async<16, Cache::cg>(p, &scratch_block[key_l * D + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
}

// FlashAttention-2 forward over the rotated scratch, one CTA per (query 64-row block, query
// head). Grid is (ceil(tokens/Br), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens,
// bottom-right causal alignment. Query rows are FWHT-rotated after staging; output rows are
// un-rotated once in FP32 in the epilogue. PV runs in BF16 (P and scratch V are both BF16).
//
// Carry (banded hq prompt route): the kernel covers only keys [key_begin, key_end) of the
// scratch, starts its online-softmax state from the carried (m, l, acc) buffers when key_begin
// > 0, and writes the state back unnormalized instead of normalizing into `out` when
// store_carry is set. The buffers use `out`'s [head_dim, q_heads, width] layout for acc and
// [q_heads, width] for m/l; acc crosses band boundaries as bf16. Single-band launches pass the
// defaults and stay on the original path.
template <typename Geometry, typename Metadata, bool Carry = false>
__launch_bounds__(kCausalPromptThreads, 1) __global__
    void causal_attention_prompt_hq_kernel(const __nv_bfloat16* __restrict__ q,
                                           const __nv_bfloat16* __restrict__ scratch_k,
                                           const __nv_bfloat16* __restrict__ scratch_v,
                                           Metadata metadata,
                                           const std::int32_t* __restrict__ positions, float scale,
                                           __nv_bfloat16* __restrict__ out, std::int32_t width,
                                           std::int32_t span, std::int32_t key_begin = 0,
                                           std::int32_t key_end   = 0x7fffffff,
                                           __nv_bfloat16* __restrict__ carry_acc = nullptr,
                                           float* __restrict__ carry_m           = nullptr,
                                           float* __restrict__ carry_l           = nullptr,
                                           std::int32_t store_carry              = 0) {
    constexpr int D             = kCausalPromptHeadDim; // 256
    constexpr int Br            = kCausalPromptBr;      // 64 query rows
    constexpr int Bc            = kCausalPromptBc;      // 64 key cols
    constexpr int Threads       = kCausalPromptThreads; // 128
    constexpr int QKNt          = Bc / 8;               // 8  QK score n-tiles
    constexpr int QKKs          = D / 16;               // 16 QK contraction steps
    constexpr int PVNt          = D / 8;                // 32 PV output n-tiles
    constexpr int PVKs          = Bc / 16;              // 4  PV contraction steps
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 128);
    if constexpr (!Carry) {
        (void)key_begin;
        (void)key_end;
        (void)carry_acc;
        (void)carry_m;
        (void)carry_l;
        (void)store_carry;
    }

    extern __shared__ __align__(16) __nv_bfloat16 causal_smem[];
    __nv_bfloat16* q_s = causal_smem;                                  // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D;                                 // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D;                                 // [Bc, D] swizzled
    std::int8_t* signs = reinterpret_cast<std::int8_t*>(v_s + Bc * D); // [D] RHT diagonal

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 Threads);
        return;
    }
    const int base_pos = positions[0];

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    const unsigned q_sbase     = smem_addr(q_s);
    const unsigned k_sbase     = smem_addr(k_s);
    const unsigned v_sbase     = smem_addr(v_s);
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as        = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r         = static_cast<unsigned>(b_rin << 4);

    hq_engine_signs_fill(signs);
    __syncthreads();

    // Stage Q into smem once via cp.async; it stays resident for the whole key loop.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads;
        const __nv_bfloat16* q_block = q + causal_prompt_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + causal_prompt_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + causal_prompt_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }
    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_limit     = Carry ? min(key_end, max_query_abs + 1) : (max_query_abs + 1);
    const int kb_begin      = Carry ? (key_begin / Bc) : 0;
    const int n_block_max   = (key_limit + Bc - 1) / Bc; // tiles [kb_begin, n_block_max)

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    if constexpr (Carry) {
        if (key_begin > 0) {
            // Resume the online softmax from the previous band's state. The carried acc is
            // stored un-rotated (out frame): load each row, FWHT-rotate into the codec frame,
            // stage through the same 8-row smem window the epilogue uses, and rebuild the C
            // fragments. Runs before Q/K(0) staging, so k_s is dead storage here.
            hq_engine_signs_fill(signs);
            __syncthreads();
            float* epi = reinterpret_cast<float*>(k_s) + warp * 8 * D;
            for (int half = 0; half < 2; ++half) {
                for (int i = 0; i < 8; ++i) {
                    const int qrow = q0 + warp_row0 + half * 8 + i;
                    float reg[8];
                    if (qrow < tokens) {
#pragma unroll
                        for (int si = 0; si < 8; ++si) {
                            reg[si] = __bfloat162float(
                                carry_acc[causal_prompt_q_index<Geometry>(q_head, si * 32 + lane,
                                                                         qrow)]);
                        }
                        hq_fwht256_sign(reg, signs, 0, lane);
                    } else {
#pragma unroll
                        for (int si = 0; si < 8; ++si) { reg[si] = 0.0f; }
                    }
#pragma unroll
                    for (int si = 0; si < 8; ++si) { epi[i * D + si * 32 + lane] = reg[si]; }
                }
                __syncwarp(FullMask);
#pragma unroll
                for (int n = 0; n < PVNt; ++n) {
                    const int d0 = n * 8 + 2 * lid;
                    if (half == 0) {
                        acc[n][0] = epi[gid * D + d0 + 0];
                        acc[n][1] = epi[gid * D + d0 + 1];
                    } else {
                        acc[n][2] = epi[gid * D + d0 + 0];
                        acc[n][3] = epi[gid * D + d0 + 1];
                    }
                }
                __syncwarp(FullMask);
            }
            // l runs as per-lane partials inside the key loop (the epilogue's warp_sum<4>
            // completes the row), so the full carried l splits evenly across the row's four
            // lanes; m is row-uniform after each tile's warp_max.
            const int row0_c = q0 + warp_row0 + gid;
            const int row1_c = row0_c + 8;
            if (row0_c < tokens) {
                const std::int64_t stat = static_cast<std::int64_t>(q_head) * width + row0_c;
                m0                      = carry_m[stat];
                l0                      = 0.25f * carry_l[stat];
            }
            if (row1_c < tokens) {
                const std::int64_t stat = static_cast<std::int64_t>(q_head) * width + row1_c;
                m1                      = carry_m[stat];
                l1                      = 0.25f * carry_l[stat];
            }
            __syncthreads();
        }
    }

    const float scale_l2 = scale * Log2E;

    ninfer::ops::cp_commit();
    causal_prompt_hq_stage_kv<Geometry>(k_s, scratch_k, kv_head, kb_begin * Bc, max_query_abs,
                                        span, tid, Carry ? key_begin : 0,
                                        Carry ? key_end : 0x7fffffff);
    ninfer::ops::cp_commit();

    // Rotate the staged q rows into the codec frame before any QK MMA reads them: bf16 ->
    // FWHT(+signs) -> bf16, the same single rounding the fill side applies to K/V. Zero-padded
    // tail rows stay exactly zero. (K(0) drains early with Q; the loop re-waits, which is free.)
    ninfer::ops::cp_wait<0>();
    __syncthreads();
    for (int row = warp; row < Br; row += Threads / 32) {
        float reg[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            reg[s] = __bfloat162float(q_s[row * D + causal_prompt_swz(row, s * 32 + lane)]);
        }
        hq_fwht256_sign(reg, signs, 0, lane);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            q_s[row * D + causal_prompt_swz(row, s * 32 + lane)] = __float2bfloat16(reg[s]);
        }
    }
    __syncthreads();


    for (int kb = kb_begin; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;

        ninfer::ops::cp_wait<0>(); // K(kb) landed
        __syncthreads();

        causal_prompt_hq_stage_kv<Geometry>(v_s, scratch_v, kv_head, k0, max_query_abs, span, tid,
                                            Carry ? key_begin : 0, Carry ? key_end : 0x7fffffff);
        ninfer::ops::cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        causal_prompt_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096),
                                                   0u, k_as, k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            causal_prompt_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                               k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1            = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile =
            (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0)) &&
            (!Carry || (k0 + Bc) <= key_end);

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const bool key0_ok = key0 <= qabs0 && (!Carry || key0 < key_end);
                const bool key1_ok = key1 <= qabs0 && (!Carry || key1 < key_end);
                score[nt][0]   = (qrow0 < tokens && key0_ok) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1_ok) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2] =
                    (qrow1 < tokens && key0 <= qabs1 && (!Carry || key0 < key_end))
                        ? score[nt][2]
                        : -CUDART_INF_F;
                score[nt][3] =
                    (qrow1 < tokens && key1 <= qabs1 && (!Carry || key1 < key_end))
                        ? score[nt][3]
                        : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0     = exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     = exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        ninfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s.
        __syncthreads();

        if (kb + 1 < n_block_max) {
            causal_prompt_hq_stage_kv<Geometry>(k_s, scratch_k, kv_head, (kb + 1) * Bc,
                                                max_query_abs, span, tid, Carry ? key_begin : 0,
                                                Carry ? key_end : 0x7fffffff);
            ninfer::ops::cp_commit();
        }

        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        {
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          causal_prompt_swz_addr(v_lane_base, 0u, v_as, v_r));
        }
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              causal_prompt_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                     ckv, v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);

    // Un-rotate each output row once in FP32, then normalize and store bf16. Each warp stages
    // its 16 rows through a private 8-row FP32 window in the (now dead) K staging buffer - half
    // the rows per pass - so the whole row is visible to the warp's inverse FWHT. Rows past
    // `tokens` rotate with garbage accumulators but only the final store is predicated, keeping
    // every shuffle converged.
    float* epi = reinterpret_cast<float*>(k_s) + warp * 8 * D;
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            const int d0 = n * 8 + 2 * lid;
            if (half == 0) {
                epi[gid * D + d0 + 0] = acc[n][0];
                epi[gid * D + d0 + 1] = acc[n][1];
            } else {
                epi[gid * D + d0 + 0] = acc[n][2];
                epi[gid * D + d0 + 1] = acc[n][3];
            }
        }
        __syncwarp(FullMask);
        for (int i = 0; i < 8; ++i) {
            // Row l lives on lanes [4i, 4i+4) of the C fragment; after the 4-lane butterfly
            // reduction any of them holds the full sum.
            const float lrow  = __shfl_sync(FullMask, half == 0 ? l0 : l1, i << 2);
            const float mrow  = __shfl_sync(FullMask, half == 0 ? m0 : m1, i << 2);
            const float inv_l = (lrow > 0.0f) ? __frcp_rn(lrow) : 0.0f;
            float reg[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) { reg[s] = epi[i * D + s * 32 + lane]; }
            hq_ifwht256_sign(reg, signs, 0, lane);
            const int qrow = q0 + warp_row0 + half * 8 + i;
            if (qrow < tokens) {
                if constexpr (Carry) {
                    if (store_carry != 0) {
                        // Hand the unnormalized un-rotated accumulator and the running softmax
                        // state to the next band; the row's lanes [4i,4i+4) agree on m/l, so
                        // one of them writes.
#pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            carry_acc[causal_prompt_q_index<Geometry>(q_head, s * 32 + lane,
                                                                     qrow)] =
                                __float2bfloat16(reg[s]);
                        }
                        if (lane == (i << 2)) {
                            const std::int64_t stat =
                                static_cast<std::int64_t>(q_head) * width + qrow;
                            carry_m[stat] = mrow;
                            carry_l[stat] = lrow;
                        }
                    } else {
#pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            out[causal_prompt_q_index<Geometry>(q_head, s * 32 + lane, qrow)] =
                                __float2bfloat16(reg[s] * inv_l);
                        }
                    }
                } else {
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        out[causal_prompt_q_index<Geometry>(q_head, s * 32 + lane, qrow)] =
                            __float2bfloat16(reg[s] * inv_l);
                    }
                }
            }
        }
        __syncwarp(FullMask);
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             Threads);
}

} // namespace ninfer::ops
