#pragma once

// hq-e8-2b split-KV causal small-T attention, tensor-core partial kernel. Structurally the
// BF16-K/BF16-V kernel (causal_attention_small_t.cuh scaffolding): every 32-key tile's K/V rows
// are group-decoded straight into the swizzled shared tile from the E8+Rice pages, the staged q
// rows are FWHT-rotated into the codec frame before QK, and the split-local accumulators are
// un-rotated once before the shared FP32 reducer consumes them, so all cache dtypes combine
// identical-frame partials. The fused append encodes each split's new K/V rows in-range with
// per-warp encoders aliasing the qkv tile (unused until q staging). One runtime-width
// instantiation per geometry covers every legal token width.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/append/hq_kernel.cuh"
#include "ops/kv_cache/hq_e8_rice_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void causal_attention_small_t_tc_partial_hq_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::uint8_t* codes_k,
    std::uint8_t* codes_v, std::uint8_t* meta_k, std::uint8_t* meta_v,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    float* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 48);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc            = WarpsPerCta;
    constexpr int Br            = Wc * 16;
    constexpr int Bc            = 32;
    constexpr int D             = kCausalHeadDim;
    constexpr int Threads       = Wc * 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);
    static_assert(WarpsPerCta * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * sizeof(float) <=
                      QkvRows * D * sizeof(__nv_bfloat16),
                  "per-warp append scratch must fit the qkv tile it aliases");

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ __align__(16) std::int8_t signs_s[kHqHeadDim];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    hq_engine_signs_fill(signs_s);
    __syncthreads();

    if constexpr (CacheInput::writes_cache) {
        // Fused append: every split encodes the new K/V rows whose positions fall in its own key
        // range before any block reads them (each key row belongs to exactly one split, so the
        // writer block is also the only reader). (token, role) units flatten over the warps and
        // the per-warp encoder scratch aliases the qkv tile, unused until q staging below.
        float* append_scratch = reinterpret_cast<float*>(qkv_s);
        for (int unit = warp; unit < valid_tokens * 2; unit += Wc) {
            const int t          = unit >> 1;
            const bool role_v    = (unit & 1) != 0;
            const std::int32_t p = pos[t];
            if (p < split_start || p >= split_end) { continue; }
            float* u = append_scratch + warp * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
            std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
            const __nv_bfloat16* src =
                (role_v ? input.v : input.k) + kv_cache_hq_src_index<Geometry>(kv_head, 0, t);
            hq_encode_row_warp(src, signs_s, 0, u, syms,
                               kv_cache_hq_row_codes_mut<Geometry>(role_v ? codes_v : codes_k,
                                                                   block_table, kv_head, p),
                               kv_cache_hq_row_meta_mut<Geometry>(role_v ? meta_v : meta_k,
                                                                  block_table, kv_head, p));
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[causal_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + causal_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    // Rotate the staged q rows into the codec frame before any QK MMA reads them: bf16 ->
    // FWHT(+signs) -> bf16, the same single rounding the fill side applies to K/V. Zero-padded
    // tail rows stay exactly zero.
    for (int row = warp; row < row_count; row += Wc) {
        float reg[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            reg[s] = __bfloat162float(qkv_s[row * D + causal_small_t_tc_swz(row, s * 32 + lane)]);
        }
        hq_fwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            qkv_s[row * D + causal_small_t_tc_swz(row, s * 32 + lane)] = __float2bfloat16(reg[s]);
        }
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + causal_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();
    int physical_page = physical_pages_s[0];
    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
        }
        // Tile source: group-decode the K/V rows straight into the swizzled tile positions.
        // Under the XOR swizzle each lattice word's 8 outputs stay contiguous, so the decoder
        // only remaps the word base (chunk ^ (key row & 7)). Rows outside this split's key
        // range are zeroed so stale shared memory (possibly NaN) never reaches the MMA path.
#pragma unroll 1
        for (int slot = tid; slot < 2 * Bc * 8; slot += Threads) {
            const bool role_v      = slot >= Bc * 8;
            const int key_l        = (slot >> 3) & (Bc - 1);
            const int lane8        = slot & 7;
            __nv_bfloat16* row_dst = (role_v ? v_s : k_s) + key_l * D;
            const int key          = k0 + key_l;
            if (key >= split_start && key < split_end) {
                hq_decode_row_group(kv_cache_hq_row_codes<Geometry>(role_v ? codes_v : codes_k,
                                                                    block_table, kv_head, key),
                                    kv_cache_hq_row_meta<Geometry>(role_v ? meta_v : meta_k,
                                                                   block_table, kv_head, key),
                                    row_dst, lane8, key_l & 7);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int chunk = ((lane8 * 4 + j) ^ (key_l & 7)) << 3;
                    store_vec(row_dst + chunk, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + causal_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        causal_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0] =
                (row0 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                    ? score[nt][3] * scale
                    : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col0)] = __float2bfloat16_rn(p00);
            p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col1)] = __float2bfloat16_rn(p01);
            p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col0)] =
                __float2bfloat16_rn(p10);
            p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col1)] =
                __float2bfloat16_rn(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(&p_sw[a_rowoff * Bc + causal_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + causal_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

    // Redistribute MMA fragments to full rows for the inverse rotation. Eight FP32 rows per
    // warp fit the dead BF16 K/V tile, so handle the two fragment row halves in turn. Preserve
    // FP32 through staging, rotation and partial stores: the shared reducer consumes floats.
    static_assert(Wc * 8 * D * sizeof(float) <= sizeof(qkv_s));
    float* acc_s = reinterpret_cast<float*>(qkv_s);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            const int d                         = n * 8 + 2 * lid;
            acc_s[(warp * 8 + gid) * D + d]     = acc[n][half * 2];
            acc_s[(warp * 8 + gid) * D + d + 1] = acc[n][half * 2 + 1];
        }
        __syncwarp();
        for (int r = 0; r < 8; ++r) {
            const int row = warp_row0 + half * 8 + r;
            if (row >= row_count) { break; }
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
            float reg[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) { reg[s] = acc_s[(warp * 8 + r) * D + s * 32 + lane]; }
            hq_ifwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, s * 32 + lane, token, split,
                                                               tokens)] = reg[s];
            }
        }
        __syncwarp();
    }
}

} // namespace ninfer::ops
