#pragma once

#include "core/pdl.cuh"

// ninfer::ops - split-KV GQA small-T attention, tensor-core partial kernel.
// Standalone from the int8 kernel (gqa_attention_decode_i8.cuh): shared scaffolding
// lives in gqa_attention_decode.cuh, but the body/append/load are not shared so the
// bf16 path can be tuned independently. Processes one KV head, one query-head
// subgroup, and one token tile; a reducer combines the split-local partials.
//
// Templated on a KV-source policy: GqaTcKVLinear stages bf16 pages with
// cp.async; GqaTcKVHq decodes each 32-key tile from the hq-e8-2b code/meta
// planes straight into the swizzled tile positions with the 8-lane cooperative
// row decoder (hq_decode_row_group), runs QK/PV on the same ldmatrix+mma path,
// and works in the codec's rotated frame: staged q rows are FWHT-rotated after
// landing (bf16 -> FWHT -> bf16, one rounding, same as the fill side) and each
// output row is un-rotated once before the partial stores, so the shared
// reducer combines original-frame partials unchanged. The fused append encodes
// the round's new K/V rows into the code planes (per-warp encoder scratch
// aliases the qkv tile, unused before q staging).

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"
#include "ops/kernel/hq_codec.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput, typename KvSource>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_bf16_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, KvSource kv,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    pdl::sync();
    static_assert(TokenTile >= 1 && TokenTile <= 8);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);
    static_assert(KvSource::hq == KvSource::rotated, "hq tiles only exist in the rotated frame");
    static_assert(!KvSource::rotated || kHqHeadDim == kGqaHeadDim);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = 32;
    constexpr int D       = kGqaHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKNt    = Bc / 8;
    constexpr int QKKs    = D / 16;
    constexpr int PVNt    = D / 8;
    constexpr int PVKs    = Bc / 16;
    // The BF16/I8 linear envelope ceiling (524288 keys) spans at most 98 pages in one 27B
    // split; the absolute U8-only envelope never stages pages here.
    constexpr int PageIds       = 128;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ std::int8_t signs_s[KvSource::rotated ? kGqaHeadDim : 1];
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
        // nullptr valid_columns = unmasked (the hq route shares one runtime-width
        // instantiation across masked and unmasked launches; the bf16 route never
        // passes null when Masked).
        if (valid_columns != nullptr) {
            const int remaining = valid_columns[batch] - column_begin;
            valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
        }    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
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
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) {
        // The hq route's public partials contract neutralizes inactive splits
        // (the shared reducer may read partials buffers that carry earlier,
        // non-zero contents); the bf16 route leaves them unwritten and relies
        // on the engine's zero-initialized partial workspace instead.
        if constexpr (KvSource::hq) { write_neutral(); }
        return;
    }

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

    if constexpr (KvSource::rotated) {
        hq_engine_signs_fill(signs_s);
        __syncthreads();
    }

    // Per-(batch) residual-window state: the side planes and ring-validity words
    // are slot-row indexed (layer views arrive pre-sliced; batch views offset by
    // the table row the kernel already selected).
    constexpr int kRingWords = static_cast<int>(kGqaHqRecentKeys) / 32;
    const std::uint32_t* ring_valid_row = nullptr;
    if constexpr (KvSource::hq) {
        if (kv.ring_valid != nullptr) {
            ring_valid_row =
                kv.ring_valid + static_cast<std::int64_t>(table_row) * kRingWords;
        }
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        if constexpr (KvSource::hq) {
            // Fused append in the hq route: every split encodes the new K/V rows whose
            // positions fall in its own key range before any block reads them (each key
            // row belongs to exactly one split, so the writer block is also the only
            // reader of those rows). (token, role) units are flattened over the warps
            // — one encode per unit, no duplicate work — and the per-warp encoder
            // scratch aliases the qkv tile, which is unused until q staging below.
            // With the residual window on, the same rows are dual-written exactly
            // (rotated bf16) into the side planes: the codec planes stay complete so
            // any side row can fall back to the codec path.
            static_assert(WarpsPerCta * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) *
                                  sizeof(float) <=
                              QkvRows * D * sizeof(__nv_bfloat16),
                          "per-warp append scratch must fit the qkv tile it aliases");
            float* append_scratch = reinterpret_cast<float*>(qkv_s);
            for (int unit = warp; unit < valid_tokens * 2; unit += Wc) {
                const int t         = unit >> 1;
                const bool role_v   = (unit & 1) != 0;
                const std::int32_t p = pos[t];
                if (p < split_start || p >= split_end) { continue; }
                float* u = append_scratch + warp * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
                std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
                const std::int64_t base =
                    static_cast<std::int64_t>(t) * Geometry::KVHeads * kGqaHeadDim;
                const __nv_bfloat16* src = (role_v ? input.v : input.k) + base +
                                           gqa_kv_new_index<Geometry>(kv_head, 0, 0);
                hq_encode_row_warp(src, signs_s, 0, u, syms,
                                   hq_row_codes_mut<Geometry>(role_v ? kv.codes_v : kv.codes_k,
                                                              block_table, kv_head, p),
                                   hq_row_meta_mut<Geometry>(role_v ? kv.meta_v : kv.meta_k,
                                                             block_table, kv_head, p),
                                   hq_dither_row_seed(kv_head, p, role_v));
                if (kv.residual_k != nullptr) {
                    hq_store_rotated_row_warp(
                        src, signs_s,
                        hq_residual_row<Geometry>(role_v ? kv.residual_v : kv.residual_k,
                                                  table_row, kv_head, p));
                    hq_ring_mark_valid(const_cast<std::uint32_t*>(ring_valid_row), p);
                }
            }
        } else {
            for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
                const int token = chunk / (D / 8);
                const int d     = (chunk - token * (D / 8)) * 8;
                const int p_tok = pos[token];
                if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                    p_tok < logical_capacity) {
                    const std::int64_t new_off = gqa_kv_new_index<Geometry>(kv_head, d, token);
                    const int lane             = tid & 31;
                    int physical_page = lane == 0 ? paged_kv_physical_page(block_table, p_tok) : 0;
                    physical_page     = __shfl_sync(FullMask, physical_page, 0);
                    const std::int64_t cache_off =
                        gqa_cache_index<Geometry>(physical_page, kv_head, d,
                                                  p_tok & kPagedKVPageMask);
                    store_vec(&kv.k[cache_off], load_vec<int4>(&input.k[new_off]));
                    store_vec(&kv.v[cache_off], load_vec<int4>(&input.v[new_off]));
                }
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();

    if constexpr (KvSource::rotated) {
        // Rotate the staged q rows into the codec frame before any QK MMA reads
        // them: bf16 -> FWHT(+signs) -> bf16, the same single rounding the fill
        // side applies to K/V. Zero-padded tail rows stay exactly zero.
        for (int row = warp; row < row_count; row += Wc) {
            float reg[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                reg[s] = __bfloat162float(qkv_s[row * D + gqa_small_t_tc_swz(row, s * 32 + lane)]);
            }
            hq_fwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                qkv_s[row * D + gqa_small_t_tc_swz(row, s * 32 + lane)] =
                    __float2bfloat16(reg[s]);
            }
        }
        __syncthreads();
    }

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
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
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
        if constexpr (KvSource::hq) {
            // Tile source: group-decode the K/V rows straight into the swizzled
            // tile positions. Under the XOR swizzle each lattice word's 8
            // outputs stay contiguous, so the decoder only remaps the word base
            // (chunk ^ (key row & 7)). Rows outside this split's key range are
            // zeroed so stale shared memory (possibly NaN) never reaches the
            // MMA path. 128 threads cover the 2 x 32 rows in four 16-row waves.
            // Sink and recent-ring rows come EXACT from the residual side planes
            // (one 16 B copy per lane chunk, same swizzle); ring slots whose
            // validity bit was cleared (rollback / prefix-restore trims) fall
            // back to the codec path.
#pragma unroll 1
            for (int slot = tid; slot < 2 * Bc * 8; slot += Threads) {
                const bool role_v = slot >= Bc * 8;
                const int key_l   = (slot >> 3) & (Bc - 1);
                const int lane8   = slot & 7;
                __nv_bfloat16* row_dst = (role_v ? v_s : k_s) + key_l * D;
                const int key = k0 + key_l;
                if (key >= split_start && key < split_end) {
                    const bool side_row =
                        kv.residual_k != nullptr &&
                        (key < static_cast<int>(kGqaHqSinkKeys) ||
                         (key >= window - static_cast<int>(kGqaHqRecentKeys) &&
                          hq_ring_slot_valid(ring_valid_row, key)));
                    if (side_row) {
                        const __nv_bfloat16* side = hq_residual_row<Geometry>(
                            role_v ? kv.residual_v : kv.residual_k, table_row, kv_head, key);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int chunk = ((lane8 * 4 + j) ^ (key_l & 7)) << 3;
                            store_vec(row_dst + chunk,
                                      load_vec<int4>(side + lane8 * 32 + j * 8));
                        }
                    } else {
                        hq_decode_row_group(
                            hq_row_codes<Geometry>(role_v ? kv.codes_v : kv.codes_k, block_table,
                                                  kv_head, key),
                            hq_row_meta<Geometry>(role_v ? kv.meta_v : kv.meta_k, block_table,
                                                 kv_head, key),
                            row_dst, lane8, key_l & 7, hq_dither_row_seed(kv_head, key, role_v));
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int chunk = ((lane8 * 4 + j) ^ (key_l & 7)) << 3;
                        store_vec(row_dst + chunk, make_int4(0, 0, 0, 0));
                    }
                }
            }
            __syncthreads();
        } else {
            if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
            }
            // Stage the bf16 K/V key tile with one cp.async wave (16B/thread, high MLP).
            // Current-step tokens come from k_new/v_new; tail slots are zeroed.
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
                const int key_l      = chunk / (D / 8);
                const int d          = (chunk - key_l * (D / 8)) * 8;
                const int key        = k0 + key_l;
                __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (key >= split_start && key < split_end) {
                    if constexpr (CacheInput::writes_cache) {
                        const int new_token = key - first_pos;
                        const bool from_new =
                            new_token >= 0 && new_token < valid_tokens && key >= first_pos;
                        if (from_new) {
                            const std::int64_t off =
                                gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                            ninfer::ops::cp_async<16>(k_dst, &input.k[off]);
                            ninfer::ops::cp_async<16>(v_dst, &input.v[off]);
                        } else {
                            const std::int64_t off = gqa_cache_index<Geometry>(
                                physical_page, kv_head, d, key & kPagedKVPageMask);
                            ninfer::ops::cp_async<16>(k_dst, &kv.k[off]);
                            ninfer::ops::cp_async<16>(v_dst, &kv.v[off]);
                        }
                    } else {
                        const std::int64_t off = gqa_cache_index<Geometry>(physical_page, kv_head,
                                                                           d,
                                                                           key & kPagedKVPageMask);
                        ninfer::ops::cp_async<16>(k_dst, &kv.k[off]);
                        ninfer::ops::cp_async<16>(v_dst, &kv.v[off]);
                    }
                } else {
                    store_vec(k_dst, make_int4(0, 0, 0, 0));
                    store_vec(v_dst, make_int4(0, 0, 0, 0));
                }
            }
            ninfer::ops::cp_commit();
            ninfer::ops::cp_wait<0>();
            __syncthreads();
        }

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
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
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
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
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
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
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
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }

    // MMA fragments hold each row in four-lane groups. Stage the final split-local
    // accumulator through shared memory so partial_acc is written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();

    if constexpr (KvSource::rotated) {
        // Partials are written in the original frame: un-rotate each output row
        // once (FP32 registers over the staged bf16 row, one extra rounding)
        // before the contiguous partial stores, so the shared reducer and the
        // other cache dtypes combine identical-frame partials.
        for (int row = warp; row < row_count; row += Wc) {
            float reg[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) { reg[s] = __bfloat162float(qkv_s[row * D + s * 32 + lane]); }
            hq_ifwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                qkv_s[row * D + s * 32 + lane] = __float2bfloat16(reg[s]);
            }
        }
        __syncthreads();
    }

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
    pdl::publish();
}

} // namespace ninfer::ops
