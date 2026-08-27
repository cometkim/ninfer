// Kernel-level timing for the engine hq kernels at realistic decode/prefill
// shapes, isolating kernel cost from engine scheduling.
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <vector>

#include "ops/kernel/gqa_attention_decode_hq.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"

using namespace ninfer::ops;

namespace {

constexpr int kRows = 262144;  // one 262k-context head-worth of rows
constexpr int kKvHeads = 4;

__global__ void gen_rows(__nv_bfloat16* out, unsigned long long seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kRows * kKvHeads * kHqHeadDim / 4;
    if (i >= total) { return; }
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    const float4 u = curand_uniform4(&st);
    const float r1 = sqrtf(-2.0f * logf(u.x));
    const float r2 = sqrtf(-2.0f * logf(u.z));
    out[i * 4 + 0] = __float2bfloat16(r1 * cosf(6.2831853f * u.y));
    out[i * 4 + 1] = __float2bfloat16(r1 * sinf(6.2831853f * u.y));
    out[i * 4 + 2] = __float2bfloat16(r2 * cosf(6.2831853f * u.w));
    out[i * 4 + 3] = __float2bfloat16(r2 * sinf(6.2831853f * u.w));
}

__global__ void identity_table(std::int32_t* table, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { table[i] = i; }
}

__global__ void decode_rows_group_bench(const std::uint8_t* codes, const std::uint8_t* meta,
                                        __nv_bfloat16* out, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = i >> 3;
    if (row >= n_rows) { return; }
    hq_decode_row_group(codes + static_cast<std::size_t>(row) * kHqRowBudgetBytes,
                        meta + static_cast<std::size_t>(row) * kHqMetaBytes,
                        out + static_cast<std::size_t>(row) * kHqHeadDim, i & 7, 0,
                        hq_dither_row_seed(0, row, false));
}

// Partial-block barrier primitives for the warp-specialized variant:
// producers arrive without waiting, consumers sync (the canonical
// producer/consumer handshake over named barriers).
__device__ __forceinline__ void bar_partial_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_partial_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// One consumer tile: QK mma -> causal/range mask -> online softmax -> p_sw
// staging -> PV mma. Shared by the V1/V2 phase-bench variants (27B geometry).
struct HqPhaseCtx {
    int kv_head;
    int split_start;
    int split_end;
    int row_count;
    float scale;
    const std::int32_t* pos;
};

__device__ __forceinline__ void hq_phase_consumer_tile(const unsigned (&af_q)[16][4],
                                                       __nv_bfloat16* p_sw,
                                                       const __nv_bfloat16* k_s,
                                                       const __nv_bfloat16* v_s, int k0,
                                                       const HqPhaseCtx& c, int warp, int lane,
                                                       float& m0, float& m1, float& l0, float& l1,
                                                       float (&acc)[32][4]) {
    constexpr int Bc = 32, D = 256;
    constexpr int QKNt = 4, QKKs = 16, PVNt = 32, PVKs = 2;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + (((lane >> 3) & 1) << 3);
    const int a_coloff  = ((lane >> 3) >> 1) << 3;
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    float score[QKNt][4];
#pragma unroll
    for (int nt = 0; nt < QKNt; ++nt) {
        score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            unsigned bf[2];
            const int brow = nt * 8 + b_rin;
            const int bcol = k * 16 + b_koff;
            ldmatrix_x2(bf[0], bf[1], smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
            mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                     af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
        }
    }
    const int row0  = warp_row0 + gid;
    const int row1  = row0 + 8;
    const int qabs0 = (row0 < c.row_count) ? c.pos[row0 / 6] : -1;
    const int qabs1 = (row1 < c.row_count) ? c.pos[row1 / 6] : -1;
    float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < QKNt; ++nt) {
        const int col0 = nt * 8 + 2 * lid;
        const int col1 = col0 + 1;
        const int key0 = k0 + col0;
        const int key1 = k0 + col1;
        score[nt][0]   = (row0 < c.row_count && key0 >= c.split_start && key0 < c.split_end &&
                          key0 <= qabs0)
                             ? score[nt][0] * c.scale
                             : -INFINITY;
        score[nt][1]   = (row0 < c.row_count && key1 >= c.split_start && key1 < c.split_end &&
                          key1 <= qabs0)
                             ? score[nt][1] * c.scale
                             : -INFINITY;
        score[nt][2]   = (row1 < c.row_count && key0 >= c.split_start && key0 < c.split_end &&
                          key0 <= qabs1)
                             ? score[nt][2] * c.scale
                             : -INFINITY;
        score[nt][3]   = (row1 < c.row_count && key1 >= c.split_start && key1 < c.split_end &&
                          key1 <= qabs1)
                             ? score[nt][3] * c.scale
                             : -INFINITY;
        bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
        bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
    }
    bm0              = warp_max<4>(bm0, FullMask);
    bm1              = warp_max<4>(bm1, FullMask);
    const float nm0  = fmaxf(m0, bm0);
    const float nm1  = fmaxf(m1, bm1);
    const float alpha0 = (m0 == -INFINITY) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
    const float alpha1 = (m1 == -INFINITY) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);
    float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
    for (int nt = 0; nt < QKNt; ++nt) {
        const int col0  = nt * 8 + 2 * lid;
        const int col1  = col0 + 1;
        const float p00 = (nm0 > -INFINITY && score[nt][0] > -INFINITY)
                              ? exp2_approx((score[nt][0] - nm0) * Log2E)
                              : 0.0f;
        const float p01 = (nm0 > -INFINITY && score[nt][1] > -INFINITY)
                              ? exp2_approx((score[nt][1] - nm0) * Log2E)
                              : 0.0f;
        const float p10 = (nm1 > -INFINITY && score[nt][2] > -INFINITY)
                              ? exp2_approx((score[nt][2] - nm1) * Log2E)
                              : 0.0f;
        const float p11 = (nm1 > -INFINITY && score[nt][3] > -INFINITY)
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
    l0  = l0 * alpha0 + bl0;
    l1  = l1 * alpha1 + bl1;
    m0  = nm0;
    m1  = nm1;
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
}

// Bench-local copy of the tensor-core hq decode kernel with runtime phase
// switches (uniform across the block, so no divergence is introduced): skips
// the group-decode tile staging (zero-filled instead) and/or the whole
// QK/softmax/PV MMA chain at the engine shape.
__global__ void hq_decode_phase_bench_kernel(
    const __nv_bfloat16* q, const std::int32_t* pos, std::int32_t tokens,
    const std::uint8_t* codes_k, const std::uint8_t* codes_v, const std::uint8_t* meta_k,
    const std::uint8_t* meta_v, const std::int32_t* block_tables, std::int32_t table_stride,
    std::int32_t column_begin, float scale, __nv_bfloat16* partial_acc, float* partial_m,
    float* partial_l, int do_decode, int do_mma) {
    using Geometry = Gqa27Geometry;
    constexpr int Wc = 4, Br = 64, Bc = 32, D = 256, Threads = 128;
    constexpr int QKNt = 4, QKKs = 16, PVNt = 32, PVKs = 2;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    __shared__ __align__(16) __nv_bfloat16 qkv_s[2 * Bc * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ std::int32_t physical_pages_s[64];
    __shared__ std::int8_t signs_s[256];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int rows    = tokens * Geometry::GroupSize;

    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_begin;
    pos += column_begin;
    const std::int32_t* block_table = block_tables;

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    const std::int32_t window    = last_pos + 1;
    const int active_splits =
        gqa_small_t_active_splits<Geometry, false>(window, gridDim.y, tokens);
    if (split >= active_splits) { return; }

    hq_engine_signs_fill(signs_s);
    __syncthreads();

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head = 0, token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < rows && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();
    for (int row = warp; row < rows; row += Wc) {
        float reg[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            reg[s] = __bfloat162float(qkv_s[row * D + gqa_small_t_tc_swz(row, s * 32 + lane)]);
        }
        hq_fwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            qkv_s[row * D + gqa_small_t_tc_swz(row, s * 32 + lane)] = __float2bfloat16(reg[s]);
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
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_splits;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_splits) : div_up(window, active_splits);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    const int first_tile  = (split_start / Bc) * Bc;
    const int key_blocks  = div_up(split_end - first_tile, Bc);

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        for (int slot = tid; slot < 2 * Bc * 8; slot += Threads) {
            const bool role_v = slot >= Bc * 8;
            const int key_l   = (slot >> 3) & (Bc - 1);
            const int lane8   = slot & 7;
            __nv_bfloat16* row_dst = (role_v ? v_s : k_s) + key_l * D;
            const int key = k0 + key_l;
            if (do_decode && key >= split_start && key < split_end) {
                hq_decode_row_group(
                    hq_row_codes<Geometry>(role_v ? codes_v : codes_k, block_table, kv_head, key),
                    hq_row_meta<Geometry>(role_v ? meta_v : meta_k, block_table, kv_head, key),
                    row_dst, lane8, key_l & 7, hq_dither_row_seed(kv_head, key, role_v));
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int chunk = ((lane8 * 4 + j) ^ (key_l & 7)) << 3;
                    store_vec(row_dst + chunk, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();
        if (do_mma) {
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
            const int qabs0 = (row0 < rows) ? pos[token0] : -1;
            const int qabs1 = (row1 < rows) ? pos[token1] : -1;
            float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = col1 + k0;
                score[nt][0] = (row0 < rows && key0 >= split_start && key0 < split_end &&
                                key0 <= qabs0)
                                   ? score[nt][0] * scale
                                   : -INFINITY;
                score[nt][1] = (row0 < rows && key1 >= split_start && key1 < split_end &&
                                key1 <= qabs0)
                                   ? score[nt][1] * scale
                                   : -INFINITY;
                score[nt][2] = (row1 < rows && key0 >= split_start && key0 < split_end &&
                                key0 <= qabs1)
                                   ? score[nt][2] * scale
                                   : -INFINITY;
                score[nt][3] = (row1 < rows && key1 >= split_start && key1 < split_end &&
                                key1 <= qabs1)
                                   ? score[nt][3] * scale
                                   : -INFINITY;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = (m0 == -INFINITY) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = (m1 == -INFINITY) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);
            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = (nm0 > -INFINITY && score[nt][0] > -INFINITY)
                                      ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                      : 0.0f;
                const float p01 = (nm0 > -INFINITY && score[nt][1] > -INFINITY)
                                      ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                      : 0.0f;
                const float p10 = (nm1 > -INFINITY && score[nt][2] > -INFINITY)
                                      ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                      : 0.0f;
                const float p11 = (nm1 > -INFINITY && score[nt][3] > -INFINITY)
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
                                smem_addr(&p_sw[a_rowoff * Bc +
                                                gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                    unsigned vf[2];
                    const int vrow = k * 16 + b_koff + b_rin;
                    const int vcol = n * 8;
                    ldmatrix_x2_t(vf[0], vf[1],
                                  smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2],
                             pf[3], vf[0], vf[1]);
                }
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < rows) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < rows) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < rows) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < rows) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();
    for (int row = warp; row < rows; row += Wc) {
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
    for (int chunk = tid; chunk < rows * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head = 0, token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

// V1: 256 threads, ALL warps run the decode bursts, warps 0-3 own the MMA
// path. Same burst/barrier structure as V0 with twice the decode parallelism
// per block (1 block/SM by registers).
//
// MEASURED (RTX 5090, 32k/grid.y=85, session 8): V1 full 0.226 ms vs V0's
// 0.190 — REFUTED. The acc-fragment register wall forces every 8-warp variant
// to 1 block/SM, so the decode chains in flight (16 rows continuous) never
// exceed V0's 2-block bursts (32 rows at ~60% duty). Do not retry this shape;
// see HANDOFF lever (b).
__global__ void hq_decode_phase8_kernel(
    const __nv_bfloat16* q, const std::int32_t* pos, std::int32_t tokens,
    const std::uint8_t* codes_k, const std::uint8_t* codes_v, const std::uint8_t* meta_k,
    const std::uint8_t* meta_v, const std::int32_t* block_tables, std::int32_t table_stride,
    std::int32_t column_begin, float scale, __nv_bfloat16* partial_acc, float* partial_m,
    float* partial_l, int do_decode, int do_mma) {
    using Geometry = Gqa27Geometry;
    constexpr int Bc = 32, D = 256, Threads = 256;
    __shared__ __align__(16) __nv_bfloat16 qkv_s[2 * Bc * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[4 * 16 * Bc];
    __shared__ std::int8_t signs_s[256];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int rows    = tokens * Geometry::GroupSize;
    const bool mma_warp = warp < 4;
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_begin;
    pos += column_begin;
    const std::int32_t* block_table = block_tables;
    const std::int32_t window = pos[tokens - 1] + 1;
    const int active_splits =
        gqa_small_t_active_splits<Geometry, false>(window, gridDim.y, tokens);
    if (split >= active_splits) { return; }

    hq_engine_signs_fill(signs_s);
    __syncthreads();
    for (int idx = tid; idx < 64 * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head = 0, token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < rows && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();
    for (int row = warp; row < rows; row += 8) {
        float reg[8];
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            reg[s2] = __bfloat162float(qkv_s[row * D + gqa_small_t_tc_swz(row, s2 * 32 + lane)]);
        }
        hq_fwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            qkv_s[row * D + gqa_small_t_tc_swz(row, s2 * 32 + lane)] = __float2bfloat16(reg[s2]);
        }
    }
    __syncthreads();

    unsigned af_q[16][4];
    if (mma_warp) {
        const int a_rin    = lane & 7;
        const int a_rowoff = a_rin + (((lane >> 3) & 1) << 3);
        const int a_coloff = ((lane >> 3) >> 1) << 3;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int arow = warp * 16 + a_rowoff;
            const int acol = k * 16 + a_coloff;
            ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                        smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
        }
    }
    __syncthreads();

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_splits;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_splits) : div_up(window, active_splits);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    const int first_tile  = (split_start / Bc) * Bc;
    const int key_blocks  = div_up(split_end - first_tile, Bc);
    const HqPhaseCtx ctx{kv_head, split_start, split_end, rows, scale, pos};

    float acc[32][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    if (mma_warp) {
#pragma unroll
        for (int n = 0; n < 32; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
        }
    }

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;
        for (int slot = tid; slot < 2 * Bc * 8; slot += Threads) {
            const bool role_v = slot >= Bc * 8;
            const int key_l   = (slot >> 3) & (Bc - 1);
            const int lane8   = slot & 7;
            __nv_bfloat16* row_dst = (role_v ? v_s : k_s) + key_l * D;
            const int key = k0 + key_l;
            if (do_decode && key >= split_start && key < split_end) {
                hq_decode_row_group(
                    hq_row_codes<Geometry>(role_v ? codes_v : codes_k, block_table, kv_head, key),
                    hq_row_meta<Geometry>(role_v ? meta_v : meta_k, block_table, kv_head, key),
                    row_dst, lane8, key_l & 7, hq_dither_row_seed(kv_head, key, role_v));
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    store_vec(row_dst + (((lane8 * 4 + j) ^ (key_l & 7)) << 3),
                              make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();
        if (mma_warp && do_mma) {
            hq_phase_consumer_tile(af_q, &p_s[warp * 16 * Bc], k_s, v_s, k0, ctx, warp, lane, m0,
                                   m1, l0, l1, acc);
        }
        __syncthreads();
    }

    if (!mma_warp) { return; }
    const int gid = lane >> 2;
    const int lid = lane & 3;
    if (lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1  = row0 + 8;
        if (row0 < rows) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < rows) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < rows) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < rows) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    bar_partial_sync(5, 128);
    for (int row = warp; row < rows; row += 4) {
        float reg[8];
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            reg[s2] = __bfloat162float(qkv_s[row * D + s2 * 32 + lane]);
        }
        hq_ifwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            qkv_s[row * D + s2 * 32 + lane] = __float2bfloat16(reg[s2]);
        }
    }
    bar_partial_sync(5, 128);
    for (int chunk = tid; chunk < rows * (D / 8); chunk += 128) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head = 0, token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

// V2: warp specialization. Warps 0-3 consume tiles (QK/softmax/PV + epilogue),
// warps 4-7 produce them one tile ahead into a double-buffered K/V tile pair,
// handshaken over named barriers; 256 threads, 1 block/SM, dynamic smem.
//
// MEASURED (RTX 5090, 32k/grid.y=85, session 8): V2 full 0.285-0.474 ms vs
// V0's 0.190 (no-decode 0.084 confirms the pipeline itself is sound) — REFUTED
// and unstable across runs. 4 dedicated decode warps at 1 block/SM decode
// SLOWER than V0's all-warp bursts at 2 blocks/SM; the review 17.2 premise of
// 16 warps/SM is unreachable because the mma warps' acc fragments cap the
// kernel at 1 block/SM. Do not retry; the >=3 G rows/s gate is closed at
// ~2.1 G and Tier 2 (V skipping) is the indicated lever.
__global__ void hq_decode_phase_ws_kernel(
    const __nv_bfloat16* q, const std::int32_t* pos, std::int32_t tokens,
    const std::uint8_t* codes_k, const std::uint8_t* codes_v, const std::uint8_t* meta_k,
    const std::uint8_t* meta_v, const std::int32_t* block_tables, std::int32_t table_stride,
    std::int32_t column_begin, float scale, __nv_bfloat16* partial_acc, float* partial_m,
    float* partial_l, int do_decode, int do_mma) {
    using Geometry = Gqa27Geometry;
    constexpr int Bc = 32, D = 256, Threads = 256;
    constexpr int kTileFloats = 2 * Bc * D;
    extern __shared__ std::uint8_t smem_raw[];
    __nv_bfloat16* tiles = reinterpret_cast<__nv_bfloat16*>(smem_raw); // [2][2*Bc*D]
    __nv_bfloat16* p_s   = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2 * kTileFloats * 2);
    std::int8_t* signs_s = reinterpret_cast<std::int8_t*>(smem_raw + 2 * kTileFloats * 2 + 4096);
    constexpr int kBarReady = 1; // 1 = buf0 tile ready, 2 = buf1
    constexpr int kBarFree  = 3; // 3 = buf0 reusable, 4 = buf1
    constexpr int kBarCons  = 5; // consumer-only epilogue syncs
    constexpr int kBarQDone = 6; // af_q ldmatrix done before first produce

    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int rows    = tokens * Geometry::GroupSize;
    const bool mma_warp = warp < 4;
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_begin;
    pos += column_begin;
    const std::int32_t* block_table = block_tables;
    const std::int32_t window = pos[tokens - 1] + 1;
    const int active_splits =
        gqa_small_t_active_splits<Geometry, false>(window, gridDim.y, tokens);
    if (split >= active_splits) { return; }

    hq_engine_signs_fill(signs_s);
    __syncthreads();
    // Stage q rows into the first tile buffer (overwritten by tile 0's decode
    // only after the consumers' af_q ldmatrix completes).
    for (int idx = tid; idx < 64 * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head = 0, token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < rows && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
        }
        tiles[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();
    for (int row = warp; row < rows; row += 8) {
        float reg[8];
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            reg[s2] = __bfloat162float(tiles[row * D + gqa_small_t_tc_swz(row, s2 * 32 + lane)]);
        }
        hq_fwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            tiles[row * D + gqa_small_t_tc_swz(row, s2 * 32 + lane)] = __float2bfloat16(reg[s2]);
        }
    }
    __syncthreads();

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_splits;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_splits) : div_up(window, active_splits);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    const int first_tile  = (split_start / Bc) * Bc;
    const int key_blocks  = div_up(split_end - first_tile, Bc);

    unsigned af_q[16][4];
    if (mma_warp) {
        const int a_rin    = lane & 7;
        const int a_rowoff = a_rin + (((lane >> 3) & 1) << 3);
        const int a_coloff = ((lane >> 3) >> 1) << 3;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int arow = warp * 16 + a_rowoff;
            const int acol = k * 16 + a_coloff;
            ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                        smem_addr(&tiles[arow * D + gqa_small_t_tc_swz(arow, acol)]));
        }
        bar_partial_arrive(kBarQDone, 256);
    } else {
        bar_partial_sync(kBarQDone, 256);
        const int ptid = tid - 128;
        for (int kb = 0; kb < key_blocks; ++kb) {
            const int p         = kb & 1;
            __nv_bfloat16* buf  = tiles + p * kTileFloats;
            __nv_bfloat16* k_dst = buf;
            __nv_bfloat16* v_dst = buf + Bc * D;
            if (kb >= 2) { bar_partial_sync(kBarFree + p, 256); }
            const int k0 = first_tile + kb * Bc;
            for (int slot = ptid; slot < 2 * Bc * 8; slot += 128) {
                const bool role_v = slot >= Bc * 8;
                const int key_l   = (slot >> 3) & (Bc - 1);
                const int lane8   = slot & 7;
                __nv_bfloat16* row_dst = (role_v ? v_dst : k_dst) + key_l * D;
                const int key = k0 + key_l;
                if (do_decode && key >= split_start && key < split_end) {
                    hq_decode_row_group(
                        hq_row_codes<Geometry>(role_v ? codes_v : codes_k, block_table, kv_head,
                                              key),
                        hq_row_meta<Geometry>(role_v ? meta_v : meta_k, block_table, kv_head,
                                             key),
                        row_dst, lane8, key_l & 7, hq_dither_row_seed(kv_head, key, role_v));
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        store_vec(row_dst + (((lane8 * 4 + j) ^ (key_l & 7)) << 3),
                                  make_int4(0, 0, 0, 0));
                    }
                }
            }
            bar_partial_arrive(kBarReady + p, 256);
        }
        return;
    }

    const HqPhaseCtx ctx{kv_head, split_start, split_end, rows, scale, pos};
    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int p = kb & 1;
        bar_partial_sync(kBarReady + p, 256);
        const __nv_bfloat16* buf = tiles + p * kTileFloats;
        if (do_mma) {
            hq_phase_consumer_tile(af_q, &p_s[warp * 16 * Bc], buf, buf + Bc * D,
                                   first_tile + kb * Bc, ctx, warp, lane, m0, m1, l0, l1, acc);
        }
        bar_partial_arrive(kBarFree + p, 256);
    }

    const int gid = lane >> 2;
    const int lid = lane & 3;
    if (lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < rows) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < rows) {
            int q_head = 0, token = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }
    __nv_bfloat16* stage = tiles; // producers are done; buf0 is free scratch
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < rows) {
            stage[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            stage[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < rows) {
            stage[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            stage[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    bar_partial_sync(kBarCons, 128);
    for (int row = warp; row < rows; row += 4) {
        float reg[8];
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            reg[s2] = __bfloat162float(stage[row * D + s2 * 32 + lane]);
        }
        hq_ifwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
        for (int s2 = 0; s2 < 8; ++s2) {
            stage[row * D + s2 * 32 + lane] = __float2bfloat16(reg[s2]);
        }
    }
    bar_partial_sync(kBarCons, 128);
    for (int chunk = tid; chunk < rows * (D / 8); chunk += 128) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head = 0, token = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&stage[row * D + d]));
        }
    }
}

}  // namespace

int main() {
    __nv_bfloat16* d_rows;
    std::uint8_t *d_codes, *d_meta;
    __nv_bfloat16 *d_out, *d_pacc;
    float *d_pm, *d_pl;
    std::int32_t *d_pos, *d_table;
    const int pages = kRows / 64;
    cudaMalloc(&d_rows, static_cast<std::size_t>(kRows) * kKvHeads * kHqHeadDim * 2);
    // Full 4-head page planes (64/8 bytes x 64 x 4 heads x pages).
    cudaMalloc(&d_codes, static_cast<std::size_t>(64) * 64 * 4 * pages);
    cudaMalloc(&d_meta, static_cast<std::size_t>(8) * 64 * 4 * pages);
    cudaMalloc(&d_out, static_cast<std::size_t>(kRows) * kHqHeadDim * 2);
    // Clean partial buffers for the decode-kernel sweeps (aliasing them into
    // the meta plane corrupts the Rice streams and invalidates the timing).
    cudaMalloc(&d_pacc, static_cast<std::size_t>(kRows) * kHqHeadDim * 2);
    cudaMalloc(&d_pm, static_cast<std::size_t>(kRows) * 4);
    cudaMalloc(&d_pl, static_cast<std::size_t>(kRows) * 4);
    cudaMalloc(&d_pos, 4);
    cudaMalloc(&d_table, static_cast<std::size_t>(pages) * 4);
    gen_rows<<<(kRows * kHqHeadDim / 4 + 255) / 256, 256>>>(d_rows, 7u);
    identity_table<<<(pages + 255) / 256, 256>>>(d_table, pages);
    const std::int32_t zero = 0;
    cudaMemcpy(d_pos, &zero, 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Encode via the fill-kernel shape (warp per row). One warp per
    // (token, kv_head, role) unit: the grid must cover tokens*KVHeads*2 units
    // (the old head-0-only grid made 7/8 of the plane never-written zero rows,
    // so the decode numbers below measured the fast path).
    constexpr int kWPB = kGqaHqFillWarps;
    const int fill_units = kRows * kKvHeads * 2;
    cudaMemset(d_codes, 0, static_cast<std::size_t>(64) * 64 * 4 * pages);
    cudaMemset(d_meta, 0, static_cast<std::size_t>(8) * 64 * 4 * pages);
    {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int rep = 0; rep < 3; ++rep) {
            gqa_attention_prefill_fill_hq_kernel<Gqa27Geometry, GqaPrefillDirectMetadata>
                <<<(fill_units + kWPB - 1) / kWPB, kWPB * 32, kGqaHqFillSmemBytes>>>(
                    d_rows, d_rows, d_pos,
                    GqaPrefillDirectMetadata{d_table}, d_codes, d_codes, d_meta, d_meta, kRows);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("encode: %.2f ms/rep for %d rows -> %.1f M rows/s\n", ms / 3, fill_units,
                    fill_units / (ms / 3) / 1e3);
    }

    // Group decoder (eight lanes per row, the engine decode-kernel shape) at
    // full occupancy.
    {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        const int grid = (kRows * 8 + 255) / 256;
        cudaEventRecord(a);
        for (int rep = 0; rep < 3; ++rep) {
            decode_rows_group_bench<<<grid, 256>>>(d_codes, d_meta, d_out, kRows);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("decode-rows(group): %.2f ms/rep for %d rows -> %.1f M rows/s\n", ms / 3,
                    kRows, kRows / (ms / 3) / 1e3);
    }

    // Append-variant timing (GqaAppendInput, the engine decode round's
    // shape): the owner split encodes this round's K/V rows inside the
    // kernel, which the sweeps above never measure. K and V encode into
    // dedicated planes (never the corpus, never each other's bytes), and the
    // appended row's meta is read back so an escalating row (three packing
    // attempts) cannot silently inflate the timing.
    {
        __nv_bfloat16 *d_q1, *d_knew, *d_vnew;
        std::int32_t* d_posa;
        std::uint8_t *d_cka, *d_cva, *d_mka, *d_mva;
        const size_t aplane_codes = static_cast<size_t>(32) * 64 * 4 * 64;
        const size_t aplane_meta  = static_cast<size_t>(32) * 64 * 4 * 8;
        cudaMalloc(&d_q1, 24 * kHqHeadDim * 2);
        cudaMalloc(&d_knew, 4 * kHqHeadDim * 2);
        cudaMalloc(&d_vnew, 4 * kHqHeadDim * 2);
        cudaMalloc(&d_posa, 4);
        cudaMalloc(&d_cka, aplane_codes);
        cudaMalloc(&d_cva, aplane_codes);
        cudaMalloc(&d_mka, aplane_meta);
        cudaMalloc(&d_mva, aplane_meta);
        cudaMemset(d_q1, 0x3C, 24 * kHqHeadDim * 2);
        cudaMemset(d_knew, 0x3C, 4 * kHqHeadDim * 2);
        cudaMemset(d_vnew, 0x3C, 4 * kHqHeadDim * 2);
        cudaMemset(d_cka, 0, aplane_codes);
        cudaMemset(d_cva, 0, aplane_codes);
        cudaMemset(d_mka, 0, aplane_meta);
        cudaMemset(d_mva, 0, aplane_meta);
        for (int window : {54, 2048}) {
            const std::int32_t last = window - 1;
            cudaMemcpy(d_posa, &last, 4, cudaMemcpyHostToDevice);
            GqaAppendInput input{d_knew, d_vnew};
            const GqaTcKVHq hq_cache{d_cka, d_cva, d_mka, d_mva};
            gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 8, 4, true, true, GqaAppendInput, GqaTcKVHq>
                <<<dim3(4, 32, 1), kGqaHqDecodeThreads>>>(
                    d_q1, input, d_posa, hq_cache, d_table, nullptr, nullptr, 32, 1, 1, 0, window,
                    0.0625f, d_pacc, d_pm, d_pl);
            cudaDeviceSynchronize();
            {
                const int page = last >> 6, off = last & 63;
                std::uint8_t hm[8];
                cudaMemcpy(hm, d_mka + static_cast<size_t>(off) * 8 +
                                   static_cast<size_t>(page) * 64 * 4 * 8,
                           8, cudaMemcpyDeviceToHost);
                std::printf("append row meta (k=%d esc=%d used=%u)\n", hm[2] & 15,
                            (hm[2] >> 4) & 3, hm[3] | ((hm[4] & 3) << 8));
            }
            cudaEvent_t a4, b4;
            cudaEventCreate(&a4);
            cudaEventCreate(&b4);
            cudaEventRecord(a4);
            for (int rep = 0; rep < 8; ++rep) {
                gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 8, 4, true, true, GqaAppendInput, GqaTcKVHq>
                    <<<dim3(4, 32, 1), kGqaHqDecodeThreads>>>(
                        d_q1, input, d_posa, hq_cache, d_table, nullptr, nullptr, 32, 1, 1, 0,
                        window, 0.0625f, d_pacc, d_pm, d_pl);
            }
            cudaEventRecord(b4);
            cudaEventSynchronize(b4);
            float ms4 = 0;
            cudaEventElapsedTime(&ms4, a4, b4);
            std::printf("decode-kernel(append) grid.y=32 window=%d: %.3f ms/call\n", window,
                        ms4 / 8);
            cudaEventDestroy(a4);
            cudaEventDestroy(b4);
        }
        cudaFree(d_q1);
        cudaFree(d_knew);
        cudaFree(d_vnew);
        cudaFree(d_posa);
        cudaFree(d_cka);
        cudaFree(d_cva);
        cudaFree(d_mka);
        cudaFree(d_mva);
    }

    // Full decode-kernel invocation shaped like the engine decode round,
    // swept over launch split counts and window sizes.
    for (int gridsplit : {4, 32, 85}) {
        for (int window : {54, 2048, 32768}) {
            std::int32_t* d_pos2;
            cudaMalloc(&d_pos2, 4);
            const std::int32_t last = window - 1;
            cudaMemcpy(d_pos2, &last, 4, cudaMemcpyHostToDevice);
            GqaCachedInput no_append{};
            const GqaTcKVHq hq_cache{d_codes, d_codes, d_meta, d_meta};
            cudaEvent_t a2, b2;
            cudaEventCreate(&a2);
            cudaEventCreate(&b2);
            cudaGetLastError();
            // Warm-up (cold launch measures setup, not steady state).
            gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 8, 4, true, true, GqaCachedInput, GqaTcKVHq>
                <<<dim3(4, gridsplit, 1), kGqaHqDecodeThreads>>>(
                    d_out, no_append, d_pos2, hq_cache, d_table, nullptr, nullptr, kRows / 64,
                    1, window, 0, window, 0.0625f, d_pacc, d_pm, d_pl);
            cudaDeviceSynchronize();
            cudaEventRecord(a2);
            for (int rep = 0; rep < 8; ++rep) {
                gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 8, 4, true, true, GqaCachedInput, GqaTcKVHq>
                    <<<dim3(4, gridsplit, 1), kGqaHqDecodeThreads>>>(
                        d_out, no_append, d_pos2, hq_cache, d_table, nullptr, nullptr,
                        kRows / 64, 1, window, 0, window, 0.0625f, d_pacc, d_pm, d_pl);
            }
            cudaEventRecord(b2);
            cudaEventSynchronize(b2);
            float ms2 = 0;
            cudaEventElapsedTime(&ms2, a2, b2);
            std::printf("decode-kernel(engine shape) grid.y=%d window=%d: %.3f ms/call\n",
                        gridsplit, window, ms2 / 8);
            cudaFree(d_pos2);
        }
    }
    // Phase attribution at the engine shape (grid.y=85, window=32768): the
    // group-decode tile staging vs the QK/softmax/PV MMA chain, plus achieved
    // occupancy of the production kernel.
    {
        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 8, 4, true, true, GqaCachedInput, GqaTcKVHq>, kGqaHqDecodeThreads, 0);
        std::printf("phase-bench occupancy: %d blocks/SM\n", blocks_per_sm);

        std::int32_t* d_pos3;
        cudaMalloc(&d_pos3, 4);
        const std::int32_t last3 = 32768 - 1;
        cudaMemcpy(d_pos3, &last3, 4, cudaMemcpyHostToDevice);
        cudaGetLastError();
        const char* names[] = {"full", "no-decode", "no-score/pv"};
        const int modes[3][2] = {{1, 1}, {0, 1}, {1, 0}};
        float times[3] = {0, 0, 0};
        for (int m = 0; m < 3; ++m) {
            // Warm-up.
            hq_decode_phase_bench_kernel<<<dim3(4, 85, 1), 128>>>(
                d_out, d_pos3, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64, 0, 0.0625f,
                d_pacc, d_pm, d_pl, modes[m][0], modes[m][1]);
            cudaDeviceSynchronize();
            cudaEvent_t a3, b3;
            cudaEventCreate(&a3);
            cudaEventCreate(&b3);
            cudaEventRecord(a3);
            for (int rep = 0; rep < 8; ++rep) {
                hq_decode_phase_bench_kernel<<<dim3(4, 85, 1), 128>>>(
                    d_out, d_pos3, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64, 0,
                    0.0625f, d_pacc, d_pm, d_pl, modes[m][0], modes[m][1]);
            }
            cudaEventRecord(b3);
            cudaEventSynchronize(b3);
            cudaEventElapsedTime(&times[m], a3, b3);
            times[m] /= 8;
            std::printf("phase-bench %-11s: %.3f ms/call\n", names[m], times[m]);
            cudaEventDestroy(a3);
            cudaEventDestroy(b3);
        }
        std::printf("phase-bench decode-only: %.3f ms/call\n", times[0] - times[1]);
        std::printf("phase-bench score/pv  : %.3f ms/call\n", times[0] - times[2]);

        // V1/V2 prototypes at the same shape (full + no-decode), plus a
        // short-window shape for the small-grid regime.
        constexpr int kWsSmem = 2 * (2 * 32 * 256 * 2) + 4 * 16 * 32 * 2 + 256;
        cudaFuncSetAttribute(hq_decode_phase_ws_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kWsSmem);
        int b8 = 0, bws = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b8, hq_decode_phase8_kernel, 256, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bws, hq_decode_phase_ws_kernel, 256,
                                                       kWsSmem);
        std::printf("phase8 occupancy: %d blocks/SM; phase-ws occupancy: %d blocks/SM\n", b8,
                    bws);
        for (int shape = 0; shape < 2; ++shape) {
            const int gsplit = shape == 0 ? 85 : 4;
            const int win    = shape == 0 ? 32768 : 200;
            std::int32_t* d_pos4;
            cudaMalloc(&d_pos4, 4);
            const std::int32_t last4 = win - 1;
            cudaMemcpy(d_pos4, &last4, 4, cudaMemcpyHostToDevice);
            const char* snames[2] = {"32k/85", "200/4"};
            for (int m = 0; m < 2; ++m) {
                const int do_decode = m == 0 ? 1 : 0;
                const char* mname   = m == 0 ? "full" : "no-decode";
                float t8 = 0, tws = 0;
                cudaEvent_t a4, b4;
                cudaEventCreate(&a4);
                cudaEventCreate(&b4);
                hq_decode_phase8_kernel<<<dim3(4, gsplit, 1), 256>>>(
                    d_out, d_pos4, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64, 0,
                    0.0625f, d_pacc, d_pm, d_pl, do_decode, 1);
                cudaDeviceSynchronize();
                cudaEventRecord(a4);
                for (int rep = 0; rep < 8; ++rep) {
                    hq_decode_phase8_kernel<<<dim3(4, gsplit, 1), 256>>>(
                        d_out, d_pos4, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64,
                        0, 0.0625f, d_pacc, d_pm, d_pl, do_decode, 1);
                }
                cudaEventRecord(b4);
                cudaEventSynchronize(b4);
                cudaEventElapsedTime(&t8, a4, b4);
                hq_decode_phase_ws_kernel<<<dim3(4, gsplit, 1), 256, kWsSmem>>>(
                    d_out, d_pos4, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64, 0,
                    0.0625f, d_pacc, d_pm, d_pl, do_decode, 1);
                cudaDeviceSynchronize();
                cudaEventRecord(a4);
                for (int rep = 0; rep < 8; ++rep) {
                    hq_decode_phase_ws_kernel<<<dim3(4, gsplit, 1), 256, kWsSmem>>>(
                        d_out, d_pos4, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64,
                        0, 0.0625f, d_pacc, d_pm, d_pl, do_decode, 1);
                }
                cudaEventRecord(b4);
                cudaEventSynchronize(b4);
                cudaEventElapsedTime(&tws, a4, b4);
                cudaEventDestroy(a4);
                cudaEventDestroy(b4);
                std::printf("phase-%s %-11s: v1 %.3f ms, v2-ws %.3f ms\n", snames[shape], mname,
                            t8 / 8, tws / 8);
            }
            cudaFree(d_pos4);
        }
        cudaFree(d_pos3);
    }

    cudaError_t e = cudaGetLastError();
    std::printf("err: %s\n", cudaGetErrorString(e));
    return 0;
}
