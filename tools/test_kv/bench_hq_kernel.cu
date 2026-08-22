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
                        out + static_cast<std::size_t>(row) * kHqHeadDim, i & 7);
}

// Bench-local copy of the engine decode kernel with runtime phase switches
// (uniform across the block, so no divergence is introduced): isolates the
// Rice K/V decode phases from the score/rescale/PV phases at the engine shape.
__global__ void hq_decode_phase_bench_kernel(
    const __nv_bfloat16* q, const std::int32_t* pos, std::int32_t tokens,
    const std::uint8_t* codes_k, const std::uint8_t* codes_v, const std::uint8_t* meta_k,
    const std::uint8_t* meta_v, const std::int32_t* block_tables, std::int32_t table_stride,
    std::int32_t column_begin, float scale, __nv_bfloat16* partial_acc, float* partial_m,
    float* partial_l, int do_decode_k, int do_score, int do_decode_v, int do_pv) {
    constexpr int kKeys = kGqaHqDecodeKeys;
    using Geometry = Gqa27Geometry;
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
    const int rows    = tokens * Geometry::GroupSize;

    const std::int32_t* block_table =
        block_tables;

    const std::int32_t first_pos = pos[column_begin];
    const std::int32_t last_pos  = pos[column_begin + tokens - 1];
    const std::int32_t window    = last_pos + 1;
    const int active_splits =
        gqa_small_t_active_splits<Geometry, false>(window, gridDim.y, tokens);
    if (split >= active_splits) { return; }

    hq_engine_signs_fill(signs);
    for (int i = tid; i < rows * kGqaHeadDim; i += kGqaHqDecodeThreads) { acc[i] = 0.0f; }
    for (int r = 0; r < rows; ++r) {
        if (tid == 0) { m_smem[r] = -INFINITY; }
    }
    __syncthreads();

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

    const std::int32_t span = window - column_begin;
    const std::int32_t kps  = div_up(span, active_splits);
    const std::int32_t key_lo = column_begin + split * kps;
    const std::int32_t key_hi = min(key_lo + kps, window);

    for (std::int32_t k0 = key_lo; k0 < key_hi; k0 += kKeys) {
        const int chunk = min(kKeys, key_hi - k0);
        if (do_decode_k) {
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
        if (do_score) {
            for (int i = tid; i < rows * kKeys; i += kGqaHqDecodeThreads) {
                const int r  = i / kKeys;
                const int kk = i - r * kKeys;
                const int token = r / Geometry::GroupSize;
                float s = 0.0f;
                if (kk < chunk && k0 + kk <= pos[column_begin + token]) {
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
        }
        if (do_decode_v) {
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
        if (do_pv) {
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
        cudaFuncSetAttribute(gqa_attention_decode_hq_kernel<Gqa27Geometry, GqaAppendInput>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(gqa_hq_decode_smem_bytes<Gqa27Geometry>()));
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
            gqa_attention_decode_hq_kernel<Gqa27Geometry, GqaAppendInput>
                <<<dim3(4, 32, 1), 256, gqa_hq_decode_smem_bytes<Gqa27Geometry>()>>>(
                    d_q1, input, d_posa, 1, d_cka, d_cva, d_mka, d_mva, d_table, nullptr,
                    nullptr, 32, 1, 0, window, 0.0625f, d_pacc, d_pm, d_pl);
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
                gqa_attention_decode_hq_kernel<Gqa27Geometry, GqaAppendInput>
                    <<<dim3(4, 32, 1), 256, gqa_hq_decode_smem_bytes<Gqa27Geometry>()>>>(
                        d_q1, input, d_posa, 1, d_cka, d_cva, d_mka, d_mva, d_table,
                        nullptr, nullptr, 32, 1, 0, window, 0.0625f, d_pacc, d_pm, d_pl);
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
            cudaEvent_t a2, b2;
            cudaEventCreate(&a2);
            cudaEventCreate(&b2);
            cudaFuncSetAttribute(
                gqa_attention_decode_hq_kernel<Gqa27Geometry, GqaCachedInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(gqa_hq_decode_smem_bytes<Gqa27Geometry>()));
            cudaGetLastError();
            // Warm-up (cold launch measures setup, not steady state).
            gqa_attention_decode_hq_kernel<Gqa27Geometry, GqaCachedInput>
                <<<dim3(4, gridsplit, 1), 256, gqa_hq_decode_smem_bytes<Gqa27Geometry>()>>>(
                    d_out, no_append, d_pos2, 1, d_codes, d_codes, d_meta, d_meta, d_table,
                    nullptr, nullptr, kRows / 64, window, 0, window, 0.0625f, d_pacc, d_pm, d_pl);
            cudaDeviceSynchronize();
            cudaEventRecord(a2);
            for (int rep = 0; rep < 8; ++rep) {
                gqa_attention_decode_hq_kernel<Gqa27Geometry, GqaCachedInput>
                    <<<dim3(4, gridsplit, 1), 256, gqa_hq_decode_smem_bytes<Gqa27Geometry>()>>>(
                        d_out, no_append, d_pos2, 1, d_codes, d_codes, d_meta, d_meta, d_table,
                        nullptr, nullptr, kRows / 64, window, 0, window, 0.0625f, d_pacc, d_pm,
                        d_pl);
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
    // Phase attribution at the engine shape (grid.y=85, window=32768): decode
    // phases vs score/rescale/PV phases, plus achieved occupancy.
    {
        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, hq_decode_phase_bench_kernel,
            kGqaHqDecodeThreads, static_cast<int>(gqa_hq_decode_smem_bytes<Gqa27Geometry>()));
        std::printf("phase-bench occupancy: %d blocks/SM\n", blocks_per_sm);

        std::int32_t* d_pos3;
        cudaMalloc(&d_pos3, 4);
        const std::int32_t last3 = 32768 - 1;
        cudaMemcpy(d_pos3, &last3, 4, cudaMemcpyHostToDevice);
        cudaFuncSetAttribute(
            hq_decode_phase_bench_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(gqa_hq_decode_smem_bytes<Gqa27Geometry>()));
        cudaGetLastError();
        const char* names[] = {"full", "no-decode", "no-score/pv"};
        const int modes[3][4] = {{1, 1, 1, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}};
        for (int m = 0; m < 3; ++m) {
            // Warm-up.
            hq_decode_phase_bench_kernel<<<dim3(4, 85, 1), 256,
                                           gqa_hq_decode_smem_bytes<Gqa27Geometry>()>>>(
                d_out, d_pos3, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64, 0, 0.0625f,
                d_pacc, d_pm, d_pl, modes[m][0], modes[m][1], modes[m][2], modes[m][3]);
            cudaDeviceSynchronize();
            cudaEvent_t a3, b3;
            cudaEventCreate(&a3);
            cudaEventCreate(&b3);
            cudaEventRecord(a3);
            for (int rep = 0; rep < 8; ++rep) {
                hq_decode_phase_bench_kernel<<<dim3(4, 85, 1), 256,
                                               gqa_hq_decode_smem_bytes<Gqa27Geometry>()>>>(
                    d_out, d_pos3, 1, d_codes, d_codes, d_meta, d_meta, d_table, kRows / 64, 0,
                    0.0625f, d_pacc, d_pm, d_pl, modes[m][0], modes[m][1], modes[m][2],
                    modes[m][3]);
            }
            cudaEventRecord(b3);
            cudaEventSynchronize(b3);
            float ms3 = 0;
            cudaEventElapsedTime(&ms3, a3, b3);
            std::printf("phase-bench %-11s: %.3f ms/call\n", names[m], ms3 / 8);
            cudaEventDestroy(a3);
            cudaEventDestroy(b3);
        }
        cudaFree(d_pos3);
    }

    cudaError_t e = cudaGetLastError();
    std::printf("err: %s\n", cudaGetErrorString(e));
    return 0;
}
