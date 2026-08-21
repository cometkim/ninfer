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

__global__ void decode_rows_kernel_bench(const std::uint8_t* codes, const std::uint8_t* meta,
                                         __nv_bfloat16* out, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = i >> 2;
    if (row >= n_rows) { return; }
    hq_decode_row_segment(codes + static_cast<std::size_t>(row) * kHqRowBudgetBytes,
                          meta + static_cast<std::size_t>(row) * kHqMetaBytes,
                          out + static_cast<std::size_t>(row) * kHqHeadDim, i & 3);
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

    // Decode (four threads per row, one 64-symbol segment each) — the engine
    // codec shape at full occupancy.
    {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        const int grid = (kRows * 4 + 255) / 256;
        cudaEventRecord(a);
        for (int rep = 0; rep < 3; ++rep) {
            decode_rows_kernel_bench<<<grid, 256>>>(d_codes, d_meta, d_out, kRows);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("decode-rows(4-way): %.2f ms/rep for %d rows -> %.1f M rows/s\n", ms / 3, kRows,
                    kRows / (ms / 3) / 1e3);
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
    cudaError_t e = cudaGetLastError();
    std::printf("err: %s\n", cudaGetErrorString(e));
    return 0;
}
