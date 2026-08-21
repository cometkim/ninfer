// 1M-key needle-retrieval verifier for the engine's hq-e8-2b KV codec
// (tools/test_kv convention). Corpus: 1,000,000 rows x 256 dims iid N(0,1);
// five high-norm needles at 5/25/50/75/95% offsets; noisy queries. The
// pipeline (RHT + E8 + fixed-budget Rice) must place every needle at the
// UNIQUE top-1 dot of the decoded corpus against an FP64 host oracle.
//
// Any miss => EXIT_FAILURE. --negative-control zeroes the payload after
// encoding and REQUIRES retrieval to fail (the verifier's failure path must
// demonstrably fire).
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <curand_kernel.h>

#include "ops/kernel/hq_codec.cuh"

using namespace ninfer::ops;

namespace {

constexpr int kRows = 1000000;
constexpr int kDim = kHqHeadDim;
constexpr int kQueries = 5;

__global__ void gen_rows(__nv_bfloat16* out, std::uint64_t seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kRows * kDim / 4;
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

__global__ void encode_rows_kernel(const __nv_bfloat16* rows, const std::int8_t* signs,
                                   std::uint8_t* codes, std::uint8_t* meta, int n_rows) {
    extern __shared__ float smem[];
    const int warp =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    if (warp >= n_rows) { return; }
    float* u    = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    hq_encode_row_warp(rows + static_cast<std::size_t>(warp) * kDim, signs, 0, u, syms,
                       codes + static_cast<std::size_t>(warp) * kHqRowBudgetBytes,
                       meta + static_cast<std::size_t>(warp) * kHqMetaBytes);
}

__global__ void decode_rows_kernel(const std::uint8_t* codes, const std::uint8_t* meta,
                                   __nv_bfloat16* out, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) { return; }
    hq_decode_row_thread(codes + static_cast<std::size_t>(i) * kHqRowBudgetBytes,
                         meta + static_cast<std::size_t>(i) * kHqMetaBytes,
                         out + static_cast<std::size_t>(i) * kDim);
}

// Rotate queries into the codec frame: one warp per query row.
__global__ void rotate_queries_kernel(const __nv_bfloat16* q, const std::int8_t* signs,
                                      __nv_bfloat16* q_rot, int n_q) {
    const int warp =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    if (warp >= n_q) { return; }
    const int lane = static_cast<int>(threadIdx.x & 31u);
    float reg[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        reg[s] = __bfloat162float(q[static_cast<std::size_t>(warp) * kDim + s * 32 + lane]);
    }
    hq_fwht256_sign(reg, signs, 0, lane);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        q_rot[static_cast<std::size_t>(warp) * kDim + s * 32 + lane] =
            __float2bfloat16(reg[s]);
    }
}

// scores[q * n + t] = <q_rot[q], row[t]> ; one thread per token.
__global__ void dots_kernel(const __nv_bfloat16* rows, const __nv_bfloat16* q_rot,
                            float* scores, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) { return; }
    for (int qi = 0; qi < kQueries; ++qi) {
        float acc = 0.0f;
        for (int d = 0; d < kDim; ++d) {
            acc += __bfloat162float(rows[static_cast<std::size_t>(t) * kDim + d]) *
                   __bfloat162float(q_rot[static_cast<std::size_t>(qi) * kDim + d]);
        }
        scores[static_cast<std::size_t>(qi) * n + t] = acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool negative_control = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--negative-control") == 0) { negative_control = true; }
    }

    std::array<int, kQueries> needle{};
    const double fracs[kQueries] = {0.05, 0.25, 0.50, 0.75, 0.95};
    for (int i = 0; i < kQueries; ++i) { needle[i] = static_cast<int>(fracs[i] * kRows); }

    __nv_bfloat16 *d_rows, *d_dec, *d_q, *d_qr;
    std::uint8_t *d_codes, *d_meta;
    std::int8_t* d_signs;
    float* d_scores;
    cudaMalloc(&d_rows, static_cast<std::size_t>(kRows) * kDim * 2);
    cudaMalloc(&d_dec, static_cast<std::size_t>(kRows) * kDim * 2);
    cudaMalloc(&d_codes, static_cast<std::size_t>(kRows) * kHqRowBudgetBytes);
    cudaMalloc(&d_meta, static_cast<std::size_t>(kRows) * kHqMetaBytes);
    cudaMalloc(&d_q, static_cast<std::size_t>(kQueries) * kDim * 2);
    cudaMalloc(&d_qr, static_cast<std::size_t>(kQueries) * kDim * 2);
    cudaMalloc(&d_signs, kDim);
    cudaMalloc(&d_scores, static_cast<std::size_t>(kQueries) * kRows * 4);

    gen_rows<<<(kRows * kDim / 4 + 255) / 256, 256>>>(d_rows, 20260821ull);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_rows(static_cast<std::size_t>(kRows) * kDim);
    cudaMemcpy(h_rows.data(), d_rows, h_rows.size() * 2, cudaMemcpyDeviceToHost);

    // Needles: unit direction * 5*sqrt(dim); queries = needle + 0.05 noise.
    std::vector<__nv_bfloat16> h_q(static_cast<std::size_t>(kQueries) * kDim);
    {
        std::mt19937 rng(0xC0FFEE);
        std::normal_distribution<float> g(0, 1);
        for (int qi = 0; qi < kQueries; ++qi) {
            const int t = needle[qi];
            double norm = 0;
            for (int d = 0; d < kDim; ++d) {
                h_rows[t * kDim + d] = __float2bfloat16(g(rng));
                const double v = __bfloat162float(h_rows[t * kDim + d]);
                norm += v * v;
            }
            norm = std::sqrt(norm);
            const double target = 5.0 * std::sqrt(256.0);
            for (int d = 0; d < kDim; ++d) {
                h_rows[t * kDim + d] =
                    __float2bfloat16(__bfloat162float(h_rows[t * kDim + d]) * (target / norm));
                h_q[qi * kDim + d] =
                    __float2bfloat16(__bfloat162float(h_rows[t * kDim + d]) + 0.05f * g(rng));
            }
        }
    }
    cudaMemcpy(d_rows, h_rows.data(), h_rows.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice);

    std::vector<std::int8_t> h_signs(kDim);
    {
        std::mt19937 srng(0x5EED);
        std::uniform_int_distribution<int> coin(0, 1);
        for (auto& s : h_signs) { s = coin(srng) ? 1 : -1; }
    }
    cudaMemcpy(d_signs, h_signs.data(), kDim, cudaMemcpyHostToDevice);

    constexpr int kWarpsPerBlock = 8;
    const size_t smem = kWarpsPerBlock * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * 4;
    encode_rows_kernel<<<(kRows + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32,
                         smem>>>(d_rows, d_signs, d_codes, d_meta, kRows);
    cudaDeviceSynchronize();

    size_t stored = static_cast<std::size_t>(kRows) * (kHqRowBudgetBytes + kHqMetaBytes) + kDim;
    const double bps = static_cast<double>(stored) * 8.0 / (kRows * kDim);
    std::printf("stored %.1f MiB -> %.3f bits/scalar (codes+meta+signs)\n",
                stored / 1048576.0, bps);

    if (negative_control) {
        cudaMemset(d_codes, 0, static_cast<std::size_t>(kRows) * kHqRowBudgetBytes);
    }

    decode_rows_kernel<<<(kRows + 255) / 256, 256>>>(d_codes, d_meta, d_dec, kRows);
    rotate_queries_kernel<<<1, 8 * 32>>>(d_q, d_signs, d_qr, kQueries);
    cudaDeviceSynchronize();
    dots_kernel<<<(kRows + 255) / 256, 256>>>(d_dec, d_qr, d_scores, kRows);
    cudaDeviceSynchronize();

    std::vector<float> scores(static_cast<std::size_t>(kQueries) * kRows);
    cudaMemcpy(scores.data(), d_scores, scores.size() * 4, cudaMemcpyDeviceToHost);

    int missed = 0;
    double cos_sum = 0;
    for (int qi = 0; qi < kQueries; ++qi) {
        double sxy = 0, sxx = 0, sqy = 0;
        long long rank = 1, ties = 0;
        const double nd = scores[qi * kRows + needle[qi]];
        for (int t = 0; t < kRows; ++t) {
            const double got = scores[qi * kRows + t];
            double ref = 0;
            for (int d = 0; d < kDim; ++d) {
                ref += static_cast<double>(__bfloat162float(h_rows[t * kDim + d])) *
                       __bfloat162float(h_q[qi * kDim + d]);
            }
            sxy += got * ref;
            sxx += got * got;
            sqy += ref * ref;
            if (got > nd) { ++rank; }
            if (got == nd) { ++ties; }
        }
        --ties;
        const double cos = sxy / (std::sqrt(sxx) * std::sqrt(sqy) + 1e-300);
        cos_sum += cos;
        std::printf("query %d (needle @ %d): rank %lld (ties %lld), dot-cosine %.6f\n",
                    qi, needle[qi], rank, ties, cos);
        if (rank != 1 || ties != 0) { ++missed; }
    }
    std::printf("aggregate dot-cosine %.6f, %d/5 unique top-1\n", cos_sum / kQueries,
                kQueries - missed);

    if (negative_control) {
        if (missed == 0) {
            std::fprintf(stderr, "NEGATIVE CONTROL FAILED TO FAIL\n");
            return EXIT_FAILURE;
        }
        std::printf("NEGATIVE CONTROL OK: %d/5 missed (failure path demonstrated)\n", missed);
        return 0;
    }
    if (missed != 0) {
        std::fprintf(stderr, "RETRIEVAL FAILURE: %d/5 needles missed at %.3f bps\n", missed, bps);
        return EXIT_FAILURE;
    }
    std::printf("RETRIEVAL PASS: 5/5 needles at unique top-1, %.3f bps\n", bps);
    return 0;
}
