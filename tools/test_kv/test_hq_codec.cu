// Standalone qualification for the hq-e8-2b row codec (tools/test_kv
// convention). Every check has a failure path: any violation exits nonzero.
//
//   1. E8int device quantizer == exhaustive FP64 brute-force nearest point
//      (both cosets; the half-integer coset must be exercised).
//   2. Row codec: device encode -> device decode reproduces the independent
//      FP64 oracle's decoded values (rotation, normalization, alpha,
//      escalation included) within bf16 rounding.
//   3. Budget invariant: every encoded row reports <= 512 used bits and
//      decode is deterministic across repeated runs.
//   4. Saturation: heavy-tailed rows never overflow (fallback engages).
//   5. Reconstruction quality: inverse-rotated rows keep cosine/snr at the
//      calibrated 2 bits/dim operating point.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ops/kernel/hq_codec.cuh"

using namespace ninfer::ops;

namespace {

int g_failed = 0;

void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++g_failed;
    }
}

double bf16_bits_to_float(unsigned short bits) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short*>(&v) = bits;
    return __bfloat162float(v);
}

// ---- device wrappers --------------------------------------------------------

__global__ void e8_quantize_kernel(const float* x, int* y, int n_blocks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) { return; }
    float x8[8];
    int y8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { x8[j] = x[i * 8 + j]; }
    hq_quantize_e8int(x8, y8);
#pragma unroll
    for (int j = 0; j < 8; ++j) { y[i * 8 + j] = y8[j]; }
}

__global__ void encode_rows_kernel(const __nv_bfloat16* rows, const std::int8_t* signs,
                                   std::uint8_t* codes, std::uint8_t* meta, int n_rows) {
    extern __shared__ float smem[];  // per warp: 256 floats + 256 uint32
    const int warp =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    if (warp >= n_rows) { return; }
    float* u    = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
    hq_encode_row_warp(rows + static_cast<std::size_t>(warp) * kHqHeadDim, signs, 0, u,
                       syms, codes + static_cast<std::size_t>(warp) * kHqRowBudgetBytes,
                       meta + static_cast<std::size_t>(warp) * kHqMetaBytes);
}

__global__ void decode_rows_kernel(const std::uint8_t* codes, const std::uint8_t* meta,
                                   __nv_bfloat16* out, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) { return; }
    hq_decode_row_thread(codes + static_cast<std::size_t>(i) * kHqRowBudgetBytes,
                         meta + static_cast<std::size_t>(i) * kHqMetaBytes,
                         out + static_cast<std::size_t>(i) * kHqHeadDim);
}

__global__ void decode_rows_group_kernel(const std::uint8_t* codes, const std::uint8_t* meta,
                                         __nv_bfloat16* out, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = i >> 3;
    if (row >= n_rows) { return; }
    hq_decode_row_group(codes + static_cast<std::size_t>(row) * kHqRowBudgetBytes,
                        meta + static_cast<std::size_t>(row) * kHqMetaBytes,
                        out + static_cast<std::size_t>(row) * kHqHeadDim, i & 7);
}

// ---- host oracles --------------------------------------------------------------

void host_fwht(double* v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                const double a = v[i + j];
                const double b = v[i + j + len];
                v[i + j]      = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}

// Exhaustive FP64 nearest point of 2*E8 (Chebyshev reach 4 per coset).
double e8_bf_nearest(const double x[8], int y_out[8]) {
    double best = 1e300;
    for (int c = 0; c < 2; ++c) {
        int base[8];
        for (int i = 0; i < 8; ++i) {
            base[i] = static_cast<int>(std::nearbyint((x[i] - c) / 2.0)) * 2 + c;
        }
        int o[7] = {-2, -2, -2, -2, -2, -2, -2};
        while (true) {
            int y[8];
            int partial = 0;
            for (int i = 0; i < 7; ++i) {
                y[i] = base[i] + 2 * o[i];
                partial += y[i];
            }
            // y7 steps by 2 from base[7]: enumerate ONLY same-parity
            // (valid E8int) candidates — mixed-parity points are not in
            // the lattice and would fake a closer "nearest".
            for (int y7 = base[7] - 4; y7 <= base[7] + 4; y7 += 2) {
                if (((partial + y7) & 3) != 0) { continue; }
                double d2 = 0.0;
                for (int i = 0; i < 8; ++i) {
                    const double e = static_cast<double>(i == 7 ? y7 : y[i]) - x[i];
                    d2 += e * e;
                }
                if (d2 < best - 1e-12) {
                    best = d2;
                    for (int i = 0; i < 7; ++i) { y_out[i] = y[i]; }
                    y_out[7] = y7;
                }
            }
            int p = 6;
            while (p >= 0 && ++o[p] > 2) {
                o[p] = -2;
                --p;
            }
            if (p < 0) { break; }
        }
    }
    return best;
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // ---- 1. E8 exactness --------------------------------------------------
    {
        const int n_blocks = 400;
        int exact = 0, odd = 0;
        for (float alpha : {1.0f, 1.45f, 6.0f}) {
            std::vector<float> hx(n_blocks * 8);
            std::vector<int> hy(n_blocks * 8);
            for (auto& v : hx) { v = gauss(rng) * alpha; }
            float* dx; int* dy;
            cudaMalloc(&dx, hx.size() * 4);
            cudaMalloc(&dy, hy.size() * 4);
            cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
            e8_quantize_kernel<<<(n_blocks + 255) / 256, 256>>>(dx, dy, n_blocks);
            cudaDeviceSynchronize();
            cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);
            cudaFree(dx);
            cudaFree(dy);
            for (int b = 0; b < n_blocks; ++b) {
                double xd[8];
                for (int i = 0; i < 8; ++i) { xd[i] = hx[b * 8 + i]; }
                int ybf[8];
                const double dbf = e8_bf_nearest(xd, ybf);
                double ddev = 0.0;
                for (int i = 0; i < 8; ++i) {
                    const double e = hy[b * 8 + i] - xd[i];
                    ddev += e * e;
                }
                check(ddev <= dbf + 1e-9 * (1.0 + dbf),
                      "E8 device point farther than brute-force nearest");
                if (ddev <= dbf + 1e-9 * (1.0 + dbf)) { ++exact; }
                if (hy[b * 8] & 1) { ++odd; }
            }
        }
        const double frac = static_cast<double>(odd) / (3.0 * n_blocks);
        std::printf("[1] E8 exact: %d/1200, odd-coset fraction %.3f\n", exact, frac);
        check(frac > 0.20 && frac < 0.80, "half-integer coset collapsed");
    }

    // ---- 2/3/4/5. Row codec roundtrip + budget + saturation + quality -----
    {
        constexpr int kNormal = 20000;
        constexpr int kHeavy  = 2000;
        constexpr int kRows   = kNormal + kHeavy;
        std::vector<float> hrows(kRows * kHqHeadDim);
        for (int r = 0; r < kRows; ++r) {
            const bool heavy = r >= kNormal;
            for (int d = 0; d < kHqHeadDim; ++d) {
                hrows[r * kHqHeadDim + d] = gauss(rng) * (heavy ? (1.0f + 9.0f * gauss(rng) * gauss(rng)) : 1.0f);
            }
        }
        std::vector<__nv_bfloat16> hrows_bf(kRows * kHqHeadDim);
        for (size_t i = 0; i < hrows.size(); ++i) { hrows_bf[i] = __float2bfloat16(hrows[i]); }

        std::vector<std::int8_t> hsigns(kHqHeadDim);
        std::mt19937 srng(0x5EED);
        std::uniform_int_distribution<int> coin(0, 1);
        for (auto& s : hsigns) { s = coin(srng) ? 1 : -1; }

        __nv_bfloat16* d_rows; std::int8_t* d_signs; std::uint8_t* d_codes; std::uint8_t* d_meta;
        __nv_bfloat16* d_out; __nv_bfloat16* d_out2;
        cudaMalloc(&d_rows, hrows_bf.size() * 2);
        cudaMalloc(&d_signs, hsigns.size());
        cudaMalloc(&d_codes, static_cast<std::size_t>(kRows) * kHqRowBudgetBytes);
        cudaMalloc(&d_meta, static_cast<std::size_t>(kRows) * kHqMetaBytes);
        cudaMalloc(&d_out, hrows_bf.size() * 2);
        cudaMalloc(&d_out2, hrows_bf.size() * 2);
        cudaMemcpy(d_rows, hrows_bf.data(), hrows_bf.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_signs, hsigns.data(), hsigns.size(), cudaMemcpyHostToDevice);

        constexpr int kWarpsPerBlock = 8;
        const size_t smem = kWarpsPerBlock * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * 4;
        encode_rows_kernel<<<(kRows + kWarpsPerBlock - 1) / kWarpsPerBlock,
                             kWarpsPerBlock * 32, smem>>>(d_rows, d_signs, d_codes, d_meta, kRows);
        cudaDeviceSynchronize();
        decode_rows_kernel<<<(kRows + 255) / 256, 256>>>(d_codes, d_meta, d_out, kRows);
        cudaDeviceSynchronize();
        decode_rows_kernel<<<(kRows + 255) / 256, 256>>>(d_codes, d_meta, d_out2, kRows);
        cudaDeviceSynchronize();
        // The 8-lane cooperative decoder (the production row decoder) must
        // be bit-identical to the sequential reference on every row.
        __nv_bfloat16* d_out4;
        cudaMalloc(&d_out4, hrows_bf.size() * 2);
        decode_rows_group_kernel<<<(kRows * 8 + 255) / 256, 256>>>(d_codes, d_meta, d_out4,
                                                                   kRows);
        cudaDeviceSynchronize();

        std::vector<std::uint8_t> hmeta(static_cast<std::size_t>(kRows) * kHqMetaBytes);
        std::vector<std::uint16_t> hout(kRows * kHqHeadDim);
        std::vector<std::uint16_t> hout2(kRows * kHqHeadDim);
        std::vector<std::uint16_t> hout4(kRows * kHqHeadDim);
        cudaMemcpy(hmeta.data(), d_meta, hmeta.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hout.data(), d_out, kRows * kHqHeadDim * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hout2.data(), d_out2, kRows * kHqHeadDim * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hout4.data(), d_out4, kRows * kHqHeadDim * 2, cudaMemcpyDeviceToHost);

        // Budget invariant + determinism + group equivalence.
        int budget_bad = 0, nondet = 0, grp_bad = 0;
        for (int r = 0; r < kRows; ++r) {
            const unsigned bits = hmeta[r * 8 + 3] | (unsigned(hmeta[r * 8 + 4] & 3) << 8);
            if (bits == 0 || bits > 512u) { ++budget_bad; }
            for (int d = 0; d < kHqHeadDim; ++d) {
                if (hout[r * kHqHeadDim + d] != hout2[r * kHqHeadDim + d]) {
                    ++nondet;
                    break;
                }
            }
            for (int d = 0; d < kHqHeadDim; ++d) {
                if (hout[r * kHqHeadDim + d] != hout4[r * kHqHeadDim + d]) {
                    ++grp_bad;
                    break;
                }
            }
        }
        std::printf("[3] budget invariant bad rows: %d, nondeterministic rows: %d, "
                    "group-mismatched rows: %d\n",
                    budget_bad, nondet, grp_bad);
        check(budget_bad == 0, "row exceeded the 512-bit budget");
        check(nondet == 0, "decode is not deterministic");
        check(grp_bad == 0, "8-lane group decode differs from sequential decode");

        // Escalation and terminal-fallback census. Escalated rows are those
        // whose first packing attempt overflowed the 512-bit budget; the
        // terminal fallback (32 bytes of 0xFF) decodes to an all-zero row, so
        // any fallback row outside a crafted corpus is silently dropped K/V.
        {
            std::vector<std::uint8_t> hcodes(static_cast<std::size_t>(kRows) * kHqRowBudgetBytes);
            cudaMemcpy(hcodes.data(), d_codes, hcodes.size(), cudaMemcpyDeviceToHost);
            int esc_normal = 0, esc_heavy = 0, fb_normal = 0, fb_heavy = 0;
            for (int r = 0; r < kRows; ++r) {
                const bool heavy = r >= kNormal;
                const int esc = (hmeta[r * 8 + 2] >> 4) & 3;
                bool fallback = true;
                for (int b = 0; b < 32; ++b) {
                    if (hcodes[static_cast<std::size_t>(r) * kHqRowBudgetBytes + b] != 0xFF) {
                        fallback = false;
                        break;
                    }
                }
                if (esc != 0) { heavy ? ++esc_heavy : ++esc_normal; }
                if (fallback) { heavy ? ++fb_heavy : ++fb_normal; }
            }
            std::printf("[3c] escalated rows: %d normal / %d heavy; terminal-fallback "
                        "(zeroed) rows: %d normal / %d heavy\n",
                        esc_normal, esc_heavy, fb_normal, fb_heavy);
            check(fb_normal == 0 && fb_heavy == 0,
                  "rows hit the terminal fallback (silently zeroed K/V) instead of the "
                  "alpha-halving rescue");
        }

        // Stored-format invariant: a row that fits the 512-bit budget at all
        // fits at k=0 with fewer-or-equal bits, and the encoder visits k=0
        // first with strict <, so every stored row must carry Rice k = 0.
        // The group decoder's unary fast path relies on this; k > 0 rows can
        // only come from a foreign encoder and take the general-k fallback.
        int k_nonzero = 0;
        for (int r = 0; r < kRows; ++r) {
            if ((hmeta[r * 8 + 2] & 0x0F) != 0) { ++k_nonzero; }
        }
        std::printf("[3b] rows with Rice k != 0: %d\n", k_nonzero);
        check(k_nonzero == 0, "encoder stored a row with Rice k != 0 (breaks the k=0 proof)");

        // Oracle comparison on a sample (the exhaustive FP64 nearest-point
        // is ~400k evaluations per word; full-corpus would run for hours).
        // Quality metrics below cover every normal row via the cheap
        // FWHT-only path.
        constexpr int kOracleSample = 200;
        double max_rel = 0.0;
        double sig = 0.0, noise = 0.0, cos_xy = 0.0, cos_xx = 0.0, cos_yy = 0.0;
        int oracle_bad = 0, row_bad = 0, tie_flips = 0;
        for (int r = 0; r < kNormal; ++r) {
            const unsigned esc = (hmeta[r * 8 + 2] >> 4) & 3u;
            std::vector<double> rot(kHqHeadDim);
            double norm = 0.0;
            for (int d = 0; d < kHqHeadDim; ++d) {
                const double x = __bfloat162float(hrows_bf[r * kHqHeadDim + d]);
                norm += x * x;
            }
            norm = std::sqrt(norm);
            for (int d = 0; d < kHqHeadDim; ++d) {
                rot[d] = __bfloat162float(hrows_bf[r * kHqHeadDim + d]) *
                         static_cast<double>(hsigns[d]);
            }
            host_fwht(rot.data(), kHqHeadDim);
            // host_fwht is UNNORMALIZED: u_scaled = fwht(x.s)*alpha/(2^esc*norm),
            // and the decoded rotated value = y*norm*(2^esc)/(alpha*sqrt(256)).
            const double scale = kHqAlpha / (static_cast<double>(1 << esc) * norm);
            const double inv = norm * static_cast<double>(1 << esc) / (kHqAlpha * 16.0);
            // One 2*E8 coordinate step in decoded space is 2*inv. The device
            // quantizes FP32-staged rows while this oracle quantizes FP64, so
            // boundary ties legitimately flip one coordinate by exactly one
            // step; anything else is an error. A row-level 2% relative-L2
            // bound catches zeroed or wrongly-scaled rows that per-coordinate
            // slack alone cannot.
            const double lattice_step = 2.0 * inv;
            double row_err2 = 0.0, row_sig2 = 0.0;
            if (r < kOracleSample) {
                for (int w = 0; w < kHqWordsPerRow; ++w) {
                    double x8[8];
                    for (int j = 0; j < 8; ++j) { x8[j] = rot[w * 8 + j] * scale; }
                    int y[8];
                    e8_bf_nearest(x8, y);
                    for (int j = 0; j < 8; ++j) {
                        const double expected = y[j] * inv;
                        const double got = bf16_bits_to_float(
                            hout[r * kHqHeadDim + w * 8 + j]);
                        const double err = std::fabs(got - expected);
                        if (err > 0.02 * std::fabs(expected) + 0.01) {
                            if (std::fabs(err - lattice_step) <= 0.1 * lattice_step) {
                                ++tie_flips;
                            } else {
                                ++oracle_bad;
                            }
                        }
                        row_err2 += err * err;
                        row_sig2 += expected * expected;
                        const double denom = std::fabs(expected) + 0.01;
                        max_rel = std::max(max_rel, err / denom);
                        sig += expected * expected;
                        noise += err * err;
                    }
                }
                if (row_sig2 > 0.0 && row_err2 > 0.0004 * row_sig2) { ++row_bad; }
            }
            // Quality in the ORIGINAL frame: inverse-rotate the decoded row
            // and compare to the source row.
            std::vector<double> g_rot(kHqHeadDim);
            for (int d = 0; d < kHqHeadDim; ++d) {
                g_rot[d] = bf16_bits_to_float(hout[r * kHqHeadDim + d]);
            }
            host_fwht(g_rot.data(), kHqHeadDim);
            for (int d = 0; d < kHqHeadDim; ++d) {
                const double s = static_cast<double>(hsigns[d]) / 16.0;
                const double src = __bfloat162float(hrows_bf[r * kHqHeadDim + d]);
                const double g = g_rot[d] * s;
                cos_xy += src * g;
                cos_xx += src * src;
                cos_yy += g * g;
            }
        }
        const double snr = 10.0 * std::log10(sig / noise);
        const double cos = cos_xy / (std::sqrt(cos_xx) * std::sqrt(cos_yy) + 1e-300);
        // Escalation fidelity: oracle-check EVERY escalated row on the corpus
        // (the census proves they exist; a fixed sample would gate on seed
        // luck), and band the count so both "escalation stopped firing" and
        // "escalation exploded" trip. On this corpus the fixed seed produces
        // 78 normal + 11 heavy escalated rows.
        int esc_oracle_bad = 0, esc_row_bad = 0, esc_count = 0;
        for (int r = 0; r < kRows; ++r) {
            const unsigned esc = (hmeta[r * 8 + 2] >> 4) & 3u;
            if (esc == 0) { continue; }
            ++esc_count;
            std::vector<double> rot(kHqHeadDim);
            double norm = 0.0;
            for (int d = 0; d < kHqHeadDim; ++d) {
                const double x = __bfloat162float(hrows_bf[r * kHqHeadDim + d]);
                norm += x * x;
            }
            norm = std::sqrt(norm);
            for (int d = 0; d < kHqHeadDim; ++d) {
                rot[d] = __bfloat162float(hrows_bf[r * kHqHeadDim + d]) *
                         static_cast<double>(hsigns[d]);
            }
            host_fwht(rot.data(), kHqHeadDim);
            const double scale = kHqAlpha / (static_cast<double>(1 << esc) * norm);
            const double inv = norm * static_cast<double>(1 << esc) / (kHqAlpha * 16.0);
            const double lattice_step = 2.0 * inv;
            double row_err2 = 0.0, row_sig2 = 0.0;
            for (int w = 0; w < kHqWordsPerRow; ++w) {
                double x8[8];
                for (int j = 0; j < 8; ++j) { x8[j] = rot[w * 8 + j] * scale; }
                int y[8];
                e8_bf_nearest(x8, y);
                for (int j = 0; j < 8; ++j) {
                    const double expected = y[j] * inv;
                    const double got = bf16_bits_to_float(hout[r * kHqHeadDim + w * 8 + j]);
                    const double err = std::fabs(got - expected);
                    if (err > 0.02 * std::fabs(expected) + 0.01 &&
                        std::fabs(err - lattice_step) > 0.1 * lattice_step) {
                        ++esc_oracle_bad;
                    }
                    row_err2 += err * err;
                    row_sig2 += expected * expected;
                }
            }
            if (row_sig2 > 0.0 && row_err2 > 0.0004 * row_sig2) { ++esc_row_bad; }
        }
        std::printf("[2] oracle mismatches: %d, tie flips: %d, bad rows: %d, max rel %.4f\n",
                    oracle_bad, tie_flips, row_bad, max_rel);
        std::printf("[5] rotated-frame decoder-vs-oracle SNR %.2f dB, original-frame cosine %.6f\n",
                    snr, cos);
        std::printf("[2b] escalated rows: %d (band 50-200), %d oracle mismatches, %d bad rows\n",
                    esc_count, esc_oracle_bad, esc_row_bad);
        check(oracle_bad == 0,
              "decoded rows deviate from the FP64 oracle beyond tie flips");
        check(row_bad == 0, "oracle rows exceed 2% relative-L2 error (zeroed or mis-scaled)");
        check(tie_flips <= 8 * kOracleSample / 200,
              "FP32/FP64 quantizer tie flips exceeded the calibration budget");
        // Nominal pooled cosine at alpha = 1.45 is ~0.938 (error orthogonal to signal,
        // energy ratio 1.137); the 0.93 floor leaves ~0.008 of margin as a tripwire.
        check(cos > 0.93,
              "original-frame cosine fell below 0.93 (nominal ~0.938 at alpha 1.45): 2 "
              "bits/dim reconstruction quality changed");
        check(esc_count >= 50 && esc_count <= 200,
              "escalation count left its corpus band (rescue behavior changed)");
        check(esc_oracle_bad == 0 && esc_row_bad == 0,
              "escalated rows deviate from the FP64 oracle");

        cudaFree(d_rows);
        cudaFree(d_signs);
        cudaFree(d_codes);
        cudaFree(d_meta);
        cudaFree(d_out);
        cudaFree(d_out2);
        cudaFree(d_out4);
    }

    // ---- 6. Synthetic Rice k > 0 rows: general-k fallback ------------------
    // The encoder never stores k > 0 (see [3b]), so these streams are written
    // by a host reference writer (no device counterpart since the parallel
    // packer replaced HqBitWriter; u32 values MSB-first, stored little-endian)
    //) to exercise the group decoder's general-k branch (remainder
    // extraction, cross-window symbols, the unary-guard tail) against
    // the sequential reference.
    {
        struct HostBitWriter {
            std::uint32_t buf[kHqRowBudgetBytes / 4] = {0};
            std::uint32_t cur = 0;
            int cur_bits = 0, written = 0;
            void put(std::uint32_t v, int n) {
                for (int i = n - 1; i >= 0; --i) {
                    cur = (cur << 1) | ((v >> i) & 1u);
                    if (++cur_bits == 32) {
                        buf[written++] = cur;
                        cur            = 0;
                        cur_bits       = 0;
                    }
                }
            }
            void put_rice(std::uint32_t z, std::uint32_t k) {
                std::uint32_t q = z >> k;
                while (q >= 32) {
                    put(0u, 31);
                    q -= 31;
                }
                put(1u, q + 1);
                if (k > 0) { put(z & ((1u << k) - 1u), static_cast<int>(k)); }
            }
            int flush() {
                if (cur_bits > 0) {
                    buf[written++] = cur << (32 - cur_bits);
                    cur            = 0;
                    cur_bits       = 0;
                }
                return written * 32;
            }
        };

        constexpr int kSynRows = 96;
        std::mt19937 srng2(0xC0FFEE);
        std::vector<std::uint8_t> syn_codes(static_cast<std::size_t>(kSynRows) * kHqRowBudgetBytes, 0);
        std::vector<std::uint8_t> syn_meta(static_cast<std::size_t>(kSynRows) * kHqMetaBytes, 0);
        for (int r = 0; r < kSynRows; ++r) {
            HostBitWriter w;
            const std::uint32_t k = 1 + (r % 3);  // k in {1, 2, 3}
            // k=1 rows can hold 256 tiny symbols; k>=2 rows exhaust the budget
            // early, which exercises the guarded tail on both decoders.
            std::uniform_int_distribution<int> zdist(0, static_cast<int>(1u << (k + 2)));
            int symbols = 0;
            while (symbols < kHqHeadDim) {
                const int budget_left =
                    kHqRowBudgetBytes * 8 - (w.written * 32 + w.cur_bits);
                if (budget_left < 2 + static_cast<int>(k) + 36) { break; }
                w.put_rice(static_cast<std::uint32_t>(zdist(srng2)), k);
                ++symbols;
            }
            int used = w.flush();
            if (r % 16 == 15) {
                // A few all-zero-stream rows with used > 0: every symbol hits
                // the unary guard on the sequential path.
                w = HostBitWriter();
                used = 256;
            }
            const std::uint16_t norm_bits = 0x3C00;  // 1.0f in fp16
            syn_meta[r * 8 + 0] = static_cast<std::uint8_t>(norm_bits & 0xFF);
            syn_meta[r * 8 + 1] = static_cast<std::uint8_t>(norm_bits >> 8);
            syn_meta[r * 8 + 2] = static_cast<std::uint8_t>(k);
            syn_meta[r * 8 + 3] = static_cast<std::uint8_t>(used & 0xFF);
            syn_meta[r * 8 + 4] = static_cast<std::uint8_t>((used >> 8) & 3);
            std::memcpy(syn_codes.data() + r * kHqRowBudgetBytes, w.buf, kHqRowBudgetBytes);
        }

        std::uint8_t* d_sc; std::uint8_t* d_sm;
        __nv_bfloat16 *d_so, *d_so2;
        cudaMalloc(&d_sc, syn_codes.size());
        cudaMalloc(&d_sm, syn_meta.size());
        cudaMalloc(&d_so, static_cast<std::size_t>(kSynRows) * kHqHeadDim * 2);
        cudaMalloc(&d_so2, static_cast<std::size_t>(kSynRows) * kHqHeadDim * 2);
        cudaMemcpy(d_sc, syn_codes.data(), syn_codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sm, syn_meta.data(), syn_meta.size(), cudaMemcpyHostToDevice);
        decode_rows_kernel<<<(kSynRows + 255) / 256, 256>>>(d_sc, d_sm, d_so, kSynRows);
        cudaDeviceSynchronize();
        decode_rows_group_kernel<<<(kSynRows * 8 + 255) / 256, 256>>>(d_sc, d_sm, d_so2, kSynRows);
        cudaDeviceSynchronize();
        std::vector<std::uint16_t> so(kSynRows * kHqHeadDim), so2(kSynRows * kHqHeadDim);
        cudaMemcpy(so.data(), d_so, so.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(so2.data(), d_so2, so2.size() * 2, cudaMemcpyDeviceToHost);
        int syn_bad = 0;
        for (int r = 0; r < kSynRows; ++r) {
            for (int d = 0; d < kHqHeadDim; ++d) {
                if (so[r * kHqHeadDim + d] != so2[r * kHqHeadDim + d]) {
                    ++syn_bad;
                    break;
                }
            }
        }
        std::printf("[6] synthetic k>0 fallback mismatched rows: %d\n", syn_bad);
        check(syn_bad == 0, "group decoder general-k fallback differs from sequential decode");
        cudaFree(d_sc);
        cudaFree(d_sm);
        cudaFree(d_so);
        cudaFree(d_so2);
    }

    if (g_failed != 0) {
        std::fprintf(stderr, "test_hq_codec: %d failures\n", g_failed);
        return EXIT_FAILURE;
    }
    std::printf("test_hq_codec: ALL PASSED\n");
    return 0;
}
