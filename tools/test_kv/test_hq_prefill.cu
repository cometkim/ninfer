// Standalone qualification for the hq-e8-2b PROMPT route: fill (quantize) ->
// scratch decode -> rotated-frame FA2 prompt kernel, judged against an
// independent FP64 attention oracle over the same decoded cache rows. The
// oracle mirrors the device bit format exactly (Rice/MSB-first, unstrip,
// FP16 norm, escalation) but evaluates the full causal attention in FP64 from
// the represented public inputs (BF16 q, decoded K/V rows). Any violation
// exits nonzero. Two scenarios: direct metadata with an identity table, and
// the engine's batch metadata with a shuffled physical page table.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>
#include <vector>

#include "hq_dither_host.h"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"

using namespace ninfer::ops;
using ninfer::kPagedKVPageSize;

namespace {

using Geo = Gqa27Geometry;
constexpr float kScale = 0.0625f;

int g_failed = 0;

void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++g_failed;
    }
}

void host_fwht(double* v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                const double a = v[i + j];
                const double b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}

// Host mirror of the engine-global RHT diagonal (hq_engine_sign).
double host_engine_sign(int d) {
    std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    return (x & 1u) ? 1.0 : -1.0;
}

// Exact host Rice/unstrip decode of one row -> FP64 rotated-frame values
// (dither included, mirroring the encode-side seeds).
void host_decode_row(const std::uint8_t* codes, const std::uint8_t* meta, int head, int pos,
                     bool role_v, std::vector<double>& out) {
    const std::uint64_t row_seed = hqhost::dither_row_seed(head, pos, role_v);
    const unsigned norm_bits =
        static_cast<unsigned>(meta[0]) | (static_cast<unsigned>(meta[1]) << 8);
    __half h;
    *reinterpret_cast<unsigned short*>(&h) = static_cast<unsigned short>(norm_bits);
    const double norm   = static_cast<double>(__half2float(h));
    const unsigned k   = meta[2] & 0x0Fu;
    const unsigned esc = (meta[2] >> 4) & 0x3u;
    const double inv_scale =
        static_cast<double>(1u << esc) / (static_cast<double>(kHqAlpha) * 16.0);
    const unsigned used =
        static_cast<unsigned>(meta[3]) | (static_cast<unsigned>(meta[4] & 0x3) << 8);
    if (used == 0) {
        out.assign(kHqHeadDim, 0.0);
        return;
    }
    // MSB-first bit cursor over the code words.
    const std::uint32_t* words = reinterpret_cast<const std::uint32_t*>(codes);
    size_t bit = 0;
    const auto next_bit = [&]() -> int {
        const size_t w = bit >> 5;
        if (w >= kHqRowBudgetBytes / 4) { ++bit; return 0; }
        const int b = static_cast<int>((words[w] >> (31 - (bit & 31))) & 1u);
        ++bit;
        return b;
    };
    for (int w = 0; w < kHqWordsPerRow; ++w) {
        unsigned z[8];
        for (int j = 0; j < 8; ++j) {
            unsigned q = 0;
            while (next_bit() == 0) {
                if (++q > kHqUnaryGuard) { break; }
            }
            unsigned r = 0;
            for (unsigned b = 0; b < k; ++b) { r = (r << 1) | static_cast<unsigned>(next_bit()); }
            z[j] = (q >= kHqUnaryGuard) ? 0u : ((q << k) | r);
        }
        // Mirror of hq_unstrip_word.
        const std::uint64_t wseed = hqhost::dither_word_seed(row_seed, w);
        int s[8];
        int p = 0;
        for (int i = 0; i < 7; ++i) {
            s[i] = (z[i] & 1u) ? -static_cast<int>((z[i] + 1u) >> 1) : static_cast<int>(z[i] >> 1);
        }
        for (int i = 0; i < 7; ++i) { p += s[i]; }
        const int t  = static_cast<int>(z[7] >> 1);
        const int t2 = (t & 1u) ? -static_cast<int>((t + 1u) >> 1) : static_cast<int>(t >> 1);
        const int c  = static_cast<int>(z[7] & 1u);
        s[7]         = ((t2 << 1) + (p & 1));
        for (int j = 0; j < 8; ++j) {
            const int y = (s[j] << 1) + c;
            out[w * 8 + j] =
                (static_cast<double>(y) + hqhost::dither(wseed, j)) * norm * inv_scale;
        }
    }
}

template <typename Metadata>
int run_scenario(int base, int width, int span, bool shuffle, unsigned seed,
                 bool residual = false) {
    const int keys    = base + width;
    const int logical = (keys + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const int phys    = logical + 5;

    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<__nv_bfloat16> hq(static_cast<size_t>(width) * Geo::QHeads * kHqHeadDim);
    std::vector<__nv_bfloat16> hk(static_cast<size_t>(keys) * Geo::KVHeads * kHqHeadDim);
    std::vector<__nv_bfloat16> hv(hk.size());
    for (auto& v : hq) { v = __float2bfloat16(gauss(rng)); }
    for (auto& v : hk) { v = __float2bfloat16(gauss(rng) * 0.8f); }
    for (auto& v : hv) { v = __float2bfloat16(gauss(rng) * 0.8f); }

    // Logical -> physical page table (shuffled when requested; page 0 is a
    // never-addressed decoy in that case).
    std::vector<std::int32_t> table(logical);
    for (int l = 0; l < logical; ++l) { table[l] = shuffle ? (l + 3) % phys : l; }

    const std::size_t plane_codes =
        static_cast<std::size_t>(kHqCodePlaneExtent) * kPagedKVPageSize * Geo::KVHeads * phys;
    const std::size_t plane_meta =
        static_cast<std::size_t>(kHqMetaPlaneExtent) * kPagedKVPageSize * Geo::KVHeads * phys;
    __nv_bfloat16 *d_q, *d_out, *d_k, *d_v, *d_sk, *d_sv;
    std::uint8_t *d_ck, *d_cv, *d_mk, *d_mv;
    std::int32_t *d_pos_a, *d_pos_b, *d_tables, *d_row;
    __nv_bfloat16 *d_rk = nullptr, *d_rv = nullptr;
    std::uint32_t* d_ring = nullptr;
    const std::size_t side_plane =
        static_cast<std::size_t>(kGqaHqSinkKeys + kGqaHqRecentKeys) * Geo::KVHeads * kHqHeadDim;
    if (residual) {
        cudaMalloc(&d_rk, side_plane * 2);
        cudaMalloc(&d_rv, side_plane * 2);
        const int ring_words = static_cast<int>(kGqaHqRecentKeys) / 32;
        cudaMalloc(&d_ring, static_cast<std::size_t>(ring_words) * 4);
        cudaMemset(d_rk, 0xCD, side_plane * 2);
        cudaMemset(d_rv, 0xCD, side_plane * 2);
        cudaMemset(d_ring, 0xFF, static_cast<std::size_t>(ring_words) * 4);
    }
    cudaMalloc(&d_q, hq.size() * 2);
    cudaMalloc(&d_out, hq.size() * 2);
    cudaMalloc(&d_k, hk.size() * 2);
    cudaMalloc(&d_v, hv.size() * 2);
    cudaMalloc(&d_ck, plane_codes);
    cudaMalloc(&d_cv, plane_codes);
    cudaMalloc(&d_mk, plane_meta);
    cudaMalloc(&d_mv, plane_meta);
    cudaMalloc(&d_sk, static_cast<std::size_t>(span) * Geo::KVHeads * kHqHeadDim * 2);
    cudaMalloc(&d_sv, static_cast<std::size_t>(span) * Geo::KVHeads * kHqHeadDim * 2);
    cudaMalloc(&d_pos_a, 4);
    cudaMalloc(&d_pos_b, 4);
    cudaMalloc(&d_tables, static_cast<std::size_t>(logical) * 4);
    cudaMalloc(&d_row, 4);
    cudaMemcpy(d_q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tables, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
    const std::int32_t pos_a = 0, pos_b = base, row0 = 0;
    cudaMemcpy(d_pos_a, &pos_a, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_b, &pos_b, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, &row0, 4, cudaMemcpyHostToDevice);
    cudaMemset(d_ck, 0, plane_codes);
    cudaMemset(d_cv, 0, plane_codes);
    cudaMemset(d_mk, 0, plane_meta);
    cudaMemset(d_mv, 0, plane_meta);
    cudaDeviceSynchronize();

    constexpr bool kBatch = std::is_same_v<Metadata, GqaPrefillBatchMetadata<false>>;
    const Metadata metadata = [&]() -> Metadata {
        if constexpr (kBatch) {
            return Metadata{d_tables, nullptr, d_row, logical};
        } else {
            return Metadata{d_tables};
        }
    }();

    const auto launch_fill = [&](const __nv_bfloat16* k, const __nv_bfloat16* v,
                                 const std::int32_t* pos, std::int32_t tokens) {
        constexpr int kWPB = kGqaHqFillWarps;
        const std::int64_t units = static_cast<std::int64_t>(tokens) * Geo::KVHeads * 2;
        gqa_attention_prefill_fill_hq_kernel<Geo, Metadata>
            <<<(units + kWPB - 1) / kWPB, kWPB * 32, kGqaHqFillSmemBytes>>>(
                k, v, pos, metadata, d_ck, d_cv, d_mk, d_mv, tokens, d_rk, d_rv, d_ring);
    };
    // Two engine-shaped chunks: history (if any) then current.
    if (base > 0) { launch_fill(d_k, d_v, d_pos_a, base); }
    launch_fill(d_k + static_cast<std::int64_t>(base) * Geo::KVHeads * kHqHeadDim,
                d_v + static_cast<std::int64_t>(base) * Geo::KVHeads * kHqHeadDim, d_pos_b,
                width);
    // Scratch decode + rotated FA2 prompt attention. When the span is smaller than the
    // visible key count, run the banded carry path (the production envelope > band case):
    // tile-aligned bands, FA2 resuming its online-softmax state between bands.
    const int keys_total  = base + width;
    const int bands       = (keys_total + span - 1) / span;
    __nv_bfloat16* d_cacc = nullptr;
    float *d_cm = nullptr, *d_cl = nullptr;
    if (bands > 1) {
        cudaMalloc(&d_cacc, static_cast<std::size_t>(width) * Geo::QHeads * kHqHeadDim * 2);
        cudaMalloc(&d_cm, static_cast<std::size_t>(width) * Geo::QHeads * 4);
        cudaMalloc(&d_cl, static_cast<std::size_t>(width) * Geo::QHeads * 4);
    }
    cudaError_t attr = cudaFuncSetAttribute(
        gqa_attention_prefill_bf16_kernel<Geo, Metadata, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kGqaPrefillRotatedSmemBytes));
    check(attr == cudaSuccess, "rotated FA2 smem opt-in rejected");
    if (bands > 1) {
        cudaError_t attr_carry = cudaFuncSetAttribute(
            gqa_attention_prefill_bf16_kernel<Geo, Metadata, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kGqaPrefillRotatedSmemBytes));
        check(attr_carry == cudaSuccess, "carry FA2 smem opt-in rejected");
    }
    const dim3 grid((width + kGqaPrefillBr - 1) / kGqaPrefillBr, Geo::QHeads, 1);
    const bool has_fresh = residual;
    for (int band = 0; band < bands; ++band) {
        const std::int32_t key_begin = band * span;
        const std::int32_t band_rows = std::min(span, keys_total - key_begin);
        if (has_fresh) {
            // REVIEW §8 D2: the current chunk's rows are staged exact from the
            // call's bf16 k/v (the A1 prompt route's fresh tensors).
            const std::int64_t fresh_units =
                static_cast<std::int64_t>(width) * Geo::KVHeads * 2;
            gqa_attention_prefill_fresh_rotate_kernel<Geo, Metadata>
                <<<static_cast<int>((fresh_units + kGqaHqFillWarps - 1) / kGqaHqFillWarps),
                   kGqaHqFillWarps * 32, kHqHeadDim>>>(
                    d_k + static_cast<std::int64_t>(base) * Geo::KVHeads * kHqHeadDim,
                    d_v + static_cast<std::int64_t>(base) * Geo::KVHeads * kHqHeadDim, d_pos_b,
                    metadata, width, span, d_sk, d_sv, key_begin, band_rows);
        }
        const int band_grid          = static_cast<int>(
            (static_cast<std::int64_t>(band_rows) * Geo::KVHeads * 2 * 8 +
             kGqaHqScratchThreads - 1) /
            kGqaHqScratchThreads);
        gqa_attention_prefill_hq_scratch_kernel<Geo, Metadata>
            <<<band_grid, kGqaHqScratchThreads>>>(d_ck, d_cv, d_mk, d_mv, metadata, d_pos_b, width,
                                                  span, d_sk, d_sv, key_begin, band_rows, d_rk,
                                                  d_rv, d_ring, has_fresh);
        if (bands == 1) {
            gqa_attention_prefill_bf16_kernel<Geo, Metadata, true>
                <<<grid, kGqaPrefillThreads, kGqaPrefillRotatedSmemBytes>>>(
                    d_q, d_sk, d_sv, metadata, d_pos_b, kScale, d_out, width, span);
        } else {
            gqa_attention_prefill_bf16_kernel<Geo, Metadata, true, true>
                <<<grid, kGqaPrefillThreads, kGqaPrefillRotatedSmemBytes>>>(
                    d_q, d_sk, d_sv, metadata, d_pos_b, kScale, d_out, width, span, key_begin,
                    key_begin + band_rows, d_cacc, d_cm, d_cl, band + 1 == bands ? 0 : 1);
        }
        cudaError_t err = cudaGetLastError();
        check(err == cudaSuccess, cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    std::printf("  kernels done (base=%d width=%d)\n", base, width);
    std::fflush(stdout);

    std::vector<__nv_bfloat16> hout(hq.size());
    cudaMemcpy(hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<std::uint8_t> hck(plane_codes), hcv(plane_codes), hmk(plane_meta), hmv(plane_meta);
    cudaMemcpy(hck.data(), d_ck, plane_codes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hcv.data(), d_cv, plane_codes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hmk.data(), d_mk, plane_meta, cudaMemcpyDeviceToHost);
    cudaMemcpy(hmv.data(), d_mv, plane_meta, cudaMemcpyDeviceToHost);

    // FP64 oracle over the same decoded rows.
    const auto row_ptr = [](std::vector<std::uint8_t>& plane, int extent, int page, int head,
                            int slot) {
        return plane.data() +
               (static_cast<std::size_t>(extent) * kPagedKVPageSize *
                    (static_cast<std::size_t>(head) + Geo::KVHeads * page) +
                static_cast<std::size_t>(extent) * slot);
    };
    std::vector<std::vector<double>> kr(Geo::KVHeads * keys), vr(Geo::KVHeads * keys);
    for (int h = 0; h < Geo::KVHeads; ++h) {
        for (int p = 0; p < keys; ++p) {
            const int page = table[p >> kPagedKVPageShift];
            const int slot = p & kPagedKVPageMask;
            kr[h * keys + p].resize(kHqHeadDim);
            host_decode_row(row_ptr(hck, kHqCodePlaneExtent, page, h, slot),
                            row_ptr(hmk, kHqMetaPlaneExtent, page, h, slot), h, p, false,
                            kr[h * keys + p]);
            vr[h * keys + p].resize(kHqHeadDim);
            host_decode_row(row_ptr(hcv, kHqCodePlaneExtent, page, h, slot),
                            row_ptr(hmv, kHqMetaPlaneExtent, page, h, slot), h, p, true,
                            vr[h * keys + p]);
        }
    }
    if (residual) {
        // Exact sources under D2: sink rows and the pre-chunk ring come from the
        // side planes (dual-written by the earlier chunk's fill); the fresh chunk
        // [base, keys) comes from the call's bf16 k/v rotated in fp64 (the device
        // rotate kernel stages fp32 — the row gates below absorb the difference).
        std::vector<__nv_bfloat16> hrk(side_plane), hrv(side_plane);
        cudaMemcpy(hrk.data(), d_rk, side_plane * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hrv.data(), d_rv, side_plane * 2, cudaMemcpyDeviceToHost);
        const int sink = static_cast<int>(kGqaHqSinkKeys);
        const int ring = static_cast<int>(kGqaHqRecentKeys);
        for (int h = 0; h < Geo::KVHeads; ++h) {
            for (int p = 0; p < keys; ++p) {
                const bool plane_row = p < sink || (p >= base - ring && p < base);
                const bool fresh_row = p >= base;
                if (!plane_row && !fresh_row) { continue; }
                if (fresh_row) {
                    std::vector<double> rot(kHqHeadDim);
                    for (int d = 0; d < kHqHeadDim; ++d) {
                        rot[d] = __bfloat162float(
                            hk[(static_cast<std::size_t>(p) * Geo::KVHeads + h) * kHqHeadDim + d]) *
                                 host_engine_sign(d);
                    }
                    host_fwht(rot.data(), kHqHeadDim);
                    for (int d = 0; d < kHqHeadDim; ++d) {
                        kr[h * keys + p][d] = rot[d] / 16.0;
                    }
                    // V is rotated too (planes store both roles rotated).
                    std::vector<double> rotv(kHqHeadDim);
                    for (int d = 0; d < kHqHeadDim; ++d) {
                        rotv[d] =
                            __bfloat162float(
                                hv[(static_cast<std::size_t>(p) * Geo::KVHeads + h) * kHqHeadDim +
                                    d]) *
                            host_engine_sign(d);
                    }
                    host_fwht(rotv.data(), kHqHeadDim);
                    for (int d = 0; d < kHqHeadDim; ++d) {
                        vr[h * keys + p][d] = rotv[d] / 16.0;
                    }
                } else {
                    const int row = p < sink ? p : sink + (p & (ring - 1));
                    const std::size_t off =
                        (static_cast<std::size_t>(row) * Geo::KVHeads + h) * kHqHeadDim;
                    for (int d = 0; d < kHqHeadDim; ++d) {
                        kr[h * keys + p][d] = __bfloat162float(hrk[off + d]);
                        vr[h * keys + p][d] = __bfloat162float(hrv[off + d]);
                    }
                }
            }
        }
    }

    double min_cos = 1.0, max_row_rel = 0.0, rel_sum = 0.0, rel_n = 0.0;
    int bad_cos = 0;
    std::vector<double> orow(kHqHeadDim), oout(kHqHeadDim);
    for (int t = 0; t < width; ++t) {
        for (int h = 0; h < Geo::QHeads; ++h) {
            const int kvh = h / Geo::GroupSize;
            const int last = base + t;
            std::vector<double> qr(kHqHeadDim);
            for (int d = 0; d < kHqHeadDim; ++d) {
                qr[d] = __bfloat162float(
                            hq[(static_cast<size_t>(t) * Geo::QHeads + h) * kHqHeadDim + d]) *
                        host_engine_sign(d);
            }
            host_fwht(qr.data(), kHqHeadDim);
            for (int d = 0; d < kHqHeadDim; ++d) { qr[d] /= 16.0; }
            double m = -1e300;
            std::vector<double> sc(last + 1);
            for (int x = 0; x <= last; ++x) {
                double dot = 0.0;
                const auto& krow = kr[kvh * keys + x];
                for (int d = 0; d < kHqHeadDim; ++d) { dot += qr[d] * krow[d]; }
                sc[x] = kScale * dot;
                m = std::max(m, sc[x]);
            }
            double l = 0.0;
            for (int x = 0; x <= last; ++x) { sc[x] = std::exp(sc[x] - m); l += sc[x]; }
            std::fill(orow.begin(), orow.end(), 0.0);
            for (int x = 0; x <= last; ++x) {
                const double p = sc[x] / l;
                const auto& vrow = vr[kvh * keys + x];
                for (int d = 0; d < kHqHeadDim; ++d) { orow[d] += p * vrow[d]; }
            }
            // Inverse rotation: butterflies first, signs and 1/sqrt(256) after
            // (R^-1 = diag(signs) . FWHT / 16; the forward is FWHT . diag(s)/16).
            for (int d = 0; d < kHqHeadDim; ++d) { oout[d] = orow[d]; }
            host_fwht(oout.data(), kHqHeadDim);
            double xy = 0.0, xx = 0.0, yy = 0.0, err2 = 0.0;
            for (int d = 0; d < kHqHeadDim; ++d) {
                const double g = __bfloat162float(
                    hout[(static_cast<size_t>(t) * Geo::QHeads + h) * kHqHeadDim + d]);
                const double o = oout[d] / 16.0 * host_engine_sign(d);
                xy += g * o;
                xx += o * o;
                yy += g * g;
                err2 += (g - o) * (g - o);
            }
            const double row_rel = std::sqrt(err2) / (std::sqrt(xx) + 1e-300);
            max_row_rel = std::max(max_row_rel, row_rel);
            rel_sum += row_rel;
            rel_n += 1.0;
            const double cos = xy / (std::sqrt(xx) * std::sqrt(yy) + 1e-300);
            min_cos = std::min(min_cos, cos);
            if (cos < 0.99) { ++bad_cos; }
        }
    }
    std::printf("[prefill base=%d width=%d span=%d shuffle=%d batch=%d] rows %d, min cos %.6f, "
                "mean row rel %.5f, max row rel %.5f, bad %d\n",
                base, width, span, shuffle ? 1 : 0, kBatch ? 1 : 0, width * Geo::QHeads, min_cos,
                rel_sum / rel_n, max_row_rel, bad_cos);
    check(bad_cos == 0, "prompt output rows below 0.99 cosine vs FP64 oracle");
    check(max_row_rel < 0.02, "prompt output rows exceed 2% row-norm deviation");

    cudaFree(d_q);
    cudaFree(d_out);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_ck);
    cudaFree(d_cv);
    cudaFree(d_mk);
    cudaFree(d_mv);
    cudaFree(d_sk);
    cudaFree(d_sv);
    cudaFree(d_pos_a);
    cudaFree(d_pos_b);
    cudaFree(d_tables);
    cudaFree(d_row);
    if (d_rk) { cudaFree(d_rk); }
    if (d_rv) { cudaFree(d_rv); }
    if (d_ring) { cudaFree(d_ring); }
    return g_failed;
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    int failed = 0;
    failed += run_scenario<GqaPrefillDirectMetadata>(100, 300, 1024, false, 20260821u);
    failed += run_scenario<GqaPrefillBatchMetadata<false>>(0, 67, 2048, true, 77u);
    // Banded carry: span below the visible key count. Boundaries tile-aligned (span
    // multiples of the 64-key FA2 tile) and the final band short (band_rows < span).
    failed += run_scenario<GqaPrefillDirectMetadata>(100, 300, 128, false, 31u);
    failed += run_scenario<GqaPrefillBatchMetadata<false>>(0, 300, 192, true, 32u);
    failed += run_scenario<GqaPrefillDirectMetadata>(0, 130, 64, false, 33u);
    // Engine-scale banded carry: the production band (262144) with a 390k-key window
    // crossing the first band boundary at tile 4096. Narrow width keeps the CPU oracle
    // affordable while exercising every large-index path.
    failed += run_scenario<GqaPrefillDirectMetadata>(390000, 8, 262144, false, 34u);
    // Residual window (WI-8): sink [0,32) + recent [keys-128, keys) rows exact from
    // the side planes, middle rows codec — single-band and banded variants.
    failed += run_scenario<GqaPrefillDirectMetadata>(100, 300, 1024, false, 71u, true);
    failed += run_scenario<GqaPrefillBatchMetadata<false>>(0, 300, 192, true, 72u, true);
    if (failed != 0) {
        std::fprintf(stderr, "test_hq_prefill: %d failures\n", failed);
        return EXIT_FAILURE;
    }
    std::printf("test_hq_prefill: ALL PASSED\n");
    return 0;
}
