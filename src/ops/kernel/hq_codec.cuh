#pragma once

// ninfer::ops - HyperQuant KV cache codec (shared device helpers): per-row
// randomized Hadamard rotation, E8 lattice true-nearest quantization,
// structural strip, and Rice entropy coding packed into a FIXED per-row byte
// budget. The format keeps the paged-KV contract's fixed-bytes-per-token
// property: every (token, kv_head) row occupies exactly kHqRowBudgetBytes of
// code plane and kHqMetaBytes of metadata plane, so capacity math, page
// addressing, and CUDA Graph address stability are unchanged from the
// fixed-width formats.
//
// Pipeline per row x (dim 256), all in the rotated frame:
//   1. norm = ||x||;  u = FWHT256(x . signs) * sqrt(256) / norm
//      (per-scalar variance ~1; signs are the fixed engine-global RHT
//      diagonal, K-role and V-role vectors of 256 signed bytes)
//   2. y = nearest point of 2*E8 for alpha * u  (both cosets tried, mod-4
//      sum parity fixed by the least-cost single +/-2 flip; this is the
//      exact Conway-Sloane decoder - the half-integer coset is NOT collapsed)
//   3. strip coset + parity redundancy -> 8 zigzagged symbols per 8-D word
//   4. Rice-pack all 32 words' symbols (256 symbols) into the row budget,
//      one Rice parameter per row chosen exactly from 8 accumulators
//
// Budget guarantee: a row whose symbols do not fit is re-encoded at
// alpha*2 (then alpha*4); a row of all-zero lattice codes always fits (256
// one-bit codes = 32 bytes <= budget), so encoding terminates in bounded
// deterministic time with no host involvement (graph-safe). Escalation is
// recorded in the metadata flags and is astronomically rare for
// post-rotation rows (Gaussian tail ~1e-5 at the shipped alpha).
//
// Storage planes per pool (one page id shared across all planes):
//   codes: U8 [kHqCodePlaneExtent=64, 64, Hkv, Nphysical]  (page-major)
//   meta : U8 [kHqMetaPlaneExtent=8, 64, Hkv, Nphysical]
// Meta row layout (8 bytes):
//   [0..1] FP16 bits of the row L2 norm (multiplier, like INT8-G64 scales)
//   [2]    Rice k (low 4 bits) | escalation count (bits 4..5) | reserved
//   [3..4] used bits: meta[3] plus bits 0..1 of meta[4] (<= 512, 10 bits)
//   [4]    bits 2..4: the 9th bit of each segment offset; bits 5..7 reserved
//   [5..7] segment offsets, low 8 bits: the BIT where symbol 64/128/192
//          starts (<= 512). Consumers decode one row with four threads, each
//          starting its serial Rice scan at these bit offsets; the bitstream
//          itself is identical to the sequential format.
//
// Dequantized row value (rotated frame): y[i] * norm / (alpha * sqrt(256)).
// Consumers rotate queries with the same signs+FWHT and un-rotate attention
// outputs once per output row (the rotation is orthogonal). Plane pointers
// are computed by the attention/fill kernels via paged_kv_element_offset
// with the extents below; this header stays pointer-free so it builds and
// qualifies standalone.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kHqHeadDim        = 256;
inline constexpr int kHqLatticeDim     = 8;
inline constexpr int kHqWordsPerRow    = kHqHeadDim / kHqLatticeDim; // 32
inline constexpr int kHqRowBudgetBytes = 64;  // 512-bit code budget = 2 bits/dim
inline constexpr int kHqMetaBytes      = 8;
inline constexpr int kHqCodePlaneExtent = kHqRowBudgetBytes;
inline constexpr int kHqMetaPlaneExtent = kHqMetaBytes;
inline constexpr int kHqMaxRiceK       = 7;
// A valid row fits 512 bits, so no legitimate unary quotient can exceed
// the budget; runs past that are zero/garbage padding and bail fast.
inline constexpr std::uint32_t kHqUnaryGuard = kHqRowBudgetBytes * 8;

// Calibrated for ~1.9 payload bits/scalar on unit-variance post-RHT rows
// (Philox Gaussian corpus, dim 256, measured on RTX 5090); kept slightly
// under the rate-matched value to leave saturation headroom.
inline constexpr float kHqAlpha = 1.45f;

// ---- zig-zag + E8int (2*E8) exact nearest point ---------------------------

__device__ __forceinline__ std::uint32_t hq_zigzag(std::int32_t v) {
    return v >= 0 ? (static_cast<std::uint32_t>(v) << 1)
                  : ((~static_cast<std::uint32_t>(v)) << 1) + 1u;
}

__device__ __forceinline__ std::int32_t hq_unzigzag(std::uint32_t z) {
    return (z & 1u) ? -static_cast<std::int32_t>((z + 1u) >> 1)
                    : static_cast<std::int32_t>(z >> 1);
}

__device__ __forceinline__ int hq_rint(float x) {
    return static_cast<int>(nearbyintf(x));
}

// Force sum(u) == 0 (mod 4) for a same-parity vector by one +/-2 flip at the
// least-confident coordinate. Exact for both E8int cosets.
template <int N>
__device__ __forceinline__ void hq_fix_parity_mod4(int (&u)[N], const float (&x)[N]) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < N; ++i) sum += u[i];
    if ((sum & 3) == 0) { return; }
    int best_j = 0;
    float best_abs = -1.0f;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const float delta = fabsf(static_cast<float>(u[i]) - x[i]);
        if (delta > best_abs) {
            best_abs = delta;
            best_j   = i;
        }
    }
    u[best_j] += (static_cast<float>(u[best_j]) >= x[best_j]) ? -2 : 2;
}

// Exact nearest point of 2*E8 = { y in Z^8 : shared parity, sum == 0 mod 4 }:
// nearest even-coset and nearest odd-coset candidates, each parity-fixed,
// closer one wins. The odd coset is the doubled half-integer coset of E8.
__device__ __forceinline__ void hq_quantize_e8int(const float (&x)[8], int (&y)[8]) {
    int u0[8], u1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        u0[i] = hq_rint(x[i] * 0.5f) * 2;
        u1[i] = hq_rint((x[i] - 1.0f) * 0.5f) * 2 + 1;
    }
    hq_fix_parity_mod4(u0, x);
    hq_fix_parity_mod4(u1, x);
    float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float e0 = static_cast<float>(u0[i]) - x[i];
        const float e1 = static_cast<float>(u1[i]) - x[i];
        d0 += e0 * e0;
        d1 += e1 * e1;
    }
    const int* pick = (d0 <= d1) ? u0 : u1;
#pragma unroll
    for (int i = 0; i < 8; ++i) { y[i] = pick[i]; }
}

// Structural strip (E8): 8 symbols from a valid lattice word.
__device__ __forceinline__ void hq_strip_word(const int (&y)[8], std::uint32_t (&z)[8]) {
    const std::uint32_t c = static_cast<std::uint32_t>(y[0] & 1);
    int s[8];
    int p = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { s[i] = (y[i] - static_cast<int>(c)) >> 1; }
#pragma unroll
    for (int i = 0; i < 7; ++i) { p += s[i]; }
    const int t = (s[7] - (p & 1)) >> 1;
#pragma unroll
    for (int i = 0; i < 7; ++i) { z[i] = hq_zigzag(s[i]); }
    z[7] = 2u * hq_zigzag(t) + c;
}

__device__ __forceinline__ void hq_unstrip_word(const std::uint32_t (&z)[8], int (&y)[8]) {
    int s[8];
    int p = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i) { s[i] = hq_unzigzag(z[i]); }
#pragma unroll
    for (int i = 0; i < 7; ++i) { p += s[i]; }
    const int t   = hq_unzigzag(z[7] >> 1);
    const int c   = static_cast<int>(z[7] & 1u);
    s[7]          = (t << 1) + (p & 1);
#pragma unroll
    for (int i = 0; i < 8; ++i) { y[i] = (s[i] << 1) + c; }
}

// ---- Rice bit primitives (row-local, word-aligned) -------------------------

__device__ __forceinline__ std::uint32_t hq_rice_bits(std::uint32_t z, std::uint32_t k) {
    return (z >> k) + 1u + k;
}

struct HqBitWriter {
    std::uint32_t cur;          // partial word being assembled MSB-first
    int cur_bits;               // bits currently in cur (0..31)
    int written;                // whole words committed to the row buffer
    std::uint32_t buf[kHqRowBudgetBytes / 4];
    bool overflow;

    __device__ HqBitWriter() : cur(0u), cur_bits(0), written(0), overflow(false) {
#pragma unroll
        for (int i = 0; i < kHqRowBudgetBytes / 4; ++i) { buf[i] = 0u; }
    }

    // Append the low n bits of v (n <= 32), MSB first, one bit at a time.
    // Deliberately simple: the funnel-shift version showed length-dependent
    // word corruption on sm_120a that resisted root-causing; at ~500 bits
    // per row the bit-loop cost is negligible for the append path.
    __device__ __forceinline__ void put(std::uint32_t v, int n) {
        if (n <= 0) { return; }
        v &= (n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
        for (int i = n - 1; i >= 0; --i) {
            cur = (cur << 1) | ((v >> i) & 1u);
            if (++cur_bits == 32) {
                if (written >= kHqRowBudgetBytes / 4) {
                    overflow = true;
                    cur_bits = 0;
                    cur      = 0u;
                    return;
                }
                buf[written++] = cur;
                cur            = 0u;
                cur_bits       = 0;
            }
        }
    }

    __device__ __forceinline__ void put_rice(std::uint32_t z, std::uint32_t k) {
        std::uint32_t q = z >> k;
        while (q >= 32) {
            put(0u, 31);
            if (overflow) { return; }
            q -= 31;
        }
        put(1u, q + 1);
        if (overflow) { return; }
        if (k > 0) { put(z & ((1u << k) - 1u), static_cast<int>(k)); }
    }

    __device__ __forceinline__ int bit_pos() const { return written * 32 + cur_bits; }

    __device__ int flush() {
        if (overflow) { return -1; }
        if (cur_bits > 0) {
            if (written >= kHqRowBudgetBytes / 4) { return -1; }
            buf[written++] = cur << (32 - cur_bits);
            cur      = 0u;
            cur_bits = 0;
        }
        return written * 32;
    }
};

struct HqBitReader {
    const std::uint32_t* words;
    int n_words;
    int word_idx;
    std::uint32_t c1, c2;
    int c1_bits, c2_bits;
    std::uint64_t reg;
    int bits_in_reg;

    __device__ HqBitReader(const std::uint32_t* w, int n)
        : words(w), n_words(n), c1(0u), c2(0u), c1_bits(0), c2_bits(0), reg(0ull),
          bits_in_reg(0) {
        word_idx = 2;
        c1       = (0 < n) ? w[0] : 0u;
        c2       = (1 < n) ? w[1] : 0u;
        c1_bits  = (0 < n) ? 32 : 0;
        c2_bits  = (1 < n) ? 32 : 0;
    }

    __device__ __forceinline__ void promote() {
        c1       = c2;
        c1_bits  = c2_bits;
        c2       = (word_idx < n_words) ? words[word_idx] : 0u;
        c2_bits  = 32;
        ++word_idx;
    }

    __device__ __forceinline__ void refill(int needed) {
        while (bits_in_reg < needed) {
            if (c1_bits == 0) { promote(); }
            const int room = 64 - bits_in_reg;
            const int take = (c1_bits < room) ? c1_bits : room;
            const std::uint32_t piece = c1 >> (32 - take);
            reg |= static_cast<std::uint64_t>(piece) << (64 - bits_in_reg - take);
            c1       = (take >= 32) ? 0u : (c1 << take);
            c1_bits -= take;
            bits_in_reg += take;
        }
    }

    __device__ __forceinline__ std::uint32_t read_bits(int n) {
        if (n <= 0) { return 0u; }
        refill(n);
        const std::uint32_t v = static_cast<std::uint32_t>(reg >> (64 - n));
        reg <<= n;
        bits_in_reg -= n;
        return v;
    }

    // Unary quotient via CLZ over the 64-bit register; corrupt all-zero runs
    // return the guard instead of walking.
    __device__ __forceinline__ std::uint32_t read_unary() {
        std::uint32_t q = 0u;
        while (true) {
            refill(1);
            std::uint64_t probe = reg;
            if (bits_in_reg < 64) {
                probe |= (1ull << (64 - bits_in_reg)) - 1ull;
            }
            const std::uint32_t lz =
                (probe == 0ull) ? 64u : static_cast<std::uint32_t>(__clzll(probe));
            q += lz;
            if (lz < static_cast<std::uint32_t>(bits_in_reg)) {
                const int consumed = static_cast<int>(lz) + 1;
                reg       = (consumed >= 64) ? 0ull : (reg << consumed);
                bits_in_reg -= consumed;
                return q;
            }
            reg         = 0ull;
            bits_in_reg = 0;
            if (q > kHqUnaryGuard) { return kHqUnaryGuard; }
        }
    }
};

// ---- warp-cooperative row codec -------------------------------------------
//
// One warp encodes or decodes one row. The encode path stages the rotated,
// alpha-scaled coordinates in shared memory (`u_scaled`, 256 floats) because
// the FWHT lives in per-lane registers while the E8 words need 8 consecutive
// coordinates in one thread.

constexpr int kHqSmemFloatsPerRow = kHqHeadDim;         // 256
constexpr int kHqSmemSymbolsPerRow = kHqHeadDim;        // 256 uint32

// In-register FWHT256 with sign pre-multiply (forward) for one row held as
// 8 register slots per lane (element e = slot*32 + lane). All shuffles run
// on the converged full warp.
__device__ __forceinline__ void hq_fwht256_sign(float (&reg)[8], const std::int8_t* signs,
                                                int sign_base, int lane) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        reg[s] *= static_cast<float>(signs[sign_base + s * 32 + lane]);
    }
#pragma unroll
    for (int len = 1; len < 32; len <<= 1) {
        const bool low = (lane & len) == 0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const float partner = __shfl_xor_sync(0xFFFFFFFFu, reg[s], len);
            reg[s] = low ? (reg[s] + partner) : (partner - reg[s]);
        }
    }
#pragma unroll
    for (int ls = 1; ls < 8; ls <<= 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if ((s & ls) == 0) {
                const float a = reg[s];
                const float b = reg[s + ls];
                reg[s]   = a + b;
                reg[s + ls] = a - b;
            }
        }
    }
    const float inv = rsqrtf(static_cast<float>(kHqHeadDim));
#pragma unroll
    for (int s = 0; s < 8; ++s) { reg[s] *= inv; }
}

// Inverse rotation: butterfly first, then signs and 1/sqrt(256).
__device__ __forceinline__ void hq_ifwht256_sign(float (&reg)[8], const std::int8_t* signs,
                                                 int sign_base, int lane) {
#pragma unroll
    for (int len = 1; len < 32; len <<= 1) {
        const bool low = (lane & len) == 0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const float partner = __shfl_xor_sync(0xFFFFFFFFu, reg[s], len);
            reg[s] = low ? (reg[s] + partner) : (partner - reg[s]);
        }
    }
#pragma unroll
    for (int ls = 1; ls < 8; ls <<= 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if ((s & ls) == 0) {
                const float a = reg[s];
                const float b = reg[s + ls];
                reg[s]   = a + b;
                reg[s + ls] = a - b;
            }
        }
    }
    const float inv = rsqrtf(static_cast<float>(kHqHeadDim));
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        reg[s] *= inv * static_cast<float>(signs[sign_base + s * 32 + lane]);
    }
}

// Encode one bf16 row (256 dims, contiguous) into codes[64] + meta[8].
// Scratch: u_scaled (256 floats) and syms (256 uint32) in shared memory,
// provided by the caller. One warp per row; lane 0 commits the outputs.
__device__ __forceinline__ void hq_encode_row_warp(const __nv_bfloat16* row,
                                                   const std::int8_t* signs, int sign_base,
                                                   float* u_scaled, std::uint32_t* syms,
                                                   std::uint8_t* codes_out,
                                                   std::uint8_t* meta_out) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    float reg[8];
    float sumsq = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int e = s * 32 + lane;
        reg[s] = __bfloat162float(row[e]);
        sumsq += reg[s] * reg[s];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sumsq += __shfl_down_sync(0xFFFFFFFFu, sumsq, off);
    }
    // __shfl_down leaves the full sum only on lane 0; broadcast it so every
    // lane scales with the SAME row norm.
    sumsq = __shfl_sync(0xFFFFFFFFu, sumsq, 0);
    const float norm = sqrtf(fmaxf(sumsq, 1e-30f));
    const float scale = kHqAlpha * sqrtf(static_cast<float>(kHqHeadDim)) / norm;
    hq_fwht256_sign(reg, signs, sign_base, lane);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        u_scaled[s * 32 + lane] = reg[s] * scale;
    }
    __syncwarp();

    int escalation = 0;
    int used_bits  = -1;
    std::uint32_t best_k = 0;
    int seg_off[3] = {0, 0, 0};
    HqBitWriter bw;
#pragma unroll 1
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (attempt > 0) {
            // Each retry doubles the effective alpha (cumulative 2x, 4x);
            // the decode side applies 1<<escalation to match.
#pragma unroll
            for (int i = lane * 8; i < lane * 8 + 8; ++i) { u_scaled[i] *= 2.0f; }
            __syncwarp();
        }
        // lane w owns word w: quantize + strip into syms.
        {
            float x8[8];
            int y8[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { x8[j] = u_scaled[lane * 8 + j]; }
            hq_quantize_e8int(x8, y8);
            std::uint32_t z8[8];
            hq_strip_word(y8, z8);
#pragma unroll
            for (int j = 0; j < 8; ++j) { syms[lane * 8 + j] = z8[j]; }
        }
        // Exact per-row Rice k from 8 shift-accumulators (warp reduction).
        std::uint32_t acc[kHqMaxRiceK + 1] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t zz = syms[lane * 8 + j];
#pragma unroll
            for (int k = 0; k <= kHqMaxRiceK; ++k) { acc[k] += zz >> k; }
        }
#pragma unroll
        for (int k = 0; k <= kHqMaxRiceK; ++k) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc[k] += __shfl_down_sync(0xFFFFFFFFu, acc[k], off);
            }
        }
        std::uint32_t best_bits = 0xFFFFFFFFu;
        best_k = 0;
#pragma unroll
        for (int k = 0; k <= kHqMaxRiceK; ++k) {
            const std::uint32_t bits = acc[k] + kHqHeadDim * (1u + k);
            if (bits < best_bits) {
                best_bits = bits;
                best_k    = static_cast<std::uint32_t>(k);
            }
        }
        // Lane 0 packs the WHOLE row from shared memory written by all lanes;
        // __shfl_*_sync in the reduction above converges execution but is not
        // a shared-memory fence, so an explicit warp barrier is required
        // before the cross-lane reads.
        __syncwarp();
        if (lane == 0) {
            bw           = HqBitWriter();
            escalation   = attempt;
            for (int w = 0; w < kHqWordsPerRow && !bw.overflow; ++w) {
                for (int j = 0; j < 8 && !bw.overflow; ++j) {
                    bw.put_rice(syms[w * 8 + j], best_k);
                }
                if (w == 7 || w == 15 || w == 23) {
                    seg_off[w == 7 ? 0 : (w == 15 ? 1 : 2)] = bw.bit_pos();
                }
            }
            used_bits = bw.flush();
        }
        used_bits = __shfl_sync(0xFFFFFFFFu, used_bits, 0);
        if (used_bits >= 0) { break; }
    }
    __syncwarp();
    best_k  = __shfl_sync(0xFFFFFFFFu, best_k, 0);
    escalation = __shfl_sync(0xFFFFFFFFu, escalation, 0);

    if (lane == 0) {
        __half h = __float2half_rn(norm);
        std::uint16_t norm_bits = *reinterpret_cast<std::uint16_t*>(&h);
        meta_out[0] = static_cast<std::uint8_t>(norm_bits & 0xFFu);
        meta_out[1] = static_cast<std::uint8_t>(norm_bits >> 8);
        std::uint32_t bits;
        if (used_bits >= 0) {
            bits = static_cast<std::uint32_t>(used_bits);
            meta_out[2] = static_cast<std::uint8_t>(best_k |
                                                    (static_cast<std::uint32_t>(escalation) << 4));
#pragma unroll
            for (int b = 0; b < kHqRowBudgetBytes; ++b) {
                codes_out[b] = reinterpret_cast<const std::uint8_t*>(bw.buf)[b];
            }
            meta_out[5] = static_cast<std::uint8_t>(seg_off[0]);
            meta_out[6] = static_cast<std::uint8_t>(seg_off[1]);
            meta_out[7] = static_cast<std::uint8_t>(seg_off[2]);
            meta_out[4] = static_cast<std::uint8_t>(
                ((used_bits >> 8) & 3u) | (((seg_off[0] >> 8) & 1u) << 2) |
                (((seg_off[1] >> 8) & 1u) << 3) | (((seg_off[2] >> 8) & 1u) << 4));
        } else {
            // Terminal fallback: all-zero lattice codes. 256 zero symbols at
            // k=0 are 256 one-bit codes = 32 bytes of 0xFF; every decoded
            // value is exactly zero regardless of norm or escalation.
            bits = 256u;
            meta_out[2] = 0u;
            meta_out[4] = 1u; // used = 256; offsets 64/128/192 fit 8 bits
#pragma unroll
            for (int b = 0; b < kHqRowBudgetBytes; ++b) {
                codes_out[b] = (b < 32) ? 0xFFu : 0x00u;
            }
            meta_out[5] = 64u;
            meta_out[6] = 128u;
            meta_out[7] = 192u;
        }
        meta_out[3] = static_cast<std::uint8_t>(bits & 0xFFu);
    }
}

// ---- engine-global RHT diagonal --------------------------------------------
//
// Deterministic engine-wide sign vector shared by the K and V roles (a single
// shared diagonal provides the isotropy the quantizer needs). Every frame
// boundary — fill encode, scratch decode, and the rotated-frame prompt
// attention kernel — must use this same diagonal.

__device__ __forceinline__ float hq_engine_sign(std::int32_t d) {
    std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    return (x & 1u) ? 1.0f : -1.0f;
}

// Fill signs[0..kHqHeadDim) from any block width (strided over blockDim.x).
__device__ __forceinline__ void hq_engine_signs_fill(std::int8_t* signs) {
    for (int i = static_cast<int>(threadIdx.x); i < kHqHeadDim;
         i += static_cast<int>(blockDim.x)) {
        signs[i] = static_cast<std::int8_t>(hq_engine_sign(i));
    }
}

// Decode one row's codes[64] + meta[8] into 256 bf16 values (rotated frame).
// ONE THREAD per row: the Rice scan is strictly serial within the row, so
// the engine stages tiles by giving each thread one row (the measured
// one-thread-per-row shape from the standalone codec benchmarks).
__device__ __forceinline__ void hq_decode_row_thread(const std::uint8_t* codes,
                                                     const std::uint8_t* meta,
                                                     __nv_bfloat16* out) {
    const std::uint16_t norm_bits = static_cast<std::uint16_t>(meta[0]) |
                                    (static_cast<std::uint16_t>(meta[1]) << 8);
    __half h;
    *reinterpret_cast<std::uint16_t*>(&h) = norm_bits;
    const float norm = __half2float(h);
    const std::uint32_t k = meta[2] & 0x0Fu;
    const std::uint32_t escalation = (meta[2] >> 4) & 0x3u;
    const float inv_scale =
        1.0f / (kHqAlpha * static_cast<float>(1u << escalation) *
                sqrtf(static_cast<float>(kHqHeadDim)));

    const unsigned used_bits =
        static_cast<unsigned>(meta[3]) | (static_cast<unsigned>(meta[4] & 0x3) << 8);
    if (used_bits == 0 || k > kHqMaxRiceK) {
        // Never-written row (zeroed metadata): decodes to exact zeros.
#pragma unroll 1
        for (int i = 0; i < kHqHeadDim; ++i) { out[i] = __float2bfloat16(0.0f); }
        return;
    }
    HqBitReader br(reinterpret_cast<const std::uint32_t*>(codes),
                   static_cast<int>((used_bits + 31) >> 5));
    std::uint32_t z[8];
#pragma unroll 1
    for (int w = 0; w < kHqWordsPerRow; ++w) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t q = br.read_unary();
            const std::uint32_t r = br.read_bits(static_cast<int>(k));
            z[j] = (q >= kHqUnaryGuard) ? 0u : ((q << k) | r);
        }
        int y[8];
        hq_unstrip_word(z, y);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[w * 8 + j] =
                __float2bfloat16(static_cast<float>(y[j]) * norm * inv_scale);
        }
    }
}

// Decode one QUARTER of a row's codes[64] + meta[8] (words
// [segment*8, segment*8+8), elements [segment*64, segment*64+64)) into bf16
// (rotated frame). FOUR THREADS per row, one segment each: each thread's
// serial Rice scan starts at the byte offset recorded by the encoder in
// meta[5..7], cutting the per-row dependency chain (and the exposed latency
// at low warp occupancy) by 4x. Produces bit-identical values to
// hq_decode_row_thread.
__device__ __forceinline__ void hq_decode_row_segment(const std::uint8_t* codes,
                                                      const std::uint8_t* meta,
                                                      __nv_bfloat16* out, int segment) {
    const std::uint16_t norm_bits = static_cast<std::uint16_t>(meta[0]) |
                                    (static_cast<std::uint16_t>(meta[1]) << 8);
    __half h;
    *reinterpret_cast<std::uint16_t*>(&h) = norm_bits;
    const float norm = __half2float(h);
    const std::uint32_t k = meta[2] & 0x0Fu;
    const std::uint32_t escalation = (meta[2] >> 4) & 0x3u;
    const float inv_scale =
        1.0f / (kHqAlpha * static_cast<float>(1u << escalation) *
                sqrtf(static_cast<float>(kHqHeadDim)));

    const unsigned used_bits =
        static_cast<unsigned>(meta[3]) | (static_cast<unsigned>(meta[4] & 0x3) << 8);
    if (used_bits == 0 || k > kHqMaxRiceK) {
        // Never-written row (zeroed metadata): decodes to exact zeros.
#pragma unroll 1
        for (int i = 0; i < kHqHeadDim / 4; ++i) {
            out[segment * (kHqHeadDim / 4) + i] = __float2bfloat16(0.0f);
        }
        return;
    }
    const unsigned start_bit =
        segment == 0 ? 0u
                     : static_cast<unsigned>(meta[4 + segment]) |
                           (((static_cast<unsigned>(meta[4]) >> (segment + 1)) & 1u) << 8);
    HqBitReader br(reinterpret_cast<const std::uint32_t*>(codes) + (start_bit >> 5),
                   (kHqRowBudgetBytes >> 2) - static_cast<int>(start_bit >> 5));
    const unsigned skip = start_bit & 31u;
    if (skip != 0u) { br.read_bits(static_cast<int>(skip)); }
    std::uint32_t z[8];
#pragma unroll 1
    for (int w = segment * 8; w < segment * 8 + 8; ++w) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t q = br.read_unary();
            const std::uint32_t r = br.read_bits(static_cast<int>(k));
            z[j] = (q >= kHqUnaryGuard) ? 0u : ((q << k) | r);
        }
        int y[8];
        hq_unstrip_word(z, y);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[w * 8 + j] =
                __float2bfloat16(static_cast<float>(y[j]) * norm * inv_scale);
        }
    }
}

} // namespace ninfer::ops
