#pragma once

// ninfer::ops - hq-e8-2b prompt-scale kernels: fill (quantize current-chunk
// K/V rows into the fixed-budget E8+Rice pages) and the scratch decode pass
// that materializes the visible history once per attention call. The prompt
// attention itself is the shared FA2 bf16 kernel (gqa_attention_prefill_bf16)
// instantiated in its rotated frame: it runs over the decoded scratch planes,
// rotates its query rows in shared memory, and un-rotates its output rows in
// the epilogue. Numerical contract: identical ideal attention oracle as the
// other cache dtypes, judged through the hq-e8-2b compute profile (rotated-
// frame decode, FP32 online softmax, single un-rotation of the output rows).
//
// Addressing: one (position, kv_head) row = 64 code bytes + 8 metadata bytes,
// both page-major via paged_kv_element_offset<kHqCodeRowBytes/8, KVHeads>.
// The RHT signs are derived from the fixed engine hash (see hq_engine_sign),
// so fill and attention agree with no extra plumbing.

#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/hq_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaHqFillWarps      = 8;
inline constexpr int kGqaHqScratchThreads = 256;

// Row base offsets inside the code/meta planes for one (position, kv_head).
template <typename Geometry>
__device__ __forceinline__ const std::uint8_t* hq_row_codes(const std::uint8_t* plane,
                                                            const std::int32_t* block_table,
                                                            std::int32_t kv_head,
                                                            std::int32_t position) {
    const std::int32_t page = block_table[position >> kPagedKVPageShift];
    const std::int32_t off  = position & kPagedKVPageMask;
    return plane + paged_kv_element_offset<kHqCodePlaneExtent, Geometry::KVHeads>(page,
                                                                                 kv_head, off, 0);
}

template <typename Geometry>
__device__ __forceinline__ const std::uint8_t* hq_row_meta(const std::uint8_t* plane,
                                                           const std::int32_t* block_table,
                                                           std::int32_t kv_head,
                                                           std::int32_t position) {
    const std::int32_t page = block_table[position >> kPagedKVPageShift];
    const std::int32_t off  = position & kPagedKVPageMask;
    return plane + paged_kv_element_offset<kHqMetaPlaneExtent, Geometry::KVHeads>(page,
                                                                                 kv_head, off, 0);
}

template <typename Geometry>
__device__ __forceinline__ std::uint8_t* hq_row_codes_mut(std::uint8_t* plane,
                                                          const std::int32_t* block_table,
                                                          std::int32_t kv_head,
                                                          std::int32_t position) {
    return const_cast<std::uint8_t*>(hq_row_codes<Geometry>(plane, block_table, kv_head, position));
}

template <typename Geometry>
__device__ __forceinline__ std::uint8_t* hq_row_meta_mut(std::uint8_t* plane,
                                                         const std::int32_t* block_table,
                                                         std::int32_t kv_head,
                                                         std::int32_t position) {
    return const_cast<std::uint8_t*>(hq_row_meta<Geometry>(plane, block_table, kv_head, position));
}

// ---- fill -------------------------------------------------------------------

// One warp quantizes one (token, kv_head, role) row. Shared layout per warp:
// 256 staging floats + 256 symbol words; one signs block per CTA. Like the
// bf16/i8 fills, only the valid prefix of the chunk is written. With the
// residual window on, the same rows are dual-written exactly (rotated bf16)
// into the side planes and the recent-ring validity bit is set; the codec
// planes stay complete so side rows can fall back to the codec path.
template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_fill_hq_kernel(const __nv_bfloat16* k,
                                                     const __nv_bfloat16* v,
                                                     const std::int32_t* positions,
                                                     Metadata metadata, std::uint8_t* codes_k,
                                                     std::uint8_t* codes_v, std::uint8_t* meta_k,
                                                     std::uint8_t* meta_v, std::int32_t tokens,
                                                     __nv_bfloat16* residual_k = nullptr,
                                                     __nv_bfloat16* residual_v = nullptr,
                                                     std::uint32_t* ring_valid = nullptr) {
    extern __shared__ float smem[];
    std::int8_t* signs = reinterpret_cast<std::int8_t*>(smem + kGqaHqFillWarps *
                                                                    (kHqSmemFloatsPerRow +
                                                                     kHqSmemSymbolsPerRow));
    hq_engine_signs_fill(signs);
    __syncthreads();

    const std::int32_t valid = metadata.valid_tokens(tokens);
    const int warp =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const std::int64_t units =
        static_cast<std::int64_t>(valid) * Geometry::KVHeads * 2;
    if (warp >= units) { return; }
    float* u_scaled = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u_scaled + kHqSmemFloatsPerRow);

    const int token = static_cast<int>(warp % valid);
    const int unit  = warp / valid;
    const int head  = unit % Geometry::KVHeads;
    const bool role_v = unit / Geometry::KVHeads != 0;
    const __nv_bfloat16* src = (role_v ? v : k) + gqa_kv_quant_src_index<Geometry>(head, 0, token);

    const std::int32_t position = positions[0] + token;
    const std::int32_t* table   = metadata.block_table();
    hq_encode_row_warp(src, signs, 0, u_scaled, syms,
                       hq_row_codes_mut<Geometry>(role_v ? codes_v : codes_k, table, head,
                                                  position),
                       hq_row_meta_mut<Geometry>(role_v ? meta_v : meta_k, table, head, position),
                       hq_dither_row_seed(head, position, role_v));
    if (residual_k != nullptr) {
        // A chunk wider than the ring contains key pairs congruent mod
        // kGqaHqRecentKeys that map to the same ring slot; the LATER key owns
        // the slot at the chunk's end window (the earlier key is already
        // outside every future recent window), so a warp whose key is
        // superseded within this chunk skips the write instead of racing it.
        const std::int32_t total = positions[0] + valid;
        if (position < static_cast<std::int32_t>(kGqaHqSinkKeys) ||
            position + static_cast<std::int32_t>(kGqaHqRecentKeys) >= total) {
            const std::int32_t slot = metadata.residual_slot();
            hq_store_rotated_row_warp(
                src, signs, hq_residual_row<Geometry>(role_v ? residual_v : residual_k, slot, head,
                                                      position));
            hq_ring_mark_valid(ring_valid == nullptr
                                   ? nullptr
                                   : ring_valid + static_cast<std::int64_t>(slot) *
                                                      (static_cast<int>(kGqaHqRecentKeys) / 32),
                               position);
        }
    }
}

// ---- scratch decode -----------------------------------------------------------

// Decode the prompt call's entire visible history [0, positions[0] + valid)
// for every kv head and both roles into contiguous bf16 scratch rows in the
// ROTATED frame: scratch[role][kv_head][position][256]. EIGHT LANES per row
// (the cooperative group decoder: unary fast path with prefix-sum symbol
// boundaries); every query tile of the shared FA2 prompt kernel then reads
// decoded bf16 rows instead of each re-walking the serial Rice stream over
// its causal prefix. Sink and recent-ring rows come EXACT from the residual
// side planes when the feature is on (one 16 B copy per lane quarter-row);
// cleared ring slots fall back to the codec path.
//
// `span` is the scratch row count per head (the execution envelope's key
// bound); the launch grid is sized for that bound and threads beyond the
// device-computed history length return immediately.
template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_hq_scratch_kernel(
    const std::uint8_t* codes_k, const std::uint8_t* codes_v,
    const std::uint8_t* meta_k, const std::uint8_t* meta_v, Metadata metadata,
    const std::int32_t* positions, std::int32_t width, std::int32_t span,
    __nv_bfloat16* scratch_k, __nv_bfloat16* scratch_v, std::int32_t key_begin = 0,
    std::int32_t band_rows = 0x7fffffff, const __nv_bfloat16* residual_k = nullptr,
    const __nv_bfloat16* residual_v = nullptr, const std::uint32_t* ring_valid = nullptr,
    bool has_fresh = false) {
    const std::int32_t tid = static_cast<std::int32_t>(blockIdx.x) * static_cast<std::int32_t>(blockDim.x) +
                             static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t keys_total = positions[0] + metadata.valid_tokens(width);
    // Banded decode: rows [0, count) of the scratch hold absolute keys
    // [key_begin, key_begin + count); count clamps to the visible window.
    const std::int32_t band_end   = min(min(keys_total, key_begin + band_rows), span + key_begin);
    const std::int32_t count      = max(0, band_end - key_begin);
    const std::int32_t units      = count * Geometry::KVHeads * 2;
    const std::int32_t unit = tid >> 3;
    if (unit >= units) { return; }
    const std::int32_t lane8 = tid & 7;
    const std::int32_t pos  = key_begin + unit / (Geometry::KVHeads * 2);
    const std::int32_t rem  = unit - (unit / (Geometry::KVHeads * 2)) * (Geometry::KVHeads * 2);
    const std::int32_t head = rem >> 1;
    const bool role_v       = (rem & 1) != 0;
    const std::int32_t* table = metadata.block_table();
    __nv_bfloat16* dst = (role_v ? scratch_v : scratch_k) +
                        (static_cast<std::int64_t>(head) * span + (pos - key_begin)) * kHqHeadDim;
    const std::int32_t slot = metadata.residual_slot();
    // Fresh-chunk rows were staged exact by gqa_attention_prefill_fresh_rotate_kernel
    // (launched before this kernel on the same stream); with fresh coverage the ring
    // serves the W keys BEFORE the chunk instead of the window tail.
    const std::int32_t fresh_from = has_fresh ? positions[0] : -1;
    const std::int32_t ring_from =
        fresh_from >= 0 ? fresh_from - static_cast<std::int32_t>(kGqaHqRecentKeys)
                        : keys_total - static_cast<std::int32_t>(kGqaHqRecentKeys);
    const bool fresh_row = fresh_from >= 0 && pos >= fresh_from;
    const bool side_row =
        !fresh_row && residual_k != nullptr &&
        (pos < static_cast<std::int32_t>(kGqaHqSinkKeys) ||
         (pos >= ring_from && pos < (fresh_from >= 0 ? fresh_from : keys_total) &&
          hq_ring_slot_valid(ring_valid == nullptr
                                 ? nullptr
                                 : ring_valid + static_cast<std::int64_t>(slot) *
                                                    (static_cast<int>(kGqaHqRecentKeys) / 32),
                             pos)));
    if (fresh_row) {
        return;
    } else if (side_row) {
        const __nv_bfloat16* side =
            hq_residual_row<Geometry>(role_v ? residual_v : residual_k, slot, head, pos);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            store_vec(dst + lane8 * 32 + j * 8, load_vec<int4>(side + lane8 * 32 + j * 8));
        }
    } else {
        hq_decode_row_group(hq_row_codes<Geometry>(role_v ? codes_v : codes_k, table, head, pos),
                            hq_row_meta<Geometry>(role_v ? meta_v : meta_k, table, head, pos), dst,
                            lane8, 0, hq_dither_row_seed(head, pos, role_v));
    }
}

// ---- fresh-chunk rotate -------------------------------------------------------
//
// REVIEW-1m-context §8 D2: prompt-phase exactness needs the chunk being
// prefilled, not just the ring. The ring holds the chunk's LAST W rows, so a
// query early in a wide chunk would otherwise see none of its own recent
// window exact. This companion pass rotates the CURRENT call's bf16 k/v rows
// (the tensors are in hand at the A1 prompt route) straight into the scratch —
// one warp per (token, kv_head, role), the same single FWHT rounding as the
// dual-write — so every prefill query has its full in-chunk recent window
// exact, plus the ring for the W keys before the chunk (the scratch kernel's
// ring bound moves to [positions[0] - W, positions[0]) when this pass ran).
// Launched per scratch band; warps whose position falls outside the band (or
// beyond the valid chunk prefix) return immediately.
template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_fresh_rotate_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
    Metadata metadata, std::int32_t width, std::int32_t span, __nv_bfloat16* scratch_k,
    __nv_bfloat16* scratch_v, std::int32_t key_begin = 0, std::int32_t band_rows = 0x7fffffff) {
    extern __shared__ std::int8_t signs_raw[];
    hq_engine_signs_fill(reinterpret_cast<std::int8_t*>(signs_raw));
    __syncthreads();
    const std::int32_t valid = metadata.valid_tokens(width);
    const int warp =
        static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const std::int64_t units = static_cast<std::int64_t>(valid) * Geometry::KVHeads * 2;
    if (warp >= units) { return; }
    const int token        = static_cast<int>(warp % valid);
    const int unit         = static_cast<int>(warp / valid);
    const int head         = unit % Geometry::KVHeads;
    const bool role_v      = unit / Geometry::KVHeads != 0;
    const std::int32_t pos = positions[0] + token;
    if (pos < key_begin || pos >= key_begin + band_rows || pos >= span + key_begin) { return; }
    const __nv_bfloat16* src = (role_v ? v : k) + gqa_kv_quant_src_index<Geometry>(head, 0, token);
    __nv_bfloat16* dst       = (role_v ? scratch_v : scratch_k) +
                        (static_cast<std::int64_t>(head) * span + (pos - key_begin)) * kHqHeadDim;
    hq_store_rotated_row_warp(src, reinterpret_cast<std::int8_t*>(signs_raw), dst);
}

inline constexpr std::size_t kGqaHqFillSmemBytes =
    kGqaHqFillWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * sizeof(float) + kHqHeadDim;

} // namespace ninfer::ops
