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
// bf16/i8 fills, only the valid prefix of the chunk is written.
template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_fill_hq_kernel(const __nv_bfloat16* k,
                                                     const __nv_bfloat16* v,
                                                     const std::int32_t* positions,
                                                     Metadata metadata, std::uint8_t* codes_k,
                                                     std::uint8_t* codes_v, std::uint8_t* meta_k,
                                                     std::uint8_t* meta_v,
                                                     std::int32_t tokens) {
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
                       hq_row_meta_mut<Geometry>(role_v ? meta_v : meta_k, table, head, position));
}

// ---- scratch decode -----------------------------------------------------------

// Decode the prompt call's entire visible history [0, positions[0] + valid)
// for every kv head and both roles into contiguous bf16 scratch rows in the
// ROTATED frame: scratch[role][kv_head][position][256]. EIGHT LANES per row
// (the cooperative group decoder: unary fast path with prefix-sum symbol
// boundaries); every query tile of the shared FA2 prompt kernel then reads
// decoded bf16 rows instead of each re-walking the serial Rice stream over
// its causal prefix.
//
// `span` is the scratch row count per head (the execution envelope's key
// bound); the launch grid is sized for that bound and threads beyond the
// device-computed history length return immediately.
template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_hq_scratch_kernel(
    const std::uint8_t* codes_k, const std::uint8_t* codes_v,
    const std::uint8_t* meta_k, const std::uint8_t* meta_v, Metadata metadata,
    const std::int32_t* positions, std::int32_t width, std::int32_t span,
    __nv_bfloat16* scratch_k, __nv_bfloat16* scratch_v) {
    const std::int32_t tid = static_cast<std::int32_t>(blockIdx.x) * static_cast<std::int32_t>(blockDim.x) +
                             static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t keys_total = positions[0] + metadata.valid_tokens(width);
    const std::int32_t units      = keys_total * Geometry::KVHeads * 2;
    const std::int32_t unit = tid >> 3;
    if (unit >= units) { return; }
    const std::int32_t lane8 = tid & 7;
    const std::int32_t pos  = unit / (Geometry::KVHeads * 2);
    const std::int32_t rem  = unit - pos * (Geometry::KVHeads * 2);
    const std::int32_t head = rem >> 1;
    const bool role_v       = (rem & 1) != 0;
    const std::int32_t* table = metadata.block_table();
    __nv_bfloat16* dst = (role_v ? scratch_v : scratch_k) +
                        (static_cast<std::int64_t>(head) * span + pos) * kHqHeadDim;
    hq_decode_row_group(hq_row_codes<Geometry>(role_v ? codes_v : codes_k, table, head, pos),
                        hq_row_meta<Geometry>(role_v ? meta_v : meta_k, table, head, pos), dst,
                        lane8);
}

inline constexpr std::size_t kGqaHqFillSmemBytes =
    kGqaHqFillWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * sizeof(float) + kHqHeadDim;

} // namespace ninfer::ops
