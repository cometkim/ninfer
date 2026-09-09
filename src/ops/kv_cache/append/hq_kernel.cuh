#pragma once

// hq-e8-2b append kernels. One warp quantizes one (token, kv_head, role) row through the
// E8-lattice + Rice codec into the fixed 64-byte code row and 8-byte metadata row; the page
// addressing is page-major through paged_kv_element_offset exactly like the fixed-width
// codecs, so capacity math and CUDA Graph address stability are unchanged. The RHT signs come
// from the fixed engine-global hash (hq_engine_signs_fill), shared with every consumer.

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/hq_e8_rice_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheHqFillWarps = 8;
inline constexpr std::size_t kKVCacheHqFillSmemBytes =
    kKVCacheHqFillWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * sizeof(float) + kHqHeadDim;

// Row base offsets inside the code/meta planes for one (position, kv_head).
template <typename Geometry>
__device__ __forceinline__ const std::uint8_t*
kv_cache_hq_row_codes(const std::uint8_t* plane, const std::int32_t* block_table,
                      std::int32_t kv_head, std::int32_t position) {
    const std::int32_t page = paged_kv_physical_page(block_table, position);
    const std::int32_t off  = position & kPagedKVPageMask;
    return plane +
           paged_kv_element_offset<kHqCodePlaneExtent, Geometry::KVHeads>(page, kv_head, off, 0);
}

template <typename Geometry>
__device__ __forceinline__ const std::uint8_t*
kv_cache_hq_row_meta(const std::uint8_t* plane, const std::int32_t* block_table,
                     std::int32_t kv_head, std::int32_t position) {
    const std::int32_t page = paged_kv_physical_page(block_table, position);
    const std::int32_t off  = position & kPagedKVPageMask;
    return plane +
           paged_kv_element_offset<kHqMetaPlaneExtent, Geometry::KVHeads>(page, kv_head, off, 0);
}

template <typename Geometry>
__device__ __forceinline__ std::uint8_t*
kv_cache_hq_row_codes_mut(std::uint8_t* plane, const std::int32_t* block_table,
                          std::int32_t kv_head, std::int32_t position) {
    return const_cast<std::uint8_t*>(
        kv_cache_hq_row_codes<Geometry>(plane, block_table, kv_head, position));
}

template <typename Geometry>
__device__ __forceinline__ std::uint8_t*
kv_cache_hq_row_meta_mut(std::uint8_t* plane, const std::int32_t* block_table, std::int32_t kv_head,
                         std::int32_t position) {
    return const_cast<std::uint8_t*>(
        kv_cache_hq_row_meta<Geometry>(plane, block_table, kv_head, position));
}

// Source row index in the [head_dim, kv_heads, tokens] K/V input tensor.
template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_hq_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kHqHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry, typename Metadata>
__global__ void kv_cache_append_full_hq_kernel(const __nv_bfloat16* __restrict__ k,
                                               const __nv_bfloat16* __restrict__ v,
                                               const std::int32_t* __restrict__ positions,
                                               Metadata metadata, std::uint8_t* codes_k,
                                               std::uint8_t* codes_v, std::uint8_t* meta_k,
                                               std::uint8_t* meta_v, std::int32_t width) {
    extern __shared__ float smem[];
    std::int8_t* signs = reinterpret_cast<std::int8_t*>(
        smem + kKVCacheHqFillWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow));
    hq_engine_signs_fill(signs);
    __syncthreads();

    const std::int32_t valid = metadata.valid_tokens(width);
    const int warp = static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const std::int64_t units = static_cast<std::int64_t>(valid) * Geometry::KVHeads * 2;
    if (warp >= units) { return; }
    float* u_scaled     = smem + (threadIdx.x >> 5) * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
    std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u_scaled + kHqSmemFloatsPerRow);

    const int token          = static_cast<int>(warp % valid);
    const int unit           = warp / valid;
    const int head           = unit % Geometry::KVHeads;
    const bool role_v        = unit / Geometry::KVHeads != 0;
    const __nv_bfloat16* src = (role_v ? v : k) + kv_cache_hq_src_index<Geometry>(head, 0, token);

    const std::int32_t position = positions[0] + token;
    const std::int32_t* table   = metadata.block_table();
    hq_encode_row_warp(
        src, signs, 0, u_scaled, syms,
        kv_cache_hq_row_codes_mut<Geometry>(role_v ? codes_v : codes_k, table, head, position),
        kv_cache_hq_row_meta_mut<Geometry>(role_v ? meta_v : meta_k, table, head, position));
}

} // namespace ninfer::ops
