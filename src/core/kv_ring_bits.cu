#include "core/kv_ring_bits.h"

#include <stdexcept>

namespace ninfer {
namespace {

__global__ void kv_ring_valid_apply_kernel(std::uint32_t* words, KvRingWords and_mask,
                                           KvRingWords or_mask) {
    const int i  = static_cast<int>(threadIdx.x);
    words[i]     = (words[i] & and_mask.w[i]) | or_mask.w[i];
}

__global__ void kv_residual_slot_copy_kernel(const char* __restrict__ src, char* __restrict__ dst,
                                             std::int64_t slot_stride, std::int64_t row_stride,
                                             std::int32_t layers, std::int32_t rows,
                                             std::int32_t source_slot,
                                             std::int32_t destination_slot) {
    // Side planes are [head_dim, kv_heads, side_rows, layers * rows]: layer L's slice for slot S
    // starts at (L * rows + S) * slot_stride, so one slot's inheritance copies `layers` separate
    // chunks of slot_stride bytes, row_stride apart.
    const std::int64_t total =
        static_cast<std::int64_t>(layers) * slot_stride;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t layer = i / slot_stride;
        const std::int64_t offset = i - layer * slot_stride;
        const std::int64_t src_base =
            (layer * rows + source_slot) * slot_stride + offset;
        const std::int64_t dst_base =
            (layer * rows + destination_slot) * slot_stride + offset;
        dst[dst_base] = src[src_base];
    }
}

} // namespace

void apply_kv_ring_valid_words(std::uint32_t* words, const KvRingWords& and_mask,
                               const KvRingWords& or_mask, int word_count, cudaStream_t stream) {
    if (words == nullptr || word_count <= 0) { return; }
    if (word_count > kKvSideWords) { throw std::invalid_argument("KV ring word count is invalid"); }
    kv_ring_valid_apply_kernel<<<1, word_count, 0, stream>>>(words, and_mask, or_mask);
}

void copy_kv_residual_slot(const void* residual_k, const void* residual_v,
                           std::int64_t slot_stride_bytes, std::int32_t layer_count,
                           std::int32_t table_rows, std::int32_t source_slot,
                           std::int32_t destination_slot, cudaStream_t stream) {
    if (residual_k == nullptr || residual_v == nullptr || slot_stride_bytes <= 0 ||
        layer_count <= 0 || table_rows <= 0 || source_slot < 0 || destination_slot < 0 ||
        source_slot >= table_rows || destination_slot >= table_rows ||
        source_slot == destination_slot) {
        return;
    }
    constexpr int kThreads = 256;
    constexpr int kBlocks  = 2048;
    auto* k_plane          = static_cast<char*>(const_cast<void*>(residual_k));
    auto* v_plane          = static_cast<char*>(const_cast<void*>(residual_v));
    kv_residual_slot_copy_kernel<<<kBlocks, kThreads, 0, stream>>>(
        k_plane, k_plane, slot_stride_bytes,
        static_cast<std::int64_t>(table_rows) * slot_stride_bytes, layer_count, table_rows,
        source_slot, destination_slot);
    kv_residual_slot_copy_kernel<<<kBlocks, kThreads, 0, stream>>>(
        v_plane, v_plane, slot_stride_bytes,
        static_cast<std::int64_t>(table_rows) * slot_stride_bytes, layer_count, table_rows,
        source_slot, destination_slot);
}

} // namespace ninfer
