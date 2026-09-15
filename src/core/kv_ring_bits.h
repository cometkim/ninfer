#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer {

// One slot row's hq residual-window validity words: [0, kKvRingWords) are the recent-ring bits
// (bit r = ring slot r holds the row the next fetch will name); word kKvRingWords bit 0 marks the
// sink rows as populated for the owning history. Passed by value so callers can hand host-computed
// masks straight to the kernel. kKvRingWords matches kCausalHqRecentKeys / 32 (asserted at the
// owning cache, which sees both definitions).
inline constexpr int kKvRingWords = 16;
inline constexpr int kKvSideWords = kKvRingWords + 1;

struct KvRingWords {
    std::uint32_t w[kKvSideWords] = {};
};

// Applies word masks to one slot row's validity words:
// words[i] = (words[i] & and_mask.w[i]) | or_mask.w[i] for i < word_count.
// Async, word_count threads.
void apply_kv_ring_valid_words(std::uint32_t* words, const KvRingWords& and_mask,
                               const KvRingWords& or_mask, int word_count, cudaStream_t stream);

// Copy every layer's side-plane rows for one slot row to another (prefix-fork inheritance when
// the source row still owns coherent rows). residual_* are the full per-cache planes
// [head_dim, kv_heads, side_rows, layers * table_rows]; slot_stride_bytes is one dim-3 element.
void copy_kv_residual_slot(const void* residual_k, const void* residual_v,
                           std::int64_t slot_stride_bytes, std::int32_t layer_count,
                           std::int32_t table_rows, std::int32_t source_slot,
                           std::int32_t destination_slot, cudaStream_t stream);

} // namespace ninfer
