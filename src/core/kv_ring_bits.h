#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer {

// One slot row's residual-ring validity words (kGqaHqRecentKeys/32, 16 at the
// shipped ring width). Passed by value so callers can hand host-computed masks
// straight to the kernel.
struct KvRingWords {
    std::uint32_t w[16] = {};
};

// Applies word masks to one slot row's residual-ring validity words:
// words[i] = (words[i] & and_mask.w[i]) | or_mask.w[i] for i < word_count.
// Async, word_count threads.
void apply_kv_ring_valid_words(std::uint32_t* words, const KvRingWords& and_mask,
                               const KvRingWords& or_mask, int word_count,
                               cudaStream_t stream);

} // namespace ninfer
