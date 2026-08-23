#include "core/kv_ring_bits.h"

namespace ninfer {
namespace {

__global__ void kv_ring_valid_apply_kernel(std::uint32_t* words, KvRingWords and_mask,
                                           KvRingWords or_mask) {
    const int i      = static_cast<int>(threadIdx.x);
    words[i]         = (words[i] & and_mask.w[i]) | or_mask.w[i];
}

} // namespace

void apply_kv_ring_valid_words(std::uint32_t* words, const KvRingWords& and_mask,
                               const KvRingWords& or_mask, int word_count,
                               cudaStream_t stream) {
    if (words == nullptr || word_count <= 0) { return; }
    kv_ring_valid_apply_kernel<<<1, word_count, 0, stream>>>(words, and_mask, or_mask);
}

} // namespace ninfer
