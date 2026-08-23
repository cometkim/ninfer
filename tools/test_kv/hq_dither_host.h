// Host mirrors of the hq codec's subtractive-dither hashes (hq_codec.cuh):
// identical uint64 arithmetic, so oracles reproduce the device dither exactly.
#pragma once

#include <cstdint>

namespace hqhost {

inline std::uint64_t dither_row_seed(int kv_head, std::int64_t position, bool role_v) {
    std::uint64_t x = 0x5DEECE66Dull ^
                      (static_cast<std::uint64_t>(kv_head) * 0x2545F4914F6CDD1Dull) ^
                      (static_cast<std::uint64_t>(position) * 0x9E3779B97F4A7C15ull) ^
                      (role_v ? 0xA5A5A5A5A5A5A5A5ull : 0x1B873593B5244C61ull);
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33;
    return x;
}

inline std::uint64_t dither_word_seed(std::uint64_t row_seed, int word) {
    std::uint64_t x = row_seed ^ (0x9E3779B97F4A7C15ull * (static_cast<std::uint64_t>(word) + 1u));
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

inline double dither(std::uint64_t word_seed, int j) {
    const std::uint32_t bits = static_cast<std::uint32_t>((word_seed >> (8 * j)) & 0xFFull);
    return static_cast<double>(bits) * (1.0 / 255.0) - 0.5;
}

} // namespace hqhost
