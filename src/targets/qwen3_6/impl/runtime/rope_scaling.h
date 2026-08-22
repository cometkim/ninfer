#pragma once
// Qwen3.6 family text RoPE scaling: the YaRN frequency table builder used by the sequence
// plan. Pure host math over the checkpoint's rope constants - no Variant instantiation.

#include "ninfer/ops/rope.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {

/**
 * YaRN frequency table (HF `_compute_yarn_parameters` semantics, beta_fast 32 / beta_slow 1,
 * truncated ramp): pairs below `low` extrapolate the linear frequency, pairs in [low, high]
 * blend extrapolation with interpolation by e = 1 - (i - low)/(high - low), and pairs above
 * `high` interpolate at inv/factor. The ramp bounds are floor/ceil of the correction dimension
 * rotary_dim * ln(original_positions / (beta * 2pi)) / (2 ln theta). The attention factor
 * 0.1 * ln(factor) + 1 travels in RopeFrequencies and scales the rotated subspace of q and k.
 */
inline ops::RopeFrequencies rope_yarn_frequencies(float theta, int rotary_dim,
                                                  std::uint32_t original_positions, float factor) {
    if (!(factor > 1.0F) || !std::isfinite(factor)) {
        throw std::invalid_argument("rope scaling: YaRN factor must exceed 1");
    }
    constexpr double kTwoPi = 6.28318530717958648;
    const double base       = static_cast<double>(theta);
    const double original   = static_cast<double>(original_positions);
    const auto correction   = [&](double beta) {
        return static_cast<double>(rotary_dim) * std::log(original / (beta * kTwoPi)) /
               (2.0 * std::log(base));
    };
    const int low  = static_cast<int>(std::floor(correction(32.0)));
    const int high = static_cast<int>(std::ceil(correction(1.0)));

    ops::RopeFrequencies frequencies;
    frequencies.attention_factor =
        static_cast<float>(0.1 * std::log(static_cast<double>(factor)) + 1.0);
    const int half = rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        const double linear = std::pow(base, -2.0 * i / rotary_dim);
        if (i < low) {
            frequencies.inv_frequency[i] = linear;
        } else if (i > high) {
            frequencies.inv_frequency[i] = linear / static_cast<double>(factor);
        } else {
            const double extrap = static_cast<double>(i - low) / static_cast<double>(high - low);
            frequencies.inv_frequency[i] =
                linear * ((1.0 - extrap) + extrap / static_cast<double>(factor));
        }
    }
    return frequencies;
}

} // namespace ninfer::targets::qwen3_6
