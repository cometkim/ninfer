#include "ninfer/ops/qk_norm_rope.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kTheta  = 1.0e7F;
constexpr float kEps    = 1.0e-6F;
// Bounded vs max(1, |expected|): two BF16 roundings (norm boundary + rotation store), the FP32
// warp-sum norm vs the double oracle, and on the q side the factor^2 rotation scale amplifying
// boundary-rounding flips (measured worst 9.6e-3 at factor 2; parity vs the chain stays exact).
constexpr double kRtol  = 1.2e-2;

struct Geometry {
    const char* label;
    int q_heads;
    int kv_heads;
};

std::size_t side_elements(int heads, int tokens) {
    return static_cast<std::size_t>(256) * static_cast<std::size_t>(heads) *
           static_cast<std::size_t>(tokens);
}

// Column-major [256, H, T] flat index.
std::size_t element_index(int heads, int token, int head, int dim) {
    return (static_cast<std::size_t>(token) * static_cast<std::size_t>(heads) +
            static_cast<std::size_t>(head)) *
               static_cast<std::size_t>(256) +
           static_cast<std::size_t>(dim);
}

std::vector<float> make_bf16(std::size_t elements, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

// FP64 oracle of the composed contract: norm = bf16(x * rsqrt(mean(x^2)+eps) * (w+1)), then the
// split-half rotation in double over the ROUNDED norm values (q rows scaled by factor^2).
std::vector<double> oracle_side(const std::vector<float>& x, const std::vector<float>& w,
                                const std::vector<int>& positions, int heads, int axes,
                                const ops::RopeFrequencies& frequencies, bool is_q) {
    const int tokens = static_cast<int>(positions.size()) / axes;
    std::vector<double> out(x.size());
    const double scale = is_q ? static_cast<double>(frequencies.attention_factor) *
                                      frequencies.attention_factor
                              : 1.0;
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            const std::size_t base = element_index(heads, token, head, 0);
            double square_sum      = 0.0;
            for (int dim = 0; dim < 256; ++dim) {
                const double value = x[base + dim];
                square_sum += value * value;
            }
            const double inv = 1.0 / std::sqrt(square_sum / 256.0 + kEps);
            std::vector<double> normed(256);
            for (int dim = 0; dim < 256; ++dim) {
                // The bf16 boundary between norm and rotation, as the sequential chain stores it.
                normed[dim] = bf16_to_f32(
                    f32_to_bf16(static_cast<float>(x[base + dim] * inv * (w[dim] + 1.0))));
            }
            for (int dim = 0; dim < 256; ++dim) {
                double value = normed[dim];
                if (dim < 64) {
                    const int pair = dim % 32;
                    // The rope contract's angle profiles: factor 1 keeps the legacy FP32
                    // position*frequency product (bit-stable engine history); any other factor
                    // computes the product in FP64. The oracle mirrors both.
                    const int axis    = axes == 3 ? pair % 3 : 0;
                    const int position = positions[static_cast<std::size_t>(axis) * tokens +
                                                   static_cast<std::size_t>(token)];
                    double angle;
                    if (frequencies.attention_factor == 1.0F) {
                        angle = static_cast<double>(static_cast<float>(position) *
                                                    static_cast<float>(
                                                        frequencies.inv_frequency[pair]));
                    } else {
                        angle = static_cast<double>(position) *
                                frequencies.inv_frequency[pair];
                    }
                    const double c       = std::cos(angle) * scale;
                    const double s       = std::sin(angle) * scale;
                    const double partner = normed[dim + (dim < 32 ? 32 : -32)];
                    value = dim < 32 ? value * c - partner * s : value * c + partner * s;
                }
                out[base + dim] = value;
            }
        }
    }
    return out;
}

int run_case(const Geometry& geometry, int tokens, int first_position, float attention_factor,
             std::uint32_t seed, int axes = 1) {
    const std::size_t q_elements = side_elements(geometry.q_heads, tokens);
    const std::size_t k_elements = side_elements(geometry.kv_heads, tokens);

    std::vector<float> q = make_bf16(q_elements, seed, -2.0F, 2.0F);
    std::vector<float> k = make_bf16(k_elements, seed + 1u, -2.0F, 2.0F);
    std::vector<float> q_weight = make_bf16(256, seed + 2u, -0.5F, 0.5F);
    std::vector<float> k_weight = make_bf16(256, seed + 3u, -0.5F, 0.5F);
    std::vector<int> positions(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(axis) * tokens +
                      static_cast<std::size_t>(token)] = first_position + token + 11 * axis;
        }
    }

    ops::RopeFrequencies frequencies = ops::rope_linear_frequencies(kTheta, 64);
    frequencies.attention_factor     = attention_factor;

    const DeviceBuffer dq   = to_device_bf16(q);
    const DeviceBuffer dk   = to_device_bf16(k);
    const DeviceBuffer dwq  = to_device_bf16(q_weight);
    const DeviceBuffer dwk  = to_device_bf16(k_weight);
    const DeviceBuffer dpos = to_device_i32(positions);
    DeviceBuffer dq_out(sizeof(std::uint16_t) * q_elements);
    DeviceBuffer dk_out(sizeof(std::uint16_t) * k_elements);
    DeviceBuffer dq_ref(sizeof(std::uint16_t) * q_elements);
    DeviceBuffer dk_ref(sizeof(std::uint16_t) * k_elements);

    Tensor tq(dq.p, DType::BF16, {256, geometry.q_heads, tokens});
    Tensor tk(dk.p, DType::BF16, {256, geometry.kv_heads, tokens});
    Tensor twq(dwq.p, DType::BF16, {256});
    Tensor twk(dwk.p, DType::BF16, {256});
    Tensor tpos(dpos.p, DType::I32,
                axes == 3 ? std::initializer_list<std::int32_t>{tokens, 3}
                          : std::initializer_list<std::int32_t>{tokens});
    Tensor tq_out(dq_out.p, DType::BF16, {256, geometry.q_heads, tokens});
    Tensor tk_out(dk_out.p, DType::BF16, {256, geometry.kv_heads, tokens});
    Tensor tq_ref(dq_ref.p, DType::BF16, {256, geometry.q_heads, tokens});
    Tensor tk_ref(dk_ref.p, DType::BF16, {256, geometry.kv_heads, tokens});

    ops::qk_norm_rope(tq, tk, twq, twk, kEps, tpos, frequencies, tq_out, tk_out, nullptr);

    // The sequential chain the fusion must reproduce bit-for-bit.
    ops::rmsnorm(tq, twq, kEps, true, tq_ref, nullptr);
    ops::rmsnorm(tk, twk, kEps, true, tk_ref, nullptr);
    ops::rope(tpos, 64, frequencies, tq_ref, tk_ref, nullptr);
    cuda_synchronize();

    const std::string label = std::string(geometry.label) + " T=" + std::to_string(tokens) +
                              " pos=" + std::to_string(first_position) +
                              " factor=" + std::to_string(attention_factor) +
                              " axes=" + std::to_string(axes);
    int failures = 0;
    const auto qn_bits  = from_device<std::uint16_t>(dq_out, q_elements);
    const auto kn_bits  = from_device<std::uint16_t>(dk_out, k_elements);
    const auto qr_bits  = from_device<std::uint16_t>(dq_ref, q_elements);
    const auto kr_bits  = from_device<std::uint16_t>(dk_ref, k_elements);
    failures += verify_exact((label + " q parity vs rmsnorm->rope chain").c_str(), qn_bits, qr_bits);
    failures += verify_exact((label + " k parity vs rmsnorm->rope chain").c_str(), kn_bits, kr_bits);

    const std::vector<double> q_oracle =
        oracle_side(q, q_weight, positions, geometry.q_heads, axes, frequencies, true);
    const std::vector<double> k_oracle =
        oracle_side(k, k_weight, positions, geometry.kv_heads, axes, frequencies, false);
    for (std::size_t i = 0; i < q_elements; ++i) {
        const double got      = bf16_to_f32(qn_bits[i]);
        const double expected = q_oracle[i];
        if (std::abs(got - expected) > kRtol * std::max(1.0, std::abs(expected))) {
            std::cerr << label << ": q oracle mismatch at " << i << " got=" << got << " expected="
                      << expected << '\n';
            ++failures;
            break;
        }
    }
    for (std::size_t i = 0; i < k_elements; ++i) {
        const double got      = bf16_to_f32(kn_bits[i]);
        const double expected = k_oracle[i];
        if (std::abs(got - expected) > kRtol * std::max(1.0, std::abs(expected))) {
            std::cerr << label << ": k oracle mismatch at " << i << " got=" << got << " expected="
                      << expected << '\n';
            ++failures;
            break;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    const Geometry geometries[] = {
        {"27b", 24, 4},
        {"35b", 16, 2},
    };
    for (const Geometry& geometry : geometries) {
        failures += run_case(geometry, 1, 0, 1.0F, 7u);
        failures += run_case(geometry, 1, 262'137, 1.0F, 11u);
        failures += run_case(geometry, 6, 1024, 1.0F, 13u);
        failures += run_case(geometry, 128, 4096, 1.0F, 17u);
        failures += run_case(geometry, 4, 524'288, 2.0F, 19u);
        // MRoPE positions [T,3] - the prefix-reuse MTP bridge path's layout (axis = pair % 3).
        failures += run_case(geometry, 1, 33'000, 1.0F, 23u, 3);
        failures += run_case(geometry, 6, 1024, 1.0F, 29u, 3);
        failures += run_case(geometry, 128, 4096, 2.0F, 31u, 3);
    }

    if (failures == 0) {
        std::cout << "PASS qk_norm_rope\n";
        return 0;
    }
    std::cout << "FAIL qk_norm_rope\n";
    return 1;
}
