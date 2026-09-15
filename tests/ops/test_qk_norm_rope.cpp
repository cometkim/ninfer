#include "ninfer/ops/qk_norm_rope.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

constexpr float kTheta = 1.0e7F;
constexpr float kEps   = 1.0e-6F;

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

// One BF16 ulp at |value| (8 significant bits).
double bf16_ulp(double value) {
    const double magnitude = std::abs(value);
    if (magnitude < 1.0e-30) return std::ldexp(1.0, -133);
    int exponent = 0;
    (void)std::frexp(magnitude, &exponent);
    return std::ldexp(1.0, exponent - 8);
}

struct OracleSide {
    std::vector<double> out;
    // Per element: the larger |n| of its rotated pair times the q scale (rotary dims only).
    std::vector<double> pair_scale;
};

// FP64 oracle of the complete formula: n = x * rsqrt(mean(x^2) + eps) * (w + 1), then the
// split-half rotation of the UNROUNDED n (no BF16 materialization of n), q rows scaled by
// attention_factor^2. The angles follow ops::rope's profiles: factor 1 keeps the FP32
// position*frequency product, any other factor computes it in FP64.
OracleSide oracle_side(const std::vector<float>& x, const std::vector<float>& w,
                       const std::vector<int>& positions, int heads, int axes,
                       const ops::RopeFrequencies& frequencies, bool is_q) {
    const int tokens   = static_cast<int>(positions.size()) / axes;
    const double scale = is_q ? static_cast<double>(frequencies.attention_factor) *
                                    static_cast<double>(frequencies.attention_factor)
                              : 1.0;
    OracleSide result{std::vector<double>(x.size()), std::vector<double>(x.size(), 0.0)};
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
            for (int dim = 0; dim < 256; ++dim) normed[dim] = x[base + dim] * inv * (w[dim] + 1.0);
            for (int dim = 0; dim < 256; ++dim) {
                double value = normed[dim];
                if (dim < 64) {
                    const int pair     = dim % 32;
                    const int axis     = axes == 3 ? pair % 3 : 0;
                    const int position = positions[static_cast<std::size_t>(axis) * tokens +
                                                   static_cast<std::size_t>(token)];
                    const double angle =
                        frequencies.attention_factor == 1.0F
                            ? static_cast<double>(
                                  static_cast<float>(position) *
                                  static_cast<float>(frequencies.inv_frequency[pair]))
                            : static_cast<double>(position) * frequencies.inv_frequency[pair];
                    const double c       = std::cos(angle) * scale;
                    const double s       = std::sin(angle) * scale;
                    const double partner = normed[dim + (dim < 32 ? 32 : -32)];
                    value = dim < 32 ? value * c - partner * s : value * c + partner * s;
                    result.pair_scale[base + dim] =
                        std::max(std::abs(normed[dim]), std::abs(partner)) * scale;
                }
                result.out[base + dim] = value;
            }
        }
    }
    return result;
}

int run_case(DeviceExecutionView execution, const Geometry& geometry, int tokens,
             int first_position, std::uint32_t seed, int axes = 1,
             const ops::RopeFrequencies* table = nullptr) {
    const std::size_t q_elements = side_elements(geometry.q_heads, tokens);
    const std::size_t k_elements = side_elements(geometry.kv_heads, tokens);

    std::vector<float> q        = make_bf16(q_elements, seed, -2.0F, 2.0F);
    std::vector<float> k        = make_bf16(k_elements, seed + 1u, -2.0F, 2.0F);
    std::vector<float> q_weight = make_bf16(256, seed + 2u, -0.5F, 0.5F);
    std::vector<float> k_weight = make_bf16(256, seed + 3u, -0.5F, 0.5F);
    std::vector<int> positions(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(axis) * tokens + static_cast<std::size_t>(token)] =
                first_position + token + 11 * axis;
        }
    }

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

    const ops::RopeFrequencies frequencies =
        table != nullptr ? *table : ops::rope_linear_frequencies(kTheta, 64);
    ops::qk_norm_rope(tq, tk, twq, twk, kEps, tpos, frequencies, tq_out, tk_out, nullptr);

    // The sequential chain, for the unrotated-dims identity and the bounded rotated difference.
    ops::rmsnorm(tq, twq, kEps, true, tq_ref, nullptr);
    ops::rmsnorm(tk, twk, kEps, true, tk_ref, nullptr);
    ops::rope(tpos, 64, frequencies, tq_ref, tk_ref, execution);
    cuda_synchronize();

    const std::string label = std::string(geometry.label) + " T=" + std::to_string(tokens) +
                              " pos=" + std::to_string(first_position) +
                              " axes=" + std::to_string(axes) +
                              (table != nullptr ? " scaled-table" : "");
    int failures            = 0;
    const auto q_bits       = from_device<std::uint16_t>(dq_out, q_elements);
    const auto k_bits       = from_device<std::uint16_t>(dk_out, k_elements);
    const auto qr_bits      = from_device<std::uint16_t>(dq_ref, q_elements);
    const auto kr_bits      = from_device<std::uint16_t>(dk_ref, k_elements);
    const OracleSide q_oracle =
        oracle_side(q, q_weight, positions, geometry.q_heads, axes, frequencies, true);
    const OracleSide k_oracle =
        oracle_side(k, k_weight, positions, geometry.kv_heads, axes, frequencies, false);

    const auto check = [&](const char* side, const std::vector<std::uint16_t>& fused,
                           const std::vector<std::uint16_t>& chain, const OracleSide& oracle) {
        int side_failures = 0;
        for (std::size_t i = 0; i < fused.size(); ++i) {
            const bool rotary    = i % 256 < 64;
            const double got     = bf16_to_f32(fused[i]);
            const double chained = bf16_to_f32(chain[i]);
            const double exact   = oracle.out[i];
            // Unrotated dims: one rounding of the same FP32 norm as rmsnorm, bit for bit.
            if (!rotary && fused[i] != chain[i]) {
                std::cerr << label << ' ' << side << ": unrotated dim differs from rmsnorm at " << i
                          << " fused=" << got << " rmsnorm=" << chained << '\n';
                ++side_failures;
                break;
            }
            // One final rounding of the FP32 evaluation: half an ulp plus FP32 arithmetic noise
            // relative to the pair's magnitude.
            const double noise = 4.0e-6 * std::max(1.0, oracle.pair_scale[i]);
            if (std::abs(got - exact) > 0.5 * bf16_ulp(exact) + noise) {
                std::cerr << label << ' ' << side << ": oracle mismatch at " << i << " got=" << got
                          << " expected=" << exact << '\n';
                ++side_failures;
                break;
            }
            // The chain additionally rounds n before rotating: its result differs by that
            // rounding carried through the rotation plus both final roundings.
            if (rotary && std::abs(got - chained) > bf16_ulp(oracle.pair_scale[i]) +
                                                        bf16_ulp(std::abs(chained)) + noise) {
                std::cerr << label << ' ' << side << ": rotated dim beyond the chain bound at " << i
                          << " fused=" << got << " chain=" << chained << '\n';
                ++side_failures;
                break;
            }
        }
        return side_failures;
    };
    failures += check("q", q_bits, qr_bits, q_oracle);
    failures += check("k", k_bits, kr_bits, k_oracle);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    DeviceContext device;
    const auto execution        = device.execution_view().on_stream(nullptr);
    int failures                = 0;
    const Geometry geometries[] = {
        {"27b", 24, 4},
        {"35b", 16, 2},
    };
    // A YaRN-shaped table (pairs 20..31 interpolated, q-side temperature): the scaled angle
    // profile and the factor^2 q scale.
    ops::RopeFrequencies scaled = ops::rope_linear_frequencies(kTheta, 64);
    for (int pair = 20; pair < 32; ++pair) { scaled.inv_frequency[pair] /= 4.0; }
    scaled.attention_factor = 1.1386294F;
    for (const Geometry& geometry : geometries) {
        failures += run_case(execution, geometry, 1, 0, 7u);
        failures += run_case(execution, geometry, 1, 262'137, 11u);
        failures += run_case(execution, geometry, 6, 1024, 13u);
        failures += run_case(execution, geometry, 128, 4096, 17u);
        failures += run_case(execution, geometry, 4, 524'288, 19u);
        // MRoPE positions [T,3] - the prefix-reuse MTP bridge path's layout (axis = pair % 3).
        failures += run_case(execution, geometry, 1, 33'000, 23u, 3);
        failures += run_case(execution, geometry, 6, 1024, 29u, 3);
        failures += run_case(execution, geometry, 128, 4096, 31u, 3);
        failures += run_case(execution, geometry, 5, 1'048'560, 37u, 1, &scaled);
        failures += run_case(execution, geometry, 7, 1'048'500, 41u, 3, &scaled);
    }

    if (failures == 0) {
        std::cout << "PASS qk_norm_rope\n";
        return 0;
    }
    std::cout << "FAIL qk_norm_rope\n";
    return 1;
}
