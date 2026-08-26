#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion conv_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 4.2e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

// FP64 oracle from the represented BF16 inputs: block-local two-tap depthwise
// convolution with per-group dynamic coefficients plus the per-channel base.
std::vector<double> conv_oracle(const std::vector<float>& hidden, const std::vector<float>& dyn,
                                const std::vector<float>& base, std::int32_t hidden_size,
                                std::int32_t columns, std::int32_t block_size,
                                std::int32_t side) {
    const int groups = hidden_size / ops::kDflash2ConvGroupSize;
    std::vector<double> expected(static_cast<std::size_t>(hidden_size) * columns);
    for (int t = 0; t < columns; ++t) {
        const int p = t % block_size;
        for (int c = 0; c < hidden_size; ++c) {
            const int g = c / ops::kDflash2ConvGroupSize;
            double acc  = 0.0;
            for (int k = 0; k < 2; ++k) {
                const double w =
                    static_cast<double>(
                        dyn[static_cast<std::size_t>(groups * 2 * side + groups * k + g) +
                            static_cast<std::size_t>(groups) * 4 * t]) +
                    static_cast<double>(
                        base[(static_cast<std::size_t>(side) * 2 + k) * hidden_size + c]);
                double x = 0.0;
                if (p - k >= 0) {
                    x = static_cast<double>(
                        hidden[static_cast<std::size_t>(c) +
                               static_cast<std::size_t>(hidden_size) * (t - k)]);
                }
                acc += w * x;
            }
            expected[static_cast<std::size_t>(c) + static_cast<std::size_t>(hidden_size) * t] =
                acc;
        }
    }
    return expected;
}

int run_case(const char* label, std::int32_t hidden_size, std::int32_t block_size,
             std::int32_t lanes, std::int32_t side, std::uint32_t seed) {
    const std::int32_t columns = block_size * lanes;
    const std::size_t count    = static_cast<std::size_t>(hidden_size) * columns;
    const int groups           = hidden_size / ops::kDflash2ConvGroupSize;
    std::vector<float> hidden(count), dyn(static_cast<std::size_t>(4 * groups) * columns),
        base(static_cast<std::size_t>(hidden_size) * 4);
    fill_uniform(hidden, seed, -8.0f, 8.0f);
    fill_uniform(dyn, seed + 1, -2.0f, 2.0f);
    fill_uniform(base, seed + 2, -0.5f, 0.5f);
    round_to_bf16(hidden);
    round_to_bf16(dyn);
    round_to_bf16(base);

    const auto expected  = conv_oracle(hidden, dyn, base, hidden_size, columns, block_size, side);
    const auto h_bits    = encode_bf16(hidden);
    const auto d_bits    = encode_bf16(dyn);
    const auto b_bits    = encode_bf16(base);
    auto hidden_buffer   = to_device(h_bits);
    auto dynamic_buffer  = to_device(d_bits);
    auto base_buffer     = to_device(b_bits);
    GuardedDeviceBuffer out_buffer(count * sizeof(std::uint16_t));

    Tensor hidden_tensor(hidden_buffer.p, DType::BF16, {hidden_size, columns});
    Tensor dynamic_tensor(dynamic_buffer.p, DType::BF16,
                          {4 * static_cast<std::int64_t>(groups), columns});
    Tensor base_tensor(base_buffer.p, DType::BF16, {2, 2, hidden_size});
    Tensor out_tensor(out_buffer.data(), DType::BF16, {hidden_size, columns});
    ops::dflash2_dynamic_conv(hidden_tensor, dynamic_tensor, base_tensor, side, block_size,
                              out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(out_buffer.data(), count), expected,
                                    conv_bf16_criterion());
    failures += out_buffer.verify_guards(label);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // Real DFlash2 shape (5120 hidden, block 8) at both sides and lane counts.
    failures += run_case("conv 5120 b8 l1 side0", 5120, 8, 1, 0, 11);
    failures += run_case("conv 5120 b8 l1 side1", 5120, 8, 1, 1, 12);
    failures += run_case("conv 5120 b8 l2 side0", 5120, 8, 2, 0, 13);
    // Degenerate blocks: a single position (both taps read the zero pad).
    failures += run_case("conv 64 b1 l1 side1", 64, 1, 1, 1, 14);
    // Wider block exercises the tap gather across positions.
    failures += run_case("conv 128 b16 l3 side0", 128, 16, 3, 0, 15);
    if (failures == 0) {
        std::cout << "dflash2_dynamic_conv: ALL PASSED\n";
        return 0;
    }
    std::cout << "dflash2_dynamic_conv: " << failures << " FAILURES\n";
    return 1;
}
