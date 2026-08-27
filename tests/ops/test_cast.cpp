#include "ninfer/ops/cast.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint32_t> fp32_bits(const std::vector<float>& values) {
    std::vector<std::uint32_t> bits(values.size());
    std::memcpy(bits.data(), values.data(), values.size() * sizeof(float));
    return bits;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> source(count);
    fill_uniform(source, seed, -64.0f, 64.0f);
    if (count >= 8) {
        const float edges[] = {-65504.0f,       -1.00390625f, -0.0f,      0.0f,
                               0.333251953125f, 1.00390625f,  3.1415927f, 65504.0f};
        std::copy(std::begin(edges), std::end(edges), source.begin());
    }

    std::vector<std::uint16_t> expected(count);
    for (std::size_t i = 0; i < count; ++i) expected[i] = f32_to_bf16(source[i]);
    const auto source_bits = fp32_bits(source);
    GuardedDeviceBuffer device_source(count * sizeof(float));
    GuardedDeviceBuffer device_destination(count * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), device_source.bytes());

    Tensor source_tensor(device_source.data(), DType::FP32, {rows, columns});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {rows, columns});
    ops::cast_fp32_to_bf16(source_tensor, destination_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact(label, from_device<std::uint16_t>(device_destination.data(), count), expected);
    const auto source_after = from_device<float>(device_source.data(), count);
    failures += verify_exact("cast source unchanged", fp32_bits(source_after), source_bits);
    failures += device_source.verify_guards("cast source");
    failures += device_destination.verify_guards("cast destination");
    return failures;
}


// The reverse promotion is exact: a represented BF16 value widens to the FP32
// whose top 16 bits are the BF16 code, so the oracle compares raw FP32 words.
int run_widen_case(const char* label, std::int32_t rows, std::int32_t columns,
                   std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> source(count);
    fill_uniform(source, seed, -64.0f, 64.0f);
    if (count >= 8) {
        const float edges[] = {-65504.0f,       -1.00390625f, -0.0f,      0.0f,
                               0.333251953125f, 1.00390625f,  3.1415927f, 65504.0f};
        std::copy(std::begin(edges), std::end(edges), source.begin());
    }
    std::vector<std::uint16_t> bf16(count);
    for (std::size_t i = 0; i < count; ++i) bf16[i] = f32_to_bf16(source[i]);
    std::vector<std::uint32_t> expected(count);
    for (std::size_t i = 0; i < count; ++i) {
        expected[i] = std::uint32_t(bf16[i]) << 16;
    }

    GuardedDeviceBuffer device_source(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_destination(count * sizeof(float));
    device_source.copy_from_host(bf16.data(), device_source.bytes());

    Tensor source_tensor(device_source.data(), DType::BF16, {rows, columns});
    Tensor destination_tensor(device_destination.data(), DType::FP32, {rows, columns});
    ops::cast_bf16_to_fp32(source_tensor, destination_tensor, nullptr);
    cuda_synchronize();

    std::vector<std::uint32_t> stored(count);
    device_destination.copy_to_host(stored.data(), count * sizeof(std::uint32_t));
    int failures = verify_exact(label, stored, expected);
    failures += device_source.verify_guards("widen source");
    failures += device_destination.verify_guards("widen destination");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("cast [1536,1]", 1536, 1, 101u);
    failures += run_case("cast [1536,257]", 1536, 257, 201u);
    failures += run_case("cast edge-length tensor", 13, 1, 301u);
    std::cout << (failures ? "FAIL" : "OK") << " cast_fp32_to_bf16\n";
    int widen_failures = 0;
    widen_failures += run_widen_case("widen [1536,1]", 1536, 1, 401u);
    widen_failures += run_widen_case("widen [1536,257]", 1536, 257, 501u);
    widen_failures += run_widen_case("widen edge-length tensor", 13, 1, 601u);
    std::cout << (widen_failures ? "FAIL" : "OK") << " cast_bf16_to_fp32\n";
    return (failures || widen_failures) ? 1 : 0;
}
