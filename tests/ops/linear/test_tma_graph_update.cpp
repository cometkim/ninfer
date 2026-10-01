// TMA Linear launches inside updated CUDA Graph executables.
//
// The engine instantiates one decode-graph executable per topology class and updates it across
// resource tiers with cudaGraphExecUpdate, and B x W >= 17 decode/verify widths select the FP8 A8
// and BF16 A16 TMA routes inside those graphs. TMA descriptor blocks are kernel parameters; on
// Windows they are staged through stream-ordered device memory instead (see
// src/ops/common/tma_descriptors.cuh), which adds allocation, staging and free nodes to captured
// graphs. Each case captures the same public Linear call over two different activation/output
// buffers, instantiates the first capture, updates it with the second, replays it twice and
// requires output bits identical to an eager call over the second buffers, with the first output
// left untouched.

#include "core/decode_graph.h"
#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/direct_bf16_weight.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    for (std::size_t index = 0; index < bits.size(); ++index) {
        const std::uint32_t hash =
            static_cast<std::uint32_t>(index) * 0x9e3779b9U ^ seed * 0x85ebca6bU;
        const int centered = static_cast<int>((hash >> 7) & 0xff) - 128;
        bits[index]        = f32_to_bf16(static_cast<float>(centered) * (1.0F / 256.0F));
    }
    return bits;
}

int run_case(const char* label, const Weight& weight, std::int32_t tokens,
             ops::LinearPolicy policy) {
    const std::int32_t n           = static_cast<std::int32_t>(weight.shape[0]);
    const std::int32_t k           = static_cast<std::int32_t>(weight.shape[1]);
    const std::string name         = std::string(label) + " [" + std::to_string(n) + "," +
                                     std::to_string(k) + "] T=" + std::to_string(tokens);
    const std::size_t output_bytes = static_cast<std::size_t>(n) * tokens * sizeof(std::uint16_t);

    DeviceBuffer first_input  = to_device(make_activation(k, tokens, 11U));
    DeviceBuffer second_input = to_device(make_activation(k, tokens, 29U));
    GuardedDeviceBuffer first_output(output_bytes);
    GuardedDeviceBuffer second_output(output_bytes);
    GuardedDeviceBuffer reference_output(output_bytes);
    const std::size_t capacity =
        ops::linear_workspace_capacity_bytes(weight.qtype, n, k, policy, tokens, tokens);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    DeviceContext context;

    const auto call = [&](DeviceBuffer& input, GuardedDeviceBuffer& output) {
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor y(output.data(), DType::BF16, {n, tokens});
        ops::linear(x, weight, y, policy, workspace, context.stream);
    };

    call(second_input, reference_output);
    cuda_synchronize(context.stream);
    const auto reference =
        from_device<std::uint16_t>(reference_output.data(), static_cast<std::size_t>(n) * tokens);

    DecodeGraphDefinition first;
    DecodeGraphDefinition second;
    DecodeGraphExecutable executable;
    cuda_synchronize();
    first.capture(context.stream, [&] { call(first_input, first_output); });
    executable.instantiate(first);
    second.capture(context.stream, [&] { call(second_input, second_output); });
    executable.update(second);

    int failures = 0;
    first_output.fill(0x5a);
    for (int replay = 0; replay < 2; ++replay) {
        second_output.fill(0xff);
        cuda_synchronize();
        executable.launch(context.stream);
        cuda_synchronize(context.stream);
        const std::string phase = name + " updated replay " + std::to_string(replay);
        failures += second_output.verify_guards(phase);
        failures += first_output.verify_guards(phase + " (first buffer)");
        const auto actual =
            from_device<std::uint16_t>(second_output.data(), static_cast<std::size_t>(n) * tokens);
        if (actual != reference) {
            std::size_t mismatches = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) mismatches += actual[i] != reference[i];
            std::cerr << phase << ": " << mismatches << " output values differ from eager\n";
            ++failures;
        }
        const auto untouched = from_device<std::uint8_t>(first_output.data(), output_bytes);
        if (std::any_of(untouched.begin(), untouched.end(),
                        [](std::uint8_t value) { return value != 0x5a; })) {
            std::cerr << phase << ": the first capture's output was written after update\n";
            ++failures;
        }
    }
    if (workspace.used() != 0) {
        std::cerr << name << ": workspace scope leaked\n";
        ++failures;
    }
    return failures;
}

int run() {
    int failures = 0;
    {
        // BF16 A16 [14336,5120]: T 33..64 selects the TMA R64T64 route.
        direct_bf16_weight::DeviceWeight weight(
            direct_bf16_weight::make_patterned(14336, 5120, 7U));
        failures += run_case("BF16 A16 TMA", weight.view(), 64, ops::LinearPolicy::A16Only);
    }
    {
        // FP8 A8 [5120,17408]: T 17..64 selects the TMA 32x64 route.
        const auto host = linear::make_fp8_weight(5120, 17408, 13U);
        DeviceBuffer payload(host.payload.size());
        payload.copy_from_host(host.payload.data(), payload.bytes);
        failures +=
            run_case("FP8 A8 TMA", host.device_weight(payload.p), 48, ops::LinearPolicy::AllowA8);
    }
    {
        // NVFP4 A4 [5120,6144]: T >= 1024 selects the TMA route.
        const auto host = linear::make_nvfp4_weight(5120, 6144, 17U);
        DeviceBuffer payload(host.payload.size());
        payload.copy_from_host(host.payload.data(), payload.bytes);
        failures += run_case("NVFP4 A4 TMA", host.device_weight(payload.p), 1024,
                             ops::LinearPolicy::AllowA4);
    }
    std::cout << (failures == 0 ? "linear TMA graph update: PASS\n"
                                : "linear TMA graph update: FAIL\n");
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "linear TMA graph update: " << error.what() << '\n';
        return 1;
    }
}
