#include "ninfer/ops/dflash2_selector_predecessors.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Exact oracle: predecessor[j, s, l] is the anchor at s = 0 and candidate j
// from slot s-1 afterwards.
std::vector<std::int32_t> predecessors_oracle(const std::vector<std::int32_t>& candidates,
                                              const std::vector<std::int32_t>& anchors,
                                              std::int32_t top_k, std::int32_t positions,
                                              std::int32_t lanes) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(top_k) * positions * lanes);
    for (int l = 0; l < lanes; ++l) {
        for (int s = 0; s < positions; ++s) {
            for (int j = 0; j < top_k; ++j) {
                const std::size_t index =
                    static_cast<std::size_t>(j) +
                    static_cast<std::size_t>(top_k) *
                        (static_cast<std::size_t>(s) * lanes + l);
                expected[index] =
                    s == 0 ? anchors[static_cast<std::size_t>(l)]
                           : candidates[static_cast<std::size_t>(j) +
                                         static_cast<std::size_t>(top_k) *
                                             ((static_cast<std::size_t>(s) - 1) * lanes +
                                              static_cast<std::size_t>(l))];
            }
        }
    }
    return expected;
}

int run_case(const char* label, std::int32_t top_k, std::int32_t positions, std::int32_t lanes,
             std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(top_k) * positions * lanes;
    std::vector<std::int32_t> candidates(count), anchors(static_cast<std::size_t>(lanes));
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < count; ++i) {
        candidates[i] = static_cast<std::int32_t>(rng() % 1000000U);
    }
    for (std::size_t i = 0; i < anchors.size(); ++i) {
        anchors[i] = static_cast<std::int32_t>(rng() % 1000000U);
    }

    const auto expected = predecessors_oracle(candidates, anchors, top_k, positions, lanes);
    auto cand_buffer    = to_device_i32(candidates);
    auto anchor_buffer  = to_device_i32(anchors);
    GuardedDeviceBuffer out_buffer(count * sizeof(std::int32_t));

    Tensor cand_tensor(cand_buffer.p, DType::I32, {top_k, positions, lanes});
    Tensor anchor_tensor(anchor_buffer.p, DType::I32, {lanes});
    Tensor out_tensor(out_buffer.data(), DType::I32, {top_k, positions, lanes});
    ops::dflash2_selector_predecessors(cand_tensor, anchor_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_exact(label, from_device<std::int32_t>(out_buffer.data(), count),
                                expected);
    failures += out_buffer.verify_guards(label);
    failures += verify_exact("candidates unchanged", from_device<std::int32_t>(cand_buffer.p, count),
                              candidates);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // Real DFlash2 shape: 16 candidates over 7 draft positions.
    failures += run_case("predecessors k16 p7 l1", 16, 7, 1, 31);
    failures += run_case("predecessors k16 p7 l2", 16, 7, 2, 32);
    // Minimal and wide lattices.
    failures += run_case("predecessors k1 p1 l1", 1, 1, 1, 33);
    failures += run_case("predecessors k8 p15 l4", 8, 15, 4, 34);
    if (failures == 0) {
        std::cout << "dflash2_selector_predecessors: ALL PASSED\n";
        return 0;
    }
    std::cout << "dflash2_selector_predecessors: " << failures << " FAILURES\n";
    return 1;
}
