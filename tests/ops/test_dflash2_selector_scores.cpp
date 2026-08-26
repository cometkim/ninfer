#include "ninfer/ops/dflash2_selector_scores.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion scores_criterion() {
    return {/*absolute*/ 1.0e-4, /*relative*/ 2.0e-5};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

// FP64 oracle from the represented inputs: codebook rows decode from BF16
// exactly, the rank dot and the unary add evaluate in double.
std::vector<double> scores_oracle(const std::vector<std::int32_t>& candidates,
                                  const std::vector<std::int32_t>& pred_ids,
                                  const std::vector<float>& unary,
                                  const std::vector<float>& hidden_proj,
                                  const std::vector<float>& successor,
                                  const std::vector<float>& predecessor, std::int32_t rank,
                                  std::int32_t top_k, std::int32_t positions,
                                  std::int32_t lanes) {
    std::vector<double> expected(static_cast<std::size_t>(top_k) * top_k * positions * lanes);
    for (int l = 0; l < lanes; ++l) {
        for (int s = 0; s < positions; ++s) {
            for (int j = 0; j < top_k; ++j) {
                const auto pred_id = static_cast<std::size_t>(
                    pred_ids[(static_cast<std::size_t>(j) * positions + s) * lanes + l]);
                for (int i = 0; i < top_k; ++i) {
                    const auto succ_id = static_cast<std::size_t>(
                        candidates[(static_cast<std::size_t>(i) * positions + s) * lanes + l]);
                    double dot = 0.0;
                    for (int r = 0; r < rank; ++r) {
                        dot += static_cast<double>(successor[succ_id * rank + r]) *
                               static_cast<double>(predecessor[pred_id * rank + r]) *
                               static_cast<double>(
                                   hidden_proj[(static_cast<std::size_t>(s) * lanes + l) * rank +
                                               r]);
                    }
                    dot += static_cast<double>(
                        unary[(static_cast<std::size_t>(i) * positions + s) * lanes + l]);
                    expected[((static_cast<std::size_t>(i) * top_k + j) * positions + s) * lanes +
                             l] = dot;
                }
            }
        }
    }
    return expected;
}

int run_case(const char* label, std::int32_t rank, std::int32_t top_k, std::int32_t positions,
             std::int32_t lanes, std::int32_t vocab, std::uint32_t seed) {
    const std::size_t kl_count = static_cast<std::size_t>(top_k) * positions * lanes;
    std::vector<float> successor(static_cast<std::size_t>(vocab) * rank),
        predecessor(static_cast<std::size_t>(vocab) * rank);
    std::vector<float> hidden_proj(static_cast<std::size_t>(rank) * positions * lanes);
    std::vector<float> unary(kl_count);
    std::vector<std::int32_t> candidates(kl_count), pred_ids(kl_count);
    fill_uniform(successor, seed, -0.5f, 0.5f);
    fill_uniform(predecessor, seed + 1, -0.5f, 0.5f);
    fill_uniform(hidden_proj, seed + 2, -1.0f, 1.0f);
    fill_uniform(unary, seed + 3, -12.0f, 12.0f);
    std::mt19937 rng(seed + 4);
    for (std::size_t index = 0; index < kl_count; ++index) {
        candidates[index] = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(vocab));
        pred_ids[index]   = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(vocab));
    }
    round_to_bf16(successor);
    round_to_bf16(predecessor);

    const auto expected = scores_oracle(candidates, pred_ids, unary, hidden_proj, successor,
                                        predecessor, rank, top_k, positions, lanes);
    auto succ_buffer    = to_device(encode_bf16(successor));
    auto pred_buffer    = to_device(encode_bf16(predecessor));
    auto hidden_buffer  = to_device_f32(hidden_proj);
    auto unary_buffer   = to_device_f32(unary);
    auto cand_buffer    = to_device_i32(candidates);
    auto predr_buffer   = to_device_i32(pred_ids);
    GuardedDeviceBuffer out_buffer(static_cast<std::size_t>(top_k) * top_k * positions * lanes *
                                   sizeof(float));

    Tensor cand_tensor(cand_buffer.p, DType::I32, {top_k, positions, lanes});
    Tensor pred_tensor(predr_buffer.p, DType::I32, {top_k, positions, lanes});
    Tensor unary_tensor(unary_buffer.p, DType::FP32, {top_k, positions, lanes});
    Tensor hidden_tensor(hidden_buffer.p, DType::FP32, {rank, positions, lanes});
    Tensor succ_tensor(succ_buffer.p, DType::BF16, {vocab, rank});
    Tensor predt_tensor(pred_buffer.p, DType::BF16, {vocab, rank});
    Tensor out_tensor(out_buffer.data(), DType::FP32, {top_k, top_k, positions, lanes});
    ops::dflash2_selector_scores(cand_tensor, pred_tensor, unary_tensor, hidden_tensor,
                                 succ_tensor, predt_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::size_t count = static_cast<std::size_t>(top_k) * top_k * positions * lanes;
    const auto stored       = from_device<float>(out_buffer.data(), count);
    const std::vector<double> stored_double(stored.begin(), stored.end());
    int failures            = verify_pointwise(label, stored_double, expected, scores_criterion());
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
    // Real DFlash2 shape: top-16 candidates over 7 draft positions.
    failures += run_case("selector r256 k16 p7 l1", 256, 16, 7, 1, 4096, 21);
    // Two lanes share the codebooks through gathered rows.
    failures += run_case("selector r256 k16 p7 l2", 256, 16, 7, 2, 4096, 22);
    // Minimal shapes: one candidate and one position collapse the lattice.
    failures += run_case("selector r64 k1 p1 l1", 64, 1, 1, 1, 512, 23);
    failures += run_case("selector r512 k8 p3 l4", 512, 8, 3, 4, 1024, 24);
    if (failures == 0) {
        std::cout << "dflash2_selector_scores: ALL PASSED\n";
        return 0;
    }
    std::cout << "dflash2_selector_scores: " << failures << " FAILURES\n";
    return 1;
}
