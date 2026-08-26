#include "ninfer/ops/dflash2_selector_scores.h"
#include "ops/op_tester.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion scores_criterion() {
    return {/*absolute*/ 1.0e-4, /*relative*/ 2.0e-5};
}

float e2m1_decode(unsigned nibble) {
    static const float magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float value                = magnitudes[nibble & 7U];
    return (nibble & 8U) != 0 ? -value : value;
}

float e4m3fn_decode(unsigned code) {
    const unsigned exponent = (code >> 3) & 0xFU;
    const unsigned fraction = code & 7U;
    const float magnitude   = exponent == 0
                                  ? static_cast<float>(fraction) * (1.0F / 512.0F)
                                  : static_cast<float>(8U + fraction) *
                                        std::ldexp(1.0F, static_cast<int>(exponent) - 10);
    return (code & 0x80U) != 0 ? -magnitude : magnitude;
}

// A weight-only NVFP4 codebook in the registered blockscale-k16-m128x4-v1 layout:
// packed E2M1 code pairs, then the E4M3FN group scales at the 256-byte-aligned
// plane, then the FP32 weight divisor.
struct Nvfp4Codebook {
    std::vector<std::uint8_t> payload;
    std::size_t scale_offset = 0;
    float divisor            = 0.0F;
    std::vector<double> decoded;  // [vocab * rank] represented values

    std::size_t code_bytes(std::size_t vocab, std::size_t rank) const {
        return vocab * rank / 2;
    }

    Nvfp4Codebook(std::size_t vocab, std::size_t rank, std::uint32_t seed) {
        const std::size_t codes    = code_bytes(vocab, rank);
        const std::size_t scales   = vocab * rank / 16;
        scale_offset               = (codes + 255U) / 256U * 256U;
        divisor                    = 3.5F;
        payload.resize(scale_offset + scales + sizeof(float));
        std::mt19937 rng(seed);
        decoded.assign(vocab * rank, 0.0);
        for (std::size_t i = 0; i < codes; ++i) {
            payload[i] = static_cast<std::uint8_t>(rng() & 0xFFU);
        }
        for (std::size_t i = 0; i < scales; ++i) {
            // Non-NaN, non-zero scale bytes keep every represented row finite and live.
            payload[scale_offset + i] = static_cast<std::uint8_t>(0x20U + (rng() % 0x30U));
        }
        std::memcpy(payload.data() + scale_offset + scales, &divisor, sizeof(divisor));
        for (std::size_t id = 0; id < vocab; ++id) {
            for (std::size_t r = 0; r < rank; ++r) {
                const unsigned pair = payload[id * rank / 2 + r / 2];
                const unsigned nibble =
                    (r % 2 == 0) ? (pair & 0xFU) : (pair >> 4);
                const double scale =
                    e4m3fn_decode(payload[scale_offset + id * rank / 16 + r / 16]);
                decoded[id * rank + r] =
                    static_cast<double>(e2m1_decode(nibble)) * scale / static_cast<double>(divisor);
            }
        }
    }

    Weight bind(const void* device_base) const {
        Weight out{};
        out.payload            = device_base;
        out.payload_bytes        = payload.size();
        out.qtype                = QType::NVFP4;
        out.group_size           = 16;
        out.ndim                 = 2;
        out.qdata                = device_base;
        out.scales               = static_cast<const std::byte*>(device_base) +
                       static_cast<std::ptrdiff_t>(scale_offset);
        const std::int32_t vocab = static_cast<std::int32_t>(decoded.size() / 256 * 256 / 256);
        (void)vocab;
        out.group              = 16;
        out.layout             = QuantLayout::BlockScaleK16M128x4;
        out.scale_dtype        = DType::FP8_E4M3FN;
        out.weight_scale_divisor = divisor;
        out.input_scale_divisor  = 1.0F;
        return out;
    }

    void set_shape(Weight& weight, std::size_t vocab, std::size_t rank) const {
        weight.n               = static_cast<std::int32_t>(vocab);
        weight.k               = static_cast<std::int32_t>(rank);
        weight.shape[0]        = static_cast<std::int32_t>(vocab);
        weight.shape[1]        = static_cast<std::int32_t>(rank);
        weight.padded_shape[0] = static_cast<std::int32_t>(vocab);
        weight.padded_shape[1] = static_cast<std::int32_t>(rank);
    }
};

// FP64 oracle from the represented inputs: codebook rows decode from the NVFP4
// codes and scales exactly, the rank dot and the unary add evaluate in double.
std::vector<double> scores_oracle(const std::vector<std::int32_t>& candidates,
                                  const std::vector<std::int32_t>& pred_ids,
                                  const std::vector<float>& unary,
                                  const std::vector<float>& hidden_proj,
                                  const std::vector<double>& successor,
                                  const std::vector<double>& predecessor, std::int32_t rank,
                                  std::int32_t top_k, std::int32_t positions,
                                  std::int32_t lanes) {
    std::vector<double> expected(static_cast<std::size_t>(top_k) * top_k * positions * lanes);
    for (int l = 0; l < lanes; ++l) {
        for (int s = 0; s < positions; ++s) {
            for (int j = 0; j < top_k; ++j) {
                const auto pred_id = static_cast<std::size_t>(
                    pred_ids[static_cast<std::size_t>(j) +
                             static_cast<std::size_t>(top_k) *
                                 (static_cast<std::size_t>(s) * lanes + l)]);
                for (int i = 0; i < top_k; ++i) {
                    const auto succ_id = static_cast<std::size_t>(
                        candidates[static_cast<std::size_t>(i) +
                                   static_cast<std::size_t>(top_k) *
                                       (static_cast<std::size_t>(s) * lanes + l)]);
                    double dot = 0.0;
                    for (int r = 0; r < rank; ++r) {
                        dot += successor[succ_id * rank + r] * predecessor[pred_id * rank + r] *
                               static_cast<double>(
                                   hidden_proj[(static_cast<std::size_t>(s) * lanes + l) * rank +
                                               r]);
                    }
                    dot += static_cast<double>(
                        unary[static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(top_k) *
                                  (static_cast<std::size_t>(s) * lanes + l)]);
                    expected[static_cast<std::size_t>(i) +
                             static_cast<std::size_t>(top_k) *
                                 (static_cast<std::size_t>(j) +
                                  static_cast<std::size_t>(top_k) *
                                      (static_cast<std::size_t>(s) * lanes + l))] = dot;
                }
            }
        }
    }
    return expected;
}

int run_case(const char* label, std::int32_t rank, std::int32_t top_k, std::int32_t positions,
             std::int32_t lanes, std::int32_t vocab, std::uint32_t seed) {
    const std::size_t kl_count = static_cast<std::size_t>(top_k) * positions * lanes;
    const std::size_t vr_count = static_cast<std::size_t>(vocab) * rank;
    std::vector<float> hidden_proj(static_cast<std::size_t>(rank) * positions * lanes);
    std::vector<float> unary(kl_count);
    std::vector<std::int32_t> candidates(kl_count), pred_ids(kl_count);
    fill_uniform(hidden_proj, seed + 2, -1.0f, 1.0f);
    fill_uniform(unary, seed + 3, -12.0f, 12.0f);
    std::mt19937 rng(seed + 4);
    for (std::size_t index = 0; index < kl_count; ++index) {
        candidates[index] = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(vocab));
        pred_ids[index]   = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(vocab));
    }

    const Nvfp4Codebook successor(static_cast<std::size_t>(vocab), rank, seed);
    const Nvfp4Codebook predecessor(static_cast<std::size_t>(vocab), rank, seed + 1);

    const auto expected = scores_oracle(candidates, pred_ids, unary, hidden_proj,
                                        successor.decoded, predecessor.decoded, rank, top_k,
                                        positions, lanes);
    auto succ_buffer    = to_device(successor.payload);
    auto pred_buffer    = to_device(predecessor.payload);
    auto hidden_buffer  = to_device_f32(hidden_proj);
    auto unary_buffer   = to_device_f32(unary);
    auto cand_buffer    = to_device_i32(candidates);
    auto predr_buffer   = to_device_i32(pred_ids);
    GuardedDeviceBuffer out_buffer(static_cast<std::size_t>(top_k) * top_k * positions * lanes *
                                   sizeof(float));

    Weight succ_weight = successor.bind(succ_buffer.p);
    successor.set_shape(succ_weight, static_cast<std::size_t>(vocab), rank);
    Weight pred_weight = predecessor.bind(pred_buffer.p);
    predecessor.set_shape(pred_weight, static_cast<std::size_t>(vocab), rank);

    Tensor cand_tensor(cand_buffer.p, DType::I32, {top_k, positions, lanes});
    Tensor pred_tensor(predr_buffer.p, DType::I32, {top_k, positions, lanes});
    Tensor unary_tensor(unary_buffer.p, DType::FP32, {top_k, positions, lanes});
    Tensor hidden_tensor(hidden_buffer.p, DType::FP32, {rank, positions, lanes});
    Tensor out_tensor(out_buffer.data(), DType::FP32, {top_k, top_k, positions, lanes});
    ops::dflash2_selector_scores(cand_tensor, pred_tensor, unary_tensor, hidden_tensor,
                                 succ_weight, pred_weight, out_tensor, nullptr);
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
