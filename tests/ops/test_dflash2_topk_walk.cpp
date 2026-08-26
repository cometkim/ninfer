#include "ninfer/ops/dflash2_selector_walk.h"
#include "ninfer/ops/dflash2_topk.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int run_topk_case(const char* label, std::int32_t rows, std::int32_t columns, std::int32_t k,
                  std::uint32_t seed) {
    std::vector<float> logits(static_cast<std::size_t>(rows) * columns);
    fill_uniform(logits, seed, -20.0f, 20.0f);
    round_to_bf16(logits);
    // Exact oracle: sort each column's (value, row) pairs by value desc, row asc.
    std::vector<std::int32_t> want_ids(static_cast<std::size_t>(k) * columns);
    std::vector<std::uint16_t> want_values(static_cast<std::size_t>(k) * columns);
    for (int t = 0; t < columns; ++t) {
        std::vector<std::pair<float, int>> entries;
        entries.reserve(static_cast<std::size_t>(rows));
        for (int r = 0; r < rows; ++r) {
            entries.emplace_back(logits[static_cast<std::size_t>(r) * columns + t], r);
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        for (int slot = 0; slot < k; ++slot) {
            want_ids[static_cast<std::size_t>(slot) * columns + t]  = entries[slot].second;
            want_values[static_cast<std::size_t>(slot) * columns + t] =
                f32_to_bf16(entries[slot].first);
        }
    }
    auto logits_buffer = to_device(encode_bf16(logits));
    GuardedDeviceBuffer ids_buffer(static_cast<std::size_t>(k) * columns * 4);
    GuardedDeviceBuffer values_buffer(static_cast<std::size_t>(k) * columns * 2);
    Tensor logits_tensor(logits_buffer.p, DType::BF16, {rows, columns});
    Tensor ids_tensor(ids_buffer.data(), DType::I32, {k, columns});
    Tensor values_tensor(values_buffer.data(), DType::BF16, {k, columns});
    ops::dflash2_topk(logits_tensor, k, ids_tensor, values_tensor, nullptr);
    cuda_synchronize();
    int failures = 0;
    failures += verify_exact(label, from_device<int>(ids_buffer.data(), static_cast<std::size_t>(k) * columns), want_ids);
    failures += verify_exact(label, from_device<std::uint16_t>(values_buffer.data(),
                                                              static_cast<std::size_t>(k) * columns),
                             want_values);
    failures += ids_buffer.verify_guards(label);
    failures += values_buffer.verify_guards(label);
    return failures;
}

int run_walk_case(const char* label, std::int32_t top_k, std::int32_t positions,
                  std::int32_t lanes, std::uint32_t seed) {
    const std::size_t sc = static_cast<std::size_t>(top_k) * top_k * positions * lanes;
    std::vector<float> scores(sc);
    fill_uniform(scores, seed, -10.0f, 10.0f);
    std::vector<std::int32_t> candidates(static_cast<std::size_t>(top_k) * positions * lanes);
    std::mt19937 rng(seed + 99);
    for (auto& c : candidates) { c = static_cast<std::int32_t>(rng() % 248320); }
    // Oracle: greedy walk with smaller-i tie-break.
    std::vector<std::int32_t> want(static_cast<std::size_t>(positions) * lanes);
    for (int l = 0; l < lanes; ++l) {
        int pred = 0;
        for (int s = 0; s < positions; ++s) {
            int best_i   = 0;
            float best_v = -1e30F;
            for (int i = 0; i < top_k; ++i) {
                const float v = scores[(((static_cast<std::size_t>(i) * top_k + pred) *
                                         positions + s) * lanes) + static_cast<std::size_t>(l)];
                if (v > best_v) {
                    best_v = v;
                    best_i = i;
                }
            }
            want[static_cast<std::size_t>(s) * lanes + l] =
                candidates[(static_cast<std::size_t>(best_i) * positions + s) * lanes + l];
            pred = best_i;
        }
    }
    auto scores_buffer    = to_device_f32(scores);
    auto candidates_buffer = to_device_i32(candidates);
    GuardedDeviceBuffer out_buffer(static_cast<std::size_t>(positions) * lanes * 4);
    Tensor scores_tensor(scores_buffer.p, DType::FP32, {top_k, top_k, positions, lanes});
    Tensor candidates_tensor(candidates_buffer.p, DType::I32, {top_k, positions, lanes});
    Tensor out_tensor(out_buffer.data(), DType::I32, {positions, lanes});
    ops::dflash2_selector_walk(scores_tensor, candidates_tensor, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_exact(label, from_device<int>(out_buffer.data(), static_cast<std::size_t>(positions) * lanes), want);
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
    failures += run_topk_case("topk 248320 c7 k16", 248320, 7, 16, 31);
    failures += run_topk_case("topk 1000 c3 k4", 1000, 3, 4, 32);
    failures += run_topk_case("topk 33 c1 k32", 33, 1, 32, 33);
    failures += run_walk_case("walk k16 p7 l1", 16, 7, 1, 41);
    failures += run_walk_case("walk k16 p7 l2", 16, 7, 2, 42);
    failures += run_walk_case("walk k4 p15 l8", 4, 15, 8, 43);
    if (failures == 0) {
        std::cout << "dflash2_topk+walk: ALL PASSED\n";
        return 0;
    }
    std::cout << "dflash2_topk+walk: " << failures << " FAILURES\n";
    return 1;
}
