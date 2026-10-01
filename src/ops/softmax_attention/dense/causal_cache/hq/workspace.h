#pragma once

// ninfer::ops::detail::hq - transient allocation recipe of the hq-e8-2b routes. The same helpers
// run against the execution WorkspaceArena and the capacity query's WorkspaceLayoutBuilder, so
// the reported capacity is exact by construction.

#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>

namespace ninfer::ops::detail::hq {

inline constexpr std::int32_t kHqWorkspaceHeadDim = 256;

// Split-KV partial state of one small-T launch: unnormalized FP32 accumulators plus the running
// maximum and denominator per (query head, column, split, batch row).
struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                           std::int32_t tokens, std::int32_t splits,
                                           std::int32_t batch_size) {
    return {
        workspace.alloc(DType::FP32, {kHqWorkspaceHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

// Prompt route state: rotated-frame BF16 K/V planes for one band of
// min(envelope, kCausalHqPromptScratchBandKeys) keys, plus the band carry (BF16 acc and FP32
// m/l over the query rows) when the envelope spans more than one band.
struct PromptScratch {
    Tensor k;
    Tensor v;
    Tensor carry_acc;
    Tensor carry_m;
    Tensor carry_l;
};

template <class Allocator>
PromptScratch allocate_prompt_scratch(Allocator& workspace, std::int32_t kv_heads,
                                      std::int32_t q_heads, std::int32_t width,
                                      std::uint32_t visible_keys) {
    const std::uint32_t band = visible_keys < kCausalHqPromptScratchBandKeys
                                   ? visible_keys
                                   : kCausalHqPromptScratchBandKeys;
    const auto rows          = static_cast<std::int32_t>(band);
    PromptScratch scratch{
        workspace.alloc(DType::BF16, {kHqWorkspaceHeadDim, kv_heads, rows}),
        workspace.alloc(DType::BF16, {kHqWorkspaceHeadDim, kv_heads, rows}),
        {},
        {},
        {},
    };
    if (band < visible_keys) {
        scratch.carry_acc = workspace.alloc(DType::BF16, {kHqWorkspaceHeadDim, q_heads, width});
        scratch.carry_m   = workspace.alloc(DType::FP32, {q_heads, width});
        scratch.carry_l   = workspace.alloc(DType::FP32, {q_heads, width});
    }
    return scratch;
}

} // namespace ninfer::ops::detail::hq
