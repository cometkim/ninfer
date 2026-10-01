#pragma once

// ninfer::ops::detail::hq - transient allocation recipe of the hq-e8-2b routes. The same helpers
// run against the execution WorkspaceArena and the capacity query's WorkspaceLayoutBuilder, so
// the reported capacity is exact by construction.

#include "core/tensor.h"

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

// Rotated-frame BF16 K/V planes of the prompt route, covering the visible-key envelope.
struct PromptScratch {
    Tensor k;
    Tensor v;
};

template <class Allocator>
PromptScratch allocate_prompt_scratch(Allocator& workspace, std::int32_t kv_heads,
                                      std::int32_t visible_keys) {
    return {
        workspace.alloc(DType::BF16, {kHqWorkspaceHeadDim, kv_heads, visible_keys}),
        workspace.alloc(DType::BF16, {kHqWorkspaceHeadDim, kv_heads, visible_keys}),
    };
}

} // namespace ninfer::ops::detail::hq
