#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/rope.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream);

// Text RoPE with the run's pair-frequency table (the checkpoint-linear table, or the YaRN table
// under rope scaling). The checkpoint's MRoPE axis mapping must be the native pair % 3.
void text_rope(const Tensor& positions, const RopeConfig& config,
               const ops::RopeFrequencies& frequencies, Tensor& x, ops::RopeSide side,
               DeviceExecutionView execution);
void text_rope(const Tensor& positions, const RopeConfig& config,
               const ops::RopeFrequencies& frequencies, Tensor& query, Tensor& key,
               DeviceExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
