#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/operands.h"
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

// `gate` (optional, out's layout) is the attention output's sigmoid gate. Returns whether the
// selected route applied it in its output epilogue; the caller applies it otherwise.
bool bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, const Tensor* gate, DeviceExecutionView execution);
bool bf16_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, const Tensor* gate, DeviceExecutionView execution);

} // namespace ninfer::ops::detail
