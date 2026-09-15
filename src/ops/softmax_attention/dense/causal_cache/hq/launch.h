#pragma once

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

// `gate` (optional, out's layout) is the attention output's sigmoid gate. Returns whether the
// selected route applied it in its output epilogue (the small-T reducer); the caller applies it
// otherwise (the prompt route).
bool hq_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, const Tensor* gate, DeviceExecutionView execution);

bool hq_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, const Tensor* gate, DeviceExecutionView execution);

} // namespace ninfer::ops::detail
