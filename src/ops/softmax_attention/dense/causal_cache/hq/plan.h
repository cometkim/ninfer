#pragma once

#include "ninfer/ops/softmax_attention.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// hq-e8-2b causal attention routes. The family is a function of the query width alone, so every
// call with W<=16 keeps one kernel topology across valid execution envelopes (the public
// CUDA Graph update contract); the envelope only sizes split grids and scratch.
//
// - SmallT: one fused append + split-KV partial launch and one reducer launch, W <= token tile.
// - ChunkedSmallT: the same pair per token-tile chunk, W <= 16.
// - Prompt: B=1, W > 16: append, then rotated-frame BF16 scratch of the visible history and the
//   FA2 kernel over it.
enum class HqKvFamily { SmallT, ChunkedSmallT, Prompt };

struct HqKvCausalPlan {
    HqKvFamily family;
    int query_heads;
    int width;
    int batch;
    // Query columns per small-T launch: 8 for 24 query heads, 6 for 16.
    int chunk_tokens;
    CausalAttentionExecutionEnvelope envelope;
};

HqKvCausalPlan make_hq_kv_causal_plan(int query_heads, int width, int batch,
                                      CausalAttentionExecutionEnvelope envelope,
                                      int multiprocessor_count);

// Exact transient capacity for every W in [min_width, max_width] at one batch size, computed with
// the same allocation recipe as hq_kv_append_attention / hq_kv_cached_attention.
std::size_t hq_kv_workspace_bytes(int query_heads, int batch, int min_width, int max_width,
                                  CausalAttentionExecutionEnvelope envelope,
                                  int multiprocessor_count);

} // namespace ninfer::ops::detail
