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

// Fork kernel-perf small-T route of the linear KV codecs (INT8, FP8, NVFP4, FP8-K/NVFP4-V): the
// tensor-core split-KV partial kernel with the fused append plus the gated reducer, two kernel
// launches like the codecs' grouped routes. `selected` says whether it serves a grouped-width
// profile (24 query heads W <= 8, 16 query heads W <= 6): only where it measured faster than the
// codec's grouped route, per envelope, so a graph tier may switch families without changing its
// topology. `splits` is its split-grid capacity, `workspace_bytes` its exact partial storage.
bool linear_kv_small_t_selected(KvCacheStorage storage, int query_heads, int width, int batch,
                                CausalAttentionExecutionEnvelope envelope);
int linear_kv_small_t_splits(KvCacheStorage storage, int query_heads, int width, int batch,
                             CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);
std::size_t linear_kv_small_t_workspace_bytes(int query_heads, int width, int splits, int batch);

// Selection depends on the envelope's upper bound, so calls under sub-envelopes of a grouped
// envelope may still take the small-T route: `reserve_bytes` is the small-T partial storage over
// that sub-envelope (0 when none). The grouped route pads its own partials up to it
// (pad_linear_kv_small_t_reserve), and each codec's capacity query includes it, so the query stays
// the exact high-water mark of a call at its envelope and still bounds every sub-envelope call.
std::size_t linear_kv_small_t_reserve_bytes(KvCacheStorage storage, int query_heads, int width,
                                            int batch, CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count);

template <class Allocator>
void pad_linear_kv_small_t_reserve(Allocator& workspace, std::size_t allocated,
                                   std::size_t reserve) {
    if (reserve > allocated) (void)workspace.alloc_bytes(reserve - allocated, 1);
}

} // namespace ninfer::ops::detail
