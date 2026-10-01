#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

#include <cstddef>

namespace ninfer::ops::detail {

// SmallT is the fork kernel-perf route (hq/plan.h, linear_kv_small_t_*), taking grouped widths
// where it measured faster.
enum class Fp8KvFamily { Grouped, ParallelGrouped, Tiled, SmallT };

struct Fp8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Fp8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
    // SmallT family only: split-grid capacity (linear_kv_small_t_splits).
    int small_t_splits = 0;
    // Grouped family: small-T partial bytes reserved for sub-envelope calls
    // (linear_kv_small_t_reserve_bytes).
    std::size_t small_t_reserve = 0;
};

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope,
                                        int multiprocessor_count);
std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope,
                                   int multiprocessor_count);

} // namespace ninfer::ops::detail
