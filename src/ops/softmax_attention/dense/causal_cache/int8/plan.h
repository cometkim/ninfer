#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

#include <cstddef>

namespace ninfer::ops::detail {

// SmallT is the fork kernel-perf route (hq/plan.h, linear_kv_small_t_*), taking grouped widths
// where it measured faster.
enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled, SmallT };

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
    // Tiled family only: key-range splits of each query tile (1 = single pass, direct output).
    int tiled_splits = 1;
    // SmallT family only: split-grid capacity (linear_kv_small_t_splits).
    int small_t_splits = 0;
    // Grouped family: small-T partial bytes reserved for sub-envelope calls
    // (linear_kv_small_t_reserve_bytes).
    std::size_t small_t_reserve = 0;
};

// Wave-fill key split of the tiled route (the fork's WI-K1a): S in 1..4, chosen only when the
// predicted waves per unit of work improve by at least 10% over the single pass.
int int8_kv_tiled_splits(int heads, int width, int multiprocessor_count);

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count);
std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

} // namespace ninfer::ops::detail
