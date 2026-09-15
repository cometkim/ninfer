#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
    // Tiled family only: key-range splits of each query tile (1 = single pass, direct output).
    int tiled_splits = 1;
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
