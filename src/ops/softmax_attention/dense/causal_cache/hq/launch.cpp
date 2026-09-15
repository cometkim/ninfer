// ninfer::ops::detail - hq-e8-2b causal attention entry points: route by query width, then the
// fork's HQ small-T (per token-tile chunk) or prompt launch with its transient workspace.
#include "ops/softmax_attention/dense/causal_cache/hq/launch.h"

#include "ops/softmax_attention/dense/causal_cache/hq/internal.h"
#include "ops/softmax_attention/dense/causal_cache/hq/plan.h"
#include "ops/softmax_attention/dense/causal_cache/hq/workspace.h"

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

std::int32_t chunk_splits(const HqKvCausalPlan& plan, std::int32_t count,
                          DeviceExecutionView execution) {
    return hq::causal_attention_split_capacity(plan.query_heads, count, KvCacheStorage::HqE8Rice2B,
                                               plan.envelope, plan.batch,
                                               execution.multiprocessor_count);
}

} // namespace

bool hq_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, const Tensor* gate, DeviceExecutionView execution) {
    const auto plan =
        make_hq_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope, execution.multiprocessor_count);
    if (plan.family == HqKvFamily::Prompt) {
        auto scope         = workspace.scope();
        const auto scratch = hq::allocate_prompt_scratch(
            workspace, cache.num_kv_heads, plan.query_heads, plan.width, envelope.max_visible_keys);
        hq::causal_attention_prompt_hq_launch(
            q, k, v, positions, valid, rows, scale, cache, scratch.k, scratch.v, scratch.carry_acc,
            scratch.carry_m, scratch.carry_l, envelope.max_visible_keys, out, execution.stream);
        return false;
    }
    // The chunked family runs the small-T pair per token-tile chunk over the full tensors; the
    // fused append writes each chunk's own columns.
    for (std::int32_t begin = 0; begin < plan.width; begin += plan.chunk_tokens) {
        const std::int32_t count = std::min(plan.chunk_tokens, plan.width - begin);
        auto chunk_scope         = workspace.scope();
        auto partial             = hq::allocate_small_t_workspace(
            workspace, plan.query_heads, count, chunk_splits(plan, count, execution), plan.batch);
        hq::causal_attention_small_t_hq_launch(
            q, k, v, positions, valid, rows, scale, cache, envelope, begin, count, partial.acc,
            partial.m, partial.l, out, gate, execution.multiprocessor_count, execution.stream);
    }
    return gate != nullptr;
}

bool hq_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, const Tensor* gate, DeviceExecutionView execution) {
    const auto plan =
        make_hq_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    if (plan.family == HqKvFamily::Prompt) {
        auto scope         = workspace.scope();
        const auto scratch = hq::allocate_prompt_scratch(
            workspace, cache.num_kv_heads, plan.query_heads, plan.width, envelope.max_visible_keys);
        hq::causal_attention_prompt_hq_attention_launch(
            q, positions, scale, cache, scratch.k, scratch.v, scratch.carry_acc, scratch.carry_m,
            scratch.carry_l, envelope.max_visible_keys, out, execution.stream);
        return false;
    }
    for (std::int32_t begin = 0; begin < plan.width; begin += plan.chunk_tokens) {
        const std::int32_t count = std::min(plan.chunk_tokens, plan.width - begin);
        auto chunk_scope         = workspace.scope();
        auto partial   = hq::allocate_small_t_workspace(workspace, plan.query_heads, count,
                                                        chunk_splits(plan, count, execution), 1);
        Tensor q_chunk = q.slice(2, begin, count);
        Tensor position_chunk = positions.slice(0, begin, count);
        Tensor out_chunk      = out.slice(2, begin, count);
        // The reducer indexes the gate like its (chunk) output.
        const Tensor gate_chunk =
            gate != nullptr
                ? gate->view({out.ne[0], out.ne[1], out.ne[2], out.ne[3]}).slice(2, begin, count)
                : Tensor{};
        hq::causal_attention_cached_small_t_hq_launch(
            q_chunk, position_chunk, scale, cache, envelope, partial.acc, partial.m, partial.l,
            out_chunk, gate != nullptr ? &gate_chunk : nullptr, execution.multiprocessor_count,
            execution.stream);
    }
    return gate != nullptr;
}

} // namespace ninfer::ops::detail
