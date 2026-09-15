#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/template_launch.cuh"
#include "ops/kv_cache/append/launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, class Input, bool Writable>
void grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                      Input input, CausalKvPartition partition, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (p.width) {
#define NINFER_INT8_GROUPED(T)                                                                     \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_INT8_GROUPED(1);
        NINFER_INT8_GROUPED(2);
        NINFER_INT8_GROUPED(3);
        NINFER_INT8_GROUPED(4);
        NINFER_INT8_GROUPED(5);
        NINFER_INT8_GROUPED(6);
        NINFER_INT8_GROUPED(7);
        NINFER_INT8_GROUPED(8);
#undef NINFER_INT8_GROUPED
    }
    throw std::logic_error("INT8 grouped plan exceeds the selected token tile");
}

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, const Tensor* gate, cudaStream_t stream) {
    const auto view =
        make_quantized_causal_cache_view<Int8KvCacheView<Input::writes_cache>>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    const auto p =
        make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys, gate);
    if (plan.query_heads == 24)
        grouped_instance<CausalD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else
        grouped_instance<CausalD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
}

template <class G, int Tokens>
void parallel_grouped(const CausalAttentionOperands& p, Int8KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   CausalCachedInput, true>(p, cache, {}, partition, partial,
                                                            stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

void execute_parallel(const CausalAttentionOperands& p, Int8KvReadView cache,
                      const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    if (plan.query_heads == 24)
        parallel_grouped<CausalD256H24Kv4, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
    else
        parallel_grouped<CausalD256H16Kv2, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
}

// Returns whether the output went through the split merge (which applies p.gate).
bool tiled(const CausalAttentionOperands& p, Int8KvReadView cache, const Int8KvCausalPlan& plan,
           WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    const int splits = plan.tiled_splits;
    CausalPartialView partial{};
    if (splits > 1)
        partial =
            allocate_causal_partials(workspace, plan.query_heads, plan.width, splits, 1).view();
    if (p.query_heads == 24)
        launch_int8_kv_tiled_mma<CausalD256H24Kv4, Int8KvTiledInstance>(p, cache, splits, partial,
                                                                        stream);
    else
        launch_int8_kv_tiled_mma<CausalD256H16Kv2, Int8KvTiledInstance>(p, cache, splits, partial,
                                                                        stream);
    return splits > 1;
}

} // namespace

bool int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, const Tensor* gate, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan           = make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                         execution.multiprocessor_count);
    if (plan.family != Int8KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        // The tiled route applies the gate only when its key split merges partial state.
        const auto p =
            make_causal_operands(q, positions, out, scale, envelope.max_visible_keys, gate);
        const auto view =
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            return tiled(p, view, plan, workspace, stream) && gate != nullptr;
        execute_parallel(p, view, plan, workspace, stream);
        return gate != nullptr;
    }
    execute_grouped(q, positions, scale, cache, &valid, &rows,
                    CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                      static_cast<const __nv_bfloat16*>(v.data)},
                    plan, workspace, out, gate, stream);
    return gate != nullptr;
}

bool int8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, const Tensor* gate, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Int8KvFamily::Tiled)
        return tiled(
                   make_causal_operands(q, positions, out, scale, envelope.max_visible_keys, gate),
                   make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), plan, workspace,
                   stream) &&
               gate != nullptr;
    if (plan.family == Int8KvFamily::ParallelGrouped)
        execute_parallel(
            make_causal_operands(q, positions, out, scale, envelope.max_visible_keys, gate),
            make_quantized_causal_cache_view<Int8KvCacheView<false>>(view), plan, workspace,
            stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, CausalCachedInput{}, plan,
                        workspace, out, gate, stream);
    return gate != nullptr;
}

} // namespace ninfer::ops::detail
