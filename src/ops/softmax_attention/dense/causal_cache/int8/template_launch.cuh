#pragma once

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/tiled_mma.cuh"
#include "ops/softmax_attention/common/causal_merge.cuh"
#include "ops/softmax_attention/common/causal_tiled_merge.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false>
void launch_int8_kv_grouped_mma(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                                Input input, CausalKvPartition partition, CausalPartialView partial,
                                cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_quantized_causal_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 12 ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("INT8 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("INT8 append requires K/V");
    constexpr auto kernel =
        int8_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries>;
    constexpr int bytes = S::kDynamicArena ? S::kArenaBytes : 0;
    if constexpr (S::kDynamicArena) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, S::kTokenTile) : 1),
                    partition.capacity, p.batch);
    CUDA_CHECK(pdl::launch_dependent(
        {grid, dim3(S::kThreads), static_cast<std::size_t>(bytes), stream}, kernel, p.q, input,
        p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales, cache.tables,
        cache.valid_columns, cache.table_rows, cache.table_stride, p.width, p.visible_capacity,
        partition, p.scale, partial.acc, partial.maximum, partial.sum));
}

// splits > 1 runs the key-range split kernel into `partial` ([D,Hq,W,splits] FP32 state) and
// merges it with causal_tiled_merge_kernel, which also applies p.gate; splits == 1 writes the
// output directly (the gate is the caller's).
template <class G, class S>
void launch_int8_kv_tiled_mma(const CausalAttentionOperands& p, Int8KvReadView cache, int splits,
                              CausalPartialView partial, cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("INT8 tiled attention requires a complete single query row");
    if (splits < 1 || splits > 4 ||
        (splits > 1 && (!partial.acc || !partial.maximum || !partial.sum)))
        throw std::invalid_argument("INT8 tiled attention: invalid key split or partial storage");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        const auto launch = [&]<bool Split>() {
            constexpr auto kernel    = int8_kv_tiled_mma_kernel<G, S, Metadata, Split>;
            static const auto status = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
            CUDA_CHECK(status);
            const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, Split ? splits : 1);
            kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
                p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                p.positions, p.scale, p.out, p.width, partial);
            CUDA_CHECK(cudaGetLastError());
        };
        if (splits == 1) {
            launch.template operator()<false>();
            return;
        }
        launch.template operator()<true>();
        // Every split owns its segment of each row's own range; the merge reads all of them.
        launch_causal_tiled_merge<G, false>(p, cache.valid_columns,
                                            CausalKvPartition{splits, splits, 0}, partial, stream);
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
}

} // namespace ninfer::ops::detail
