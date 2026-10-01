#pragma once

// ninfer::ops::detail::hq - private launch prototypes of the hq-e8-2b causal attention routes.
// This is the HQ subset of the fork's former shared causal_cache/launch.h; the per-type entry
// points that the dispatcher calls are in hq/launch.h, the route and capacity policy in hq/plan.h.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::hq {

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
    // Physical SM count of the executing device; bounds the split grid (see hq/plan.cpp).
    std::int32_t multiprocessor_count = 0;
};

// Split-grid capacity of one small-T launch over an execution envelope. The device-side active
// split policy (small_t_common.cuh) never exceeds it.
std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size,
                                             std::int32_t multiprocessor_count);

void causal_attention_small_t_hq_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out,
    std::int32_t multiprocessor_count, cudaStream_t stream);

void causal_attention_cached_small_t_hq_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, std::int32_t multiprocessor_count, cudaStream_t stream);

void causal_attention_prompt_hq_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& positions, const Tensor& valid_columns,
                                       const Tensor& table_rows, float scale,
                                       PagedKVBatchLayerView cache, const Tensor& scratch_k,
                                       const Tensor& scratch_v, Tensor& out, cudaStream_t stream);

void causal_attention_prompt_hq_attention_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 const Tensor& scratch_k, const Tensor& scratch_v,
                                                 Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail::hq
