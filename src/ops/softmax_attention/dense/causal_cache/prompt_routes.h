#pragma once

// Private dtype launch boundary. Keep kernel instantiations in prompt_bf16.cu and
// prompt_i8.cu so the heavy prompt routes compile independently.
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

namespace ninfer::ops::detail {
template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_bf16_launch_for(const Tensor& q, const Tensor& positions, float scale,
                                             const CacheView& cache, Metadata metadata, Tensor& out,
                                             Tensor& partial_acc, Tensor& partial_m,
                                             Tensor& partial_l, std::int32_t split_count,
                                             cudaStream_t stream);
template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_i8_launch_for(const Tensor& q, const Tensor& positions, float scale,
                                           const CacheView& cache, Metadata metadata, Tensor& out,
                                           Tensor& partial_acc, Tensor& partial_m,
                                           Tensor& partial_l, std::int32_t split_count,
                                           cudaStream_t stream);
} // namespace ninfer::ops::detail
