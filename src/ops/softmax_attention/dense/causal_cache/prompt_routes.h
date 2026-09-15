#pragma once

// Private dtype launch boundary; heavy kernels instantiate only in the dtype TUs.
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

namespace ninfer::ops::detail {
template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_bf16_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream);
template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_i8_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream);
} // namespace ninfer::ops::detail
