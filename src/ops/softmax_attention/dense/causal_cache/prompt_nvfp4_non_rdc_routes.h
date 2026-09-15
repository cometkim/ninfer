#pragma once
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"
namespace ninfer::ops::detail {
template <typename Geometry, typename CacheView, typename Metadata>
void prompt_nvfp4_non_rdc_geometry_launch(const Tensor& q, const Tensor& positions, float scale, const CacheView& cache,
                Metadata metadata, Tensor& out, cudaStream_t stream);
}
