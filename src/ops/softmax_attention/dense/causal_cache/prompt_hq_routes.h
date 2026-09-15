#pragma once
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"
namespace ninfer::ops::detail {
template <typename Geometry, typename CacheView, typename Metadata>
void prompt_hq_geometry_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                         const Tensor& positions, float scale, CacheView cache, Metadata metadata,
                         const Tensor& scratch_k, const Tensor& scratch_v,
                         const Tensor& carry_acc, const Tensor& carry_m, const Tensor& carry_l,
                         std::uint32_t visible_keys, Tensor& out, cudaStream_t stream);
}
