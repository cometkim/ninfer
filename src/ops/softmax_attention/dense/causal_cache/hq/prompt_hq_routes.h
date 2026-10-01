#pragma once
// ninfer::ops::detail - hq-e8-2b prompt geometry launcher, defined in prompt_hq_impl.cuh and
// explicitly instantiated per geometry (prompt_hq_h24.cu, prompt_hq_h16.cu) for the direct and
// the masked/unmasked batched metadata.
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/dense/causal_cache/hq/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/hq/internal.h"

namespace ninfer::ops::detail::hq {

template <typename Geometry, typename CacheView, typename Metadata>
void prompt_hq_geometry_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, float scale, CacheView cache,
                               Metadata metadata, const Tensor& scratch_k, const Tensor& scratch_v,
                               const Tensor& carry_acc, const Tensor& carry_m,
                               const Tensor& carry_l, std::uint32_t visible_keys, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail::hq
