// ninfer::ops - hq-e8-2b small-T partial-kernel route, 35B-A3B geometry. Split from
// the dtype route so the codec-heavy kernel compiles per geometry in parallel.
#include "ops/launcher/gqa_attention_decode_hq_routes.cuh"

namespace ninfer::ops::detail {

template void gqa_small_t_partial_hq<Gqa35Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
template void gqa_small_t_partial_hq<Gqa35Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

} // namespace ninfer::ops::detail
