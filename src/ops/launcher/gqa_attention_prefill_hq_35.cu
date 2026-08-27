// ninfer::ops - hq-e8-2b prompt/prefill routes, 35B-A3B geometry. Split from the
// dtype route so the codec-heavy kernels compile per geometry in parallel.
#include "ops/launcher/gqa_attention_prefill_hq_routes.cuh"

namespace ninfer::ops::detail {

#define NINFER_GQA_PREFILL_HQ35_INSTANTIATE(CACHE_VIEW, METADATA) \
    template void gqa_prefill_attention_hq<Gqa35Geometry, CACHE_VIEW, METADATA>( \
        const Tensor&, const Tensor&, float, const CACHE_VIEW&, METADATA, const Tensor&, \
        const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, \
        const Tensor&, std::uint32_t, const Tensor&, const Tensor&, const Tensor&, \
        std::int32_t, Tensor&, cudaStream_t); \
    template void gqa_prefill_append_hq<Gqa35Geometry, CACHE_VIEW, METADATA>( \
        const Tensor&, const Tensor&, const Tensor&, CACHE_VIEW, METADATA, cudaStream_t); \

NINFER_GQA_PREFILL_HQ35_INSTANTIATE(PagedKVLayerView, GqaPrefillDirectMetadata)
NINFER_GQA_PREFILL_HQ35_INSTANTIATE(PagedKVBatchLayerView, GqaPrefillBatchMetadata<false>)
NINFER_GQA_PREFILL_HQ35_INSTANTIATE(PagedKVBatchLayerView, GqaPrefillBatchMetadata<true>)
#undef NINFER_GQA_PREFILL_HQ35_INSTANTIATE

} // namespace ninfer::ops::detail
