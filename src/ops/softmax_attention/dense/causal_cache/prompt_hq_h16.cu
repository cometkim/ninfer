#include "prompt_hq_impl.cuh"
namespace ninfer::ops::detail {
template void prompt_hq_geometry_launch<CausalD256H16Kv2, PagedKVLayerView, PagedKVDirectMetadata>(const Tensor& q, const Tensor& k, const Tensor& v,
                         const Tensor& positions, float scale, PagedKVLayerView cache, PagedKVDirectMetadata metadata,
                         const Tensor& scratch_k, const Tensor& scratch_v,
                         const Tensor& carry_acc, const Tensor& carry_m, const Tensor& carry_l,
                         std::uint32_t visible_keys, Tensor& out, cudaStream_t stream);
template void prompt_hq_geometry_launch<CausalD256H16Kv2, PagedKVBatchLayerView, PagedKVBatchMetadata<false>>(const Tensor& q, const Tensor& k, const Tensor& v,
                         const Tensor& positions, float scale, PagedKVBatchLayerView cache, PagedKVBatchMetadata<false> metadata,
                         const Tensor& scratch_k, const Tensor& scratch_v,
                         const Tensor& carry_acc, const Tensor& carry_m, const Tensor& carry_l,
                         std::uint32_t visible_keys, Tensor& out, cudaStream_t stream);
template void prompt_hq_geometry_launch<CausalD256H16Kv2, PagedKVBatchLayerView, PagedKVBatchMetadata<true>>(const Tensor& q, const Tensor& k, const Tensor& v,
                         const Tensor& positions, float scale, PagedKVBatchLayerView cache, PagedKVBatchMetadata<true> metadata,
                         const Tensor& scratch_k, const Tensor& scratch_v,
                         const Tensor& carry_acc, const Tensor& carry_m, const Tensor& carry_l,
                         std::uint32_t visible_keys, Tensor& out, cudaStream_t stream);
}
