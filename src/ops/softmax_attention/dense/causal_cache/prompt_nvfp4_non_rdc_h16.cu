#include "prompt_nvfp4_non_rdc_impl.cuh"
namespace ninfer::ops::detail {
template void prompt_nvfp4_non_rdc_geometry_launch<CausalD256H16Kv2, PagedKVLayerView, PagedKVDirectMetadata>(const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
                PagedKVDirectMetadata metadata, Tensor& out, cudaStream_t stream);
template void prompt_nvfp4_non_rdc_geometry_launch<CausalD256H16Kv2, PagedKVBatchLayerView, PagedKVBatchMetadata<false>>(const Tensor& q, const Tensor& positions, float scale, const PagedKVBatchLayerView& cache,
                PagedKVBatchMetadata<false> metadata, Tensor& out, cudaStream_t stream);
template void prompt_nvfp4_non_rdc_geometry_launch<CausalD256H16Kv2, PagedKVBatchLayerView, PagedKVBatchMetadata<true>>(const Tensor& q, const Tensor& positions, float scale, const PagedKVBatchLayerView& cache,
                PagedKVBatchMetadata<true> metadata, Tensor& out, cudaStream_t stream);
}
