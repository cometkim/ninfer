// ninfer::ops::detail - nvfp4 small-T launcher instantiations (build-speed TU split).
#include "ops/softmax_attention/dense/causal_cache/small_t_nvfp4_tc_launch.h"

namespace ninfer::ops::detail {

template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 1, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 1, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 3, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 5, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 6, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 7, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H24Kv4, 8, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 1, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 3, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 5, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_nvfp4_partial<CausalD256H16Kv2, 6, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
