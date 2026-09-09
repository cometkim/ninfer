// ninfer::ops::detail - bf16 small-T launcher instantiations (build-speed TU split).
#include "ops/softmax_attention/dense/causal_cache/small_t_tc_launch.h"

namespace ninfer::ops::detail {

template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 1, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 2, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 3, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 4, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 5, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 6, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 7, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H24Kv4, 8, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 1, 2, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 2, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 3, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 4, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 5, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, false, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, false, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, false, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, true, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);
template void launch_tc_partial_bf16<CausalD256H16Kv2, 6, 4, true, true, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail