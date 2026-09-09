#include "ops/softmax_attention/dense/causal_cache/small_t_hq_tc_launch.h"

namespace ninfer::ops::detail {

template void causal_attention_small_t_hq_launch_for<CausalD256H16Kv2, CausalAppendInput>(
    const Tensor&, CausalAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

template void causal_attention_small_t_hq_launch_for<CausalD256H16Kv2, CausalCachedInput>(
    const Tensor&, CausalCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const CausalSmallTInvocation&, CausalAttentionExecutionEnvelope, Tensor&, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
