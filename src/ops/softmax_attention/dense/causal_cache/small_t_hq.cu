// HQ small-T dispatcher; heavy kernels are instantiated once per geometry.
#include "ops/softmax_attention/dense/causal_cache/small_t_hq_tc_launch.h"

namespace ninfer::ops::detail {


void causal_attention_small_t_hq_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (width < 1 || width > hq_token_tile<CausalD256H24Kv4>()) {
        throw std::invalid_argument("causal_attention_small_t_hq_launch: unsupported T");
    }
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_hq_launch_for<CausalD256H24Kv4>(q, input, positions, scale, cache,
                                                                 invocation, envelope, partial_acc,
                                                                 partial_m, partial_l, out, stream);
        return;
    }
    if (width > hq_token_tile<CausalD256H16Kv2>()) {
        throw std::invalid_argument("causal_attention_small_t_hq_launch: unsupported T");
    }
    causal_attention_small_t_hq_launch_for<CausalD256H16Kv2>(q, input, positions, scale, cache,
                                                             invocation, envelope, partial_acc,
                                                             partial_m, partial_l, out, stream);
}

void causal_attention_cached_small_t_hq_launch(const Tensor& q, const Tensor& positions,
                                               float scale, const PagedKVLayerView& cache,
                                               CausalAttentionExecutionEnvelope envelope,
                                               Tensor& partial_acc, Tensor& partial_m,
                                               Tensor& partial_l, Tensor& out,
                                               cudaStream_t stream) {
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_hq_launch_for<CausalD256H24Kv4>(
            q, input, positions, scale, batch_cache, invocation, envelope, partial_acc, partial_m,
            partial_l, out, stream);
        return;
    }
    causal_attention_small_t_hq_launch_for<CausalD256H16Kv2>(
        q, input, positions, scale, batch_cache, invocation, envelope, partial_acc, partial_m,
        partial_l, out, stream);
}

} // namespace ninfer::ops::detail
