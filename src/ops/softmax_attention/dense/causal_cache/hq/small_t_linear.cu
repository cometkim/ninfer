// Linear-codec small-T dispatcher of the fork kernel-perf routes; the heavy kernels are
// instantiated once per codec and geometry (small_t_<codec>_h24.cu, small_t_<codec>_h16.cu).
#include "ops/softmax_attention/dense/causal_cache/hq/small_t_linear_tc_launch.h"

namespace ninfer::ops::detail::hq {
namespace {

template <typename CacheInput>
void launch_for_storage(const Tensor& q, CacheInput input, const Tensor& positions, float scale,
                        PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                        CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc,
                        Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<SmallTCodec Codec>() {
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_small_t_linear_launch_for<Codec, CausalD256H24Kv4>(
                q, input, positions, scale, cache, invocation, envelope, partial_acc, partial_m,
                partial_l, out, stream);
        } else {
            causal_attention_small_t_linear_launch_for<Codec, CausalD256H16Kv2>(
                q, input, positions, scale, cache, invocation, envelope, partial_acc, partial_m,
                partial_l, out, stream);
        }
    };
    switch (cache.storage) {
    case KvCacheStorage::Int8Group64:
        launch.template operator()<SmallTCodec::Int8>();
        return;
    case KvCacheStorage::Fp8E4M3Row256:
        launch.template operator()<SmallTCodec::Fp8>();
        return;
    case KvCacheStorage::Nvfp4Group16:
        launch.template operator()<SmallTCodec::Nvfp4>();
        return;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        launch.template operator()<SmallTCodec::K8v4>();
        return;
    default:
        break;
    }
    throw std::invalid_argument("causal_attention_small_t_linear_launch: unsupported KV codec");
}

} // namespace

void causal_attention_small_t_linear_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, const Tensor* gate, std::int32_t multiprocessor_count,
    cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns        = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows           = &table_rows,
        .full_width           = q.ne[2],
        .column_begin         = 0,
        .width                = q.ne[2],
        .batch_size           = q.ne[3],
        .multiprocessor_count = multiprocessor_count,
        .gate                 = gate,
    };
    launch_for_storage(q, input, positions, scale, cache, invocation, envelope, partial_acc,
                       partial_m, partial_l, out, stream);
}

void causal_attention_cached_small_t_linear_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, const Tensor* gate, std::int32_t multiprocessor_count,
    cudaStream_t stream) {
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns        = nullptr,
        .table_rows           = nullptr,
        .full_width           = q.ne[2],
        .column_begin         = 0,
        .width                = q.ne[2],
        .batch_size           = 1,
        .multiprocessor_count = multiprocessor_count,
        .gate                 = gate,
    };
    launch_for_storage(q, input, positions, scale, single_row_paged_kv_batch_view(cache),
                       invocation, envelope, partial_acc, partial_m, partial_l, out, stream);
}

} // namespace ninfer::ops::detail::hq
