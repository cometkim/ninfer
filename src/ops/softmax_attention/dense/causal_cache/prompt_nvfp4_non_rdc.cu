#include "prompt_nvfp4_non_rdc_routes.h"
// Non-RDC ownership for the warp-specialized NVFP4 causal prompt kernel.
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {


template <typename CacheView, typename Metadata>
void dispatch(const Tensor& q, const Tensor& positions, float scale, const CacheView& cache,
              Metadata metadata, Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        prompt_nvfp4_non_rdc_geometry_launch<CausalD256H24Kv4>(q, positions, scale, cache, metadata, out, stream);
        return;
    }
    prompt_nvfp4_non_rdc_geometry_launch<CausalD256H16Kv2>(q, positions, scale, cache, metadata, out, stream);
}

} // namespace

void causal_attention_prompt_nvfp4_kernel_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch(q, positions, scale, cache, metadata, out, stream);
}

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor& q, const Tensor& positions,
                                                       const Tensor& valid_columns,
                                                       const Tensor& table_rows, float scale,
                                                       const PagedKVBatchLayerView& cache,
                                                       Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        dispatch(q, positions, scale, cache, metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
