#include "prompt_hq_routes.h"
// ninfer::ops::detail - hq-e8-2b prompt/prefill launch ownership: fill (shared with the append
// family), one scratch decode of the visible history, then the rotated-frame FA2 kernel over
// the scratch. Heavy kernels instantiate in separate per-geometry translation units.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kv_cache/append/launch.h"

#include "ops/common/math.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void causal_attention_prompt_hq_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& positions, const Tensor& valid_columns,
                                       const Tensor& table_rows, float scale,
                                       PagedKVBatchLayerView cache, const Tensor& scratch_k,
                                       const Tensor& scratch_v, const Tensor& carry_acc,
                                       const Tensor& carry_m, const Tensor& carry_l,
                                       std::uint32_t visible_keys, Tensor& out,
                                       cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        const auto launch_geometry = [&]<typename Geometry>() {
            prompt_hq_geometry_launch<Geometry>(q, k, v, positions, scale, cache, metadata,
                                          scratch_k, scratch_v, carry_acc, carry_m, carry_l,
                                          visible_keys, out, stream);
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            launch_geometry.template operator()<CausalD256H24Kv4>();
        } else {
            launch_geometry.template operator()<CausalD256H16Kv2>();
        }
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void causal_attention_prompt_hq_attention_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 const Tensor& scratch_k, const Tensor& scratch_v,
                                                 const Tensor& carry_acc, const Tensor& carry_m,
                                                 const Tensor& carry_l,
                                                 std::uint32_t visible_keys, Tensor& out,
                                                 cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data),
                                         cache.slot};
    const Tensor no_input;
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        prompt_hq_geometry_launch<CausalD256H24Kv4>(q, no_input, no_input, positions, scale, cache,
                                              metadata, scratch_k, scratch_v, carry_acc, carry_m,
                                              carry_l, visible_keys, out, stream);
        return;
    }
    prompt_hq_geometry_launch<CausalD256H16Kv2>(q, no_input, no_input, positions, scale, cache,
                                          metadata, scratch_k, scratch_v, carry_acc, carry_m,
                                          carry_l, visible_keys, out, stream);
}

} // namespace ninfer::ops::detail
