// ninfer::ops::detail - hq-e8-2b prompt/prefill launch ownership: fill (shared with the append
// family), one scratch decode of the visible history, then the rotated-frame FA2 kernel over
// the scratch. The codec-heavy kernels dominate this route's compile time, so each geometry
// instantiates in this one TU.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kv_cache/append/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/hq_kernel.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_hq.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::size_t kHqPromptSmemBytes = kCausalPromptSmemBytes + kHqHeadDim;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_hq_attention(const Tensor& q, const Tensor& positions, float scale, CacheView cache,
                         Metadata metadata, const Tensor& scratch_k, const Tensor& scratch_v,
                         Tensor& out, cudaStream_t stream) {
    if (scratch_k.data == nullptr || scratch_v.data == nullptr || scratch_k.dtype != DType::BF16 ||
        scratch_v.dtype != DType::BF16) {
        throw std::invalid_argument("causal_attention prompt: hq-e8-2b scratch is missing");
    }
    static const bool smem_attribute = [] {
        const cudaError_t status = cudaFuncSetAttribute(
            causal_attention_prompt_hq_kernel<Geometry, Metadata>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kHqPromptSmemBytes));
        return status == cudaSuccess;
    }();
    if (!smem_attribute) {
        throw std::runtime_error("causal_attention prompt: hq-e8-2b shared memory is unavailable");
    }

    // Materialize the visible history once (rotated-frame bf16, eight lanes per row through the
    // cooperative group decoder), then run the rotated FA2 prompt kernel over the scratch.
    const auto tokens      = static_cast<std::int32_t>(q.ne[2]);
    const auto span        = static_cast<std::int32_t>(scratch_k.ne[2]);
    const auto units_bound = static_cast<std::int64_t>(span) * Geometry::KVHeads * 2 * 8;
    const int scratch_grid = static_cast<int>(
        div_up(units_bound, static_cast<std::int64_t>(kCausalPromptHqScratchThreads)));
    causal_attention_prompt_hq_scratch_kernel<Geometry, Metadata>
        <<<scratch_grid, kCausalPromptHqScratchThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), tokens, span,
            static_cast<__nv_bfloat16*>(scratch_k.data),
            static_cast<__nv_bfloat16*>(scratch_v.data));
    CUDA_CHECK(cudaGetLastError());

    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_hq_kernel<Geometry, Metadata>
        <<<attention_grid, kCausalPromptThreads, kHqPromptSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(scratch_k.data),
            static_cast<const __nv_bfloat16*>(scratch_v.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, span);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_attention_prompt_hq_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& positions, const Tensor& valid_columns,
                                       const Tensor& table_rows, float scale,
                                       PagedKVBatchLayerView cache, const Tensor& scratch_k,
                                       const Tensor& scratch_v, Tensor& out, cudaStream_t stream) {
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
            launch_hq_attention<Geometry>(q, positions, scale, cache, metadata, scratch_k,
                                          scratch_v, out, stream);
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
                                                 Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_hq_attention<CausalD256H24Kv4>(q, positions, scale, cache, metadata, scratch_k,
                                              scratch_v, out, stream);
        return;
    }
    launch_hq_attention<CausalD256H16Kv2>(q, positions, scale, cache, metadata, scratch_k,
                                          scratch_v, out, stream);
}

} // namespace ninfer::ops::detail
