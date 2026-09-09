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
void launch_hq_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                         const Tensor& positions, float scale, CacheView cache, Metadata metadata,
                         const Tensor& scratch_k, const Tensor& scratch_v,
                         const Tensor& carry_acc, const Tensor& carry_m, const Tensor& carry_l,
                         std::uint32_t visible_keys, Tensor& out, cudaStream_t stream) {
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
    static const bool smem_attribute_carry = [] {
        const cudaError_t status = cudaFuncSetAttribute(
            causal_attention_prompt_hq_kernel<Geometry, Metadata, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kHqPromptSmemBytes));
        return status == cudaSuccess;
    }();
    if (!smem_attribute || !smem_attribute_carry) {
        throw std::runtime_error("causal_attention prompt: hq-e8-2b shared memory is unavailable");
    }

    // Materialize the visible history (rotated-frame bf16, eight lanes per row through the
    // cooperative group decoder), then run the rotated FA2 prompt kernel over the scratch.
    // Scratch wider than `span` keys runs in sequential carry bands: each band decodes its keys
    // into the band-local scratch rows and the FA2 kernel resumes/writes back the online-softmax
    // state (m, l, unnormalized acc) between bands. With the residual window on, the fresh-rotate
    // pass stages the current chunk's bf16 rows exact per band and the scratch kernel reads
    // sink/recent rows from the side planes.
    const auto tokens   = static_cast<std::int32_t>(q.ne[2]);
    const auto span     = static_cast<std::int32_t>(scratch_k.ne[2]);
    const auto visible  = static_cast<std::int32_t>(visible_keys);
    const bool has_residual = cache.residual_k.data != nullptr;
    // Fresh-chunk staging needs this call's bf16 K/V rows; the read-only cached route has none,
    // so its queries keep the ring-tail window instead of the in-chunk window.
    const bool has_fresh = has_residual && k.data != nullptr && v.data != nullptr;
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    const int bands = static_cast<int>(div_up(visible, span));
    if (bands > 1 && (span % kCausalPromptBc) != 0) {
        // Band bases index the FA2 key-tile grid; a mid-tile base would stage scratch rows below
        // the band. The production band (262144) is tile-aligned.
        throw std::invalid_argument("causal_attention prompt: scratch band is not tile-aligned");
    }
    for (int band = 0; band < bands; ++band) {
        const std::int32_t key_begin = band * span;
        const std::int32_t band_rows = span < visible - key_begin ? span : visible - key_begin;
        if (has_fresh) {
            const std::int64_t fresh_units =
                static_cast<std::int64_t>(tokens) * Geometry::KVHeads * 2;
            const int fresh_warps = kKVCacheHqFillWarps;
            const int fresh_grid =
                static_cast<int>(div_up(fresh_units, static_cast<std::int64_t>(fresh_warps)));
            causal_attention_prompt_hq_fresh_rotate_kernel<Geometry, Metadata>
                <<<fresh_grid, fresh_warps * 32, kHqHeadDim, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata, tokens, span,
                    static_cast<__nv_bfloat16*>(scratch_k.data),
                    static_cast<__nv_bfloat16*>(scratch_v.data), key_begin, band_rows);
            CUDA_CHECK(cudaGetLastError());
        }
        const std::int64_t units_bound =
            static_cast<std::int64_t>(band_rows) * Geometry::KVHeads * 2 * 8;
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
                static_cast<__nv_bfloat16*>(scratch_v.data), key_begin, band_rows,
                has_residual ? static_cast<const __nv_bfloat16*>(cache.residual_k.data) : nullptr,
                has_residual ? static_cast<const __nv_bfloat16*>(cache.residual_v.data) : nullptr,
                has_residual ? static_cast<const std::uint32_t*>(cache.side_words.data) : nullptr,
                has_fresh);
        CUDA_CHECK(cudaGetLastError());
        if (bands == 1) {
            causal_attention_prompt_hq_kernel<Geometry, Metadata>
                <<<attention_grid, kCausalPromptThreads, kHqPromptSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(scratch_k.data),
                    static_cast<const __nv_bfloat16*>(scratch_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, span);
        } else {
            causal_attention_prompt_hq_kernel<Geometry, Metadata, true>
                <<<attention_grid, kCausalPromptThreads, kHqPromptSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(scratch_k.data),
                    static_cast<const __nv_bfloat16*>(scratch_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, span, key_begin,
                    key_begin + band_rows, static_cast<__nv_bfloat16*>(carry_acc.data),
                    static_cast<float*>(carry_m.data), static_cast<float*>(carry_l.data),
                    band + 1 == bands ? 0 : 1);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

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
            launch_hq_attention<Geometry>(q, k, v, positions, scale, cache, metadata,
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
        launch_hq_attention<CausalD256H24Kv4>(q, no_input, no_input, positions, scale, cache,
                                              metadata, scratch_k, scratch_v, carry_acc, carry_m,
                                              carry_l, visible_keys, out, stream);
        return;
    }
    launch_hq_attention<CausalD256H16Kv2>(q, no_input, no_input, positions, scale, cache,
                                          metadata, scratch_k, scratch_v, carry_acc, carry_m,
                                          carry_l, visible_keys, out, stream);
}

} // namespace ninfer::ops::detail
