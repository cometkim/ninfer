#pragma once

// ninfer::ops::detail - hq-e8-2b prompt/prefill launch routes, shared by the
// per-geometry instantiation TUs gqa_attention_prefill_hq_{27,35}.cu. The
// codec-heavy scratch kernel dominates this route's compile time, so each
// geometry is explicitly instantiated in its own translation unit.

#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_hq(const Tensor& q, const Tensor& positions, float scale,
                              const CacheView& cache, Metadata metadata, const Tensor& scratch_k,
                              const Tensor& scratch_v, Tensor& out, cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    if (scratch_k.data == nullptr || scratch_v.data == nullptr ||
        scratch_k.dtype != DType::BF16 || scratch_v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention prompt: hq-e8-2b scratch is missing");
    }
    static const cudaError_t attr_rot = cudaFuncSetAttribute(
        gqa_attention_prefill_bf16_kernel<Geometry, Metadata, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kGqaPrefillRotatedSmemBytes));
    CUDA_CHECK(attr_rot);
    // Materialize the visible history once (rotated-frame bf16, eight lanes
    // per row via the cooperative group decoder), then run the shared FA2
    // prompt kernel over the scratch.
    const auto tokens       = static_cast<std::int32_t>(q.ne[2]);
    const auto span         = static_cast<std::int32_t>(scratch_k.ne[2]);
    const auto units_bound  = static_cast<std::int64_t>(span) * Geometry::KVHeads * 2 * 8;
    const int scratch_grid  = static_cast<int>(
        div_up(units_bound, static_cast<std::int64_t>(kGqaHqScratchThreads)));
    gqa_attention_prefill_hq_scratch_kernel<Geometry, Metadata>
        <<<scratch_grid, kGqaHqScratchThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(cache_k.data),
            static_cast<const std::uint8_t*>(cache_v.data),
            static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), tokens, span,
            static_cast<__nv_bfloat16*>(scratch_k.data),
            static_cast<__nv_bfloat16*>(scratch_v.data));
    CUDA_CHECK(cudaGetLastError());
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    gqa_attention_prefill_bf16_kernel<Geometry, Metadata, true>
        <<<attention_grid, kGqaPrefillThreads, kGqaPrefillRotatedSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(scratch_k.data),
            static_cast<const __nv_bfloat16*>(scratch_v.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, span);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_hq(const Tensor& k, const Tensor& v, const Tensor& positions,
                           CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    Tensor& cache_k_meta    = cache.k_scale_pages;
    Tensor& cache_v_meta    = cache.v_scale_pages;
    constexpr int kFillBlock = kGqaHqFillWarps * 32;
    const std::int64_t fill_units =
        static_cast<std::int64_t>(tokens) * Geometry::KVHeads * 2;
    const int fill_grid =
        static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kGqaHqFillWarps)));
    gqa_attention_prefill_fill_hq_kernel<Geometry, Metadata>
        <<<fill_grid, kFillBlock, kGqaHqFillSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
            static_cast<std::uint8_t*>(cache_k_meta.data),
            static_cast<std::uint8_t*>(cache_v_meta.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
