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
                              const CacheView& cache, Metadata metadata, const Tensor& new_k,
                              const Tensor& new_v, const Tensor& scratch_k,
                              const Tensor& scratch_v, const Tensor& carry_acc,
                              const Tensor& carry_m, const Tensor& carry_l,
                              std::uint32_t visible_keys, const Tensor& partial_acc,
                              const Tensor& partial_m, const Tensor& partial_l,
                              std::int32_t split_count, Tensor& out, cudaStream_t stream) {
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
    static const cudaError_t attr_carry = cudaFuncSetAttribute(
        gqa_attention_prefill_bf16_kernel<Geometry, Metadata, true, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kGqaPrefillRotatedSmemBytes));
    CUDA_CHECK(attr_carry);
    // Materialize the visible history (rotated-frame bf16, eight lanes per row via the
    // cooperative group decoder), then run the shared FA2 prompt kernel over the scratch.
    // Scratch wider than `span` keys runs in sequential carry bands: each band decodes its
    // keys into the band-local scratch rows and the FA2 kernel resumes/writes back the
    // online-softmax state (m, l, unnormalized acc) between bands.
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const auto span   = static_cast<std::int32_t>(scratch_k.ne[2]);
    const auto visible = static_cast<std::int32_t>(visible_keys);
    const int bands = static_cast<int>(div_up(visible, span));
    if (bands > 1 && (span % kGqaPrefillBc) != 0) {
        // Band bases index the FA2 key-tile grid; a mid-tile base would stage scratch
        // rows below the band. The production band (262144) is tile-aligned.
        throw std::invalid_argument("gqa_attention prompt: scratch band is not tile-aligned");
    }
    // Key-splitting applies only to the single-band launch: banded runs chain (m, l, acc)
    // state sequentially across bands, so their launches stay whole.
    const std::int32_t bands_split = (bands == 1) ? split_count : 1;
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                              static_cast<unsigned>(Geometry::QHeads),
                              static_cast<unsigned>(bands_split));
    // Fresh-chunk exactness (REVIEW §8 D2): with the residual window on, the A1
    // route also rotates the current chunk's bf16 k/v rows straight into each
    // scratch band; the scratch kernel then reads its in-chunk recent window
    // from those exact rows and the ring bound moves to the W keys before the
    // chunk. Cached-input routes pass empty tensors and keep tail coverage.
    const bool has_fresh =
        cache.residual_k.data != nullptr && new_k.data != nullptr && new_v.data != nullptr;
    for (int band = 0; band < bands; ++band) {
        const std::int32_t key_begin = band * span;
        const std::int32_t band_rows = min(span, visible - key_begin);
        if (has_fresh) {
            const auto fresh_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads * 2;
            const int rotate_grid = static_cast<int>(
                div_up(fresh_units, static_cast<std::int64_t>(kGqaHqFillWarps)));
            gqa_attention_prefill_fresh_rotate_kernel<Geometry, Metadata>
                <<<rotate_grid, kGqaHqFillWarps * 32, kHqHeadDim, stream>>>(
                    static_cast<const __nv_bfloat16*>(new_k.data),
                    static_cast<const __nv_bfloat16*>(new_v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata, tokens, span,
                    static_cast<__nv_bfloat16*>(scratch_k.data),
                    static_cast<__nv_bfloat16*>(scratch_v.data), key_begin, band_rows);
            CUDA_CHECK(cudaGetLastError());
        }
        const auto units_bound = static_cast<std::int64_t>(band_rows) * Geometry::KVHeads * 2 * 8;
        const int scratch_grid = static_cast<int>(
            div_up(units_bound, static_cast<std::int64_t>(kGqaHqScratchThreads)));
        gqa_attention_prefill_hq_scratch_kernel<Geometry, Metadata>
            <<<scratch_grid, kGqaHqScratchThreads, 0, stream>>>(
                static_cast<const std::uint8_t*>(cache_k.data),
                static_cast<const std::uint8_t*>(cache_v.data),
                static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
                static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
                static_cast<const std::int32_t*>(positions.data), tokens, span,
                static_cast<__nv_bfloat16*>(scratch_k.data),
                static_cast<__nv_bfloat16*>(scratch_v.data), key_begin, band_rows,
                static_cast<const __nv_bfloat16*>(cache.residual_k.data),
                static_cast<const __nv_bfloat16*>(cache.residual_v.data),
                static_cast<const std::uint32_t*>(cache.ring_valid.data), has_fresh);
        CUDA_CHECK(cudaGetLastError());
        if (bands == 1) {
            gqa_attention_prefill_bf16_kernel<Geometry, Metadata, true>
                <<<attention_grid, kGqaPrefillThreads, kGqaPrefillRotatedSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(scratch_k.data),
                    static_cast<const __nv_bfloat16*>(scratch_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, span, 0, 0x7fffffff, nullptr,
                    nullptr, nullptr, 0, static_cast<float*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    bands_split);
            CUDA_CHECK(cudaGetLastError());
            if (bands_split > 1) {
                const dim3 reduce_grid(static_cast<unsigned>(tokens),
                                       static_cast<unsigned>(Geometry::QHeads), 1u);
                gqa_attention_prefill_reduce_kernel<Geometry>
                    <<<reduce_grid, kGqaPrefillHeadDim, 0, stream>>>(
                        static_cast<const float*>(partial_acc.data),
                        static_cast<const float*>(partial_m.data),
                        static_cast<const float*>(partial_l.data), scale, tokens, bands_split,
                        static_cast<__nv_bfloat16*>(out.data));
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            gqa_attention_prefill_bf16_kernel<Geometry, Metadata, true, true>
                <<<attention_grid, kGqaPrefillThreads, kGqaPrefillRotatedSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(scratch_k.data),
                    static_cast<const __nv_bfloat16*>(scratch_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, span, key_begin,
                    key_begin + band_rows,
                    static_cast<__nv_bfloat16*>(carry_acc.data),
                    static_cast<float*>(carry_m.data), static_cast<float*>(carry_l.data),
                    band + 1 == bands ? 0 : 1);
        }
        CUDA_CHECK(cudaGetLastError());
    }
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
            static_cast<std::uint8_t*>(cache_v_meta.data), tokens,
            static_cast<__nv_bfloat16*>(cache.residual_k.data),
            static_cast<__nv_bfloat16*>(cache.residual_v.data),
            static_cast<std::uint32_t*>(cache.ring_valid.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
