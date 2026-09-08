#pragma once

// ninfer::ops::detail - shared launcher templates for the w8 small-T MMA route (the build-speed
// TU split, re-derived on the restructured tree). The token-width tables and the vocabulary
// tiles expand in their own TUs; the dispatcher references them through extern-template
// declarations and never re-expands the contraction.

#include "core/pdl.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kBlocks), dim3(Schedule::kThreads), 0, stream},
        w8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule, W8ContiguousOutput,
                              W8SmallTMmaStoreEpilogue, W8SmallTMmaIdentityRows, false, false>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output,
            W8SmallTMmaStoreEpilogue{}, W8SmallTMmaIdentityRows{}, ActiveTokens));
    CUDA_CHECK(cudaGetLastError());
}

template <int Capacity>
void launch_vocabulary_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = W8VocabularyProjectionGeometry;
    using Schedule = W8SmallTMmaSchedule<Capacity <= 32 ? 8 : 4, Capacity, 2,
                                         W8SmallTMmaScaleAccess::Shared>;
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(Geometry::kOutputRows / 16), dim3(Schedule::kThreads), 0, stream},
        w8_small_t_mma_kernel<Geometry, Capacity, Schedule, W8ContiguousOutput,
                              W8SmallTMmaStoreEpilogue, W8SmallTMmaIdentityRows, false, true>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, W8SmallTMmaStoreEpilogue{},
            W8SmallTMmaIdentityRows{}, x.ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

extern template void launch_exact<W8MtpInputProjectionGeometry, 1>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 3>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 4>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 5>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 6>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 7>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 9>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 10>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 11>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 12>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 13>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 14>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 15>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 17>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 18>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 19>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 20>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 21>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 22>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 23>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 25>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 26>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 27>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 28>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 29>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 30>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 31>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 33>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 34>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 35>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 36>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 37>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 38>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 39>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 41>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 42>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 43>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 44>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 45>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 46>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 47>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpInputProjectionGeometry, 48>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 1>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 3>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 4>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 5>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 6>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 7>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 9>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 10>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 11>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 12>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 13>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 14>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 15>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 17>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 18>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 19>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 20>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 21>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 22>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 23>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 25>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 26>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 27>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 28>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 29>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 30>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 31>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 33>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 34>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 35>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 36>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 37>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 38>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 39>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 41>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 42>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 43>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 44>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 45>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 46>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 47>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionProjectionGeometry, 48>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 1>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 3>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 4>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 5>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 6>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 7>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 9>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 10>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 11>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 12>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 13>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 14>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 15>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 17>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 18>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 19>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 20>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 21>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 22>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 23>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 25>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 26>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 27>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 28>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 29>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 30>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 31>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 33>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 34>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 35>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 36>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 37>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 38>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 39>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 41>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 42>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 43>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 44>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 45>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 46>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 47>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpAttentionOutputGeometry, 48>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 1>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 3>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 4>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 5>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 6>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 7>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 9>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 10>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 11>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 12>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 13>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 14>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 15>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 17>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 18>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 19>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 20>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 21>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 22>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 23>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 25>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 26>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 27>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 28>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 29>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 30>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 31>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 33>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 34>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 35>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 36>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 37>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 38>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 39>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 41>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 42>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 43>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 44>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 45>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 46>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 47>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 48>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 49>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 50>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 51>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpGateUpProjectionGeometry, 52>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 1>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 3>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 4>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 5>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 6>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 7>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 9>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 10>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 11>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 12>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 13>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 14>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 15>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 17>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 18>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 19>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 20>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 21>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 22>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 23>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 25>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 26>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 27>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 28>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 29>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 30>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 31>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 33>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 34>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 35>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 36>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 37>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 38>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 39>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 41>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 42>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 43>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 44>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 45>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 46>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 47>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8MtpDownProjectionGeometry, 48>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 1>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 3>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 4>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 5>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 6>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 7>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 9>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 10>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 11>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 12>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 13>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 14>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 15>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 17>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 18>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 19>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 20>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 21>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 22>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 23>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 25>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 26>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 27>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 28>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 29>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 30>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 31>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 33>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 34>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 35>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 36>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 37>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 38>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 39>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 41>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 42>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 43>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 44>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 45>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 46>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 47>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W835bMtpProjectionGeometry, 48>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 1>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 3>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 4>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 5>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 6>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 7>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 9>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 10>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 11>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 12>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 13>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 14>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 15>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 17>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 18>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 19>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 20>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 21>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 22>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 23>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 25>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 26>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 27>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 28>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 29>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 30>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 31>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 33>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 34>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 35>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 36>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 37>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 38>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 39>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 41>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 42>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 43>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 44>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 45>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 46>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 47>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 48>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 49>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 50>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 51>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 52>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<W8DFlash2AttentionProjectionGeometry, 53>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_vocabulary_tile<8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_vocabulary_tile<16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_vocabulary_tile<24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_vocabulary_tile<32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_vocabulary_tile<40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail