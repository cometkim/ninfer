#pragma once

// ninfer::ops::detail - shared launcher template for the NVFP4 A16 small-T route (the build-speed
// TU split, mirroring the w8 small-T family). The DFlash2 drafter geometries expand in their own
// TU; the dispatcher references them through extern-template declarations and never re-expands
// the contraction. The six target-model geometries remain implicitly instantiated in the
// dispatcher TU they have always expanded in.

#include "core/device.h" // CUDA_CHECK
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;

    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
            Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

extern template void launch_exact<Nvfp4DFlash2FeatureGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<Nvfp4DFlash2QkvGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<Nvfp4DFlash2AttnOutGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<Nvfp4DFlash2ConvProjGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
extern template void launch_exact<Nvfp4DFlash2SelectorGeometry, 2>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);

#define NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(TOKEN)                                                     \
    extern template void launch_exact<Nvfp4DFlash2FeatureGeometry, TOKEN>(                           \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    extern template void launch_exact<Nvfp4DFlash2QkvGeometry, TOKEN>(                               \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    extern template void launch_exact<Nvfp4DFlash2AttnOutGeometry, TOKEN>(                           \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    extern template void launch_exact<Nvfp4DFlash2ConvProjGeometry, TOKEN>(                          \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    extern template void launch_exact<Nvfp4DFlash2SelectorGeometry, TOKEN>(                          \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);

NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(3)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(4)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(5)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(6)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(7)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(8)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(9)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(10)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(11)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(12)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(13)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(14)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(15)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(16)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(17)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(18)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(19)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(20)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(21)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(22)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(23)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(24)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(25)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(26)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(27)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(28)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(29)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(30)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(31)
NINFER_NVFP4_DFLASH2_EXTERN_TOKEN(32)

#undef NINFER_NVFP4_DFLASH2_EXTERN_TOKEN

} // namespace ninfer::ops::detail
