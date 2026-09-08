#include "ops/linear/nvfp4/nvfp4_small_t_launch.h"

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers = make_launchers<Geometry>(
        std::make_index_sequence<kNvfp4LastSmallT - kNvfp4FirstSmallT + 1>{});
    return kLaunchers;
}

} // namespace

#define NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(TOKEN)                                                \
    template void launch_exact<Nvfp4DFlash2FeatureGeometry, TOKEN>(                                  \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    template void launch_exact<Nvfp4DFlash2QkvGeometry, TOKEN>(                                      \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    template void launch_exact<Nvfp4DFlash2AttnOutGeometry, TOKEN>(                                  \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    template void launch_exact<Nvfp4DFlash2ConvProjGeometry, TOKEN>(                                 \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);                                        \
    template void launch_exact<Nvfp4DFlash2SelectorGeometry, TOKEN>(                                 \
        const Tensor&, const Weight&, Tensor&, cudaStream_t);

NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(2)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(3)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(4)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(5)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(6)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(7)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(8)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(9)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(10)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(11)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(12)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(13)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(14)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(15)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(16)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(17)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(18)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(19)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(20)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(21)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(22)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(23)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(24)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(25)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(26)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(27)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(28)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(29)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(30)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(31)
NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN(32)

#undef NINFER_NVFP4_DFLASH2_INSTANTIATE_TOKEN

void launch_nvfp4_small_t_dflash2(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::DFlash2Feature:
        launchers<Nvfp4DFlash2FeatureGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DFlash2Qkv:
        launchers<Nvfp4DFlash2QkvGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DFlash2AttnOut:
        launchers<Nvfp4DFlash2AttnOutGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DFlash2ConvProj:
        launchers<Nvfp4DFlash2ConvProjGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DFlash2Selector:
        launchers<Nvfp4DFlash2SelectorGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        break;
    }
    throw std::invalid_argument("nvfp4 small-T DFlash2: not a DFlash2 problem");
}

} // namespace ninfer::ops::detail
