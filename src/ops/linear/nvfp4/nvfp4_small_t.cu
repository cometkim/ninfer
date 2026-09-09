#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"
#include "ops/linear/nvfp4/nvfp4_small_t_launch.h"

#include <array>
#include <cstddef>
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

void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launchers<Nvfp4AttnInputGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launchers<Nvfp4GdnInputGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launchers<Nvfp4MlpGateUpGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launchers<Nvfp4Residual6144Geometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launchers<Nvfp4Residual17408Geometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DFlash2Feature:
    case Nvfp4Problem::DFlash2Qkv:
    case Nvfp4Problem::DFlash2AttnOut:
    case Nvfp4Problem::DFlash2ConvProj:
    case Nvfp4Problem::DFlash2Selector:
        launch_nvfp4_small_t_dflash2(x, weight, out, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
