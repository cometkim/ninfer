// DFlash2 drafter matrices for the NVFP4-encoded draft module: weight-only A16 routes
// (gemv single token, exact small-T families, 32-token chunks for wider extents). The draft
// never quantizes activations, so no A4 route is registered for these shapes.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_dflash2_geometry.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

// The fork's measured DFlash2 schedules, expressed on the unified A16 templates with the same
// parameters (Nvfp4GemvSchedule/Nvfp4SimtSchedule had the identical parameter lists).
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
template <int Tokens>
using Exact =
    Nvfp4A16SimtSchedule<(Tokens >= 8 && Tokens <= 16) ? 16 : 4, 1, 2, 16, Tokens, 1,
                         Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                         Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C2        = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4        = Nvfp4A16SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C32       = Nvfp4A16SimtSchedule<4, 1, 2, 16, 16, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::TokenTilesContiguous, 3>;
using FullChunk = Nvfp4A16SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;

inline constexpr std::int32_t kChunkTokens = 32;
inline constexpr int kFirstExact           = 5;
inline constexpr int kLastExact            = 28;

template <class Geometry, std::size_t... Offsets>
constexpr auto exact_launchers(std::index_sequence<Offsets...>) {
    return std::array<Nvfp4Launch, sizeof...(Offsets)>{
        &nvfp4_linear_a16_simt<Geometry, kFirstExact + static_cast<int>(Offsets),
                               Exact<kFirstExact + static_cast<int>(Offsets)>, true>...};
}

template <class Geometry>
Nvfp4Launch select_a16(std::int32_t tokens) {
    static constexpr auto exact =
        exact_launchers<Geometry>(std::make_index_sequence<kLastExact - kFirstExact + 1>{});
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens == kChunkTokens) return nvfp4_linear_a16_simt<Geometry, 32, FullChunk, true>;
    if (tokens >= kFirstExact && tokens <= kLastExact)
        return exact[static_cast<std::size_t>(tokens - kFirstExact)];
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens <= kChunkTokens) return nvfp4_linear_a16_simt<Geometry, 32, C32, false>;
    throw std::logic_error("nvfp4 DFlash2 A16 chunk exceeds shape capacity");
}

// Wider extents run as consecutive 32-token chunks over column slices of x and out.
template <class Geometry>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += kChunkTokens) {
        const std::int32_t count = std::min(kChunkTokens, x.ne[1] - offset);
        const Tensor input       = x.slice(1, offset, count);
        Tensor output            = out.slice(1, offset, count);
        select_a16<Geometry>(count)(input, weight, output, stream);
    }
}

template <class Geometry>
const Nvfp4LinearShape make_dflash2_shape() {
    return {Geometry::kOutputRows, Geometry::kInputRows, launch_a16<Geometry>, nullptr,
            [](std::int32_t, std::int32_t) { return false; }};
}

} // namespace

const Nvfp4LinearShape kNvfp4DFlash2Feature  = make_dflash2_shape<Nvfp4DFlash2FeatureGeometry>();
const Nvfp4LinearShape kNvfp4DFlash2Qkv      = make_dflash2_shape<Nvfp4DFlash2QkvGeometry>();
const Nvfp4LinearShape kNvfp4DFlash2AttnOut  = make_dflash2_shape<Nvfp4DFlash2AttnOutGeometry>();
const Nvfp4LinearShape kNvfp4DFlash2ConvProj = make_dflash2_shape<Nvfp4DFlash2ConvProjGeometry>();
const Nvfp4LinearShape kNvfp4DFlash2Selector = make_dflash2_shape<Nvfp4DFlash2SelectorGeometry>();

} // namespace ninfer::ops::detail
