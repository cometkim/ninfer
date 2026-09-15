#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_dflash2_geometry.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4DFlash2QkvGeometry;
// One 6144-row contraction stored as q [4096,T], k [1024,T] and v [1024,T].
using Output = LinearBf16SegmentedOutput<4096, 1024, 1024>;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, cudaStream_t);

// The split-output epilogue owns the family's measured low-T warp mapping (the same crossover
// as the Text attention-input small-T route).
template <int ActiveTokens>
struct Nvfp4DFlash2AttentionSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 8 ? 16 : 8);
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    using Type =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
};

Output split_output(Tensor& q, Tensor& k, Tensor& v) {
    return Output{{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                   static_cast<__nv_bfloat16*>(v.data)}};
}

void launch_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                   cudaStream_t stream) {
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight), split_output(q, k, v), LinearIdentityEpilogue{}, stream);
}

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    using Schedule = typename Nvfp4DFlash2AttentionSmallTProductionSchedule<ActiveTokens>::Type;
    launch_nvfp4_a16_simt<
        Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight), split_output(q, k, v), LinearIdentityEpilogue{}, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kNvfp4LastSmallT - kNvfp4FirstSmallT + 1>{});

} // namespace

void nvfp4_dflash2_attn_input(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                              Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        const Tensor x_slice      = x.slice(1, token_begin, active);
        Tensor q_slice            = q.slice(1, token_begin, active);
        Tensor k_slice            = k.slice(1, token_begin, active);
        Tensor v_slice            = v.slice(1, token_begin, active);
        if (active == 1) {
            launch_decode(x_slice, weight, q_slice, k_slice, v_slice, stream);
        } else {
            kLaunchers[active - kNvfp4FirstSmallT](x_slice, weight, q_slice, k_slice, v_slice,
                                                   stream);
        }
    }
}

} // namespace ninfer::ops::detail
