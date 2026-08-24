// ninfer::ops - qk_norm_rope launcher: fixed Text D256/R64 geometries.
#include "ops/launcher/qk_norm_rope.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"
#include "ops/kernel/rope.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 128;

} // namespace

void qk_norm_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_weight,
                         const Tensor& k_weight, float eps, const Tensor& positions,
                         const RopeFrequencies& frequencies, Tensor& q_out, Tensor& k_out,
                         cudaStream_t stream) {
    const int tokens     = static_cast<int>(positions.ne[0]);
    const int rows       = static_cast<int>(q.ne[1] + k.ne[1]) * tokens;
    constexpr int kWarps = kBlock / 32;
    const int blocks     = (rows + kWarps - 1) / kWarps;
    const auto launch    = [&]<RopeKernelMode Mode, int QHeads, int KHeads>() {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(static_cast<unsigned>(blocks)), dim3(kBlock), 0, stream},
            rope_norm_fused_kernel<Mode, QHeads, KHeads, kBlock>,
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat162*>(q_weight.data),
            static_cast<const __nv_bfloat162*>(k_weight.data), eps,
            static_cast<const std::int32_t*>(positions.data), frequencies,
            static_cast<__nv_bfloat16*>(q_out.data), static_cast<__nv_bfloat16*>(k_out.data),
            tokens));
    };
    const auto launch_for_geometry = [&]<int QHeads, int KHeads>() {
        if (positions.ne[1] == 3) {
            launch.template operator()<RopeKernelMode::TextMrope, QHeads, KHeads>();
        } else {
            launch.template operator()<RopeKernelMode::Text1D, QHeads, KHeads>();
        }
    };
    if (q.ne[1] == 24 && k.ne[1] == 4) {
        launch_for_geometry.template operator()<24, 4>();
    } else {
        launch_for_geometry.template operator()<16, 2>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
