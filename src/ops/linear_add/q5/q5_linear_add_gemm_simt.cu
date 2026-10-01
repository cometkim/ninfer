#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Cols, int Stride>
void launch_split2(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual_out.data);
    const auto stride = std::int64_t(residual_out.nb[1] / sizeof(__nv_bfloat16));
    if constexpr (Cols >= 2) {
        // Fork kernel-perf: T=2..4 (MTP verify) keeps the fork's rowsplit split2 kernel, which
        // measured faster than the unified direct-SIMT schedule in the verify graph on RTX 5090
        // (N=5120 K=17408 T=4: 44.6 vs 48.4 us). It writes token-major rows of a contiguous
        // output; other layouts take the unified route.
        if (stride == residual_out.ne[0]) {
            q5_rowsplit_gemm_simt_split2_kernel<Q5RowSplitSimtSchedule, Cols, Stride / 1024, Stride,
                                                false, 0, true>
                <<<dim3(static_cast<unsigned>(residual_out.ne[0])), 2 * 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const std::uint8_t*>(w.qdata),
                    static_cast<const std::uint8_t*>(w.qhigh),
                    static_cast<const std::uint8_t*>(w.scales), data, residual_out.ne[0], x.ne[0],
                    x.ne[1], w.padded_shape[1], Stride / 1024);
            return;
        }
    }
    using Schedule = Q5A16DirectSimtSchedule<1, Cols, 2, 8, 16, Stride, true>;
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, w),
                                        LinearBf16StridedOutput{data, stride, 0},
                                        LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}

template <int Cols>
void dispatch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (w.k == 6144) {
        launch_split2<Cols, 6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_split2<Cols, 17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
    }
}

template <class Launch>
void dispatch_cols(std::int32_t cols, Launch&& launch) {
    switch (cols) {
#define NINFER_Q5_LINEAR_ADD_EXACT(COLS)                                                           \
    case COLS:                                                                                     \
        launch.template operator()<COLS>();                                                        \
        return
        NINFER_Q5_LINEAR_ADD_EXACT(1);
        NINFER_Q5_LINEAR_ADD_EXACT(2);
        NINFER_Q5_LINEAR_ADD_EXACT(3);
        NINFER_Q5_LINEAR_ADD_EXACT(4);
#undef NINFER_Q5_LINEAR_ADD_EXACT
    default:
        throw std::invalid_argument("q5 linear_add split2: T must be in [1,4]");
    }
}

} // namespace

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    dispatch_cols(x.ne[1], [&]<int Cols>() { dispatch_shape<Cols>(x, w, residual_out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
