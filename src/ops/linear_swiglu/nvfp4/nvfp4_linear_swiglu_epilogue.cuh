#pragma once
#include "ops/common/math.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
template <int RowsPerBranch>
struct Nvfp4SwiGluRows {
    static constexpr bool kPaired = true;
    // Fork kernel-perf: one gate/up row pair per warp selects the fork's dedicated pair kernel
    // (nvfp4_a16_gemv_branch_pair_kernel) for the T=1 decode.
    static constexpr bool kBranchPairs = RowsPerBranch == 1;

    __device__ __forceinline__ int weight_row(int begin, int row, int rows) const {
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? rows / 2 : 0);
    }
};

struct Nvfp4SwiGluEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, silu(gate) * up);
    }
};
} // namespace ninfer::ops::detail
