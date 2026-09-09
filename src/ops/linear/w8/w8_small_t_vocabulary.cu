// ninfer::ops::detail - w8 vocabulary tile launcher instantiations (build-speed TU split).
#include "ops/linear/w8/w8_small_t_launch.h"

namespace ninfer::ops::detail {

template void launch_vocabulary_tile<8>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
template void launch_vocabulary_tile<16>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
template void launch_vocabulary_tile<24>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
template void launch_vocabulary_tile<32>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);
template void launch_vocabulary_tile<40>(
    const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail