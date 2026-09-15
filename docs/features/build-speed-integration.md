# Parallel attention compilation for the cumulative fork

This branch owns the compilation split adapted to the cumulative fork's attention routes.
It provides separate BF16 and INT8 prompt translation units while retaining
the fork's INT8 split-reduction launch and workspace contract.
Non-RDC NVFP4 prompt kernels are separated into H24 and H16 geometry units.
HyperQuant prompt kernels also have H24/H16 units, including carry and residual routes.

This guide is self-contained: the standalone `feat/build-speed` branch is not a prerequisite.
Follow the [build contract](../maintainer/build-system.md) and build with `cmake --build build -j`.
Route source ownership remains in `src/ops/softmax_attention/sources.cmake`.
Keep geometry instantiations out of the dispatchers and retain each route's RDC policy.
These compilation boundaries preserve launch mathematics; no runtime speedup is asserted.
