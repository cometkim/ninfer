# Parallel attention compilation for the cumulative fork

This branch owns the compilation split adapted to the cumulative fork's attention routes.
HyperQuant (`hq-e8-2b`) prompt kernels are instantiated in separate H24 and H16 geometry
units (`prompt_hq_h24.cu`, `prompt_hq_h16.cu`), including the carry and residual routes,
the same split the HQ Small-T kernels use; `prompt_hq.cu` keeps only the route selection.

Upstream `dev` compiles each KV type's causal attention in its own translation units
(`src/ops/softmax_attention/dense/causal_cache/<type>/`), which supersedes this branch's
earlier BF16/INT8 prompt split and its non-RDC NVFP4 H24/H16 prompt split. The INT8
split-reduction launch and workspace contract of `feat/kernel-perf` is unchanged.

This guide is self-contained: the standalone `feat/build-speed` branch is not a prerequisite.
Follow the [build contract](../maintainer/build-system.md) and build with `cmake --build build -j`.
Route source ownership remains in `src/ops/softmax_attention/sources.cmake`.
Keep geometry instantiations out of the dispatchers and retain each route's RDC policy.
These compilation boundaries preserve launch mathematics; no runtime speedup is asserted.
