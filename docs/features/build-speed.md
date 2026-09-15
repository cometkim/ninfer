# Parallel attention compilation

This standalone branch splits heavy causal-attention template instantiations into separate files.
BF16 and INT8 prompt launchers compile in distinct translation units.
Non-RDC NVFP4 prompt launchers are split by H24 and H16 geometry.
The dispatcher retains direct and masked metadata variants and existing launch mathematics.

Configure using the [build contract](../maintainer/build-system.md), then
run `cmake --build build -j` to allow the build tool to compile independent units in parallel.
Source ownership remains in `src/ops/softmax_attention/sources.cmake`.
Keep new dtype/geometry instantiations in their owning route files, not in dispatchers.
This branch does not add Windows or HyperQuant support and claims no inference speedup.
