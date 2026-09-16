# Windows platform support

This branch adds MSVC/Windows support for the existing single-GPU CUDA engine.
It covers native artifact IO, Python positional IO, portable wide-integer arithmetic,
console/logging behavior, platform tests and MSVC-compatible CUDA host boundaries.
Maintained Jinja resources retain LF bytes across Windows checkouts.

Use CMake 3.28+, a CUDA-compatible MSVC C++20 toolchain and an `sm_120a` CUDA toolkit.
Make FFmpeg and the product dependencies discoverable as described in the
[build contract](../maintainer/build-system.md); dependencies are not downloaded by CMake.
From an environment with MSVC and CUDA available, configure Ninja with
`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`, then `cmake --build build -j`.
Keep machine-specific compiler and dependency paths in `CMakeUserPresets.json`.
Enable `BUILD_TESTING=ON` and select an existing managed Python environment for tests.
The port does not change the supported GPU architecture or introduce another artifact format.
