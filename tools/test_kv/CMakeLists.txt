cmake_minimum_required(VERSION 3.24)

# Must precede project() so the cache entry wins over CMake's default
# architecture detection (a stale default like compute_75 produces PTX-only
# fatbins that this machine's driver refuses to JIT). Mirrors the engine's
# 120a-only policy.
set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")

project(ninfer-hq-kv-tests LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR "this harness supports only CMAKE_CUDA_ARCHITECTURES=120a; got '${CMAKE_CUDA_ARCHITECTURES}'")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Standalone qualification harness for the hq-e8-2b KV codec. It compiles the
# codec header directly (no engine link) so codec iterations do not pay the
# full engine build cycle.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)

add_executable(test_hq_codec test_hq_codec.cu)
add_executable(verify_hq_retrieval verify_hq_retrieval.cu)
add_executable(bench_hq_kernel bench_hq_kernel.cu)
add_executable(test_hq_prefill test_hq_prefill.cu)
add_executable(test_hq_decode test_hq_decode.cu)

