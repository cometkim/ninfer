#pragma once

// Kernel-parameter transport for TMA descriptor blocks: structs of CUtensorMap, alignas(128).
//
// Upstream passes a block as a `__grid_constant__` by-value kernel parameter, and that remains the
// path everywhere except Windows. MSVC cannot lay out a by-value parameter of an over-aligned type
// in the cudafe1 host stub (C2719), so on _WIN32 the kernel takes a pointer to a device-resident
// copy instead:
//
// - The copy is staged on the consuming stream by a one-thread-per-word kernel whose by-value
//   parameter carries the block as 16-byte uint4 words. Kernel parameters are snapshotted into CUDA
//   graph nodes, so every graph replay re-stores exactly the captured descriptors. A
//   cudaMemcpyAsync from the launcher's stack would bake a dead host pointer into the graph, and
//   pinned host staging is illegal under capture.
// - The device block is allocated and freed with cudaMallocAsync/cudaFreeAsync on that same stream.
//   A NULL-stream free is unordered against a non-blocking compute stream, and the pool could
//   recycle the block while the TMA unit is still reading it (a half-overwritten map deadlocks the
//   kernel's mbarrier transaction wait).
// - The block is written through the generic proxy and read by the tensormap proxy. The stream pool
//   hands back recently used addresses, so the TMA-issuing thread acquires every map with
//   `fence.proxy.tensormap::generic.acquire` before its first use.
//
// The parameter spelling depends only on _WIN32, never on __CUDA_ARCH__, so the host stub and the
// device code always agree on the kernel signature.

#include "core/device.h" // CUDA_CHECK

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <type_traits>

#ifdef _WIN32
#    define NINFER_TMA_DESCRIPTORS_PARAM(Type) const Type* __restrict__
#else
#    define NINFER_TMA_DESCRIPTORS_PARAM(Type) const __grid_constant__ Type
#endif

namespace ninfer::ops::detail {

// Device view of the block for the TMA-issuing thread; call it once before the first TMA load.
template <class Descriptors>
__device__ __forceinline__ const Descriptors& tma_descriptor_block(const Descriptors& block) {
    return block;
}

template <class Descriptors>
__device__ __forceinline__ const Descriptors& tma_descriptor_block(const Descriptors* block) {
    static_assert(sizeof(Descriptors) % sizeof(CUtensorMap) == 0);
    const auto* maps = reinterpret_cast<const unsigned char*>(block);
#pragma unroll
    for (int offset = 0; offset < static_cast<int>(sizeof(Descriptors));
         offset += static_cast<int>(sizeof(CUtensorMap))) {
        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                     :
                     : "l"(maps + offset)
                     : "memory");
    }
    return *block;
}

#ifdef _WIN32

template <int Words>
struct TmaDescriptorWords {
    uint4 words[Words];
};

template <int Words>
__global__ void
tma_store_descriptor_words_kernel(uint4* __restrict__ destination,
                                  const __grid_constant__ TmaDescriptorWords<Words> payload) {
    if (blockIdx.x == 0 && threadIdx.x < Words) {
        destination[threadIdx.x] = payload.words[threadIdx.x];
    }
}

#endif

// Host-side owner of one launch's descriptor block. `get()` is the argument for a
// NINFER_TMA_DESCRIPTORS_PARAM kernel parameter; it stays valid until this object is destroyed,
// which must happen after the last launch that uses it has been enqueued on `stream`.
template <class Descriptors>
class TmaDescriptorArgument {
public:
    static_assert(std::is_trivially_copyable_v<Descriptors>);
    static_assert(sizeof(Descriptors) % sizeof(uint4) == 0);

#ifdef _WIN32
    TmaDescriptorArgument(const Descriptors& descriptors, cudaStream_t stream) : stream_(stream) {
        constexpr int kWords = static_cast<int>(sizeof(Descriptors) / sizeof(uint4));
        static_assert(kWords <= 32, "one warp stores the descriptor words");
        CUDA_CHECK(
            cudaMallocAsync(reinterpret_cast<void**>(&device_), sizeof(Descriptors), stream));
        TmaDescriptorWords<kWords> payload{};
        std::memcpy(payload.words, &descriptors, sizeof(Descriptors));
        tma_store_descriptor_words_kernel<kWords>
            <<<1, 32, 0, stream>>>(reinterpret_cast<uint4*>(device_), payload);
        CUDA_CHECK(cudaGetLastError());
    }

    ~TmaDescriptorArgument() {
        // Stream-ordered after every launch enqueued while this object was alive. Never throws:
        // a discarded capture unwinds through here on an invalidated capturing stream.
        if (device_ != nullptr) { (void)cudaFreeAsync(device_, stream_); }
    }

    [[nodiscard]] const Descriptors* get() const noexcept { return device_; }

private:
    Descriptors* device_ = nullptr;
    cudaStream_t stream_ = nullptr;
#else
    TmaDescriptorArgument(const Descriptors& descriptors, cudaStream_t) : value_(descriptors) {}

    // Copied into the kernel parameter (and into a graph node during capture) at each launch.
    [[nodiscard]] const Descriptors& get() const noexcept { return value_; }

private:
    Descriptors value_;
#endif

public:
    TmaDescriptorArgument(const TmaDescriptorArgument&)            = delete;
    TmaDescriptorArgument& operator=(const TmaDescriptorArgument&) = delete;
};

} // namespace ninfer::ops::detail
