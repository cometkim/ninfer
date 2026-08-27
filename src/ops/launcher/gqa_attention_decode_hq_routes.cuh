#pragma once

// ninfer::ops::detail - hq-e8-2b small-T partial-kernel launch route, shared by
// the per-geometry instantiation TUs gqa_attention_decode_hq_{27,35}.cu. The
// codec-heavy hq instantiation of the tensor-core kernel dominates this
// route's compile time, so each geometry is explicitly instantiated in its own
// translation unit.

#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_decode_hq.cuh"
#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheInput>
void gqa_small_t_partial_hq(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    // Static shared memory only (~36.6 KB: qkv tile 32 KB + P 4 KB + page ids +
    // RHT signs); no dynamic-smem opt-in attribute is needed.
    // TokenTile=8: the hq route's single runtime-width instantiation covers every
    // decode/verify width 1..8, so a width-8 block verify (draft 7 + bonus) is one
    // pass; GroupSize 6 (27B: 48 rows) and 8 (35B: 64 rows = Br) both fit the row tile.
    CUDA_CHECK(pdl::launch_dependent(
        {grid, dim3(kGqaHqDecodeThreads), 0, stream},
        gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, 8, 4, true, true, CacheInput,
                                                    GqaTcKVHq>,
        static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data),
            GqaTcKVHq{static_cast<std::uint8_t*>(cache.k_pages.data),
                      static_cast<std::uint8_t*>(cache.v_pages.data),
                      static_cast<std::uint8_t*>(cache.k_scale_pages.data),
                      static_cast<std::uint8_t*>(cache.v_scale_pages.data),
                      static_cast<__nv_bfloat16*>(cache.residual_k.data),
                      static_cast<__nv_bfloat16*>(cache.residual_v.data),
                      static_cast<std::uint32_t*>(cache.ring_valid.data)},
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.width, invocation.full_width,
            invocation.column_begin, logical_capacity, scale,
            static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
