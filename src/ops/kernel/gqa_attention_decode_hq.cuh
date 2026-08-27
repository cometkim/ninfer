#pragma once

// ninfer::ops - hq-e8-2b small-T split-KV partial kernel (decode/verify).
// The hq route is a KV-source policy of the tensor-core small-T kernel
// (gqa_attention_decode_bf16.cuh): each 32-key tile is group-decoded from the
// code/meta planes straight into the swizzled shared-memory positions, QK/PV
// run on ldmatrix+mma in the codec's rotated frame, and output rows are
// un-rotated once so the shared reducer combines original-frame partials
// unchanged. Masks, splits, neutral partials, the fused append, and the
// batch/column offsets are the TC kernel's own contracts.
//
// Deliberately ONE runtime-width instantiation per (Geometry, CacheInput):
// TokenTile=8 covers every decode/verify width 1..8 (the phased loops are
// width-agnostic), MultiBatch=true is exact for batch_size 1 (batch 0 offsets
// vanish), and Masked=true reads valid_columns=nullptr as unmasked. The
// codec-heavy instantiation keeps the CUDA front-end's memory footprint
// bounded; per-geometry TUs compile in parallel. Launch sites instantiate
// gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, 8, 4, true, true,
// CacheInput, GqaTcKVHq> with kGqaHqDecodeThreads threads and no dynamic smem.
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaHqDecodeThreads = 128;

} // namespace ninfer::ops
