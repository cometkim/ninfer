#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Per-column top-k selection over a logits matrix: for each column t of
 * `logits` [rows, columns], selects the k largest values with their row ids.
 *
 * Ties break toward the SMALLER row id. `ids` is I32 [K, columns] (largest
 * first), `values` is BF16 [K, columns] holding the corresponding `logits`
 * entries bit-exactly. `logits` is contiguous BF16 [rows, columns]; K is in
 * [1, 64]; rows is in [1, 2^25]; columns in [1, 64]. Inputs and outputs must
 * not overlap. The represented selection is exactly the iterative
 * largest-first removal of the BF16-ordered values (BF16 compare order equals
 * IEEE order; -0 and +0 tie as equal). This is an exact selection Op: ids and
 * values are unique per output column and are a deterministic function of the
 * represented inputs. There is no workspace or persistent state effect.
 */
void dflash2_topk(const Tensor& logits, std::int32_t k, Tensor& ids, Tensor& values,
                  cudaStream_t stream);

} // namespace ninfer::ops
