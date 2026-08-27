#include "ops/linear/nvfp4/nvfp4_format.h"

#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ninfer/ops/dflash2_selector_predecessors.h"
#include "ninfer/ops/dflash2_selector_scores.h"
#include "ninfer/ops/dflash2_selector_walk.h"
#include "ninfer/ops/dflash2_topk.h"

#include "ops/launcher/dflash2_draft.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require_contiguous_bf16(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::BF16 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous non-null BF16");
    }
}

void require_contiguous_i32(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::I32 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous non-null I32");
    }
}

void require_contiguous_f32(const Tensor& t, const char* op, const char* name) {
    if (t.dtype != DType::FP32 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous non-null FP32");
    }
}

} // namespace

void dflash2_dynamic_conv(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                          std::int32_t side, std::int32_t block_size, Tensor& out,
                          cudaStream_t stream) {
    constexpr const char* op = "dflash2_dynamic_conv";
    require_contiguous_bf16(hidden, op, "hidden");
    require_contiguous_bf16(dynamic, op, "dynamic");
    require_contiguous_bf16(base, op, "base");
    require_contiguous_bf16(out, op, "out");
    const std::int64_t hidden_size = hidden.ne[0];
    const std::int64_t columns     = hidden.ne[1];
    if (hidden.ne[2] != 1 || hidden.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": hidden must be [hidden, columns]");
    }
    if (hidden_size % kDflash2ConvGroupSize != 0) {
        throw std::invalid_argument(std::string(op) + ": hidden size must be divisible by 16");
    }
    if (dynamic.ne[0] != 4 * (hidden_size / kDflash2ConvGroupSize) || dynamic.ne[1] != columns ||
        dynamic.ne[2] != 1 || dynamic.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": dynamic must be [4*hidden/16, columns]");
    }
    if (base.ne[0] != 2 || base.ne[1] != 2 || base.ne[2] != hidden_size || base.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": base must be [2, 2, hidden]");
    }
    if (out.ne[0] != hidden_size || out.ne[1] != columns || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": out must be [hidden, columns]");
    }
    if (side < 0 || side > 1) {
        throw std::invalid_argument(std::string(op) + ": side must be 0 or 1");
    }
    if (block_size < 1 || block_size > 64 || columns % block_size != 0 ||
        columns / block_size > 8) {
        throw std::invalid_argument(std::string(op) +
                                    ": block_size must be in [1,64] dividing columns into "
                                    "[1,8] lanes");
    }
    if (columns == 0) { return; }
    detail::dflash2_dynamic_conv_launch(hidden, dynamic, base, side, block_size, out, stream);
}

void dflash2_selector_scores(const Tensor& candidates, const Tensor& predecessor_ids,
                             const Tensor& unary, const Tensor& hidden_proj,
                             const Weight& successor_rows, const Weight& predecessor_rows,
                             Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "dflash2_selector_scores";
    require_contiguous_i32(candidates, op, "candidates");
    require_contiguous_i32(predecessor_ids, op, "predecessor_ids");
    require_contiguous_f32(unary, op, "unary");
    require_contiguous_f32(hidden_proj, op, "hidden_proj");
    require_contiguous_f32(out, op, "out");
    const std::int64_t top_k     = candidates.ne[0];
    const std::int64_t positions = candidates.ne[1];
    const std::int64_t lanes     = candidates.ne[2];
    const std::int64_t rank      = hidden_proj.ne[0];
    const std::int64_t vocab     = successor_rows.n;
    if (candidates.ne[3] != 1 || predecessor_ids.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": candidate tensors must be [K, P, L]");
    }
    if (predecessor_ids.ne[0] != top_k || predecessor_ids.ne[1] != positions ||
        predecessor_ids.ne[2] != lanes) {
        throw std::invalid_argument(std::string(op) +
                                    ": predecessor_ids must match candidates shape");
    }
    if (unary.ne[0] != top_k || unary.ne[1] != positions || unary.ne[2] != lanes ||
        unary.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": unary must be [K, P, L]");
    }
    if (hidden_proj.ne[1] != positions || hidden_proj.ne[2] != lanes || hidden_proj.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": hidden_proj must be [R, P, L]");
    }
    if (out.ne[0] != top_k || out.ne[1] != top_k || out.ne[2] != positions ||
        out.ne[3] != lanes) {
        throw std::invalid_argument(std::string(op) + ": out must be [K, K, P, L]");
    }
    if (top_k < 1 || top_k > 32 || rank < 16 || rank > 512 || positions < 1 || positions > 15 ||
        lanes < 1 || lanes > 8 || vocab < 1 || rank % 64 != 0 || vocab % 128 != 0) {
        throw std::invalid_argument(std::string(op) + ": supported domain violated");
    }
    detail::validate_nvfp4_weight(successor_rows, op);
    detail::validate_nvfp4_weight(predecessor_rows, op);
    if (successor_rows.k != rank || predecessor_rows.n != vocab || predecessor_rows.k != rank) {
        throw std::invalid_argument(std::string(op) +
                                    ": codebooks must both be [vocab, rank] NVFP4 weights");
    }
    if (top_k * positions * lanes == 0) { return; }
    detail::dflash2_selector_scores_launch(candidates, predecessor_ids, unary, hidden_proj,
                                           successor_rows, predecessor_rows, out, stream);
}

void dflash2_topk(const Tensor& logits, std::int32_t k, Tensor& ids, Tensor& values,
                  cudaStream_t stream) {
    constexpr const char* op = "dflash2_topk";
    require_contiguous_bf16(logits, op, "logits");
    require_contiguous_i32(ids, op, "ids");
    require_contiguous_bf16(values, op, "values");
    const std::int64_t rows    = logits.ne[0];
    const std::int64_t columns = logits.ne[1];
    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": logits must be [rows, columns]");
    }
    if (k < 1 || k > 64 || rows < k || rows > (std::int64_t{1} << 25) || columns < 1 ||
        columns > 64) {
        throw std::invalid_argument(std::string(op) + ": supported domain violated");
    }
    if (ids.ne[0] != k || ids.ne[1] != columns || ids.ne[2] != 1 || ids.ne[3] != 1 ||
        values.ne[0] != k || values.ne[1] != columns || values.ne[2] != 1 || values.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": ids/values must be [k, columns]");
    }
    detail::dflash2_topk_launch(logits, k, ids, values, stream);
}

void dflash2_selector_walk(const Tensor& scores, const Tensor& candidates, Tensor& out,
                           cudaStream_t stream) {
    constexpr const char* op = "dflash2_selector_walk";
    require_contiguous_f32(scores, op, "scores");
    require_contiguous_i32(candidates, op, "candidates");
    require_contiguous_i32(out, op, "out");
    const std::int64_t top_k     = scores.ne[0];
    const std::int64_t positions = scores.ne[2];
    const std::int64_t lanes     = scores.ne[3];
    if (scores.ne[1] != top_k) {
        throw std::invalid_argument(std::string(op) + ": scores must be [K, K, P, L]");
    }
    if (top_k < 1 || top_k > 32 || positions < 1 || positions > 15 || lanes < 1 || lanes > 8) {
        throw std::invalid_argument(std::string(op) + ": supported domain violated");
    }
    if (candidates.ne[0] != top_k || candidates.ne[1] != positions ||
        candidates.ne[2] != lanes || candidates.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": candidates must be [K, P, L]");
    }
    if (out.ne[0] != positions || out.ne[1] != lanes || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": out must be [P, L]");
    }
    detail::dflash2_selector_walk_launch(scores, candidates, out, stream);
}

void dflash2_selector_predecessors(const Tensor& candidates, const Tensor& anchors, Tensor& out,
                                   cudaStream_t stream) {
    constexpr const char* op = "dflash2_selector_predecessors";
    require_contiguous_i32(candidates, op, "candidates");
    require_contiguous_i32(anchors, op, "anchors");
    require_contiguous_i32(out, op, "out");
    const std::int64_t top_k     = candidates.ne[0];
    const std::int64_t positions = candidates.ne[1];
    const std::int64_t lanes     = candidates.ne[2];
    if (candidates.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": candidates must be [K, P, L]");
    }
    if (top_k < 1 || top_k > 32 || positions < 1 || positions > 15 || lanes < 1 || lanes > 8) {
        throw std::invalid_argument(std::string(op) + ": supported domain violated");
    }
    if (anchors.ne[0] != lanes || anchors.ne[1] != 1 || anchors.ne[2] != 1 || anchors.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": anchors must be [L]");
    }
    if (out.ne[0] != top_k || out.ne[1] != positions || out.ne[2] != lanes || out.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": out must be [K, P, L]");
    }
    detail::dflash2_selector_predecessors_launch(candidates, anchors, out, stream);
}

} // namespace ninfer::ops
