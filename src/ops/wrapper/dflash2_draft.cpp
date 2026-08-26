#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ninfer/ops/dflash2_selector_scores.h"

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
                             const Tensor& successor_rows, const Tensor& predecessor_rows,
                             Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "dflash2_selector_scores";
    require_contiguous_i32(candidates, op, "candidates");
    require_contiguous_i32(predecessor_ids, op, "predecessor_ids");
    require_contiguous_f32(unary, op, "unary");
    require_contiguous_f32(hidden_proj, op, "hidden_proj");
    require_contiguous_bf16(successor_rows, op, "successor_rows");
    require_contiguous_bf16(predecessor_rows, op, "predecessor_rows");
    require_contiguous_f32(out, op, "out");
    const std::int64_t top_k    = candidates.ne[0];
    const std::int64_t positions = candidates.ne[1];
    const std::int64_t lanes     = candidates.ne[2];
    const std::int64_t rank      = hidden_proj.ne[0];
    const std::int64_t vocab     = successor_rows.ne[0];
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
    if (successor_rows.ne[1] != rank || successor_rows.ne[2] != 1 || successor_rows.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": successor_rows must be [vocab, R]");
    }
    if (predecessor_rows.ne[0] != vocab || predecessor_rows.ne[1] != rank ||
        predecessor_rows.ne[2] != 1 || predecessor_rows.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": predecessor_rows must be [vocab, R]");
    }
    if (out.ne[0] != top_k || out.ne[1] != top_k || out.ne[2] != positions ||
        out.ne[3] != lanes) {
        throw std::invalid_argument(std::string(op) + ": out must be [K, K, P, L]");
    }
    if (top_k < 1 || top_k > 32 || rank < 16 || rank > 512 || positions < 1 || positions > 15 ||
        lanes < 1 || lanes > 8 || vocab < 1) {
        throw std::invalid_argument(std::string(op) + ": supported domain violated");
    }
    if (top_k * positions * lanes == 0) { return; }
    detail::dflash2_selector_scores_launch(candidates, predecessor_ids, unary, hidden_proj,
                                           successor_rows, predecessor_rows, out, stream);
}

} // namespace ninfer::ops
