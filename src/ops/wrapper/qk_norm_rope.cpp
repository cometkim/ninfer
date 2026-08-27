// ninfer::ops - qk_norm_rope wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/qk_norm_rope.h"

#include "ops/launcher/qk_norm_rope.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;

void require_side(const Tensor& x, const Tensor& weight, std::int32_t heads, const char* op,
                  const char* name) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": " + name + "/weight must be BF16");
    }
    if (x.ne[0] != kHeadDim || x.ne[1] != heads || x.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be [256," +
                                    std::to_string(heads) + ",T]");
    }
    if (weight.ne[0] != kHeadDim || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": " + name + " weight must be [256]");
    }
    if (!x.is_contiguous() || !weight.is_contiguous() || x.data == nullptr ||
        weight.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + "/weight must be contiguous");
    }
}

} // namespace

void qk_norm_rope(const Tensor& q, const Tensor& k, const Tensor& q_weight,
                  const Tensor& k_weight, float eps, const Tensor& positions,
                  const RopeFrequencies& frequencies, Tensor& q_out, Tensor& k_out,
                  cudaStream_t stream) {
    constexpr const char* op = "qk_norm_rope";
    const std::int32_t tokens = q.ne[2];
    if (!(q.ne[1] == 24 && k.ne[1] == 4) && !(q.ne[1] == 16 && k.ne[1] == 2)) {
        throw std::invalid_argument(std::string(op) +
                                    ": head geometry must be (24,4) or (16,2)");
    }
    const std::int32_t q_heads = q.ne[1];
    const std::int32_t k_heads = k.ne[1];
    require_side(q, q_weight, q_heads, op, "q");
    require_side(k, k_weight, k_heads, op, "k");
    require_side(q_out, q_weight, q_heads, op, "q_out");
    require_side(k_out, k_weight, k_heads, op, "k_out");
    if (k.ne[2] != tokens || q_out.ne[2] != tokens || k_out.ne[2] != tokens || tokens < 1) {
        throw std::invalid_argument(std::string(op) + ": side token counts must match and be >=1");
    }
    if (positions.dtype != DType::I32 || positions.ne[0] != tokens ||
        (positions.ne[1] != 1 && positions.ne[1] != 3) || positions.ne[2] != 1 ||
        positions.ne[3] != 1 || !positions.is_contiguous() || positions.data == nullptr) {
        throw std::invalid_argument(std::string(op) +
                                    ": positions must be contiguous I32 [T] or [T,3]");
    }
    if (!std::isfinite(eps) || eps < 0.0f) {
        throw std::invalid_argument(std::string(op) + ": eps must be finite and non-negative");
    }
    if (!(frequencies.attention_factor > 0.0f) || !std::isfinite(frequencies.attention_factor)) {
        throw std::invalid_argument(std::string(op) + ": attention factor must be positive");
    }

    detail::qk_norm_rope_launch(q, k, q_weight, k_weight, eps, positions, frequencies, q_out,
                                k_out, stream);
}

} // namespace ninfer::ops
