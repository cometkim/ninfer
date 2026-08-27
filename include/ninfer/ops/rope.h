#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/** Maximum supported rotary pairs; RopeFrequencies storage is sized by this bound. */
inline constexpr int kRopeMaxPairs = 128;

/**
 * Pair-frequency table driving every rope mode. `inv_frequency[i]` is the cycle frequency of
 * rotary pair i; only [0, rotary_dim/2) entries are read. `attention_factor` is a q-side
 * temperature: the rotated dimensions of every q row scale by its square while k rows rotate
 * unscaled, so cached K is factor-free and the scores of the rotated dimensions scale by
 * attention_factor^2; 1.0F leaves the rotation untouched. Unrotated dimensions are never
 * affected. The two host builders cover the linear checkpoint tables; YaRN-shaped tables are
 * constructed by the owning target.
 */
struct RopeFrequencies {
    double inv_frequency[kRopeMaxPairs] = {};
    float attention_factor              = 1.0F;
};

/** Linear table theta^(-2i/rotary_dim), rotary_dim <= 2 * kRopeMaxPairs. */
RopeFrequencies rope_linear_frequencies(float theta, int rotary_dim);

/** Vision 2-D table: 36 pairs wrapping an 18-entry theta^(-2j/36) ladder. */
RopeFrequencies rope_vision_frequencies(float theta);

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * cos and sin carry `frequencies.attention_factor` squared on the q path and unmodified on the
 * k path (the factor-free-K contract above). Dimensions [rotary_dim,head_dim) are unchanged.
 * Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*frequencies.inv_frequency[i].
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with the
 *   same table as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with the wrapped vision table.
 *
 * Numerical profiles of the angle phi: the DFlash domain reduces positions[t] * frequency in
 * FP64 before sincosf. Text and Vision at attention_factor 1 compute the product in FP32 - the
 * exact legacy route, bit-stable across engine rebuilds. Text at any other factor (position
 * scaling past the checkpoint's trained range) computes and 2*pi-reduces the angle in FP64
 * before sincosf; the legacy FP32 product loses ~0.03 rad on the lowest pairs at 1M positions.
 *
 * positions is contiguous; frequencies.attention_factor is positive and finite. Q/K tensors are
 * BF16 [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an
 * optional padded token stride. The registered optimized domains are D256/R64 Text Q/K head
 * geometries 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and
 * k must not overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of
 * the supplied Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64
 * from the represented inputs. The updated BF16 values are promoted and compared directly with
 * that result; output storage rounding belongs to the Op's numerical criterion, not the oracle.
 * Unrotated dimensions remain bit-exact. Private kernel arithmetic is implementation-defined.
 * The Op uses no workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, const RopeFrequencies& frequencies, Tensor& q,
          Tensor& k, cudaStream_t stream);

/** Operand side of the single-tensor rope form; selects whether the factor scale applies. */
enum class RopeSide : std::uint8_t {
    /** Rotated dims scale by attention_factor squared (queries). */
    Query,
    /** Rotation stays unscaled — the factor-free cached-K contract (keys). */
    Key,
};

// Single-tensor form with the same formula and storage contract; `side` selects q semantics
// (attention_factor squared on the rotated dims) or k semantics (unscaled, cacheable keys).
// The head count comes directly from x.
void rope(const Tensor& positions, int rotary_dim, const RopeFrequencies& frequencies, Tensor& x,
          RopeSide side, cudaStream_t stream);

} // namespace ninfer::ops
