#pragma once

// ninfer::ops::detail - private launch prototypes for gqa_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class GqaAttentionRoute { SmallT, ChunkedSmallT, Prompt };

struct GqaSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope);

// Per-dtype small-T partial-kernel routes. Each route lives in its own
// translation unit (gqa_attention_decode_<dtype>.cu) so the fat kernel families
// compile in parallel; every route is explicitly instantiated there for both
// geometries and both cache-input modes, so callers see declarations only.
template <typename Geometry, typename CacheInput>
void gqa_small_t_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                              PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                              std::int32_t logical_capacity, std::int32_t splits,
                              Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                              cudaStream_t stream);

template <typename Geometry, typename CacheInput>
void gqa_small_t_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t implementation_window,
                            std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                            Tensor& partial_l, cudaStream_t stream);

template <typename Geometry, typename CacheInput>
void gqa_small_t_partial_hq(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits,
                            Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                            cudaStream_t stream);

// Per-dtype prompt/prefill routes, each compiled in its own translation unit
// (gqa_attention_prefill_<dtype>.cu) and explicitly instantiated there for both
// geometries, both cache views, and every metadata variant. The hq attention
// route owns the one-shot bf16 scratch planes; every other route ignores them.
template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_bf16(const Tensor& q, const Tensor& positions, float scale,
                                const CacheView& cache, Metadata metadata, Tensor& out,
                                const Tensor& partial_acc, const Tensor& partial_m,
                                const Tensor& partial_l, std::int32_t split_count,
                                cudaStream_t stream);

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_i8(const Tensor& q, const Tensor& positions, float scale,
                              const CacheView& cache, Metadata metadata, Tensor& out,
                              const Tensor& partial_acc, const Tensor& partial_m,
                              const Tensor& partial_l, std::int32_t split_count,
                              cudaStream_t stream);

// Key-split policy for the INT8 prompt kernel (ROADMAP WI-K1a): splitting the key range into S
// segments multiplies the (q_block, head) item count without extra KV traffic; per-segment
// online-softmax partials are merged by gqa_attention_prefill_reduce_kernel. S minimizes the
// wave-fill model ceil(items*S/#SMs)/S over S in {1..4} and stays at 1 unless that predicts a
// >=10% attention-time reduction; S=1 keeps the single-pass path bit-identical.
std::int32_t gqa_prefill_split_count(std::int32_t width, std::int32_t q_heads);

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_hq(const Tensor& q, const Tensor& positions, float scale,
                              const CacheView& cache, Metadata metadata, const Tensor& new_k,
                              const Tensor& new_v, const Tensor& scratch_k,
                              const Tensor& scratch_v, const Tensor& carry_acc,
                              const Tensor& carry_m, const Tensor& carry_l,
                              std::uint32_t visible_keys, const Tensor& partial_acc,
                              const Tensor& partial_m, const Tensor& partial_l,
                              std::int32_t split_count, Tensor& out, cudaStream_t stream);

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_bf16(const Tensor& k, const Tensor& v, const Tensor& positions,
                             CacheView cache, Metadata metadata, cudaStream_t stream);

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_i8(const Tensor& k, const Tensor& v, const Tensor& positions,
                           CacheView cache, Metadata metadata, cudaStream_t stream);

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_hq(const Tensor& k, const Tensor& v, const Tensor& positions,
                           CacheView cache, Metadata metadata, cudaStream_t stream);

// Largest verify width one small-T kernel pass covers for a cache dtype: the hq-e8-2b
// instantiation tiles eight tokens (width-8 block verify); the BF16/I8 kernels keep
// their tuned six-token tile. Wider widths chunk at this size.
[[nodiscard]] std::int32_t gqa_small_t_chunk_tokens(DType cache_dtype);

[[nodiscard]] bool gqa_attention_uses_small_t(std::int32_t tokens, DType cache_dtype);

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size, DType cache_dtype,
                                              GqaExecutionEnvelope envelope);

const char* gqa_attention_route_name(GqaAttentionRoute route);

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, const Tensor& gate, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream);

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions,
                                         const Tensor& gate, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream);

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 const Tensor& scratch_k, const Tensor& scratch_v,
                                 const Tensor& carry_acc, const Tensor& carry_m,
                                 const Tensor& carry_l, std::uint32_t visible_keys,
                                 const Tensor& partial_acc, const Tensor& partial_m,
                                 const Tensor& partial_l, std::int32_t split_count, Tensor& out,
                                 cudaStream_t stream);

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream);

// scratch_k/scratch_v are the hq-e8-2b one-shot decode planes (BF16
// [head_dim, kv_heads, span]); they must be empty tensors for every other
// cache dtype. Envelopes wider than the scratch span band through the carry tensors.
void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, const Tensor& scratch_k,
                                           const Tensor& scratch_v, const Tensor& carry_acc,
                                           const Tensor& carry_m, const Tensor& carry_l,
                                           std::uint32_t visible_keys,
                                           const Tensor& partial_acc, const Tensor& partial_m,
                                           const Tensor& partial_l, std::int32_t split_count,
                                           Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
