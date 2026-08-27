#pragma once

#include "core/linear_attention_state.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::int32_t kKvQuantGroup = 64;

// hq-e8-2b runtime KV codec tag: (DType::U8, quant_group == kKvHqQuantGroup)
// identifies the fixed-budget E8+Rice format. Codes planes carry 64 bytes
// per (token, kv_head) row; the metadata planes (viewed through the
// *_scale_pages slots) carry 8 bytes = head_dim / kKvHqQuantGroup rows, so
// the shared scale-plane extent formula keeps holding.
inline constexpr std::int32_t kKvHqQuantGroup    = 32;
inline constexpr std::int32_t kKvHqCodeRowBytes  = 64;
inline constexpr std::int32_t kKvHqMetaRowBytes  = 8;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    DType kv_dtype                          = DType::BF16;
    std::int32_t kv_quant_group             = 0;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    LinearAttentionStatePoolSpec linear_attention;
};

struct PagedKVCacheLayout {
    PagedKVPoolLayout pool;
    // hq-e8-2b residual window (empty regions for every other dtype): exact
    // rotated-frame bf16 side planes [head_dim, kv_heads, sink+recent, layers *
    // table_rows] (layer l's plane is the dim-3 slice [l*table_rows,
    // (l+1)*table_rows)) plus kGqaHqRecentKeys/32 U32 recent-ring validity
    // words per slot row [words, table_rows] shared by every layer.
    TensorRegion residual_k;
    TensorRegion residual_v;
    TensorRegion ring_valid;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table,
                     std::int32_t slot) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
    std::int32_t slot_ = 0;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] PagedKVPool& pool() noexcept { return pool_; }

    [[nodiscard]] const PagedKVPool& pool() const noexcept { return pool_; }

    [[nodiscard]] bool residual_enabled() const noexcept { return residual_k_.data != nullptr; }

    // Exact post-restore ring validity: after trimming a retained bundle to `base` keys,
    // ring slot r stays valid iff the last key it wrote (largest key < retained_keys with
    // key % ring == r) lies inside the new recent window [base - ring, base). A full reset
    // (base 0) clears every bit.
    void revalidate_residual_ring(std::int32_t row, std::uint32_t retained_keys,
                                  std::uint32_t base, cudaStream_t stream = nullptr);

    // Clears the ring bits of the slots written by keys [first_key, end_key) — rejected
    // speculative drafts whose rows clobbered older still-recent keys.
    void invalidate_residual_ring(std::int32_t row, std::uint32_t first_key,
                                  std::uint32_t end_key, cudaStream_t stream = nullptr);

    [[nodiscard]] PagedKVCacheView execution_view(const PagedKVAllocation& allocation) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table,
                                              std::int32_t slot) const;

    PagedKVPool pool_;
    Tensor residual_k_;
    Tensor residual_v_;
    Tensor ring_valid_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    std::int32_t head_dim_     = 0;
    DType dtype_               = DType::BF16;
    std::int32_t quant_group_  = 0;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;
    LinearAttentionStatePoolLayout linear_attention;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;
    LinearAttentionStatePool linear_attention;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3_6
