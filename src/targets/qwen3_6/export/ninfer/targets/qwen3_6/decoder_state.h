#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    // hq-e8-2b residual window (empty regions when the feature is off): BF16 side planes
    // [head_dim, kv_heads, sink+recent rows, layers * table_rows] for K and V plus the per-slot
    // validity words [kKvSideWords, table_rows] shared by all layers.
    TensorRegion residual_k;
    TensorRegion residual_v;
    TensorRegion side_words;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t table_rows   = 0;
    PagedKVStorageLayout layer_storage;

    [[nodiscard]] std::size_t payload_bytes() const noexcept {
        return pages.payload_bytes() + residual_k.region.bytes + residual_v.region.bytes +
               side_words.region.bytes;
    }
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
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table, std::int32_t slot) noexcept;

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

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

    [[nodiscard]] bool residual_enabled() const noexcept { return residual_k_.data != nullptr; }

    // Residual-window lifecycle. Every method is a stream-ordered no-op when the feature is off.
    // A ring slot stays valid across a backward trim only if its LAST WRITTEN key (the largest
    // key < retained_keys congruent to the slot) still lies inside the sequence's new recent
    // window [base - ring, base); `revalidate` recomputes exactly that set. `invalidate` clears
    // the slots written by [first_key, end_key). `clear_side_rows` drops every bit including the
    // sink-valid mark (stale rows after a slot handover). `copy_side_rows` inherits another
    // slot's rows (prefix-fork destination), keeping that slot's sink mark.
    void revalidate_residual_ring(std::int32_t row, std::uint32_t retained_keys,
                                  std::uint32_t base, cudaStream_t stream);
    void invalidate_residual_ring(std::int32_t row, std::uint32_t first_key,
                                  std::uint32_t end_key, cudaStream_t stream);
    void clear_side_rows(std::int32_t row, cudaStream_t stream);
    void copy_side_rows(std::int32_t source_row, std::int32_t destination_row,
                        cudaStream_t stream);
    // Validity-word row copy (companion to copy_side_rows for fork inheritance).
    void copy_side_words(std::int32_t source_row, std::int32_t destination_row,
                         cudaStream_t stream);

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table,
                                              std::int32_t slot) const;

    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    Tensor residual_k_;
    Tensor residual_v_;
    Tensor side_words_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    std::int32_t table_rows_   = 0;
    PagedKVStorageLayout layer_storage_;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3_6
