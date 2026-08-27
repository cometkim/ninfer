#include <ninfer/targets/qwen3_6/decoder_state.h>

#include "core/kv_ring_bits.h"

#include <ninfer/ops/gqa_attention.h>

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

constexpr int kRingWords = static_cast<int>(ops::kGqaHqRecentKeys) / 32;
constexpr KvRingWords kZeroWords{};
static_assert(kRingWords <= 16, "ring validity words exceed the by-value kernel parameter");

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::int32_t table_rows,
                              std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool hq        = dtype == DType::U8;
    const bool quantized = dtype == DType::I8 || hq;
    if ((!quantized && (dtype != DType::BF16 || quant_group != 0)) ||
        (dtype == DType::I8 &&
         (quant_group != kKvQuantGroup || head_dim % quant_group != 0)) ||
        (hq && (quant_group != kKvHqQuantGroup || head_dim != 256))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) * (quantized ? 4ULL : 2ULL));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (hq) {
            pool_spec.planes.push_back({DType::U8, kKvHqCodeRowBytes, kv_heads, 256});
            pool_spec.planes.push_back({DType::U8, kKvHqCodeRowBytes, kv_heads, 256});
            pool_spec.planes.push_back({DType::U8, kKvHqMetaRowBytes, kv_heads, 256});
            pool_spec.planes.push_back({DType::U8, kKvHqMetaRowBytes, kv_heads, 256});
        } else {
            pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
            pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
            if (quantized) {
                pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
                pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            }
        }
    }
    PagedKVCacheLayout layout;
    layout.pool = plan_paged_kv_pool(builder, pool_spec);
    if (hq) {
        // Residual window storage (WI-8): exact side rows are slot-indexed, so every
        // layer's plane is one dim-3 slice of a single [.., layers * table_rows] tensor
        // and the per-slot ring words are shared by all layers (appends are
        // position-driven and therefore layer-uniform).
        const std::int32_t side_rows =
            static_cast<std::int32_t>(ops::kGqaHqSinkKeys + ops::kGqaHqRecentKeys);
        const std::int32_t slots     = static_cast<std::int32_t>(layers) * table_rows;
        layout.residual_k =
            builder.add_tensor(DType::BF16, {head_dim, kv_heads, side_rows, slots}, 256,
                               "Paged KV residual k plane");
        layout.residual_v =
            builder.add_tensor(DType::BF16, {head_dim, kv_heads, side_rows, slots}, 256,
                               "Paged KV residual v plane");
        layout.ring_valid =
            builder.add_tensor(DType::I32, {kRingWords, table_rows, 1, 1}, 256,
                               "Paged KV residual ring validity");
    }
    layout.layers      = layers;
    layout.max_context = capacity;
    layout.kv_heads    = kv_heads;
    layout.head_dim    = head_dim;
    layout.dtype       = dtype;
    layout.quant_group = quant_group;
    return layout;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kv_table_rows, spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), dtype_(layout.dtype),
      quant_group_(layout.quant_group) {
    if (layout.residual_k.region.bytes != 0 || layout.residual_v.region.bytes != 0 ||
        layout.ring_valid.region.bytes != 0) {
        if (dtype_ != DType::U8 || layout.residual_k.region.bytes == 0 ||
            layout.residual_v.region.bytes == 0 || layout.ring_valid.region.bytes == 0) {
            throw std::logic_error("Paged KV residual layout is inconsistent");
        }
        residual_k_ = layout.residual_k.bind(backing);
        residual_v_ = layout.residual_v.bind(backing);
        ring_valid_ = layout.ring_valid.bind(backing);
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table,
                                   std::int32_t slot) noexcept
    : cache_(&cache), block_table_(block_table), slot_(slot) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_, slot_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table(), allocation.bound_row());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table,
                                          std::int32_t slot) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8 || dtype_ == DType::U8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    const std::int32_t rows  = pool_.table_row_count();
    PagedKVLayerView view{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .residual_k    = residual_k_.data == nullptr
                             ? Tensor()
                             : residual_k_.slice(3, static_cast<std::int32_t>(layer) * rows + slot, 1),
        .residual_v    = residual_v_.data == nullptr
                             ? Tensor()
                             : residual_v_.slice(3, static_cast<std::int32_t>(layer) * rows + slot, 1),
        .ring_valid    = ring_valid_.data == nullptr ? Tensor() : ring_valid_.slice(1, slot, 1),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
    return view;
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8 || dtype_ == DType::U8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    const std::int32_t rows  = pool_.table_row_count();
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .residual_k    = residual_k_.data == nullptr
                             ? Tensor()
                             : residual_k_.slice(3, static_cast<std::int32_t>(layer) * rows, rows),
        .residual_v    = residual_v_.data == nullptr
                             ? Tensor()
                             : residual_v_.slice(3, static_cast<std::int32_t>(layer) * rows, rows),
        .ring_valid    = ring_valid_,
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
    };
}

void PagedKVCache::revalidate_residual_ring(std::int32_t row, std::uint32_t retained_keys,
                                            std::uint32_t base, cudaStream_t stream) {
    if (ring_valid_.data == nullptr) { return; }
    if (row < 0 || row >= pool_.table_row_count()) {
        throw std::out_of_range("Paged KV residual row is out of range");
    }
    const std::int64_t ring   = static_cast<std::int64_t>(ops::kGqaHqRecentKeys);
    const std::int64_t window = static_cast<std::int64_t>(base);
    KvRingWords keep{};
    for (std::int64_t r = 0; r < ring; ++r) {
        // Largest key < retained_keys congruent to r mod ring: the last writer of
        // slot r before the trim. It stays valid iff it lies inside the sequence's
        // new recent window [base - ring, base).
        if (retained_keys == 0 || static_cast<std::int64_t>(retained_keys) - 1 < r) { continue; }
        const std::int64_t key =
            r + (static_cast<std::int64_t>(retained_keys) - 1 - r) / ring * ring;
        if (key >= window - ring && key < window) {
            keep.w[static_cast<std::size_t>(r / 32)] |= 1u << (r % 32);
        }
    }
    apply_kv_ring_valid_words(static_cast<std::uint32_t*>(ring_valid_.data) +
                                  static_cast<std::size_t>(row) * kRingWords,
                              keep, kZeroWords, kRingWords, stream);
}

void PagedKVCache::invalidate_residual_ring(std::int32_t row, std::uint32_t first_key,
                                            std::uint32_t end_key, cudaStream_t stream) {
    if (ring_valid_.data == nullptr) { return; }
    if (row < 0 || row >= pool_.table_row_count()) {
        throw std::out_of_range("Paged KV residual row is out of range");
    }
    if (end_key <= first_key) { return; }
    const std::uint32_t ring = ops::kGqaHqRecentKeys;
    KvRingWords clear_words{};
    for (std::uint32_t key = first_key; key < end_key && key < first_key + ring; ++key) {
        const std::uint32_t r = key & (ring - 1);
        clear_words.w[r / 32] |= 1u << (r % 32);
    }
    KvRingWords keep{};
    for (int w = 0; w < kRingWords; ++w) { keep.w[w] = ~clear_words.w[w]; }
    apply_kv_ring_valid_words(static_cast<std::uint32_t*>(ring_valid_.data) +
                                  static_cast<std::size_t>(row) * kRingWords,
                              keep, kZeroWords, kRingWords, stream);
}

std::size_t PagedKVCacheLayout::payload_bytes() const noexcept {
    return pool.payload_bytes() + residual_k.region.bytes + residual_v.region.bytes +
           ring_valid.region.bytes;
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
