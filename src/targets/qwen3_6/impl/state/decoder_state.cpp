#include <ninfer/targets/qwen3_6/decoder_state.h>

#include "core/device.h"
#include "core/kv_ring_bits.h"

#include "ninfer/ops/softmax_attention.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

static_assert(kKvRingWords == static_cast<int>(ops::kCausalHqRecentKeys) / 32,
              "KV ring word width must match the hq recent-window constant");

constexpr int kSideRows = static_cast<int>(ops::kCausalHqSinkKeys + ops::kCausalHqRecentKeys);

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, KvCacheStorage storage,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const PagedKVStorageLayout layer_storage = paged_kv_storage_layout(storage, head_dim);

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    KVPageGeometry geometry;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * layer_storage.planes_per_layer());
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {layer_storage.key.data_dtype, layer_storage.key.data_leading_extent, kv_heads, 256});
        geometry.planes.push_back({layer_storage.value.data_dtype,
                                   layer_storage.value.data_leading_extent, kv_heads, 256});
        if (layer_storage.key.has_scale()) {
            geometry.planes.push_back({layer_storage.key.scale_dtype,
                                       layer_storage.key.scale_leading_extent, kv_heads, 256});
        }
        if (layer_storage.value.has_scale()) {
            geometry.planes.push_back({layer_storage.value.scale_dtype,
                                       layer_storage.value.scale_leading_extent, kv_heads, 256});
        }
    }
    PagedKVCacheLayout layout;
    layout.pages = plan_device_kv_page_pool(
        builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                      .geometry         = std::move(geometry)});
    layout.execution_tables = plan_kv_execution_tables(
        builder,
        KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows});
    if (storage == KvCacheStorage::HqE8Rice2B) {
        // Residual window storage (hq-e8-2b only): exact side rows are slot-indexed, so every
        // layer's plane is one dim-3 slice of a single [head_dim, kv_heads, side_rows,
        // layers * table_rows] tensor and the per-slot validity words are shared by all layers
        // (appends are position-driven and therefore layer-uniform).
        const std::int32_t slots = static_cast<std::int32_t>(layers) * table_rows;
        layout.residual_k =
            builder.add_tensor(DType::BF16, {head_dim, kv_heads, kSideRows, slots}, 256,
                               "Paged KV residual k plane");
        layout.residual_v =
            builder.add_tensor(DType::BF16, {head_dim, kv_heads, kSideRows, slots}, 256,
                               "Paged KV residual v plane");
        layout.side_words =
            builder.add_tensor(DType::I32, {kKvSideWords, table_rows, 1, 1}, 256,
                               "Paged KV residual validity words");
    }
    layout.layers        = layers;
    layout.max_context   = capacity;
    layout.kv_heads      = kv_heads;
    layout.table_rows    = table_rows;
    layout.layer_storage = layer_storage;
    return layout;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      table_rows_(layout.table_rows), layer_storage_(layout.layer_storage) {
    if (pages_.plane_count() !=
        static_cast<std::size_t>(layers_) * layer_storage_.planes_per_layer()) {
        throw std::invalid_argument("Paged KV layer plane inventory is inconsistent");
    }
    if (layout.residual_k.region.bytes != 0 || layout.residual_v.region.bytes != 0 ||
        layout.side_words.region.bytes != 0) {
        if (layer_storage_.storage != KvCacheStorage::HqE8Rice2B ||
            layout.residual_k.region.bytes == 0 || layout.residual_v.region.bytes == 0 ||
            layout.side_words.region.bytes == 0) {
            throw std::logic_error("Paged KV residual layout is inconsistent");
        }
        residual_k_ = layout.residual_k.bind(backing);
        residual_v_ = layout.residual_v.bind(backing);
        side_words_ = layout.side_words.bind(backing);
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

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()), row.row_index());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table,
                                          std::int32_t slot) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const std::size_t stride        = layer_storage_.planes_per_layer();
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(layer_storage_.key.has_scale());
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = layer_storage_.key.has_scale() ? pages_.plane(k_scale_index) : Tensor(),
        .v_scale_pages = layer_storage_.value.has_scale() ? pages_.plane(v_scale_index) : Tensor(),
        .residual_k    = residual_k_,
        .residual_v    = residual_v_,
        .side_words    = side_words_,
        .slot          = slot,
        .block_table   = block_table,
        .head_dim      = layer_storage_.head_dim,
        .num_kv_heads  = kv_heads_,
        .storage       = layer_storage_.storage,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    const PagedKVLayerView direct = layer_view(layer, Tensor(), 0);
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .residual_k    = direct.residual_k,
        .residual_v    = direct.residual_v,
        .side_words    = direct.side_words,
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
    };
}

void PagedKVCache::revalidate_residual_ring(std::int32_t row, std::uint32_t retained_keys,
                                            std::uint32_t base, cudaStream_t stream) {
    if (side_words_.data == nullptr) { return; }
    if (row < 0 || row >= table_rows_) {
        throw std::out_of_range("Paged KV residual row is out of range");
    }
    const std::int64_t ring   = static_cast<std::int64_t>(ops::kCausalHqRecentKeys);
    const std::int64_t window = static_cast<std::int64_t>(base);
    KvRingWords keep{};
    for (std::int64_t r = 0; r < ring; ++r) {
        // Largest key < retained_keys congruent to r mod ring: the last writer of slot r before
        // the trim. It stays valid iff it lies inside the sequence's new recent window
        // [base - ring, base).
        if (retained_keys == 0 || static_cast<std::int64_t>(retained_keys) - 1 < r) { continue; }
        const std::int64_t key =
            r + (static_cast<std::int64_t>(retained_keys) - 1 - r) / ring * ring;
        if (key >= window - ring && key < window) {
            keep.w[static_cast<std::size_t>(r / 32)] |= 1u << (r % 32);
        }
    }
    apply_kv_ring_valid_words(static_cast<std::uint32_t*>(side_words_.data) +
                                  static_cast<std::size_t>(row) * kKvSideWords,
                              keep, KvRingWords{}, kKvSideWords, stream);
}

void PagedKVCache::invalidate_residual_ring(std::int32_t row, std::uint32_t first_key,
                                            std::uint32_t end_key, cudaStream_t stream) {
    if (side_words_.data == nullptr) { return; }
    if (row < 0 || row >= table_rows_) {
        throw std::out_of_range("Paged KV residual row is out of range");
    }
    if (end_key <= first_key) { return; }
    const std::uint32_t ring = ops::kCausalHqRecentKeys;
    KvRingWords clear_words{};
    for (std::uint32_t key = first_key; key < end_key && key < first_key + ring; ++key) {
        const std::uint32_t r = key & (ring - 1);
        clear_words.w[r / 32] |= 1u << (r % 32);
    }
    KvRingWords keep{};
    for (int w = 0; w < kKvSideWords; ++w) { keep.w[w] = ~clear_words.w[w]; }
    apply_kv_ring_valid_words(static_cast<std::uint32_t*>(side_words_.data) +
                                  static_cast<std::size_t>(row) * kKvSideWords,
                              keep, KvRingWords{}, kKvSideWords, stream);
}

void PagedKVCache::clear_side_rows(std::int32_t row, cudaStream_t stream) {
    if (side_words_.data == nullptr) { return; }
    if (row < 0 || row >= table_rows_) {
        throw std::out_of_range("Paged KV residual row is out of range");
    }
    apply_kv_ring_valid_words(static_cast<std::uint32_t*>(side_words_.data) +
                                  static_cast<std::size_t>(row) * kKvSideWords,
                              KvRingWords{}, KvRingWords{}, kKvSideWords, stream);
}

void PagedKVCache::copy_side_rows(std::int32_t source_row, std::int32_t destination_row,
                                  cudaStream_t stream) {
    if (side_words_.data == nullptr) { return; }
    if (source_row < 0 || source_row >= table_rows_ || destination_row < 0 ||
        destination_row >= table_rows_) {
        throw std::out_of_range("Paged KV residual row is out of range");
    }
    copy_kv_residual_slot(residual_k_.data, residual_v_.data,
                          static_cast<std::int64_t>(kSideRows) * kv_heads_ *
                              layer_storage_.head_dim * sizeof(std::uint16_t),
                          static_cast<std::int32_t>(layers_), table_rows_, source_row,
                          destination_row, stream);
}

void PagedKVCache::copy_side_words(std::int32_t source_row, std::int32_t destination_row,
                                   cudaStream_t stream) {
    if (side_words_.data == nullptr) { return; }
    if (source_row < 0 || source_row >= table_rows_ || destination_row < 0 ||
        destination_row >= table_rows_) {
        throw std::out_of_range("Paged KV residual row is out of range");
    }
    if (source_row == destination_row) { return; }
    auto* words = static_cast<std::uint8_t*>(side_words_.data);
    CUDA_CHECK(cudaMemcpyAsync(words + static_cast<std::size_t>(destination_row) * kKvSideWords * 4,
                               words + static_cast<std::size_t>(source_row) * kKvSideWords * 4,
                               kKvSideWords * 4, cudaMemcpyDeviceToDevice, stream));
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
