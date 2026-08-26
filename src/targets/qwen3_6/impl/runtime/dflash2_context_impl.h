#include "targets/qwen3_6/impl/runtime/dflash2_context.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

DFlash2PersistentState::DFlash2PersistentState(DeviceSpan backing,
                                               const DFlash2PersistentLayout& layout)
    : local(backing, layout.local),
      rewrite_checkpoint_local(backing, layout.rewrite_checkpoint_local),
      prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (local.layer_count() != DFlash2Config::layers ||
        rewrite_checkpoint_local.layer_count() != DFlash2Config::layers ||
        local.capacity() != DFlash2Config::local_capacity ||
        rewrite_checkpoint_local.capacity() != DFlash2Config::local_capacity ||
        local.num_kv_heads() != DFlash2Config::kv_heads ||
        rewrite_checkpoint_local.num_kv_heads() != DFlash2Config::kv_heads ||
        local.head_dim() != DFlash2Config::head_dim ||
        rewrite_checkpoint_local.head_dim() != DFlash2Config::head_dim ||
        local.lane_capacity() != rewrite_checkpoint_local.lane_capacity() ||
        local.lane_capacity() < static_cast<std::int32_t>(1)) {
        throw std::invalid_argument("DFlash2 persistent cache layout is invalid");
    }
}

CyclicKVCacheLayerView DFlash2PersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

void DFlash2PersistentState::save_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream) {
    rewrite_checkpoint_local.copy_lane_from(local, lane, stream);
}

void DFlash2PersistentState::restore_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream) {
    local.copy_lane_from(rewrite_checkpoint_local, lane, stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
