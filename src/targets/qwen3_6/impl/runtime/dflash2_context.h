#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct DFlash2PersistentState {
    CyclicKVCache local;
    CyclicKVCache rewrite_checkpoint_local;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor pending_features;

    DFlash2PersistentState(DeviceSpan backing, const DFlash2PersistentLayout& layout);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    void save_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream);
    void restore_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream);
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
