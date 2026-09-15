#include "ops/softmax_attention/dense/causal_cache/hq/plan.h"

#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/hq/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/hq/internal.h"
#include "ops/softmax_attention/dense/causal_cache/hq/workspace.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace hq {
namespace {

constexpr int kMaximumSmallTWidth = 16;

// Upper bound of the device-side split policy (causal_small_t_default_splits in
// small_t_common.cuh) over one window. Graph calls pass their replay interval, eager calls an
// exact window.
template <typename Geometry>
std::int32_t split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}

// The policy is monotonic inside these finite segments and may drop when crossing a boundary.
// Evaluating every segment end plus both interval ends gives the exact interval maximum.
template <typename Geometry>
std::int32_t launch_capacity(CausalAttentionExecutionEnvelope envelope) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits = split_upper_bound<Geometry>(static_cast<std::int32_t>(window));
        capacity          = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

// The fork's compile-time split ceiling (85 x split scale) is two CTAs per SM per KV head on the
// 170-SM RTX 5090. Deriving the launch bound from the executing device keeps that occupancy on
// other SM counts and is identical at 170 SMs; the device policy never exceeds the launch bound.
std::int32_t device_split_bound(std::int32_t multiprocessor_count, std::int32_t kv_heads) {
    return std::max<std::int32_t>(1, 2 * multiprocessor_count / kv_heads);
}

// W=1 above 1024 visible keys runs the narrow tile (four CTAs per SM), whose device split
// policy (causal_hq_narrow_active_splits) doubles the grid up to the 256-split reducer limit.
std::int32_t narrow_decode_capacity(std::int32_t capacity, std::int32_t tokens,
                                    CausalAttentionExecutionEnvelope envelope) {
    if (tokens == 1 && envelope.max_visible_keys > 1024) { return std::min(256, 2 * capacity); }
    return capacity;
}

} // namespace

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size,
                                             std::int32_t multiprocessor_count) {
    if (cache_storage != KvCacheStorage::HqE8Rice2B || multiprocessor_count <= 0 || tokens < 1 ||
        tokens > (q_heads == 24 ? 8 : 6) || batch_size < 1 || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("hq-e8-2b attention split capacity: invalid profile");
    }
    if (q_heads == CausalD256H24Kv4::QHeads) {
        const std::int32_t capacity =
            std::min(launch_capacity<CausalD256H24Kv4>(envelope),
                     device_split_bound(multiprocessor_count, CausalD256H24Kv4::KVHeads));
        if (batch_size > 1) {
            // Keep complete grids within one or two waves: 160 target CTAs at 170 SMs leave room
            // for the indivisible 4*B group, including B=3/5/6/7. The page bound is the fork's
            // shared small-T policy, kept so the split counts stay identical.
            const std::int32_t target_ctas = multiprocessor_count * 16 / 17;
            const std::int32_t grid_limit  = div_up(target_ctas, 4 * batch_size);
            const std::int32_t page_limit =
                div_up(static_cast<std::int32_t>(envelope.max_visible_keys), 3968);
            return std::min(capacity, std::max({4, grid_limit, page_limit}));
        }
        return narrow_decode_capacity(capacity, tokens, envelope);
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        const std::int32_t capacity =
            std::min(launch_capacity<CausalD256H16Kv2>(envelope),
                     device_split_bound(multiprocessor_count, CausalD256H16Kv2::KVHeads));
        return narrow_decode_capacity(capacity, tokens, envelope);
    }
    throw std::invalid_argument("hq-e8-2b attention split capacity: unsupported head geometry");
}

} // namespace hq

HqKvCausalPlan make_hq_kv_causal_plan(int heads, int width, int batch,
                                      CausalAttentionExecutionEnvelope envelope,
                                      int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > hq::kMaximumSmallTWidth) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument("hq-e8-2b attention: invalid plan inputs");
    }
    const int tile          = heads == 24 ? 8 : 6;
    const HqKvFamily family = width <= tile                      ? HqKvFamily::SmallT
                              : width <= hq::kMaximumSmallTWidth ? HqKvFamily::ChunkedSmallT
                                                                 : HqKvFamily::Prompt;
    return {family, heads, width, batch, tile, envelope};
}

std::size_t hq_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                  CausalAttentionExecutionEnvelope envelope,
                                  int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, hq::kMaximumSmallTWidth); ++width) {
        const auto plan =
            make_hq_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        for (int begin = 0; begin < width; begin += plan.chunk_tokens) {
            const int count  = std::min(plan.chunk_tokens, width - begin);
            const int splits = hq::causal_attention_split_capacity(
                heads, count, KvCacheStorage::HqE8Rice2B, envelope, batch, multiprocessor_count);
            WorkspaceLayoutBuilder layout;
            (void)hq::allocate_small_t_workspace(layout, heads, count, splits, batch);
            maximum = std::max(maximum, layout.peak_bytes(1));
        }
    }
    if (max_width > hq::kMaximumSmallTWidth) {
        (void)make_hq_kv_causal_plan(heads, max_width, batch, envelope, multiprocessor_count);
        // The band carry grows with the query width, so the widest prompt call bounds it.
        WorkspaceLayoutBuilder layout;
        (void)hq::allocate_prompt_scratch(layout, heads == 24 ? 4 : 2, heads, max_width,
                                          envelope.max_visible_keys);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
