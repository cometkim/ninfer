#include "ops/softmax_attention/dense/causal_cache/hq/plan.h"

#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/hq/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/hq/internal.h"
#include "ops/softmax_attention/dense/causal_cache/hq/workspace.h"

#include <algorithm>
#include <array>
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

// Host mirror of the linear codecs' device split policies (causal_small_t_policy_splits): INT8
// adds the fork's measured T=5 and T>=6 specializations to the shared tiers; FP8 rows on the
// 24-head geometry use every split for single-token windows past 8198 keys (the NVFP4 and K8V4
// launches keep the shared tiers, which bound their device policy).
template <typename Geometry>
std::int32_t linear_split_count(KvCacheStorage storage, std::int32_t window, std::int32_t tokens) {
    if (storage == KvCacheStorage::Int8Group64) {
        if (tokens == 5 && window > 128 && window <= 512) {
            return div_up(window, 32 / Geometry::SmallTSplitScale);
        }
        if (tokens >= 6 && window > 128 && window <= 160) {
            constexpr std::int32_t kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
            return div_up(window, kKeysPerSplit);
        }
        if (tokens >= 6 && window > 5000 && window <= 8198) {
            const std::int32_t splits   = div_up(window, 192 / Geometry::SmallTSplitScale);
            constexpr std::int32_t kMin = 4 * Geometry::SmallTSplitScale;
            constexpr std::int32_t kMax = 42 * Geometry::SmallTSplitScale;
            return std::min(std::max(splits, kMin), kMax);
        }
    }
    if constexpr (Geometry::SmallTSplitScale == 1) {
        if (storage == KvCacheStorage::Fp8E4M3Row256 && tokens == 1 && window > 8198) {
            return Geometry::SmallTMaximumSplits;
        }
    }
    return split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t linear_launch_capacity(KvCacheStorage storage,
                                    CausalAttentionExecutionEnvelope envelope,
                                    std::int32_t tokens) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        capacity = std::max(capacity, linear_split_count<Geometry>(
                                          storage, static_cast<std::int32_t>(window), tokens));
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

std::int32_t causal_attention_split_capacity_linear(KvCacheStorage cache_storage,
                                                    std::int32_t q_heads, std::int32_t tokens,
                                                    CausalAttentionExecutionEnvelope envelope,
                                                    std::int32_t batch_size,
                                                    std::int32_t multiprocessor_count) {
    const bool linear = cache_storage == KvCacheStorage::Int8Group64 ||
                        cache_storage == KvCacheStorage::Fp8E4M3Row256 ||
                        cache_storage == KvCacheStorage::Nvfp4Group16 ||
                        cache_storage == KvCacheStorage::Fp8KeyNvfp4Value;
    if (!linear || multiprocessor_count <= 0 || tokens < 1 || tokens > (q_heads == 24 ? 8 : 6) ||
        batch_size < 1 || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("small-T attention split capacity: invalid profile");
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        // The fork bounds the 16-head grid by the split policy alone at every batch size.
        return std::min(linear_launch_capacity<CausalD256H16Kv2>(cache_storage, envelope, tokens),
                        device_split_bound(multiprocessor_count, CausalD256H16Kv2::KVHeads));
    }
    if (q_heads != CausalD256H24Kv4::QHeads) {
        throw std::invalid_argument("small-T attention split capacity: unsupported geometry");
    }
    const std::int32_t capacity =
        std::min(linear_launch_capacity<CausalD256H24Kv4>(cache_storage, envelope, tokens),
                 device_split_bound(multiprocessor_count, CausalD256H24Kv4::KVHeads));
    if (batch_size == 1) { return capacity; }
    // Keep complete grids within one or two waves (the fork's per-codec targets of 160 or 320
    // CTAs at 170 SMs, scaled to the executing device), leaving room for the indivisible 4*B
    // group; the page bound keeps every split's page staging inside the kernels' 128-entry window.
    const bool narrow              = tokens <= 5;
    const bool two_waves           = cache_storage == KvCacheStorage::Int8Group64
                                         ? narrow || envelope.max_visible_keys > 4096
                                         : cache_storage == KvCacheStorage::Nvfp4Group16 && narrow;
    const std::int32_t one_wave    = multiprocessor_count * 16 / 17;
    const std::int32_t target_ctas = two_waves ? 2 * one_wave : one_wave;
    const std::int32_t grid_limit  = div_up(target_ctas, 4 * batch_size);
    const std::int32_t page_limit =
        div_up(static_cast<std::int32_t>(envelope.max_visible_keys), 3968);
    return std::min(capacity, std::max({4, grid_limit, page_limit}));
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

namespace {

// Graph resource tier bound (visible keys) up to which the fork's small-T kernel serves a profile,
// per batch class (B = 1, 2, 3-4, 5-8) and query width (W = 1..8 for 24 query heads, 1..6 for 16);
// 0 = never. Measured on the RTX 5090 against the codec's grouped route: interleaved op sweeps
// under graph replay (warm cache, fragmented pages, envelope [1, tier], windows just above the
// previous tier, mid-tier and at the top; B=3-4 measured at 4, B=5-8 at 8). The bound maximizes
// the summed log speed ratio over the tiers it covers, kept when that gain reaches 1% and each
// extension adds at least 0.5%.
using SmallTBounds = std::array<std::array<std::uint32_t, 8>, 4>;

constexpr SmallTBounds kInt8H24  = {{
    {8192, 16384, 16384, 16384, 16384, 16384, 8192, 8192},
    {4096, 16384, 16384, 32768, 32768, 8192, 32768, 16384},
    {8192, 16384, 8192, 32768, 8192, 4096, 4096, 4096},
    {2048, 8192, 4096, 4096, 16384, 4096, 4096, 4096},
}};
constexpr SmallTBounds kInt8H16  = {{
    {128, 4096, 4096, 4096, 2048, 2048, 0, 0},
    {2048, 2048, 2048, 2048, 2048, 512, 0, 0},
    {512, 512, 512, 512, 128, 512, 0, 0},
    {128, 512, 128, 128, 128, 128, 0, 0},
}};
constexpr SmallTBounds kFp8H24   = {{
    {2048, 4096, 4096, 4096, 4096, 4096, 4096, 4096},
    {4096, 4096, 16384, 4096, 16384, 16384, 32768, 32768},
    {2048, 2048, 4096, 16384, 4096, 16384, 16384, 16384},
    {128, 2048, 4096, 2048, 4096, 4096, 8192, 4096},
}};
constexpr SmallTBounds kFp8H16   = {{
    {128, 0, 0, 2048, 2048, 512, 0, 0},
    {512, 512, 0, 512, 512, 512, 0, 0},
    {512, 128, 0, 0, 128, 0, 0, 0},
    {128, 0, 0, 0, 0, 0, 0, 0},
}};
constexpr SmallTBounds kNvfp4H24 = {{
    {8192, 8192, 8192, 8192, 8192, 4096, 8192, 4096},
    {4096, 32768, 8192, 8192, 16384, 8192, 8192, 16384},
    {2048, 16384, 16384, 8192, 4096, 4096, 4096, 8192},
    {2048, 8192, 32768, 32768, 32768, 4096, 2048, 2048},
}};
constexpr SmallTBounds kNvfp4H16 = {{
    {2048, 2048, 2048, 2048, 2048, 2048, 0, 0},
    {2048, 2048, 0, 0, 2048, 2048, 0, 0},
    {512, 0, 512, 0, 512, 512, 0, 0},
    {512, 128, 0, 0, 128, 128, 0, 0},
}};
constexpr SmallTBounds kK8v4H24  = {{
    {2048, 4096, 4096, 4096, 4096, 4096, 4096, 2048},
    {2048, 16384, 16384, 8192, 16384, 8192, 32768, 8192},
    {2048, 8192, 8192, 8192, 16384, 32768, 4096, 32768},
    {128, 8192, 8192, 4096, 8192, 4096, 16384, 16384},
}};
constexpr SmallTBounds kK8v4H16  = {{
    {512, 128, 0, 2048, 0, 2048, 0, 0},
    {0, 512, 0, 512, 512, 512, 0, 0},
    {512, 512, 0, 512, 512, 512, 0, 0},
    {128, 128, 0, 128, 128, 0, 0, 0},
}};

// Graph tiers bound the visible keys by the tier end plus the verify width (DFlash2 and MTP
// verify calls see at most 16 keys past their tier's frontier).
constexpr std::uint32_t kTierWidthSlack = 16;

// Inclusive visible-key bound below which the profile takes the small-T route; 0 = never.
std::uint32_t small_t_visible_bound(KvCacheStorage storage, int query_heads, int width, int batch) {
    if (width < 1 || width > (query_heads == 24 ? 8 : 6) || batch < 1 || batch > 8) return 0;
    const bool h24             = query_heads == 24;
    const SmallTBounds* bounds = nullptr;
    switch (storage) {
    case KvCacheStorage::Int8Group64:
        bounds = h24 ? &kInt8H24 : &kInt8H16;
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        bounds = h24 ? &kFp8H24 : &kFp8H16;
        break;
    case KvCacheStorage::Nvfp4Group16:
        bounds = h24 ? &kNvfp4H24 : &kNvfp4H16;
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        bounds = h24 ? &kK8v4H24 : &kK8v4H16;
        break;
    default:
        return 0;
    }
    const int batch_class     = batch == 1 ? 0 : batch == 2 ? 1 : batch <= 4 ? 2 : 3;
    const std::uint32_t bound = (*bounds)[batch_class][width - 1];
    return bound == 0 ? 0 : bound + kTierWidthSlack;
}

} // namespace

bool linear_kv_small_t_selected(KvCacheStorage storage, int query_heads, int width, int batch,
                                CausalAttentionExecutionEnvelope envelope) {
    const std::uint32_t bound = small_t_visible_bound(storage, query_heads, width, batch);
    return bound != 0 && envelope.max_visible_keys <= bound;
}

int linear_kv_small_t_splits(KvCacheStorage storage, int query_heads, int width, int batch,
                             CausalAttentionExecutionEnvelope envelope, int multiprocessor_count) {
    return hq::causal_attention_split_capacity_linear(storage, query_heads, width, envelope, batch,
                                                      multiprocessor_count);
}

std::size_t linear_kv_small_t_workspace_bytes(int query_heads, int width, int splits, int batch) {
    WorkspaceLayoutBuilder layout;
    (void)hq::allocate_small_t_workspace(layout, query_heads, width, splits, batch);
    return layout.peak_bytes(1);
}

std::size_t linear_kv_small_t_reserve_bytes(KvCacheStorage storage, int query_heads, int width,
                                            int batch, CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count) {
    const std::uint32_t bound = small_t_visible_bound(storage, query_heads, width, batch);
    if (bound == 0 || envelope.min_visible_keys > bound) return 0;
    const CausalAttentionExecutionEnvelope small_t{envelope.min_visible_keys,
                                                   std::min(envelope.max_visible_keys, bound)};
    return linear_kv_small_t_workspace_bytes(
        query_heads, width,
        linear_kv_small_t_splits(storage, query_heads, width, batch, small_t, multiprocessor_count),
        batch);
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
