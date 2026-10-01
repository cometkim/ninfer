#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/hq/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
constexpr int kTiledQueryRows         = Int8KvTiledInstance::kQueryRows;
} // namespace

int int8_kv_tiled_splits(int heads, int width, int multiprocessor_count) {
    const std::int64_t items =
        static_cast<std::int64_t>((width + kTiledQueryRows - 1) / kTiledQueryRows) * heads;
    const std::int64_t sms = multiprocessor_count;
    // A single pass that fits one wave gains nothing from splitting keys.
    if (items <= sms) return 1;
    const double single = static_cast<double>((items + sms - 1) / sms);
    double best_ratio   = single;
    int best            = 1;
    for (int splits = 2; splits <= 4; ++splits) {
        const double ratio = static_cast<double>((items * splits + sms - 1) / sms) / splits;
        if (ratio < best_ratio) {
            best_ratio = ratio;
            best       = splits;
        }
    }
    return best_ratio > 0.9 * single ? 1 : best;
}

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count) {
    if (multiprocessor_count <= 0 || (heads != 24 && heads != 16) || width < 1 || batch < 1 ||
        batch > 8 || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumLinearVisibleKeys)
        throw std::invalid_argument("INT8 attention: invalid plan inputs");
    constexpr int grouped_limit = Int8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? Int8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Int8KvFamily::ParallelGrouped
                                                                     : Int8KvFamily::Tiled;
    const int tiles =
        family == Int8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    const std::int64_t sms      = multiprocessor_count;
    const auto budget =
        heads == 24 || width <= 4 ||
                causal_query_tiles_underfill_sms(independent_tiles, multiprocessor_count)
            ? 2 * sms
            : sms;
    CausalKvPartition partition{1, causal_partition_target(budget, independent_tiles)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift    = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity     = partition.active(envelope.max_visible_keys);
    const int tiled_splits = family == Int8KvFamily::Tiled
                                 ? int8_kv_tiled_splits(heads, width, multiprocessor_count)
                                 : 1;
    Int8KvCausalPlan plan{family, heads, width, batch, envelope, partition, tiled_splits};
    // Grouped widths take the fork small-T route where it measured faster; a grouped plan
    // reserves the partials its sub-envelope small-T calls need.
    if (family == Int8KvFamily::Grouped) {
        if (linear_kv_small_t_selected(KvCacheStorage::Int8Group64, heads, width, batch,
                                       envelope)) {
            plan.family         = Int8KvFamily::SmallT;
            plan.small_t_splits = linear_kv_small_t_splits(
                KvCacheStorage::Int8Group64, heads, width, batch, envelope, multiprocessor_count);
        } else {
            plan.small_t_reserve = linear_kv_small_t_reserve_bytes(
                KvCacheStorage::Int8Group64, heads, width, batch, envelope, multiprocessor_count);
        }
    }
    return plan;
}

std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            make_int8_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == Int8KvFamily::Tiled) continue;
        if (plan.family == Int8KvFamily::SmallT) {
            maximum = std::max(maximum, linear_kv_small_t_workspace_bytes(
                                            heads, width, plan.small_t_splits, batch));
            continue;
        }
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, plan.partition.capacity, batch);
        // A grouped call pads its partials up to the small-T reserve of its envelope.
        maximum = std::max({maximum, layout.peak_bytes(1), plan.small_t_reserve});
    }
    // The tiled key split keeps one split count per query-tile interval of widths, with partial
    // storage growing in width, and the count is not monotone across intervals (a 948-column
    // tail can split four ways where a 1024-column chunk splits three), so every interval's last
    // width is evaluated.
    for (int begin = std::max(min_width, kGroupedPrefillMaxWidth + 1); begin <= max_width;) {
        const int last =
            std::min(max_width, (begin + kTiledQueryRows - 1) / kTiledQueryRows * kTiledQueryRows);
        const int splits = int8_kv_tiled_splits(heads, last, multiprocessor_count);
        if (splits > 1) {
            WorkspaceLayoutBuilder layout;
            (void)allocate_causal_partials(layout, heads, last, splits, 1);
            maximum = std::max(maximum, layout.peak_bytes(1));
        }
        begin = last + 1;
    }
    return maximum;
}

} // namespace ninfer::ops::detail
