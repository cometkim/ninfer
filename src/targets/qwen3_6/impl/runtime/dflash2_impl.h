#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "ninfer/ops/cast.h"
#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ninfer/ops/dflash2_selector_predecessors.h"
#include "ninfer/ops/dflash2_selector_scores.h"
#include "ninfer/ops/dflash2_selector_walk.h"
#include "ninfer/ops/dflash2_topk.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash2_state(const PrefillContext& state) {
    if (state.dflash2 == nullptr || !state.execution.model.dflash2.has_value()) {
        throw std::logic_error("DFlash2 schedule requires DFlash2 weights and state");
    }
}

DFlash2PersistentState& dflash2_state(PrefillContext& state) {
    require_dflash2_state(state);
    return *state.dflash2;
}

DFlash2PersistentState& dflash2_state(DFlash2BatchContext& state) { return state.dflash2; }

DFlash2PersistentState& dflash2_state(DFlash2AppendContext& state) { return state.dflash2; }

template <class V>
DFlashFeatureSink prefill_dflash2_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash2) {
        throw std::logic_error("DFlash2 feature capture is unavailable for this target");
    } else {
        require_dflash2_state(state);
        using Config = typename V::DFlash2Config;
        return DFlashFeatureSink{
            .features        = &dflash2_state(state).prefill_features,
            .positions       = &dflash2_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_dflash2_sink_impl(DFlash2BatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash2) {
        throw std::logic_error("DFlash2 feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlash2Config;
        return DFlashFeatureSink{
            .batch_features      = &dflash2_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

// Projects one exact target-feature block and injects its per-layer K/V into the cyclic cache.
// The fused query_key_value parent is evaluated in full and its K/V row blocks extracted;
// positions are absolute and the rope table is the drafter's own unscaled 1e7 table.
template <class V, class Context>
void dflash2_append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash2) {
        throw std::logic_error("DFlash2 context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlash2Config;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch) {
            throw std::invalid_argument("DFlash2 context append inputs are invalid");
        }
        // One cyclic lane can commit at most its window; a wider exact prefix keeps the newest
        // local_capacity tokens, which is exactly the SWA state at the new frontier.
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash2 oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        Tensor local_counts    = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash2 prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({Config::feature_rows, columns}),
                    state.execution.model.dflash2->feature_projection, projected,
                    state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash2->context_norm, Config::rms_epsilon,
                     true, context, state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash2->layers.at(static_cast<std::size_t>(layer));
            Tensor layer_context =
                replace_local_window ? context.slice(1, local_offset, local_width) : context;
            Tensor layer_positions =
                replace_local_window ? positions.slice(0, local_offset, local_width) : positions;
            const int layer_columns = local_width * batch;
            auto layer_roots =
                workspace_recipe::dflash2_context_layer<Config>(state.execution.work,
                                                                layer_columns);
            ops::linear(layer_context, weight.query_key_value, layer_roots.qkv,
                        state.execution.device.stream);
            ops::extract_bf16_columns(layer_roots.qkv, Config::query_size, layer_roots.key_raw,
                                      state.execution.device.stream);
            ops::extract_bf16_columns(layer_roots.qkv, Config::query_size + Config::kv_size,
                                      layer_roots.value, state.execution.device.stream);
            Tensor key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
            ops::rmsnorm(layer_roots.key_raw.view({Config::head_dim, Config::kv_heads,
                                                   layer_columns}),
                         weight.key_norm, Config::rms_epsilon, false, key,
                         state.execution.device.stream);
            static const ops::RopeFrequencies frequencies =
                ops::rope_linear_frequencies(Config::rope_theta, Config::head_dim);
            ops::rope(layer_positions.view({layer_columns}), Config::head_dim, frequencies, key,
                      ops::RopeSide::Key, state.execution.device.stream);
            ops::kv_cache_append_prefix(
                key.view({Config::head_dim, Config::kv_heads, local_width, batch}),
                layer_roots.value.view({Config::head_dim, Config::kv_heads, local_width, batch}),
                layer_positions.view({local_width, batch}), local_counts, lanes, local_envelope,
                dflash2_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                state.execution.device.stream);
        }
    }
}

template <class V>
void dflash2_propose_batch_impl(DFlash2BatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlash2Envelopes envelopes) {
    if constexpr (!V::supports_dflash2) {
        throw std::logic_error("DFlash2 proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlash2Config;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes               = frame.lanes.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        if (state.execution.proposal_head != ProposalHead::Full) {
            throw std::logic_error("DFlash2 requires the full-vocabulary proposal head");
        }
        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            const auto& weight =
                state.execution.model.dflash2->layers.at(static_cast<std::size_t>(layer));
            {
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash2_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                ops::linear(roots.hidden, weight.attention_conv_projection, roots.dynamic,
                            state.execution.device.stream);
                ops::dflash2_dynamic_conv(roots.hidden, roots.dynamic, weight.attention_conv_base,
                                          0, width, roots.conv_hidden,
                                          state.execution.device.stream);
                ops::linear(roots.conv_hidden, weight.query_key_value, roots.qkv,
                            state.execution.device.stream);
                ops::extract_bf16_columns(roots.qkv, 0, roots.query_raw,
                                          state.execution.device.stream);
                ops::extract_bf16_columns(roots.qkv, Config::query_size, roots.key_raw,
                                          state.execution.device.stream);
                ops::extract_bf16_columns(roots.qkv, Config::query_size + Config::kv_size,
                                          roots.value, state.execution.device.stream);
                Tensor query_normed = roots.query.view(
                    {Config::head_dim, Config::query_heads, columns});
                ops::rmsnorm(roots.query_raw.view({Config::head_dim, Config::query_heads, columns}),
                             weight.query_norm, Config::rms_epsilon, false, query_normed,
                             state.execution.device.stream);
                Tensor key_normed = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(roots.key_raw.view({Config::head_dim, Config::kv_heads, columns}),
                             weight.key_norm, Config::rms_epsilon, false, key_normed,
                             state.execution.device.stream);
                static const ops::RopeFrequencies frequencies =
                    ops::rope_linear_frequencies(Config::rope_theta, Config::head_dim);
                ops::rope(positions.view({columns}), Config::head_dim, frequencies, query_normed,
                          key_normed, state.execution.device.stream);
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                ops::swa(
                    query_normed.view({Config::head_dim, Config::query_heads, width, batch_size}),
                    key_normed.view({Config::head_dim, Config::kv_heads, width, batch_size}),
                    roots.value.view({Config::head_dim, Config::kv_heads, width, batch_size}),
                    positions, valid_columns, lanes, Config::attention_scale,
                    dflash2_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    envelopes.local, state.execution.work, attention_batch,
                    state.execution.device.stream);
                ops::linear(roots.attention, weight.attention_output, roots.projected,
                            state.execution.device.stream);
                ops::dflash2_dynamic_conv(roots.projected, roots.dynamic, weight.attention_conv_base,
                                          1, width, roots.conv_attention,
                                          state.execution.device.stream);
                ops::residual_add(roots.conv_attention, residual, state.execution.device.stream);
            }
            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash2_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                ops::linear(roots.hidden, weight.mlp_conv_projection, roots.dynamic,
                            state.execution.device.stream);
                ops::dflash2_dynamic_conv(roots.hidden, roots.dynamic, weight.mlp_conv_base, 0,
                                          width, roots.conv_hidden, state.execution.device.stream);
                ops::linear_swiglu(roots.conv_hidden, weight.gate_up, roots.intermediate,
                                   ops::LinearPolicy::A16Only, state.execution.work,
                                   state.execution.device.stream);
                ops::linear(roots.intermediate, weight.down, roots.projected,
                            state.execution.device.stream);
                ops::dflash2_dynamic_conv(roots.projected, roots.dynamic, weight.mlp_conv_base, 1,
                                          width, roots.conv_projected,
                                          state.execution.device.stream);
                ops::residual_add(roots.conv_projected, residual, state.execution.device.stream);
            }
        }

        const std::int32_t draft_columns = static_cast<std::int32_t>(k) * batch_size;
        Tensor packed =
            state.execution.work.alloc(DType::BF16, {Config::hidden, draft_columns});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden =
            state.execution.work.alloc(DType::BF16, {Config::hidden, draft_columns});
        ops::rmsnorm(packed, state.execution.model.dflash2->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);

        Tensor logits = state.execution.work.alloc(
            DType::BF16, {TextConfig::output_rows, draft_columns});
        ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                    state.execution.device.stream);
        const auto top_k = static_cast<std::int32_t>(Config::selector_top_k);
        Tensor candidate_ids = state.execution.work.alloc(DType::I32, {top_k, draft_columns});
        Tensor candidate_values = state.execution.work.alloc(DType::BF16, {top_k, draft_columns});
        ops::dflash2_topk(logits, top_k, candidate_ids, candidate_values,
                          state.execution.device.stream);
        Tensor candidates = candidate_ids.view({top_k, static_cast<std::int32_t>(k), batch_size});
        Tensor unary = state.execution.work.alloc(
            DType::FP32, {top_k, static_cast<std::int32_t>(k), batch_size});
        ops::cast_bf16_to_fp32(
            candidate_values.view({top_k, static_cast<std::int32_t>(k), batch_size}), unary,
            state.execution.device.stream);
        Tensor predecessors = state.execution.work.alloc(
            DType::I32, {top_k, static_cast<std::int32_t>(k), batch_size});
        ops::dflash2_selector_predecessors(candidates, anchors, predecessors,
                                           state.execution.device.stream);
        Tensor hidden_proj = state.execution.work.alloc(
            DType::BF16, {Config::selector_rank, draft_columns});
        ops::linear(proposal_hidden, state.execution.model.dflash2->selector_hidden, hidden_proj,
                    state.execution.device.stream);
        Tensor hidden_proj_f32 = state.execution.work.alloc(
            DType::FP32, {Config::selector_rank, static_cast<std::int32_t>(k), batch_size});
        ops::cast_bf16_to_fp32(
            hidden_proj.view({Config::selector_rank, static_cast<std::int32_t>(k), batch_size}),
            hidden_proj_f32, state.execution.device.stream);
        Tensor scores = state.execution.work.alloc(
            DType::FP32, {top_k, top_k, static_cast<std::int32_t>(k), batch_size});
        ops::dflash2_selector_scores(candidates, predecessors, unary, hidden_proj_f32,
                                     state.execution.model.dflash2->selector_successor,
                                     state.execution.model.dflash2->selector_predecessor, scores,
                                     state.execution.device.stream);
        ops::dflash2_selector_walk(scores, candidates, drafts, state.execution.device.stream);
        state.execution.work.reset();
    }
}

auto dflash2_decode_batch_body(DFlash2BatchContext& state, std::int32_t batch_size,
                              std::uint32_t k, DFlash2Envelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash2 decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents          = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.lanes.slice(0, 0, batch_size);
        Tensor append_positions = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits    = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlash2Config::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash2_state(state).pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        dflash2_append_context_impl<Variant>(state, compact_features, append_positions,
                                             append_counts, lanes, envelopes.append);

        dflash2_propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids,
                                            state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0,
                         state.execution.rope_frequencies, {},
                         &state.text_cache);
        DFlashFeatureSink sink =
            batch_dflash2_sink_impl<Variant>(state, lanes, valid_columns, width, batch_size);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_positions,
                                 .valid_columns   = valid_columns,
                                 .kv_table_rows   = text_rows,
                                 .lanes           = lanes,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = drafts,
                                 .current_extents = extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                                 .feature_sink    = &sink,
                             },
                             target_envelope);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash2_feature_sink(PrefillContext& state,
                                       DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_dflash2_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash2_append_context(DFlash2AppendContext& state, const Tensor& features,
                            const Tensor& positions, const Tensor& commit_counts,
                            const Tensor& lanes,
                            ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    dflash2_append_context_impl<Variant>(state, features, positions, commit_counts, lanes, envelope);
}

void dflash2_append_context(PrefillContext& state, const Tensor& features,
                            const Tensor& positions, const Tensor& commit_counts,
                            const Tensor& lanes,
                            ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    dflash2_append_context_impl<Variant>(state, features, positions, commit_counts, lanes, envelope);
}

void capture_dflash2_decode_batch(DFlash2BatchContext& state, std::int32_t batch_size,
                                  std::uint32_t k, DFlash2Envelopes envelopes,
                                  ops::GqaExecutionEnvelope target_envelope,
                                  DecodeGraphDefinition& definition) {
    auto body = dflash2_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash2_decode_batch(DFlash2BatchContext& state, std::int32_t batch_size, std::uint32_t k,
                          DFlash2Envelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                          DecodeGraphExecutable* executable) {
    auto body = dflash2_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
