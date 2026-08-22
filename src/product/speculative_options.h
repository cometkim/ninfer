#pragma once

#include "ninfer/types.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}


/** Parsed `--rope-scaling` spec: factor 0 disables scaling; a factor above 1 selects YaRN. */
struct RopeScalingSpec {
    float factor      = 0.0F;
    float temperature = 0.1F;  // attention factor = temperature * ln(factor) + 1
    float beta_fast   = 32.0F;
    float beta_slow   = 1.0F;
};

namespace rope_scaling_detail {

[[nodiscard]] inline float parse_field(std::string_view text, const char* what) {
    float value = 0.0F;
    const auto* first = text.data();
    const auto* last  = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc() || result.ptr != last || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid rope-scaling ") + what + ": " +
                                    std::string(text));
    }
    return value;
}

} // namespace rope_scaling_detail

/**
 * Accepts `none`, `yarn:F`, and `yarn:F` extended with comma-separated `t=<c>` (attention
 * temperature coefficient, default 0.1), `bf=<n>` (beta_fast, default 32), and `bs=<n>`
 * (beta_slow, default 1), each at most once. Engine-side validation owns the value ranges.
 */
[[nodiscard]] inline RopeScalingSpec parse_rope_scaling(std::string_view value) {
    if (value == "none") { return {}; }
    constexpr std::string_view prefix = "yarn:";
    if (value.substr(0, prefix.size()) != prefix) {
        throw std::invalid_argument("invalid rope-scaling (expected none or yarn:F[,...]): " +
                                    std::string(value));
    }
    RopeScalingSpec spec;
    const std::string_view rest = value.substr(prefix.size());
    const auto first_comma      = rest.find(',');
    spec.factor = rope_scaling_detail::parse_field(rest.substr(0, first_comma == std::string_view::npos
                                                                    ? rest.size()
                                                                    : first_comma),
                                                   "factor");
    std::string_view fields = first_comma == std::string_view::npos
                                  ? std::string_view{}
                                  : rest.substr(first_comma + 1);
    bool have_temperature = false;
    bool have_fast        = false;
    bool have_slow        = false;
    while (!fields.empty()) {
        const auto comma   = fields.find(',');
        const auto field   = fields.substr(0, comma == std::string_view::npos ? fields.size() : comma);
        const auto equals  = field.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument("invalid rope-scaling field (expected t=, bf=, or bs=): " +
                                        std::string(field));
        }
        const auto key   = field.substr(0, equals);
        const auto value_text = field.substr(equals + 1);
        if (key == "t" && !std::exchange(have_temperature, true)) {
            spec.temperature = rope_scaling_detail::parse_field(value_text, "temperature");
        } else if (key == "bf" && !std::exchange(have_fast, true)) {
            spec.beta_fast = rope_scaling_detail::parse_field(value_text, "beta_fast");
        } else if (key == "bs" && !std::exchange(have_slow, true)) {
            spec.beta_slow = rope_scaling_detail::parse_field(value_text, "beta_slow");
        } else {
            throw std::invalid_argument("unknown or repeated rope-scaling field: " +
                                        std::string(field));
        }
        fields = comma == std::string_view::npos ? std::string_view{} : fields.substr(comma + 1);
    }
    return spec;
}

} // namespace ninfer::product
