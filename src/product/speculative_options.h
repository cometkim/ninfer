#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

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


[[nodiscard]] inline float parse_rope_scaling(std::string_view value) {
    if (value == "none") { return 0.0F; }
    constexpr std::string_view prefix = "yarn:";
    if (value.substr(0, prefix.size()) == prefix) {
        const std::string factor_text(value.substr(prefix.size()));
        std::size_t consumed = 0;
        float factor = 0.0F;
        try {
            factor = std::stof(factor_text, &consumed);
        } catch (const std::exception&) {
            throw std::invalid_argument("invalid rope-scaling factor: " + factor_text);
        }
        if (consumed != factor_text.size()) {
            throw std::invalid_argument("invalid rope-scaling factor: " + factor_text);
        }
        return factor;
    }
    throw std::invalid_argument("invalid rope-scaling (expected none or yarn:F): " +
                                std::string(value));
}

} // namespace ninfer::product
