#!/usr/bin/env bash
set -uo pipefail

# Master model-card campaign on the single GPU: for each artifact, the
# GPQA-Diamond + AIME26 quality rounds and the LongBench v2 RoPE cells.
# Terminal-Bench 2.1 and SWE-Bench Pro run separately from WSL against a
# manually launched server (eval/agentic/).
#
# usage: eval/run_master_campaign.sh [<label> ...]
#   default labels: nvfp4qat nvfp4full   (models/qwen3_8_27b_<label>.ninfer)

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

labels=("$@")
if [[ ${#labels[@]} -eq 0 ]]; then
    labels=(nvfp4qat nvfp4full)
fi

campaign_log="eval/runs-card/campaign-$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p eval/runs-card

for label in "${labels[@]}"; do
    artifact="models/qwen3_8_27b_${label}.ninfer"
    if [[ ! -f "${artifact}" ]]; then
        echo "missing artifact: ${artifact}" | tee -a "${campaign_log}"
        exit 1
    fi
    echo "[$(date -u +%FT%TZ)] === ${label}: quality rounds ===" | tee -a "${campaign_log}"
    if bash eval/run_card_quality.sh "${artifact}" "${label}" 42 43 44 \
        >>"${campaign_log}" 2>&1; then
        echo "[$(date -u +%FT%TZ)] ${label} quality rounds OK" | tee -a "${campaign_log}"
    else
        echo "[$(date -u +%FT%TZ)] ${label} quality rounds FAILED (see ${campaign_log})" | tee -a "${campaign_log}"
    fi
    echo "[$(date -u +%FT%TZ)] === ${label}: LongBench v2 cells ===" | tee -a "${campaign_log}"
    if bash eval/run_card_lbv2.sh "${artifact}" "${label}" \
        >>"${campaign_log}" 2>&1; then
        echo "[$(date -u +%FT%TZ)] ${label} LBv2 OK" | tee -a "${campaign_log}"
    else
        echo "[$(date -u +%FT%TZ)] ${label} LBv2 FAILED (see ${campaign_log})" | tee -a "${campaign_log}"
    fi
done
echo "[$(date -u +%FT%TZ)] campaign complete" | tee -a "${campaign_log}"
