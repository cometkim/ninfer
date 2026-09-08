#!/usr/bin/env bash
set -euo pipefail

# Model-card LongBench v2 campaign: the three RoPE-profile cells (short @ native
# 262k int8, medium @ 524k hq yarn:2, long @ 786k hq yarn:3) in full-capability
# mode - thinking ON under the card's registered sampling, one seed per round.
# One fresh server per cell; the server's default thinking mode is ON (the
# cells request enable_thinking explicitly).
#
# usage: eval/run_card_lbv2.sh <artifact-path> <label> [seed ...]
#   default seeds: 42 43 44
#   LBV2_CELLS: optional space-separated subset of suite names to run
#   (e.g. "lbv2_long_768k_full" to complete a single interrupted cell).

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${repo_dir}/build-ninja/apps/ninfer-serve.exe"
eval_python="${repo_dir}/eval/.venv/Scripts/python.exe"
config_template="${repo_dir}/eval/configs/qwen3_8_27b_card_lbv2.yaml"
log_dir="${repo_dir}/eval/server-logs"
port=18080

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <artifact-path> <label> [seed ...]" >&2
    exit 2
fi
artifact="$1"
label="$2"
shift 2
seeds=("$@")
if [[ ${#seeds[@]} -eq 0 ]]; then
    seeds=(42 43 44)
fi

for required_file in "${server_bin}" "${artifact}" "${eval_python}" "${config_template}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "missing required file: ${required_file}" >&2
        exit 1
    fi
done
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
mkdir -p -- "${log_dir}"

# suite|kv-dtype|max-context|rope-scaling
cells=(
    "lbv2_short_native|int8|262144|none"
    "lbv2_medium_524k|hq-e8-2b|524288|yarn:2"
    "lbv2_long_768k_full|hq-e8-2b|786432|yarn:3"
)

for seed in "${seeds[@]}"; do
    config="${repo_dir}/eval/runs-card/lbv2-${label}-seed${seed}.yaml"
    mkdir -p -- "${repo_dir}/eval/runs-card"
    sed "s/__SEED__/${seed}/g" "${config_template}" >"${config}"
    for cell in "${cells[@]}"; do
        IFS='|' read -r suite kv max_context rope <<<"${cell}"
        if [[ -n "${LBV2_CELLS:-}" ]] && ! [[ " ${LBV2_CELLS} " == *" ${suite} "* ]]; then
            continue
        fi
        rope_args=()
        if [[ "${rope}" != "none" ]]; then
            rope_args=(--rope-scaling "${rope}")
        fi
        run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        server_log="${log_dir}/lbv2-${label}-seed${seed}-${suite}-${run_stamp}.server.log"
        if curl --fail --silent --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "port ${port} already has a healthy service; stop it first" >&2
            exit 1
        fi
        echo "=== ${label} seed ${seed}: ${suite} (kv=${kv} ctx=${max_context} rope=${rope}, thinking on) ==="
        "${server_bin}" "${artifact}" \
            --host 127.0.0.1 --port "${port}" \
            --model-id qwen3.8-27b \
            --max-context "${max_context}" \
            --kv-capacity "${max_context}" \
            --max-concurrency 1 \
            --max-pending-requests 2 \
            --pending-timeout-ms 86400000 \
            --prefill-chunk 1024 \
            --kv-dtype "${kv}" \
            --spec mtp --draft-tokens 3 \
            "${rope_args[@]}" \
            >"${server_log}" 2>&1 &
        server_pid=$!
        cleanup() {
            if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
                kill -TERM "${server_pid}"; wait "${server_pid}" || true
            fi
        }
        trap cleanup EXIT
        ready=0
        for ((attempt = 1; attempt <= 240; ++attempt)); do
            if curl --fail --silent --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
                ready=1; break
            fi
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                wait "${server_pid}" || true
                echo "server exited before ready; see ${server_log}" >&2
                exit 1
            fi
            sleep 2
        done
        [[ "${ready}" == 1 ]] || { echo "server not ready in 480s; see ${server_log}" >&2; exit 1; }
        PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval run \
            --config "${config}" --suite "${suite}"
        kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        server_pid=""
    done
done
