# HANDOFF — cross-session work state

Read this with `AGENTS.md` before planning work; update it before ending a session that changed
anything material. The 2026-08-27 reorganization replaced the session-by-session diary (sessions
2–28, 2026-08-21→27) with this document plus `ROADMAP.md` (track status) and the per-track
roadmaps; the former REVIEW-*.md review documents are deleted with their verdicts folded into
the roadmaps. The full pre-reorganization history is recoverable from branch
`dev-preorg-20260827` and `backup/20260827/*.bundle`.

## Branch map (2026-08-27 rebuild)

| branch | base | commits over base | PR target |
|---|---|---|---|
| feat/build-speed | natpate/master (b686696e) | +1 (195c5dc7) | natpate PR #2 (rebased over their compressed-KV merge; TUs compile-verified) |
| feat/build-speed-upstream | upstream/master (feaf4dd0) | +1 (d41d617f) | upstream (same split, message adjusted; to open) |
| feat/hyperquant | old natpate/master + feat/build-speed(3dc028f4) | +22 | natpate lineage |
| feat/1m-context | feat/hyperquant | +12 | upstream or natpate |
| feat/qwen3.8-nvfp4full | upstream/master (feaf4dd0) | +10 | upstream |
| feat/msvc-test-constexpr | upstream/master (2190c4a1) | +1 (8a30534e) | upstream |
| feat/kernel-perf | feat/1m-context | +14 | fork stack (natpate lineage; content folded into the dflash2 squash) |
| feat/dflash2 | feat/kernel-perf + merge(feat/qwen3.8-nvfp4full) + merge(upstream/master 2190c4a1) | +2 merges, +7 | fork stack (integration tip) |
| cometkim/dev | fork base 310e815f | +2 (squash dflash2 + fork-layer restore) | never PR'd |

2026-08-28 upstream sync (feaf4dd0 → 2190c4a1, 68 commits): master fast-forwarded;
`feat/msvc-test-constexpr` rebased (upstream's refactored GDN tests + the constexpr→const fix);
`feat/dflash2` absorbed master via merge with the fork op/runtime lineage authoritative —
upstream's execution-ownership refactor, public engine API change (`OutputConsumerMode`),
renamed `softmax_attention` op tree, fp8/int8-hadamard KV, host context cache + prefix
resource scheduling, and serve rewrites are superseded by the fork lineage; upstream's
standalone additions kept (tools/streaming_http, tools/ninfer_serve python client,
tools/bench TTFT suite + fixtures). Deferred for targeted adoption on fork structure:
serving fixes (text-part tool results, tool-arg schema typing, warmup-gated readiness,
stream perf), frontend one-pass media perf, oversized-prompt rejection, control-token
provenance, host-KV/context-cache scheduling. Pre-sync tips under
`refs/backup/pre-sync-20260828/*`. Dev rebuild parity: exact 40-file fork-base set
(both documented rebuild fixups repeated: duplicated `handle_props` dropped, pre-reorg
REVIEW/ROADMAP-1m-context records not reintroduced). `feat/build-speed` (natpate form)
and `feat/build-speed-upstream` (upstream PR form, feaf4dd0 base) were not rebased —
their TU-split content lives in the fork lineage independently; rebase the PR forms
only when opening PRs.

Convention notes: squashes applied in stack order (build-speed, hyperquant, 1m-context,
nvfp4full, msvc-test-constexpr, kernel-perf, dflash2); lineage-boundary squash conflicts
resolve toward the incoming branch (it stacks on the previous one); the one manual fix beyond
that was dropping a duplicated `handle_props` the webui×artifact merge reintroduces.
**natpate tracking is FROZEN (owner, 2026-08-27)**: the fork tracks upstream only; natpate's
role ended with the windows-port layer (both of its commits are absorbed into natpate's master).
Consequently `git diff feat/build-speed cometkim/dev` legitimately lists natpate's newer master
content (their compressed-KV e8 work) — the one deliberate parity-rule exception; dev does not
carry it. Local-only branches: `dev-preorg-20260827` (pre-rebuild tip), `dev-prerebuild`,
`scrub-backup`; bundles in `backup/20260827/` (untracked).

## Current state

**DFlash2 executes end-to-end with native acceptance and beats MTP everywhere measured** — the
active project's items 1–4 are landed and verified (see `ROADMAP-dflash2.md` for the tables):
+30% committed t/s over MTP3 at 8k greedy (157.6 vs 121.2), +52% at the 258k native envelope
(82.1 vs 54.0 tok/s decode), acceptance flat-to-up to 258k, vision + prefix-reuse clean. Stage-3
(MTP purge) is a live owner decision. The artifact is `models/qwen3_8_27b_nvfp4full.ninfer`
(v2, 18.07 GiB, DFlash2 module in NVFP4).

Everything else is stable and verified: hq-e8-2b KV (parity with int8 quality at 390–400k,
exact needles to 592k), 524k held / 768k engineering-ready (KL instrument pending), kernel-perf
K1/K2 landed (pp65536 int8 +14.2%; PDL publish rule; fusions), serving + CLI through one Engine
route. Regression notes: `ninfer_qwen3_6_frontend_test` fails at clean head (fixture-hash drift,
pre-existing) and `ninfer_openai_schema_test` crashes 0xc0000409 (pre-existing class) — both
outside current scope, recorded.

## Remaining work (priority order)

Mirrors `ROADMAP.md`: (1) DFlash2 stage-3 decision inputs; (2) the KL instrument for 768k;
(3) hq Tier-2 decode selection; (4) kernel-perf open items (fusion block, K1c, K1(d)/nvfp4-KV
decision, K3-on-Linux, K4, vLLM A/B); (5) host-kv / model-opt when scheduled. The DFlash2
forward-distribution debugging is DONE — the one defect (context_norm `unit_offset`) is fixed;
do not re-derive the session-27 hypothesis chain from the old diary.

## Verification & measurement cautions

1. **GPU discipline**: verify the GPU is quiet before trusting a cell (a screensaver once
   depressed every cell ~5–6%); run-to-run noise is real (±0.5% decode, ±6% prefill) — compare
   with same-binary alternating A/B pairs, never across sessions. Kill strays with
   `taskkill //PID <pid> //T` (the MSYS `$!` is not the Windows pid) and confirm the GPU is
   free between cells.
2. **Fingerprint discipline**: any schedule/numerics change needs a strong identity gate — the
   greedy token-stream byte-diff against a recorded stream is primary; an acceptance-ratio
   fingerprint (nvfp4full+int8 MTP3 0.4285714286 full head; groupwise-int 0.3080808081; the
   `--lm-head-draft` variant 0.4022988506) detects but cannot prove identity. Bench cells seed
   decode with a one-token prefill: ALWAYS add a real-prompt greedy smoke (the PDL entry-trigger
   bug passed every bench fingerprint and garbled real prompts).
3. **Rebuild both apps** (`ninfer` and `ninfer-serve`) after any engine change before serving
   conclusions — a stale server once "reproduced" a serving-specific regression that was pure
   staleness. `--Target ninfer` does not relink the server.
4. **Prefix-reuse repro** after rope/context plumbing: serve request A fresh, then B extending
   the conversation; expect `reuse=restore_turn_checkpoint` and a coherent MTP rate. The
   qk_norm_rope MRoPE regression (req 3 of a serving session) was invisible to every CLI smoke.
5. **[d0,d1] tensors are channel-fastest** (element (c,t) at c + H·t): new [d0,d1,d2] ops need
   at least one fixture with UNEQUAL dims so a transposed gather cannot pass — four dflash2
   kernels once passed their suites while gathering wrong elements in the engine.
6. **PDL rule** (a contract): producers publish at kernel END after their stores; consumers
   wait before the first dependent read, including early-exit paths.
7. **Windows editing/build**: non-ASCII literals go in as hex escapes (`\xef\xbf\xbd`, not raw
   UTF-8 bytes); Git Bash heredocs mangle large/escaped content — write files with proper tools.
   Never pipe long-running GPU programs through `head`/`tail` (writer hang with VRAM held; log
   to files). `constexpr std::sqrt` is not C++20 (use `const`; IEEE sqrt is correctly rounded);
   the `/utf-8` flags are fork-local, never in a feat branch.
8. **Engine internals**: positional aggregate init silently nulls trailing members — when
   adding a field to an aggregate like `ExecutionCore`, grep ALL construction sites; a
   reference-typed member fails compilation instead. No `cudaStreamSynchronize`/`cudaMemcpy`
   inside captured graph bodies — debug probes go in `static bool once`-guarded blocks outside
   the graph or in eager runs, and are stripped before commit.
9. **Module conventions**: the DFlash2 module follows HF Qwen3 plain-`w` RMSNorm (NOT the fork
   target's (1+w) Text-norm); hq's zero-tail invariant (decoders read either `used` form); the
   swa T=8 L=96 case's 1-ulp online-softmax deviation is understood (criterion 1e-3 with
   rationale inline).

## Build & tools

```bash
powershell -ExecutionPolicy Bypass -File configure-ninja.ps1      # once per build dir
powershell -ExecutionPolicy Bypass -File build-ninja.ps1 [-Target <name>]
cd tools/test_kv && powershell -ExecutionPolicy Bypass -File build-ninja-tk.ps1   # standalone hq suite
```

Tests and benchmarks ON by default (`-NoTests`/`-NoBenchmarks` opt out); the GQA launcher TUs
are split per dtype (and per geometry for the hq codec kernels) — new launcher template
instantiations belong in the per-dtype route TUs, not the dispatchers. Full-build ctest state
on this box: 84/91 pass; the two known failures are the pre-existing frontend fixture-hash and
openai-schema abort noted above. Eval harness: `eval/.venv` (uv, Python 3.12; EvalScope
1.10.0), `PYTHONPATH=eval eval/.venv/Scripts/python.exe -m ninfer_eval …`, configs under
`eval/configs/`, run records under `eval/runs/` (local-only). Profiler outputs under
`profiles/` (local-only, gitignored). Serving quickstarts: the root `qwen3.8-27b-*.bat` presets
(262k/524k/786k convention).

## Local prerequisites

Artifacts: `models/qwen3_8_27b_nvfp4full.ninfer` (v2 with the DFlash2 module; conversion report
beside it), `models/qwen3_8_27b.ninfer` (groupwise-int; both MTP smokes gated on it),
official `models/qwen3_8_27b_nvfp4.ninfer` (perf runs). Prompts:
`tests/fixtures/longcontext/longprompt_*.json` (labels are NOMINAL char-ratios; true counts:
"32k"=24,732, "262k"=202,514, "390k"=304,222, "500k"=390,033, "592k"=592,558, "1m"=1,029,898).
DFlash2 source + llama.cpp harness artifacts under `~/Workspace/llama-wi7/` (outside the repo);
the converted artifact is self-contained (sources deletable once no re-conversion is needed).
GPU power limit is 450 W until reboot (set for the eval campaign).
