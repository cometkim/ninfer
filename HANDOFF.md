# HANDOFF — cross-session work state

Read with AGENTS.md before planning work. Earlier records remain recoverable from
refs/backup/pre-sync-20260908/*, backup/20260908/all-refs.bundle, and
refs/backup/post-rebase-20260909/{dev,1m-context}.

## Current state (2026-09-09)

Ports 1–11 are landed. The 1M-envelope investigation, standalone 1m-context re-port and dev
rebuild are complete. Both 786k and 1M presets boot and serve with the local WebUI. The idle
measurement campaign below supersedes old performance/acceptance records, including the
previously claimed cumulative +77% gain.

| branch | tip | base / role |
|---|---|---|
| master | a16b6442 | upstream semantic base |
| feat/msvc-test-constexpr | 62acee7e | master |
| feat/windows-port | 985907a4 | master; Windows/TMA/build |
| feat/webui | 81e077f4 | windows-port |
| feat/mtp7 | e3bf30db | master; MTP K1–K7 |
| feat/hyperquant | 5be7eb60 | windows-port; self-contained, PDL-free HQ |
| feat/1m-context | ae0eeb87 | hyperquant; plain rmsnorm → rope |
| feat/dflash2 | 90fd8a3b | windows-port; unified NVFP4 draft execution |
| feat/qwen3.8-nvfp4full | bc36e9c2 | dflash2 |
| feat/qwen3.8-nvfp4qat | c17ccc30 | dflash2 |
| feat/kernel-perf | 6c3fdbf4 | master; fusions, PDL, prompt/TU splits |

Dev starts at fork base **83ddf726**, with exactly one squash(feat/*) per branch in this order:
msvc-test-constexpr, windows-port, webui, mtp7, hyperquant, 1m-context, dflash2,
qwen3.8-nvfp4full, qwen3.8-nvfp4qat, kernel-perf. The integration fold retains PDL HQ/DFlash
forms and the splice tool. Dev-only supporting commits add same-binary measurement controls,
local preset budgets and session documentation; preserve these or fold their content deliberately.
The runtime/preset tip is 209cf315, followed by the session-documentation commit containing this
file. The validated standalone build remains in ../ninfer-1m-report on codex/1m-report;
its two untracked Ninja wrappers are local build aids, not feature content.

Each feature has its own README tip commit. All ten intermediate squashes passed the README
skeleton-plus-union check for rows and bullets. Independent feature lineages mean a final
single-feature-to-dev diff includes sibling content: account for it and for qualified fused/PDL
integration forms explicitly. The rebuilt tree preserves the previous integration except the
intended 1M fixes/docs, measurement controls and presets. Removed the accidentally tracked
generated Testing/Temporary/CTestCostData.txt. Do not blindly reuse whole-file conflict
resolutions: cached resolutions can pull kernel-perf back into the standalone 1M branch.

## 1M decode investigation and correction

The historical ~3 tok/s log had **4.07 GiB unbanded workspace**, 13.3 GiB runtime reservation
and only 766 MiB device headroom. Its two full-capacity BF16 scratch planes account for 4 GiB
at 1M keys (2 × 1048576 × 4 heads × 256 × 2 bytes). The banded source uses **1.08 GiB active
workspace** at both 524k and 1M. The old executable was stale relative to that source. The evidence
points to memory pressure from the stale workspace allocation; WDDM spilling itself was not
captured in that historical run.

Whole-inference Nsight Systems node traces, nvfp4full/HQ, a short real prompt and 32 generated
tokens, compared 524288/yarn:2 and 1048576/yarn:4:

| quantity | 524288 envelope | 1048576 envelope |
|---|---:|---:|
| HQ kernel launches | 544 | 544 |
| HQ median kernel time | 21.919 µs | 21.887 µs |
| active workspace | 1.08 GiB | 1.08 GiB |
| non-speculative graph allowance | 12 MiB | 12 MiB |

27B attention split capacity is capped at 85. Per-round grids/splits follow live history and use
the same initial graph interval here; there is no current envelope-scaled decode explosion.
Initial 128-token envelope pairs measured 81.2/81.1 and 80.6/79.6 tok/s (524k/1M), establishing
that the old cliff is absent, not measuring long-history speed.

Attribution exposed a separate numerical bug: load-time execution_core() omitted the trailing
RoPE-frequency table from its aggregate initializer. Plain rope rejected zero tables; the fused
route silently captured identity rotations. ExecutionCore::rope_frequencies is now a required
reference, and every construction binds the owned table, including load-time graph capture.
The fix lives in the 1M feature and dev squash. Fresh paired measurements use the corrected
binary. Pre-fix token/acceptance fingerprints are stale.

The re-port carries WI-8 dither/window, YaRN/banding/lanes and retrieval content from
e648f178/8f79eaa9/4008d8de, adapted to hyperquant's plain interfaces, with RopeSide for single
query/key routes. It adds no fused qk_norm_rope operation or PDL dependency. CLI/serve docs
now describe rope-scaling and execution-envelope limits.

Verified on the standalone branch: rope + rope-scaling + HQ codec + all 26 HQ attention
scenarios pass; retrieval gives 5/5 unique top-1 needles at 2.250 bps.
Engine builds and runs a coherent 32-token groupwise-int HQ 1M/yarn:4 real-prompt
response through graphs. On dev, softmax-attention and qk_norm_rope suites pass with controls
enabled and disabled. All four fusion/PDL combinations produce identical greedy real-prompt
output and MTP3 acceptance (42/63; positions 18/14/10).

## Working 786k / 1M presets

qwen3.8-27b-nvfp4full-hq-e8-2b-786k.bat and the -1m.bat choose local models/webui when
present, otherwise auto-download. Passing both --webui and --webui-dir still triggers download;
the offline route must use only --webui-dir. The local bundle has 66 files. Both presets use
two pinned host-state slots and 1024 MiB host KV: the default eight slots plus 8 GiB pinned KV
failed allocation on this 32 GiB host. This retains the cache tier with an explicit local budget.

Live acceptance used exact preset engine flags with a loopback test port: /props, /v1/models
and a non-streaming chat completion succeeded on both. Vision is enabled; 786k also loads
DFlash2 K7, while 1M uses yarn:4 without speculation. Auto capacity reaches the full pool:

| preset | page groups | runtime reservation | free GPU memory |
|---|---:|---:|---:|
| 786432 | 12288/12288 | 8.93 GiB | 4.08 GiB |
| 1048576 | 16384/16384 | 10.6 GiB | 3.06 GiB |

The corrected 1M server decoded 64 tokens at **80.1 tok/s** from a 60-token request. This
proves execution at the envelope; it does not establish quality or speed with a populated
1M-token history. Synthetic codec retrieval is not an end-to-end long-context quality gate.

## Idle paired measurements

RTX 5090, sm_120a, Windows, MSVC 19.44.35228, nvcc 13.3.73, CMake 4.4.2, Ninja wrappers.
JSON reports CUDA runtime/driver API 13.4. Exact artifacts: models/qwen3_8_27b_nvfp4full.ninfer
and models/qwen3_8_27b_nvfp4qat.ninfer (v2).

One frozen ninfer_bench.exe, sequential processes, three consecutive GPU-utilization samples
≤5% before each process, one warmup + three measured repetitions, two alternating A/B pairs
per comparison. Command shape:

```text
--weights <artifact> -n 128 --max-ctx <envelope> --kv-dtype <profile> -r 3 --warmup 1 -o json
```

Add `--spec mtp --draft-tokens K` for K>0.
These are **tg128 seed-token workloads**: 8k/262k are allocated envelopes, not populated history
lengths. Acceptance is corpus-specific. Pair gains are means of B/A ratios, not historical records.

NINFER_BENCH_FUSIONS=0 selects plain q/k rmsnorm→rope and separate sigmoid output gating;
it does not disable upstream GDN fusion. NINFER_BENCH_PDL=0 removes dependent-launch
serialization attributes while retaining the same compiled kernels. Process-fixed controls
default to enabled; see bench/README.md.

Fusion/PDL contribution, nvfp4full MTP3 INT8. Each row is its own adjacent A/B comparison;
the range gives the two observed pair gains, not a confidence interval.

| envelope | change | A tok/s | B tok/s | mean paired gain | pair range |
|---|---|---:|---:|---:|---:|
| 8192 | fusion on, PDL off | 127.62 | 131.22 | +2.84% | +1.78 to +3.89% |
| 8192 | PDL on, fusion off | 130.33 | 130.72 | +0.30% | +0.10 to +0.49% |
| 8192 | PDL on, fusion on | 129.48 | 131.57 | +1.62% | +1.32 to +1.91% |
| 262144 | fusion on, PDL off | 128.49 | 129.77 | +1.00% | +0.50 to +1.50% |
| 262144 | PDL on, fusion off | 127.01 | 128.82 | +1.46% | -0.34 to +3.25% |
| 262144 | PDL on, fusion on | 129.68 | 129.93 | +0.20% | +0.02 to +0.38% |

Fusion yields a modest gain. PDL gains are small and vary with the comparison; the 262k
unfused pair changes sign between repeats. These results do not support the old cumulative
+77% attribution. All fusion/PDL cells retain 41.76% MTP acceptance.

MTP K1–K7 at the 8192 envelope, INT8, fusions and PDL enabled. Every K2–K7 cell is paired
with a fresh K1 control; K1 is shown beside each width instead of treating a distant control
as interchangeable. Round time includes the measured decode interval divided by MTP rounds.
K1 acceptance is 67.11% for nvfp4full and 60.76% for QAT.

| profile | K | paired K1 tok/s | K tok/s | paired gain | K ms/round | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| nvfp4full | 2 | 122.66 | 140.07 | +14.19% | 15.49 | 58.47% |
| nvfp4full | 3 | 122.31 | 130.66 | +6.83% | 17.19 | 41.76% |
| nvfp4full | 4 | 122.65 | 130.73 | +6.59% | 18.47 | 35.41% |
| nvfp4full | 5 | 122.44 | 107.39 | -12.29% | 20.20 | 23.45% |
| nvfp4full | 6 | 123.60 | 106.76 | -13.62% | 21.80 | 22.29% |
| nvfp4full | 7 | 121.88 | 131.06 | +7.54% | 22.20 | 27.81% |
| nvfp4qat | 2 | 122.32 | 122.06 | -0.21% | 15.42 | 43.38% |
| nvfp4qat | 3 | 120.69 | 115.08 | -4.65% | 17.12 | 32.12% |
| nvfp4qat | 4 | 119.92 | 116.01 | -3.26% | 18.09 | 27.80% |
| nvfp4qat | 5 | 119.95 | 104.10 | -13.21% | 20.16 | 21.85% |
| nvfp4qat | 6 | 119.95 | 96.87 | -19.24% | 21.31 | 18.23% |
| nvfp4qat | 7 | 120.58 | 105.08 | -12.84% | 22.15 | 19.52% |

For this corpus, nvfp4full favors K2; QAT favors K1, with K2 effectively tied. Round time
increases gradually with width. The old large missing-route cliff claim is not supported
by these clean pairs; acceptance materially affects the resulting width ranking.

MTP0/MTP7 records, INT8, both controls enabled:

| profile | envelope | MTP0 tok/s | MTP7 tok/s | paired gain | K7 tokens/round | K7 ms/round |
|---|---:|---:|---:|---:|---:|---:|
| nvfp4full | 8192 | 81.18 | 128.64 | +58.46% | 2.909 | 22.61 |
| nvfp4full | 262144 | 81.28 | 128.14 | +57.65% | 2.909 | 22.70 |
| nvfp4qat | 8192 | 84.16 | 104.56 | +24.24% | 2.327 | 22.26 |
| nvfp4qat | 262144 | 84.73 | 104.97 | +23.88% | 2.327 | 22.17 |

HQ versus other KV profiles, nvfp4full MTP3, both controls enabled. Different KV codecs can
change greedy acceptance, so throughput combines kernel cost and the resulting token path.
The entire tg128 history fits in HQ's BF16 residual window; these HQ cells measure that
window route and allocation envelope, not steady-state compressed long-history attention.

| envelope | comparator | comparator tok/s | HQ tok/s | HQ paired gain | comparator acceptance | HQ acceptance |
|---|---|---:|---:|---:|---:|---:|
| 8192 | int8 | 129.46 | 134.45 | +3.86% | 41.76% | 46.25% |
| 8192 | k8v4 | 131.19 | 134.27 | +2.35% | 43.37% | 46.25% |
| 8192 | nvfp4 | 136.28 | 134.65 | -1.19% | 46.54% | 46.25% |
| 262144 | int8 | 128.50 | 133.72 | +4.06% | 41.76% | 46.25% |
| 262144 | k8v4 | 129.53 | 134.45 | +3.80% | 43.37% | 46.25% |
| 262144 | nvfp4 | 135.88 | 135.35 | -0.39% | 46.54% | 46.25% |

All 112 process cells completed successfully, with 336 measured repetitions. Prelaunch idle
checks passed for every cell. Small differences remain subject to run-to-run variation; these
short seed-token cells do not establish long-history throughput or model quality.

Local ignored provenance: profiles/bench/post-rebase-20260909/paired.ps1, summarize.py,
paired JSON/metadata, envelope traces, branch tests and preset responses. Historical loaded or
unpaired cells, width-cliff claims and old record comparisons must not be reused.

## Preserved ports and cautions

- Upstream remains the semantic authority for DFlash2 selector walk, K1..15 × B1..8, both heads,
  serving, host/prefix scheduling and the five original KV profiles. All fork ports are retained.
- Kernel-perf contains qk norm/rope and sigmoid-gate fusions, INT8 fp16-PV and INT8/BF16 prompt
  key splits, PDL publish-at-end, per-dtype/per-geometry launcher TUs and per-request error
  boundaries. Do not restore misordered BF16 partials from intermediate port history.
- V2 NVFP4 DFlash2 execution is complete through one unified binder. Module directories were
  renamed to upstream names in place; *.head.bak backups remain beside artifacts, payloads and
  offsets unchanged. The pre-rebase selector misread K16M128x4 scales as row-major; its old
  3.70 tok/round result is invalid evidence for preferring the former fork walk.
- models/qwen3_8_27b_nvfp4full_w8.ninfer is the v3 66-object W8 splice, made by
  tools.convert.qwen3_8_27b.splice_w8_module preserving base payloads. The groupwise-int
  models/qwen3_8_27b.ninfer has no draft companion; official nvfp4 is the pre-suffix core form.
  Full PR-form W8-module conversions are not prerequisites for the executable v2 files.
- HQ retains half-cell subtractive dither, BF16 sink32/recent512 residual window, absolute
  ownership epochs, rejected-draft invalidation and coherent prefix inheritance. Prompt scratch
  bands at 262144 keys with online-softmax carry. Linear KV caps at 524288; only 27B HQ permits 1M.
- BF16 CUDA-graph Engine execution fails cudaGraphExecUpdateFailure on clean upstream on this
  box; this is separate from the corrected RoPE binding. The old context-materialize replay
  failure was a test-only stream-ordering race, already fixed, not an engine defect.
- PDL producers publish after stores; consumers wait before dependent reads on all paths. No
  host synchronization/copies inside captures. Required aggregate members must not default away.

## Next work and build discipline

1. On the next kernel-perf rebase, drop its duplicated Windows TMA fix (already in windows-port).
2. For long-context quality, use populated prompts on the corrected binary. Old coding/model-card
   campaigns need a fresh lane choice and rerun before being advertised as post-rebase results.

Use powershell -ExecutionPolicy Bypass -File configure-ninja.ps1 once per build directory and
after CMakeLists edits; then build-ninja.ps1 [-Target <name>]. No developer prompt or numeric
job limit. Converter Python: ~/Workspace/ninfer-src/cvenv/Scripts/python.exe.

A local Ninja cache had zero header dependencies on the runtime variant object, so a header fix
did not rebuild it. The MSVC include prefix must match compiler locale; inspect ninja -t deps
when incremental builds appear stale. Both runtime variants were explicitly rebuilt and now
have tracked header dependencies before this campaign's binary froze. A no-op link is not
proof that a changed template header reached the executable.

GPU jobs never overlap. Use hidden PowerShell Start-Process with stdout/stderr files, retain
Handle before WaitForExit (PS5), and check ExitCode. LNK1104 usually means a stray holds the
exe. Run ctest from build-ninja and compute-sanitizer.exe directly. Use apply_patch or file-based
UTF-8 Python scripts for C/C++ edits; Git Bash heredocs and escaped inline Python mangle
newline literals. Never implicitly download or regenerate local artifacts.
