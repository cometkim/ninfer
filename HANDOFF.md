# HANDOFF — hq-e8-2b KV cache integration + 1M-context track, continue in the next session

This document hands off the HyperQuant KV cache (`--kv-dtype hq-e8-2b`) engine integration work.
Read it, then start from the "Remaining work" section. The former `AUDIT-hyperquant.md` (pitfall
checklist) and `REVIEW-hyperquant.md` (the 17-section review that drove sessions 2–8) were deleted
at the 2026-08-23 extraction; their live content is folded into "Remaining work" and the cautions
below, and historical "REVIEW §N" / "AUDIT" references in the session records point at those
deleted documents.

## Branch map (2026-08-23 night rebuild, v2 - master-based after PR #2 fix)

PR #2 (natpate/ninfer-windows, head feat/build-speed, base natpate/master) exposed that a
natpate/dev-based rebuild silently dropped webui + meta.n_ctx from the PR - natpate's master
has the port core but NOT those (dev-only there). The stack is therefore based on
natpate/master (536be6d1) so the PR carries its complete content; natpate's dev dflash2 line
is deliberately NOT carried (it flows here when natpate merges dev->master). Pushed with
--force-with-lease:

| branch | base | commits over base | PR target |
|---|---|---|---|
| feat/windows-port | natpate/master | +2 (webui, meta.n_ctx; absorbed by natpate dev already) | not PR-active |
| feat/build-speed | natpate/master | +1 (launcher TU split) = PR #2 head, ONE commit | natpate PR #2 |
| feat/qwen3.8-nvfp4full | upstream/master | +10 (0 behind upstream) | upstream |
| feat/hyperquant | feat/build-speed | +22 (sessions 2-9) | natpate lineage |
| feat/1m-context | feat/hyperquant | +11 (WI-1/2/3/3b/8, envelope, frontend fixes) | upstream or natpate |
| cometkim/dev | feat/windows-port | fork-base commit, then +4 squashes on top | never PR'd |

Stack top builds clean with ninfer_gqa_attention_test passing (worktree check). The /utf-8
charset flags are a FORK-LOCAL commit on dev (CP949 locale workaround, per PR feedback - never
in a feat branch). Known deltas
vs the pre-rebuild dev: the duplicate handle_props definition in the old http_server lineage
is gone; feat/1m-context's frontend replacement literal is raw UTF-8 bytes (identical under
/utf-8) while dev keeps the hex-escape form; dev no longer carries natpate's dev-branch
dflash2 (rebase when they merge it to master). Local backups: backup/20260823/*.

2026-08-24 path scrub + rebuild: local absolute paths removed everywhere (model references
-> `models/`, eval prerequisites -> repository-relative `eval/data/`, toolchain fallbacks
-> PATH/env resolution only), and dev rebuilt in convention order - the scrubbed fork-base
commit (fork docs + tooling-policy commit folded in) sits BELOW the four squashes, so every
future rebuild inherits the scrubbed base. Local branches `scrub-backup` (pre-scrub tip) and
`dev-prerebuild` (pre-rebuild tip) hold the old states, not pushed.

## Current state (summary)

**WI-8 LANDED AND VERIFIED (session 16): hq-e8-2b needle retrieval is CLEAN at 32k / 304k /
390k / 592k true tokens (yarn:2 / yarn:4) — the >262k codec garble that blocked M2 since session
10 is fixed by the BF16 sink+recent residual window (W=512) plus half-cell subtractive dither.
Perf at the same cells is BETTER than session 8 (tg128 71.7→80.6; pp32k+tg64 decode 58.1→62.2 —
exact-row 16 B copies are cheaper than Rice decode). The bf16/int8 linear envelope is raised to
524,288 keys (int8 at 390,033 retrieves exactly; fits to ~430k beside nvfp4full weights).**

| probe (nvfp4full + hq, greedy, verified build) | tokens | retrieval | prefill t/s | decode t/s |
|---|---:|---|---:|---:|
| 32k needle, factor 1 | 24,772 | ✓ exact | 7,882 | 63.4 |
| "390k" file, yarn:2 | 304,262 | ✓ exact | 1,636 | 24.9 |
| "500k" file, yarn:2 | 390,073 | ✓ exact (was garbled under every setting) | 1,384 | 20.8 |
| "500k"+MTP3, yarn:2 | 390,073 | ✓ exact, 46/64 accepted | 1,384 | 62.9 committed |
| "592k" file, yarn:4, 1M pool auto | 592,558 | ✓ exact (re-verified post-D2 at 921.9 t/s) | 916–922 | 15.2 |
| "1m" file d50, yarn:4 | 1,029,898 | ✗ fluent token soup | 517.5 | 8.8 |
| "1m" file d20, yarn:4 | 1,029,899 | ✗ same soup (depth-independent) | 531.0 | — |
| 390k int8 (envelope raised) | 390,073 | ✓ exact (Q1a made permanent) | — | — |

**1M boundary (session 16 close)**: the clean/garble cliff sits in (592k, 1.03M]. The 1M failure is
DEPTH-INDEPENDENT (d20 ≡ d50) and its signature is confident multilingual token soup from token 1 —
unlike the pre-WI-8 codec failures (immediate EOS / short garbage). No int8 control exists at 1M
(does not fit), so attribution is open with dense-YaRN×4-at-1M as the leading hypothesis (ROADMAP
§5's named risk: Qwen's own 1M deployment pairs YaRN with DCA + MInference sparse prefill; dense
YaRN at 1M was never validated on this checkpoint) and the hq noise floor at ~1M distractors
secondary. Next session's discriminators, in cost order: YaRN factor/temperature grid at 1M
(§5k's prior "temperature doesn't rescue" was codec-class; this signature differs), an ~800k
bracket cell, then lbv2_long@1M. If dense YaRN is the binding constraint, the honest paths are
WI-6 (MInference-style sparse prefill) or revisiting the 1M goal's scope. Fixtures:
`longprompt_1m_needle{,_d20}.json` (~1.03M tokens, tiled from the 592k/500k/262k tangles).

Engine cells (RTX 5090, nvfp4full weights, session 8, same-session int8 pairs):

| cell | INT8 | HQ-E8-2B | ratio |
|---|---:|---:|---:|
| MTP0 tg128 | 76.5 | 71.7 | 0.938 |
| MTP0 pp8192+tg128 | 73.1 | 65.1 | 0.891 |
| MTP0 pp32768+tg64 | 69.6 | 58.05 | **0.834** |
| MTP3 pp2048+128 | 218.6 | 208.1 | 0.952 |
| MTP3 pp32k+64 | 208.8 | 177.9 | **0.852** |

CLI decode (nvfp4full, session 7): 52.0 @32k, ~35 @128k (interpolated, unmeasured), 25.3 @262k;
MTP3 @8k 183.5. KV payload @262k: 2.25 GiB (int8 8.25). Cumulative decode gains sessions 2–8:
32k 28.6 → 52.0 (+82%), 262k 9.1 → 25.3 (+178%), tg128 49.9 → 70.7 (+42%), MTP3@8k → 183.5. The
pre-implementation expectation for 2 bps KV was ≤1.3–1.4× int8 decode cost at 262k (capacity,
not speed, is the primary win) — measured hq now sits below that band at every length. The
remaining long-context gap is the per-step whole-window decode, bound by the dependent 8-lane
chains at 8 warps/SM (§5e); Tier 2 (exact V skipping) is the next lever.

Session-3 `ninfer_bench` matrix for the record (official nvfp4 artifact, r=3 warmup=1,
max_ctx 66560, CUDA-graph decode, no MTP; superseded by the §5c campaign tables):

| bench | bf16 | int8 | hq baseline | hq (session 3) |
|---|---|---|---|---|
| pp512 | 6604 | 6699 | 6508 | 6476 |
| pp2048 | 8111 | 7968 | 8102 | 7960 |
| pp8192 | 7636 | 7508 | 7516 | 7420 |
| pp32768 | 6037 | 5899 | 5724 | 5655 |
| pp65536 | 4667 | 4573 | 4303 | 4242 |
| tg128 | 60.6 | 61.1 | 49.9 | **53.5** |
| pp2048+tg128 | 8154 / 60.2 | 7998 / 60.7 | 8141 / 48.6 | 8038 / **53.4** |
| pp8192+tg128 | 7545 / 59.2 | 7424 / 60.0 | 7510 / 42.7 | 7353 / **50.2** |
| pp32768+tg64 | 5943 / 55.3 | 5852 / 58.4 | 5714 / 28.6 | 5625 / **41.9** |

MTP3 @8k: 147.3 tok/s (spec accept 96/128), up from 108 before the append flatten.

## This session (sessions 2–8; see §5b–§5f for the later addenda)

### 1. Correctness fixes from REVIEW §2 (all oracle-gated, fix-before-PR)

- **Batch/column offsets** (`gqa_attention_decode_hq.cuh`): the kernel now mirrors the bf16
  small-T kernel — `column_base = column_begin + batch·full_width` applied to q, pos, and the
  append sources; partial tensors offset by batch. Fixes silent wrong attention for batch>1
  (serving) and ChunkedSmallT.
- **Split partition over the whole window**: keys [0, window) partitioned across active splits
  (was [column_begin, window), which skipped earlier chunks' keys). `valid_columns` handling now
  subtracts `column_begin` like bf16.
- **`l_smem` read-before-init** fixed (zeroed at init); NaN-inherited smem could silently drop a
  split through the reducer's `tile_l > 0` guard.
- **Fused append now encodes every valid token owned by the split** (was: only the last token —
  broken for MTP verify widths ≥ 2), reading rows from the full-width frame with proper offsets.
- **New gate `tools/test_kv/test_hq_decode.cu`**: fill → decode kernel → host partial combine vs
  an FP64 oracle over device-decoded K/V rows, in the kernel's own compute profile (rotated-frame
  scores, one un-rotation). 7 scenarios: tokens=1/3, batch=2 with disjoint tables,
  column_begin=6 chunked shape, partial valid_columns, smem-trash determinism (NaN pattern),
  append variant. ALL PASSED (min cos 0.999993, max row rel 0.44%).

### 2. Decode performance: 8-lane cooperative group decoder

Phase attribution (bench-local kernel copy with runtime phase switches) showed the Rice K/V
decode phases were 84% of the decode kernel (0.74 of 0.89 ms @32k) and an upfront-prefetch
experiment ruled out load latency (~9%): the cost was the serial 64-symbol ALU dependency chain
per segment thread at 1 block/SM.

REVIEW §3.1's proof holds and is now asserted in the tests: **every stored row has Rice k = 0**
(a row fitting the 512-bit budget at all fits at k=0; the encoder picks minimum bits with strict
<; 22k-row corpus + synthetic checks confirm 0 rows with k≠0). With k=0 every 1-bit terminates a
symbol, so boundaries are prefix-sum computable.

`hq_decode_row_group` (hq_codec.cuh, 8 lanes per row, one 64-bit window each):
- window = two u32 code words concatenated MSB-first (**the words are u32 values stored
  little-endian — a raw u64 load scrambles the bit order; this cost an hour to find**);
- k==0 fast path: `popcll` per window + 3-round `__shfl_up_sync(..., 8)` exclusive prefix sum
  (symbol index base), a segmented scan of (all-zero, trailing zeros) for the carry-in unary run,
  then per-lane `clzll` decode staging zigzag codes as u16 into the output row itself;
- k>0 general fallback (defensive; in-tree encoder never stores it): speculative scan + 8-step
  fixup chain, tested against the sequential decoder on synthetic host-encoded streams;
- after `__syncwarp`, each lane unstrips four complete E8 words from the staged row (static word
  ownership — no cross-lane exchange for words spanning lane boundaries);
- bit-identical to `hq_decode_row_thread` on the 22k corpus (test_hq_codec [3]); the 4-way
  `hq_decode_row_segment` decoder is deleted (superseded; the sequential decoder remains as the
  oracle).

Wired into the decode kernel's K/V chunk phases (256 threads = all 32 chunk rows in one wave)
and the prefill scratch kernel (launcher grid ×8). Standalone engine-shape decode kernel:
window 32768 grid.y=85: **0.896 → 0.31–0.37 ms** (≈2.9×, ~850 M rows/s in-kernel); decode-only
phase time 0.755 → 0.178 ms. Score/PV phases (0.155 ms) are now the co-limiter.

### 3. Follow-up review (REVIEW §11) — addressed

- F1 test blind spot FIXED: scenario C now uses non-identity `table_rows` ({1,0}) with
  per-batch cache content (a wrong table lookup can no longer pass by coincidence); new
  scenario H (window 2560) exercises the multi-chunk online-softmax rescale chain; new
  scenario I verifies the out-of-range guard.
- F4: zero-tail invariant documented (codec header + caution 3); stale K/V-role sign comment
  and the "u64 load" wording fixed.
- F7: out-of-range positions now write neutral partials (bf16 parity; `(void)logical_capacity`
  gone); HANDOFF ratio-direction wording corrected; padding caution softened.
- F5 measured (see lever d): append-variant bench row added; nsys profile captured.
- Open, folded into later levers per the review's own order: F2 (scratch-kernel smem staging
  — 256 scattered 2-byte global stores per row; likely the remaining prefill gap at 128k/262k),
  F3 (general-k fallback via the sequential decoder to shed ~16+ registers; fold into lever c),
  F6 (flatten the append's per-token loop; fold into lever d), 35B bench geometry row,
  FP16 norm clamp, op-level tests under tests/ops, real-model quality gate.

### 4. Follow-up review §12 (session 4) — N1/N2 fixed, N3 guarded

- **N2 (escalation rescue was inverted — silently zeroed rows) FIXED and measured.** The
  census proved it: 78 normal + 11 heavy rows of the 22k corpus hit the terminal fallback
  (all-zero decode) and ZERO rows ever escalated-and-fit, exactly the review's proof (a row
  fits iff sum(z) <= 256; the old retry DOUBLED the symbols). The encoder now halves the
  staged coordinates per retry (`*= 0.5f`) and both decoders multiply by `1<<escalation`
  (test oracles flipped with them). After the fix: the same 89 rows escalate-and-fit
  (fallback 0/0), oracle max rel 0.9970 -> 0.0040, rotated-frame SNR **23.07 -> 55.03 dB** —
  the zeroed rows were the dominant error term in every prior quality number. Heavy-row
  escalation is now oracle-gated (test [2b]). NOTE: all previously reported hq quality
  metrics predate this fix.
- **N1+F6 (append encoded each row twice, two serial passes) FIXED**: (token, role) units
  flattened over all 8 warps, one encode per unit, scratch in `kv_smem`; the 4 KB static
  `append_smem` deleted; one barrier after the loop (block-uniform). Standalone append
  bench 0.175 -> 0.129 ms @window 54 (the dedup removed parallel-duplicate work, the
  flatten removed the pass serialization; one serial encode ~78-110 us remains — §3.3's
  target). MTP widths: 12 serial encodes -> 2 rounds.
- **N3 guards added**: the append bench uses dedicated K/V code planes (no corpus mutation,
  no K/V aliasing) and reads back the appended row's meta (k=0 esc=0 used=512 — no
  escalation inflation of the timing).
- **Engine cells after N1+N2** (official artifact, r=3, QUIET GPU): tg128 53.5 (+3.1),
  pp2048+tg128 53.4 (+7%), pp8192+tg128 50.2 (+5%), pp32768+tg64 41.9 (+5.5%), MTP3@8k
  147.3 (+36%), prefill at baseline. An earlier same-binary run had shown tg128 flat at
  50.1-50.3 — that was CONTAMINATION: a 3D screensaver (Mystify.scr) was running and
  depressed every cell ~5-6% (see caution 5b). nsys medians (286 -> 206 us/instance) were
  captured under the same contamination; the relative kernel-level comparison stands, but
  re-profile on a verified-quiet GPU before using absolute instance times.
- N4: oracle else-branch reindented; the append barrier's block-uniformity documented
  inline. Still open: 35B-geometry scenarios/tests, FP16 norm clamp, op-level tests,
  real-model quality gate (now MORE important: rerun with the N2 fix).

## File map (this session)

- `src/ops/kernel/hq_codec.cuh` — `hq_decode_row_group` (+ `hq_group_read`,
  `hq_group_scan_window` helpers); `hq_decode_row_segment` deleted; k==0 invariant documented.
- `src/ops/kernel/gqa_attention_decode_hq.cuh` — group decoder in K/V phases; all §2 fixes.
- `src/ops/kernel/gqa_attention_prefill_hq.cuh` — scratch kernel on the group decoder.
- `src/ops/launcher/gqa_attention_prefill_hq_routes.cuh` — scratch grid ×8 threads/row.
- `tools/test_kv/test_hq_decode.cu` — NEW decode-kernel oracle gate (7 scenarios).
- `tools/test_kv/test_hq_codec.cu` — group bit-identity, k==0 corpus assert, synthetic k>0
  fallback rows; segment comparison removed.
- `tools/test_kv/bench_hq_kernel.cu` — group-decode rate bench + phase-attribution kernel
  (full / no-decode / no-score-pv at engine shape).
- `tools/test_kv/build-ninja-tk.ps1` — Ninja fast-path wrapper for the standalone suite
  (imports vcvars itself, like configure-ninja.ps1).

## Verified measurements (standalone, tools/test_kv)

- decode-rows(group, full occupancy): ~195–198 M rows/s; encode (fill shape): 57–63 M units/s
- decode kernel (engine shape, grid.y=85): window 54 → 0.034–0.038 ms; 2048 → 0.037–0.043 ms;
  32768 → **0.31–0.37 ms** (baseline 0.896–0.99)
- phase attribution @32k/85: full 0.316–0.381 · no-decode 0.154–0.155 · decode-only 0.178
- run-to-run engine noise is real: pp8192 swung ±6% across identical binaries; decode cells are
  tight (±0.5%). A/B with same-binary reruns, not across rebuilds.

## Build

Fast path (plain PowerShell, no developer prompt; tests and benchmarks ON by default,
`-NoTests`/`-NoBenchmarks` opt out; MSVC and nvcc host compiles run under `/utf-8`):

```bash
powershell -ExecutionPolicy Bypass -File configure-ninja.ps1      # once per build dir
powershell -ExecutionPolicy Bypass -File build-ninja.ps1 [-Target <name>]
```

The Visual Studio generator flow (`build-windows`, `build-live.ps1`) remains supported;
toolchains resolve from PATH/environment only (`VCPKG_ROOT` or vcpkg on PATH, `CUDA_PATH` required).
Standalone suite (no engine build; Ninja fast path, no developer prompt needed):
```bash
cd tools/test_kv
powershell -ExecutionPolicy Bypass -File build-ninja-tk.ps1
./build-ninja/test_hq_codec.exe      # ALL PASSED
./build-ninja/test_hq_prefill.exe    # ALL PASSED
./build-ninja/test_hq_decode.exe     # ALL PASSED (7 scenarios)
./build-ninja/verify_hq_retrieval.exe
./build-ninja/bench_hq_kernel.exe
```
Models: official `models/qwen3_8_27b_nvfp4.ninfer`
(perf runs); local `models/qwen3_8_27b_nvfp4full.ninfer` (smoke). Prompts `longprompt_{32k,128k,262k}.json`.
Engine bench: `./build-ninja/bench/ninfer_bench.exe --weights <artifact> --kv-dtype hq-e8-2b -p ... -pg ... -r 3 --warmup 1 --max-ctx 66560`.
CLI sweep: `./build-ninja/apps/ninfer.exe <artifact> --kv-dtype hq-e8-2b --messages longprompt_32k.json --greedy --max-new 128 --max-context 65536 --raw-output` (weights is positional).

## Session-5 addendum (2026-08-22 evening): 2x2x2 bench campaign + accuracy campaign state

- **Accuracy campaign (GPQA first)**: interrupted at 22/198 reviewed, 86.4% running. Band
  CORRECTED per review S6: the 84.34% figure was Qwen3.6-27B nvfp4, not 3.8 — the Qwen3.8-27B
  comparators are 89.39% (nvfp4full, int8 KV, this exact doc profile temp 0.6/penalty 1.0,
  this box, EvalScope 1.9.0) and 90.40% (official nvfp4, int8 KV, temp 1.0/penalty 0.0, model
  card). Confirmation band is therefore ~89-90% ± ~2 (binomial SE at n=198); at n=22 the
  interim is ±14 and is not evidence either way. Treat a final score < 85% as a signal worth a
  second seed or the paired int8 run. Harness here is EvalScope 1.10.0 (framework-pinned).
  The run is resumable:
  relaunch the hq server (flags in profiles/bench/campaign-20260822.md context / see below),
  then `PYTHONPATH=eval eval/.venv/Scripts/python.exe -m ninfer_eval resume --run
  20260822T093946Z-cae7691f`. Config: eval/configs/qwen3_8_27b_nvfp4_kv_accuracy.yaml
  (per-phase suites gpqa / aime / multimodal, doc-aligned sampling). Per the owner: no int8
  control run — confirmation against the band, not paired comparison.
- **Bench campaign (official ninfer_bench, one idle session)**: nvfp4 x nvfp4full weights,
  int8 x hq-e8-2b KV, MTP0 full matrix + MTP3 decode cells. Full record with GPU telemetry
  (clocks/temp/power/util/mem per run, no throttling): profiles/bench/campaign-20260822.md —
  LOCAL-ONLY (profiles/ is gitignored by design); the headline numbers live here and the raw
  outputs on the bench box.
  Headlines: nvfp4full is the faster profile everywhere (+24-32% prefill, +9-14% MTP0 decode);
  hq/int8 decode ratio 0.81->0.66 (nvfp4) and 0.79->0.52 (nvfp4full) from tg128 to 32k — the
  hq attention kernel is weights-independent, so it dominates more on the faster weights;
  hq prefill 91-98% of int8 on nvfp4, wider and noisier on nvfp4full at >=8k. Session-absolute
  numbers run ~10-14% above the afternoon (cold quiet GPU) — compare within the campaign only.
- **Review S7 REFUTED by probe + code (do not "fix" it)**: `plan.capacity` IS the
  per-sequence ceiling (options.max_context); the pool is `plan.kv_capacity`. The workspace
  envelope was already correctly per-sequence; the scratch is the documented constant ~1.0 GiB
  at 252,928. A c=1 server probe (pool 252,928, runtime 3.73 GiB) vs the eval server (pool
  505,856, runtime 6.41 GiB) confirms the pool marginal is the KV planes themselves:
  16 layers x 576 B/token/layer = 9,216 B/token (2.17 GiB at c=1). The earlier session-5
  memory analysis here wrongly dropped the 16-layer factor (claimed 0.28 GiB KV at 505k and
  1.9 GiB scratch) — the 6.41 GiB runtime is ~4.34 GiB KV planes + ~1.0 GiB scratch + ~1.0 GiB
  fixed (MTP/workspaces/graphs). The same error invalidated the "1M-context nvfp4full+hq fits
  in ~12 GiB" claim: at a 1M envelope the KV planes alone are ~9.5 GiB (plus the 4 GiB
  one-shot scratch WI-2 targets) — the 1M envelope does NOT fit trivially on 32 GB; that is
  what ROADMAP-1m-context.md's work items exist for. A hazard comment now sits at the
  envelope line (capacity vs kv_capacity is an easy misread).

### 5b. Session 7: parallel encoder (lever d) — the LDL/STL check decided it

The pre-build check the review asked for: SASS showed the encoder's `HqBitWriter::buf`
(runtime-indexed u32 array) made the append-input decode kernel and the fill kernel carry
~47k LDL/STL each (cached-input kernel: 16) — the encoder was local-memory-bound, not
chain-bound. With the k=0 invariant the parallel packer collapses to almost nothing: symbols
stay in registers (lane w owns lattice word w), a 5-round shfl prefix-sum of per-lane bit
lengths yields every symbol's start and the row total BEFORE any write (no overflow
rollback), and packing is 8 atomicOr terminator bits per lane into a zeroed 16-word staging
row (a k=0 Rice stream is zeros with 256 one-bits). The k-selection accumulators and the
bit-serial writer are deleted; HqBitReader (sequential decoder) kept. Bit-identical output
by construction — all four gates pass with identical census/oracle numbers.

- SASS: append kernel 47k -> 35 local ops; fill kernel -> 19.
- Fill encode: 61 -> 350 M units/s (5.7x).
- Append-variant decode kernel: 0.129 -> 0.064 ms @window 54.
- META NOTE: `used` now reports the EXACT bit total (e.g. 485) where the old writer
  word-padded to 512 (codes byte-identical; both decoders read either form identically via
  the zero-tail invariant). This dissolves the S9 worst-case caveat: the bench row was always
  a typical ~485-bit row.
- ENGINE (official artifact, same quiet evening as the 2x2 campaign): tg128 53.5 -> **70.7**
  (int8-parity: int8 71.1 same evening), pp8192+tg128 50.2 -> 64.8, pp32768+tg64 41.9 -> 51.8,
  MTP3@8k 147 -> 183.5 (int8 201), CLI decode 32k 42.2 -> 52.0, 262k 22.9 -> 25.3. Prefill up
  ~10% at 64k (fill kernel). The in-engine append cost was far above the single-owner-block
  standalone view; per-step saving at tg128 ~3.5 ms (16 x ~0.22 ms).
- Cumulative hq decode (sessions 2-7): 32k 28.6 -> 52.0 (+82%), 128k 16.9 -> ~35 (UNMEASURED — interpolate from
  the 32k/262k CLI cells until a 128k run with its int8 counterpart), 262k 9.1 -> 25.3 (+178%), tg128 49.9 -> 70.7 (+42%), MTP3@8k -> 183.5.
  hq/int8 same-evening: tg128 0.99, 32k+64 0.77, MTP3 0.91.
- Remaining Tier-1 levers unchanged in priority: (a) TC tile-source kernel (score/PV now
  dominates the decode kernel), (c) occupancy. The 2x2 campaign's nvfp4full numbers predate
  this change; rerun the campaign matrix before publishing any cross-config table.

### 5c. Session 7 campaign rerun (post-parallel-encoder, 2x2 matrix)

Same method as the evening campaign, rerun ~3 h later on the same binaries + the encoder fix.
IMPORTANT: the int8 cells drifted -5-7% between the two campaigns (sustained benching, temp
78->83 C) - read ratios WITHIN this campaign only. Telemetry: no throttling; nvfp4full_hq hit
96% util avg, mem peak 19.96 GiB (lightest config).

| decode cell | nvfp4full hq | nvfp4full int8 | ratio | nvfp4 hq | nvfp4 int8 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| tg128 | **75.6** | 75.2 | **1.005** | 64.1 | 66.3 | 0.97 |
| pp2048+tg128 | 74.3 | 73.7 | 1.008 | 63.8 | 65.8 | 0.97 |
| pp8192+tg128 | 68.6 | 73.1 | 0.94 | 59.8 | 64.8 | 0.92 |
| pp32768+tg64 | 53.7 | 70.5 | 0.76 | 47.9 | 62.3 | 0.77 |
| MTP3 tg128 | 113.0 | 123.1 | 0.92 | 96.1 | 96.6 | 0.99 |
| MTP3 pp2048+128 | 189.3 | 216.6 | 0.87 | 176.7 | 188.5 | 0.94 |
| MTP3 pp32k+64 | 126.1 | 209.6 | 0.60 | 119.2 | 189.4 | 0.63 |

Headline: **nvfp4full+hq is now the joint-best short-context configuration** - tg128 75.6
ties int8 (75.2) and beats nvfp4+int8 (66.3); prefill 10.4k/9.5k tok/s at 2k/8k (92-95% of
nvfp4full int8). The remaining hq gap is concentrated at long context (0.76-0.77 MTP0, 0.60
MTP3 @32k) - exactly lever (a)'s score/PV domain. Full record:
profiles/bench/*_matrix2.txt, *_mtp3v2.txt, gpu_stats_campaign2.csv (local).

### 5d. Session 7 padding + review §16 doc items

Row-stride padding (one element per key row, kKeyStride = 257): breaks the score loop's
32-way bank conflict (all lanes at same d, formerly 512 B stride = one bank). Decode kernel
@32k/grid.y=85: 0.31 -> 0.283 ms (~9%); engine nvfp4full+hq pp32768+tg64 53.7 -> 55.6 (+3.5%),
pp8192+tg128 68.6 -> 69.2. Smem +64 B (under the 97,600 limit).

Review §16 verdict: packer correct and exact; short-context gap closed (tg128 1.005);
S9 retracted (used was word-padded). Remaining gap is score/PV + window decode at long
context (0.60-0.77 at 32k). Re-based estimate for levers (a)+(c): 32k MTP0 ~55.6 -> ~66
vs int8 70.5 (ratio ~0.94). §16.4 doc items resolved (caution 3, stale comments, 128k
label, HostBitWriter naming). §16.5: short-context parity confirmed; INT4-G64 still the
better default at <=524k because the long-context hq gap is the window decode.

### 5e. Session 8: lever (a) — hq decode as a tensor-core tile-source kernel

The hq decode kernel is now a **KV-source policy of the bf16 TC kernel** (REVIEW §4, exactly
as designed). `gqa_attention_decode_bf16.cuh` is templated on `KvSource`
(`GqaTcKVLinear` / `GqaTcKVHq`, defined in `gqa_attention_decode.cuh`): the hq policy
group-decodes each 32-key tile straight into the swizzled `k_s`/`v_s` positions
(`hq_decode_row_group(..., xor_chunk = key row & 7)` — the swizzle is a single XOR on the
element index, `e ^ (xc << 3)`, preserving each lattice word's 8 contiguous outputs), runs
QK/PV on the inherited ldmatrix+mma path in the rotated frame (q rows FWHT-rotated after
staging, output rows un-rotated once before the partial stores, prefill-FA2-style hooks),
and the fused append encodes into the code planes with per-warp scratch aliasing the qkv
tile. The old scalar 256-thread kernel in `gqa_attention_decode_hq.cuh` is deleted; that
header now documents the single runtime-width instantiation
(`<Geometry, 6, 4, /*MultiBatch*/true, /*Masked*/true, CacheInput, GqaTcKVHq>`, 128 threads,
~36.75 KB STATIC smem, `__launch_bounds__(128, 2)` → measured 2 blocks/SM; the 97 KB
dynamic-smem carve-up and its attr are gone). Masks/splits/neutral-partials/reducer are the
TC kernel's own contracts; the graph grid is unchanged so the layouts tiering stands.

Two contract notes landed during debugging:
- **hq writes neutral partials for inactive splits** (`split >= active`), like the old hq
  kernel — the shared reducer may see partials buffers carrying earlier non-zero contents;
  the bf16 policy still returns silently and relies on the zero-initialized engine partial
  workspace. test_hq_decode enforces the hq side (fresh cudaMallocs recycle old contents
  between scenarios; without the neutrals, scenarios D/E/G read garbage).
- `Masked=true` with `valid_columns == nullptr` means unmasked — the hq route shares one
  instantiation; the null branch must NOT subtract column_begin (that bug neutralized
  every split for chunked shapes: scenario D).

**Standalone (tools/test_kv, all four gates pass; test_hq_decode unchanged, 10 scenarios,
min cos 0.999990, max row rel 0.52%):**
- decode kernel @32k/grid.y=85: 0.283 → **0.204 ms** (phase copy: full 0.195, no-decode
  0.072, **decode-only 0.123** = 2.13 G rows/s over 262144 rows/layer, **score/PV 0.032**
  — was ~0.155)
- append-variant kernel @window 54: 0.070 → 0.024-0.027 ms
- decode-rows(group) microbench 192.7 M rows/s (baseline ~199; the first swizzle mapping
  cost 18%, the single-XOR form recovered it)
- grid.y=4/32 sweeps REGRESSED at small grids (w=54: 0.015-0.016 → 0.024): the 128-thread
  block reaches its design point only when 2 blocks co-reside per SM

**Engine (nvfp4full, r=3, quiet GPU, same-session int8 pairs):**

| cell | hq | int8 (same session) | ratio | §5c ratio |
|---|---:|---:|---:|---:|
| MTP0 tg128 | 71.7 | 76.5 | 0.938 | 1.005 |
| MTP0 pp8192+tg128 | 65.1 | 73.1 | 0.891 | 0.94 |
| MTP0 pp32768+tg64 | **58.05** | 69.6 | **0.834** | 0.76 |
| MTP3 pp2048+128 | 208.1 | 218.6 | 0.952 | 0.87 |
| MTP3 pp32k+64 | **177.9** | 208.8 | **0.852** | 0.60 |

Long context (lever (a)'s target domain) and MTP3 improved strongly (32k MTP0 55.6 → 58.05;
MTP3 @32k 126.1 → 177.9, +41%). **Short context gave back ~6%** (tg128 1.005 → 0.938):
at tg128 the grid is 4 heads × 4 splits = 16 blocks on 170 SMs — every block runs alone,
so the 128-thread kernel has half the old 256-thread kernel's decode parallelism exactly
where co-residency never happens (the standalone small-grid regression is the same effect).
The bf16 route passes its greedy-32k smoke on the official Desktop artifact.

**Gate verdict (§14.3/§16.5):** decode-only is 0.123 ms ≈ 2.13 G rows/s, above the ≤0.10 ms
/ ≥3 G rows/s post-(a)+(c) band edge (the review's (a)-alone estimate ~0.19 for the whole
kernel was hit exactly; (c)'s 2 blocks/SM came free with the static-smem rewrite). Per the
review's rule the design floor is nearly confirmed: after lever (b) below, the next lever is
Tier 2 row-skipping (§14.4), not more homogeneous kernel work.

**Review §17 (post-landed) received and addressed:** verdict is "landed as designed and
correct" (swizzle plumbing, group geometry, rotated-frame hooks, and contracts all checked
against the diff). §17.2's re-read of the numbers: old and new kernels both run 8 warps/SM,
so the unchanged decode-only 0.122/0.123 ms means the decode phase is bound by dependent
8-lane chains in flight (latency), not ALU issue — (c)'s assumed decode-phase halving never
existed, 0.834 is what (a) alone buys, and the tg128 give-back is the same effect at 4
warps/SM. §17.4 items closed this session: `ninfer_gqa_attention_test` (A1/A3 oracle,
bf16+i8 routes) and `ninfer_bidirectional_gqa_attention_test` both PASS (BUILD_TESTING=ON
now set on build-ninja; the greedy-32k smoke was not the only bf16 evidence anymore); the
bf16 launcher now documents the Masked⇒non-null valid_columns invariant at its dispatch.
The phase-attribution copy in bench_hq_kernel remains a bench-local copy by choice — folding
phase switches into the production template would put bench knobs in the product path;
revisit only if it drifts again.

**Lever (b) MEASURED AND REFUTED (session 8, post-review-17).** Two prototypes were built in
the bench-local phase kernel (production kernel untouched) and measured at 32k/grid.y=85
against V0 = the production kernel (full 0.190 ms, decode-only 0.121):
- **V1** — 256 threads, all warps run the decode bursts, warps 0-3 own the mma path: full
  0.226-0.321 ms across runs (no-decode 0.084). Worse than V0 at every run.
- **V2** — true warp specialization: warps 0-3 consume tiles, warps 4-7 produce them one tile
  ahead into a double-buffered K/V pair (named-barrier handshakes, 68.9 KB dynamic smem,
  256 threads): full 0.285-0.474 ms, unstable across runs (no-decode 0.079-0.084 confirms
  the pipeline mechanics are sound — the producer decode itself is what slows).
Neither fixes the small-grid regime either (200/4 window: v1 0.039, v2 0.032 vs production
~0.024). Root cause: the mma warps' acc fragments (128 f32 registers) cap every 8-warp
variant at 1 block/SM, so the review's 16 warps/SM premise is unreachable — 4 dedicated
decode warps continuous (16 rows in flight) never beat V0's all-warp bursts at 2 blocks/SM
(32 rows at ~60% duty). An unroll experiment inverted en route: the compiler's partial
unrolling of the decode slot loop provides cross-iteration ILP the latency-bound chains
need — `#pragma unroll 1` on the prototypes cost 30-40%, and explicit full-unroll on V0
changed nothing (already unrolled).

**Gate now CLOSED per §17.3's own rule: decode-only stays ~0.12 ms = 2.1 G rows/s; the
chains are the limit of the homogeneous TC route. Tier 2 (exact V skipping, REVIEW §14.4)
is the next decode lever.** The small-window split tier (more hq splits below ~4k to grow
the 16-block tg128 grid) remains the only measured-path fallback for the tg128 give-back
and touches the shared host/device split policy — separate focused work if taken.

### 5f. Session 9 (2026-08-23): extraction, squash rebuild, review fold-in

- All hq production work from sessions 2–8 cherry-picked onto `feat/hyperquant` (17 commits on
  `d039a02a` → tip `11bb3ff3`): the decode-kernel correctness fixes, the cooperative group
  decoder, the escalation-rescue fix, the parallel unary packer, the key-row padding, the TC
  tile-source kernel, the warp-spec refutation bench, the `tools/test_kv` suite, the envelope
  semantics comment, and the kv-dtype accuracy campaign config. HANDOFF-only hunks dropped; the
  branch is PR-ready pending items 3–4 below.
- `cometkim/dev` rebuilt per the fork convention (fork base → squash(build-speed) →
  squash(qwen3.8-nvfp4full) → squash(hyperquant) → one fork-docs commit); the tree was verified
  identical to the pre-rebuild tip `f1178757`. Both branches pushed to origin.
- `AUDIT-hyperquant.md` and `REVIEW-hyperquant.md` deleted; live content folded into this file
  (Tier 2 and page-selection designs in Remaining work 1; small open items in 3–5). Final review
  verdicts for the record: lever (a) landed as designed and correct; lever (b) refuted by
  measurement; the homogeneous-TC decode floor is the dependent 8-lane chains at 8 warps/SM —
  2.13 G rows/s ≈ 10% of the ALU issue-peak estimate, i.e. latency-bound, not ALU-bound — so
  the decode-phase gate is closed and Tier 2 is the only path to ≥3 G rows/s.

### 5g. Session 10 (2026-08-23): 1M-context M1 landed (WI-1 envelope + WI-3 YaRN)

**M1 is verified end-to-end**: `--kv-dtype hq-e8-2b --max-context 524288 --kv-capacity auto
--rope-scaling yarn:2` boots, prefills (906 tok/s on the smoke prompt), decodes (76 tok/s
tiny-context), and MTP3 verifies (170 tok/s, 21/30 accepted, "Paris"). The 1M envelope also
boots with the one-shot scratch (`--kv-capacity 1048576 --rope-scaling yarn:4`: 569 MiB free,
0 headroom — WI-2 is still required for auto-capacity and margin, but it is not a boot blocker).
A REAL ≥512k-token prefill has not been run yet (no fixture).

- **WI-1** (`ace0eae0`): `kNativeContext` 262144 → 1048576; the GQA op bounds envelopes per
  dtype — U8 reaches the new absolute ceiling (block-table addressing + linear scratch, no
  staged page tables), BF16/I8 stay at `kGqaAttentionMaximumLinearVisibleKeys = 262144` (64
  staged page ids per split). test_hq_decode scenario J (window 262,208) and the per-dtype
  ceiling contract in `test_gqa_attention` pin it. int32 audit clean: scratch plane ne max
  1.07e9, block tables 16,384 pages, splits still clamp at 85.
- **WI-3** (`0d33db2d`): `ops::rope` takes a `RopeFrequencies` table (128 double frequencies +
  attention factor multiplying cos and sin) instead of theta; host builders
  `rope_linear_frequencies`/`rope_vision_frequencies`; the family plan resolves
  `--rope-scaling yarn:F` (CLI + serve) via `rope_yarn_frequencies` (HF ramp 14/22 at dim 64,
  base 1e7, original 262144; attention factor 0.1·ln F + 1) and threads it Program →
  ExecutionCore → TextContext into all Text/MTP rope sites. Vision/DFlash keep linear tables;
  DFlash rejects scaling. Angle profiles: factor 1 keeps the legacy FP32 product (flag-off
  bit-stability; linear tables bit-match the baked constants — pinned); scaled factors compute
  and 2π-reduce in FP64 (the FP32 product loses ~0.03 rad on low pairs at 1M).
  Tests: `ninfer_rope_test` (oracle consumes the table + factor; YaRN-shaped cases at
  1,048,575) and the new `ninfer_qwen3_6_27b_rope_scaling_test` (factor-4/2 tables bit-exact vs
  independently computed references = ROADMAP §6; legacy text/vision tables bit-exact; dflash
  reference corrects one ~2e-13-low legacy constant).
- **Build note**: build-ninja was reconfigured WITHOUT `-TuLog` (the deleted tu_wrap.py launch
  wrapper had poisoned the cache); BUILD_TESTING=ON preserved.
- **Debug record**: engine segfaulted at any context because five POSITIONAL inline
  `ExecutionCore{...}` aggregates in `program_impl.h` (lines ~1500/1531/1789/1920/2081) were
  missed when the rope pointer was added — positional aggregate init silently nulls trailing
  members. All five now pass `&rope_frequencies`. When adding an ExecutionCore field, grep for
  `{device, model, work,` — the lambda is not the only construction site.
- **Long-context sanity (2026-08-23, greedy single-needle probes, nvfp4full + hq)**: speed at
  390k tokens / yarn:2 — prefill 1330–1348 tok/s (the session-4 fit predicts ~1,590 at 390k:
  ~15% UNDER, the quadratic term is ~23% larger; REVIEW R5 re-fits 1M to ~31 min), decode
  20–21.5 tok/s (≈0.07 ms/1k keys ⇒ 1M ≈ ~10.5 tok/s — genuinely ahead of the session-4
  projection of ~8). Retrieval ladder (needle at depth, magic code):
  32k factor-1 ✓ clean; 32k yarn:4 ✓ clean (no static-YaRN short regression here); 128k yarn:2
  ✓ clean; 202k yarn:2 ✓ clean; **390k yarn:2 FRAGILE** — depth 20% ✓ clean, depth 80% ✓ noisy
  prefix + code, depth 50% ✗ immediate `<|im_end|>`; 390k yarn:4 also failed that probe; 390k
  UNSCALED (OOD control) partially retrieved. Read: dense YaRN on this checkpoint is clean
  through ~200k true tokens (~0.77x native) and degrades by ~1.5x native; the >262k mechanical
  path is fine (the unscaled control retrieves through the same prefill). Probe fixtures:
  `longprompt_{32k,128k,262k,500k}_needle.json` (generator: tiled word-tangle, needle record
  with a magic code, seeds 20260823/4242, depth 0.5; d20/d80 variants regenerable).
- **Engine robustness fix (found by the 524k gate)**: the first `lbv2_medium` run died at
  sample 10 — the model emitted a byte-fallback token with an invalid UTF-8 leading byte, the
  frontend THREW from `feed_token_bytes`, the exception escaped the round loop, and
  `fail_all` permanently downed the engine (all later requests 503). Fixed: the generated
  stream's UTF-8 handling is now lossy — one invalid byte publishes U+FFFD and resynchronizes
  (`terminalize` already used that replacement for budget-cut codepoints); incomplete
  multi-byte tails still hold for the next batch. Rerun: **15/15 samples, zero server errors**,
  including the previously fatal sample. Remaining open engine question (recorded, not
  addressed): per-request output errors should fail the REQUEST, not reach `fail_all` at all.
- **First 524k real-document gate (`lbv2_medium`, limit 15, yarn:2 + hq)**: accuracy 0.40
  (6/15; n=15 ⇒ SE ~±0.13) on LongBench-v2 medium (real docs, up to ~500k tokens). No
  collapse — consistent with the single-probe picture (fine through ~200k, marginal past) but
  not conclusive; the native-range control (`lbv2_short`, 180 samples) and the full medium run
  are the attribution cells. Run record: `eval/runs/20260822T180902Z-*` (local).
- **LongBench v2 smoke (end-to-end, 2026-08-23)**: the `lbv2_short_native` suite ran one real
  sample against the hq 262k server — dataset auto-downloaded (ZhipuAI/LongBench-v2, no local
  prerequisites), 38,468-token document, rule-scored `ANSWER:` extraction clean; the model
  demonstrably read the document (cited its section/pages/quotes) and answered A vs gold D.
  Caution: `ninfer-serve.exe` had gone stale while only `ninfer` was rebuilt after the
  ExecutionCore fix — after any engine-construction change, rebuild BOTH apps before serving.
- **Review R1/R2/R7 addressed (`7f49a4d5`)**: max-context beyond the linear envelope rejects at
  option validation for bf16/int8 with a fix-naming message; static-YaRN-below-native and
  unscaled-past-native carry an operator note through MemorySummary into the CLI summary and
  serve request log (with the resolved factor); `ExecutionCore::rope_frequencies` is now a
  reference member so an omitted positional aggregate fails to compile (the session-10 segfault
  class); README states the per-dtype envelope. Verified: int8+524288 rejects, both notes print,
  rope/gqa/builder tests green.
- **R6 ablations (2026-08-23)**: R6.2 as written (int8 KV at 390k) is not runnable — the linear
  envelope R1 pins caps bf16/int8 at 262,144 keys (the 64-page-id staging is a real kernel
  limit: 390k/85 splits ≈ 72 pages > 64). Closest runnable discriminators, both executed:
  (A1) 390k d50 with `yarn:1.5` (factor tracks length, effective ~260k): FAILS with corrupted
  fragments ("72PU-AL", "58virtual") — not a clean factor-policy miss; (A2) true-262,070-token
  needles, hq vs int8, depths 50%/80%: ALL FOUR cells retrieve the exact code cleanly — **the
  hq codec is exonerated through its full native envelope**, and the 390k fragility is a
  strictly-beyond-262k phenomenon (every config there degraded somehow: yarn 1.5/2/4 failed d50
  differently, unscaled partially retrieved, yarn:2 was clean at d20/d80 — single-probe noise
  dominates; the multi-sample suites decide).
- **Long-context eval suite added**: `eval/configs/qwen3_8_27b_long_context.yaml` (EvalScope
  needle_haystack, rule judge, thinking off): native int8/hq 8k–128k baselines, 524k yarn:2,
  1M yarn:4, and the 32k yarn:4 short-prompt regression. LOCAL PREREQUISITES: the ModelScope
  needle corpus `local_path` and the exact HF tokenizer `tokenizer_path` must be pointed at
  real local directories before the first run.

### 5i. Session 11 close: review N1 re-applied, attribution cells run

- **N1 (re-applied after the reviewer's force-push regressed it)**: the generated-stream
  U+FFFD replacement literal is the EF BF BD hex escape (raw bytes in the source); the
  force-pushed intermediate had the double-encoded mojibake back in frontend.cpp and had
  dropped the test - both restored here. `test_utf8_replacement_bytes` feeds an invalid
  0xFF lead byte and asserts exactly one replacement character then continued generation
  (the split-codepoint hold case was already covered). The frontend TU now COMPILES on this
  box (the CP949 C2001 blocker is gone under the `/utf-8` build flag) but cannot EXECUTE
  here: `official_tokenizer()` and the template fixtures read the upstream maintainer's
  absolute Linux paths (`/home/neroued/models/...`), so `read_file` throws and the binary
  aborts (0xC0000409, cdb-confirmed unhandled C++ EH) before any test runs. Execution
  remains Linux-fixture-verified. Editing lesson, now three-for-three: non-ASCII byte
  literals go through the Edit tool as hex escapes - the Git Bash heredoc mangled every
  escape attempt.
- **Attribution cells - codec cleared on real documents**: `lbv2_short` at native 262k, the
  same 60 questions: int8 0.40 vs hq 0.3833 (a one-question difference). With the needle A2
  result, the hq-e8-2b codec is indistinguishable from int8 at its full native envelope on
  synthetic retrieval and LongBench v2; lbv2_medium 524k yarn:2 sits at 0.40 (n=15) - no
  real-document collapse past native. Remaining cells: full lbv2_medium (215) and
  lbv2_long at 1M, plus envelope-failed count reporting (review N5). Run records under
  eval/runs (local).
- **Queued for next session** (roadmap WI-3b/WI-7 specify them): ~~the q-side factor move and
  the yarn temperature parser/grid, and the WI-7 MTP gate~~ — landed and measured in §5k.
  Remaining: the N2 request-scoped error boundary, the bf/bs ramp grid only if a new signal
  justifies it, and the WI-7 (ii) DFlash acceptance trace on llama.cpp.

### 5j. Session 12 (2026-08-23): /utf-8 charset flags, YaRN ablation cell, build docs

- **`/utf-8` build flags landed** (MSVC `C,CXX` + nvcc host `-Xcompiler=/utf-8`), after
  validating that every one of the 15 non-ASCII sources is valid UTF-8. This lifts the
  frontend TU's long-standing CP949 C2001 compile blocker on this box. The TU now compiles
  and links; executing it here still aborts (0xC0000409) because `official_tokenizer()`
  and the template fixtures read the upstream maintainer's absolute `/home/neroued/...`
  paths — execution stays Linux-fixture-verified. Includes the N1 escape-form literal
  (the pulled `minor review` commit had regressed it back to raw EF BF BD bytes).
- **configure-ninja quoting bug found and fixed**: unquoted `-DFLAG=$var` in a
  native-command invocation with backtick continuations passes the LITERAL string under
  Windows PowerShell (reproduced with `& cmd.exe /c echo`) — `$bench` had always been
  passed literally and CMake truthy-coerced any non-false string, so `-NoBenchmarks`
  never actually worked. All `-D` arguments are quoted now; the cache verifies
  `BUILD_TESTING=ON` / `NINFER_BUILD_BENCHMARKS=ON` genuinely. Tests are ON by default
  (`-NoTests` opts out); the dead `-TuLog` plumbing (untracked `tools/tu_wrap.py`) is
  removed from both wrapper scripts.
- **README build section is fork-first now**: the Ninja wrapper pair leads (what it
  enables, machine fallbacks, `/utf-8`), the plain Visual Studio generator flow follows;
  clone URLs point at `cometkim/ninfer`; an accidentally duplicated intro/model-table
  block was removed. The stale hand-export VS flow in this file's Build section was
  replaced with the wrapper flow.
- **YaRN ablation (`lbv2_medium`, same first-15 medium samples, hq KV)**: native 262k
  unscaled = 0.3333 (5/15) vs 524k `yarn:2` = 0.40 (6/15). Sample 9 (425,593 tokens) is
  0 in both cells — envelope-rejected at native (input_tokens 0), prefilled-but-immediate-
  EOS at yarn. Paired-14: native 5/14 = 0.357 vs yarn 6/14 = 0.429; the five shared
  correct samples are identical (0,3,4,6,8) and the single differential is sample 11
  (many-shot ICL: unscaled exhausted the 1024-token budget mid-reasoning, yarn answered
  correctly). **No measurable YaRN penalty on real-document MCQ at 62k–242k-token prompts;
  n=14 cannot resolve better than ~±20pp, so the honest claim is "indistinguishable",
  not "yarn wins".** Suite `lbv2_medium_native` (limit 15) added to the long-context
  config with the pairing note. Run records: `20260822T203443Z-17dc527d` (native),
  `20260822T180902Z-5840026a` (yarn) (local).

### 5k. Session 13 (2026-08-23): WI-3b q-side YaRN temperature + tunable ramp (i + ii)

- **q-side factor move landed**: `RopeFrequencies.attention_factor` is now a q-side temperature —
  the rope kernel scales the rotated dims of every q row by its square and leaves k rows
  unscaled, so cached K is factor-free (`(f·q)·(f·k) == (f²·q)·k`; same numerics up to one bf16
  rounding; factor 1 stays a bit-exact no-op, flag-off bit-stability preserved). The
  single-tensor rope form gained an explicit `RopeSide` — its real callers are BOTH sides
  (MTP k append `text_context_impl.h` → Key, MTP draft q → Query, DFlash context k → Key), so a
  one-sided default would have double-scaled MTP attention. Oracle updated (`test_rope` q/k
  sides + a scaled single-q case at 1M positions); `test_rope_scaling` covers the temperature
  and ramp parameterization plus rejections. Both green.
- **Parser**: `--rope-scaling yarn:F[,t=c][,bf=n][,bs=n]` (CLI + serve; `RopeScalingSpec` in
  `product/speculative_options.h`; from_chars, duplicate/unknown fields rejected). Builder
  `rope_yarn_frequencies(theta, dim, original, factor, temperature=0.1, beta_fast=32,
  beta_slow=1)` with ramp clamps (review R4). Threaded through EngineOptions → layouts →
  Program → MemorySummary (request log + CLI summary show non-default knobs). docs/cli.md
  updated. NOTE: the t/bf/bs knobs only take effect with factor > 1 (validated engine-side).
- **Enabling fixes found because the default build was actually exercised**: `bench/ops/
  rope_bench.cu` had been unbuildable since session 10 (theta-float rope API gone, and the
  non-template `rope_generic_kernel` in rope.cuh ODR-collided between the bench and launcher
  TUs — now `inline`); two of the never-compiled-on-MSVC test TUs fixed (`std::aligned_alloc`
  → `_aligned_malloc` shim in test_gdn_replay_records; missing `ninfer::` qualification in
  test_openai_schema from upstream ca0763ba). REMAINING pre-existing MSVC-portability failures
  in upstream test TUs (not this change): constexpr `std::sqrt` in constant expressions
  (test_gdn_replay_fold ×2, test_vision_attention, test_gated_delta_net_replay_record) —
  GCC allows it, MSVC does not. Targeted builds remain the verified path on this box.
- **Measurements (2026-08-23 evening, nvfp4full + hq, greedy, exclusive GPU)**. Fixture note:
  the `longprompt_*` labels are NOMINAL (char-ratio); actual tokenizer counts: "32k"=24,732,
  "262k"=202,514, "390k"=304,222 (new file: the 500k tangle truncated; needle at ~64% depth),
  "500k"=390,033.
  - **Temperature grid** (needle retrieval, yarn:2): 304k tokens — t ∈ {0.1, 0.15, 0.2, 0.25}
    ALL retrieve the exact code (prefill 1622 tok/s, decode 24.8 tok/s); 390k tokens — t=0.1
    AND t=0.25 both garble then im_end ("7/", " u 薪nok"). 202k: t=0.1 and t=0.25 both clean;
    24.7k yarn:4 clean. **Verdict: the q-side port is non-regressive everywhere tested, and
    the ~1.5x-native garble is NOT a temperature problem in 0.1–0.25 — do not spend more sweep
    budget there; the 304k cell is a new clean record for the >native regime.**
  - **MTP gates (WI-7 i)**: 202k tokens — d2 2.26 tok/round (63.0% accept, 60.5 tok/s), d3
    **2.58** (52.8%, 63.5), d4 2.66 (41.4%, 62.4), d5 2.75 (35.0%, 59.3) — d3 is the
    throughput optimum; 304k d3 — **2.71 tok/round (57.1%, 52.0 tok/s over 63 rounds)**,
    i.e. acceptance does NOT decay from 202k to 304k; 390k d3 — garbled regime (1.0 tok/round,
    0% accept), matching the non-MTP garble there. **Read: MTP acceptance tracks output
    coherence, not raw context; the deep-context MTP multiplier stays available wherever
    generation is coherent.**
  - **Engine bug found and fixed**: MTP + any prompt whose final visible keys exceed 262,144
    failed engine construction post-WI-2 ("CUDA Graph preparation consumed 427-437 MiB,
    exceeding the planned allowance of 82 MiB") — the big-visible MTP graph tier was
    calibrated pre-banding at <=262k. Tier ladder now 12/82/512/1024 MiB at
    4k/262k/524k/1M visible keys (512 = measured 427-437 at ~304k + margin; 1024 is an
    unmeasured extrapolation). Verified: 304k and 390k MTP boots + full runs.
  - Left deliberately: the bf/bs ramp grid (no temperature signal at the failing length makes
    it low-priority), review R3's explicit precision-profile flag.

### 5l. Session 14 (2026-08-23): review §6 answered — Q1 attribution, Q4 boot, Q5 local tests

Review `REVIEW-1m-context.md` §6 (commit `37ed005b`) addressed in its own order:

- **Q5 CLOSED — the frontend suite runs and passes on this box** (26 tests, rc=0, first time
  anywhere outside the reference box). Three blockers fell on the way: (1) `NINFER_TEST_FIXTURES`
  now redirects the tokenizer trio (extracted from the local 3.8 artifact with the new
  `tools/artifact/extract_frontend_fixtures.py`; landed in the natpate fork's `hf/Qwen3.8-27B`;
  the needle suites now carry their own copy under `eval/data/`); (2) **the committed replacement test could never
  have run anywhere** — its `std::string("\xff", 1)` vocab entry cannot survive the JSON
  dump/parse round-trip (nlohmann validates UTF-8 both ways); the production mechanism is the
  byte-level alphabet (tokenizer always decodes vocab chars through the GPT-2 byte table), so
  the vocab entry is `"ÿ"` (U+00FF -> raw 0xFF) and the test drives the full public API; (3)
  **the chat-template fixtures hashed wrong under `core.autocrlf=true`** — the pinned digests
  are for LF bytes; `.gitattributes` now marks `tests/fixtures/frontend/*.jinja` `-text` and
  the files re-checkout byte-exact. The runner also gained a diagnostic `catch` in `main`
  (unhandled fixture throws used to surface as a bare 0xC0000409).
- **Q1a DECISIVE — the >262k garble is the hq codec, not YaRN.** With the reviewer's one-off
  patch (linear ceiling 262144→524288 + `PageIds` 64→128 in the bf16/i8 decode kernels,
  reverted after the run; 3 lines, recorded here), **int8 KV at 390,033 tokens retrieves the
  needle exactly** (`--kv-dtype int8 --max-context 393216 --kv-capacity 393216`, greedy) while
  hq at the same prompt garbles under every temperature and sampling mode. Every earlier 390k
  failure (yarn 1.5/2/4, unscaled, sampled) ran on hq. This inverts the "codec exonerated,
  YaRN/model limits" reading: the codec is clean THROUGH native (A2, lbv2) and becomes the
  binding constraint past it — per-vector bias compounding over the long window, the paper's
  Table 6 mechanism. Next accuracy lever: the BF16 sink/recent residual window (R6.2), or a
  product decision to raise the bf16/i8 envelope (the experiment patch is the whole change;
  it also needs the R1 ceiling-contract test and docs updated). Side-datum: int8 decode
  measured 49.3 tok/s at 390k vs hq ~20.6 — the hq long-window decode gap widens past native,
  reinforcing Tier 2 as the decode priority.
- **Q1c — not greedy collapse**: temperature 0.6 / top-p 0.95 / seed 42 at 390k garbles the
  same way greedy does.
- **Q4 — MTP3 boots at the 1M envelope**: 592,518-token prompt under
  `--max-context 1048576 --rope-scaling yarn:4` constructs, prefills, and decodes; the
  >524288-visible graph class used 256 KiB against the 1 GiB tier (conservative but safe —
  no construction failure). Fixture `longprompt_592k_needle.json` (500k+262k tangles
  concatenated). Output garbles there (hq codec, consistent with Q1a).
- **Q1b skipped**: no groupwise-int artifact on this box (17 GB download out of scope for
  this session); still the cleanest weight-quant control when one is available.
- Builder comment nit (§6.2): the HF-vs-ours `high`-clamp divergence for exotic beta_slow is
  now documented at the builder.

### 5m. Session 15 (2026-08-23): review §7 pulled; WI-8 design frozen at the kernel seams

Review `114008e5` (§7 + ROADMAP WI-8) absorbed: the attribution is confirmed (Q1a), the
int8-envelope product option is recorded as parallel, and WI-8 lever 1 (BF16 sink + recent
residual window) is the M2 gate. Vision-at-1M was probed in passing: vision costs ~1.3 GiB
all-in (0.27 weights + ~1 GiB planned workspace) and the 1M hq envelope still resolves the
full pool with 1.93 GiB free — vision is NOT a misfit at any hq envelope.

**WI-8 design, verified against the current kernel source (insertion points exact):**

- Constants (public, `include/ninfer/ops/gqa_attention.h`): `kGqaHqSinkKeys = 32`,
  `kGqaHqRecentKeys = 128` — BOTH must be 32-aligned (tile size) so no residual/hq tile
  straddles a boundary; 32 not 8 for the sink to keep tiles whole (24 extra exact rows are
  free). W=128 keeps 4 full tiles; ring slot = `key & 127` (power of two).
- Storage: per (sequence slot, layer) two planes `[S+W][KVHeads][256]` bf16 — **K stored
  RHT-ROTATED** (the TC kernel runs QK in the rotated frame; residual rows must match, and
  rotated storage also matches the prefill scratch frame), V unrotated. ~64 KiB/token/layer ⇒
  (S+W) ≈ 8.7 MiB per slot per layer-family — megabytes; persistent layout ownership, one
  plane pair per sequence slot. View seam: add `Tensor residual_k/residual_v` to
  `PagedKVLayerView`/`PagedKVBatchLayerView` (`src/core/paged_kv_cache.h`); EMPTY tensor =
  feature off (bit-compatible default; the wrapper validates shape/dtype when non-empty).
- **Decode tile fetch** (`gqa_attention_decode_bf16.cuh` ~302-331): inside the hq branch's
  per-slot loop, when `key < S` (sink row `key`) or `key >= window - W` (ring row
  `key & (W-1)`), replace `hq_decode_row_group(...)` with an exact bf16 row copy using the
  else-branch's store pattern: lane8 owns row elements `[lane8*32, lane8*32+32)`, chunks
  `((lane8*4+j) ^ (key_l & 7)) << 3`, one 16 B `store_vec(load_vec(side_row + lane8*32 + j*8))`
  per j — ~10 lines, same smem swizzle, zero kernel-structure change. MultiBatch: side base
  `+ batch * (S+W) * KVHeads * 256`.
- **Decode fused append** (~184-212): alongside `hq_encode_row_warp`, dual-write the exact
  row — V straight from `input.v`; K rotated first (`hq_fwht256_sign` on 8 registers, the
  same single rounding as the encode staging), then bf16 store to the ring slot. ~20 lines.
- **Prefill**: (a) A2 append (`gqa_kv_append` hq route) dual-writes the same way — the ring
  fills from each chunk's fresh bf16 K/V, exactly as §7.2 specifies; (b) the scratch kernel
  (`gqa_attention_prefill_hq.cuh`) gets the same third source at its row decode — residual
  rows copy from the side planes instead of codec-decoding, so FA2 attention over sink+recent
  is exact in the prompt phase too (the 390k failure shows up in prompt-phase attention, not
  just decode; the window must cover both phases).
- Gates: `test_hq_decode`/`test_hq_prefill` scenarios with residual planes populated (oracle
  uses exact rows for those positions); then 390k/592k probes, 304k non-regression,
  `lbv2_medium` at 524k. If the window is insufficient: subtractive dither on K (WI-8 lever 2).

Not started: the implementation itself. The seams above were read and verified line-level;
nothing in the tree implements them yet.

### 5n. Session 16 (2026-08-23 night): WI-8 implemented — residual window + dither; the >262k garble is fixed

Lever 1 AND lever 2 of the frozen §5m design, plus the int8 envelope raise, all landed in one
session and verified end-to-end on a clean rebuild.

**Design deltas from the frozen §5m notes (decided from the code, all three forced):**
- **V is stored ROTATED, like K** (the frozen note said "V unrotated" — inconsistent with its own
  pure-copy fetch spec: both consumers run QK AND PV over rotated rows and un-rotate only the
  output, so a pure-copy side row must be rotated for both roles).
- **W = 512, not 128** (32-aligned power of two; memory is not the constraint — 38 MiB/slot/pool;
  exact-row fetches are cheaper than Rice decode, so decode got FASTER).
- **A per-slot ring-validity bitmap exists** (kGqaHqRecentKeys/32 U32 words per (pool, slot),
  layer-shared): ring slots hold "the row the next fetch will name", which backward frontier
  moves break. Maintained: append dual-writes set bits; full-reset admission clears (revalidate
  formula: keep slot r iff its last-written key lies in [base−ring, base)); MTP rollback clears
  the rejected drafts' slots; retained-bundle claims keep the lane (bind_row = lane), so planes
  follow ownership with NO copy. Codec planes stay complete — every side row can fall back.
- **Chunk-internal ring race guard**: a fill chunk wider than the ring contains key pairs
  congruent mod W mapping to the same slot; a warp whose key is superseded within the chunk
  (position + ring < chunk end) skips the dual-write — the later warp owns the slot at the
  chunk-end window and no fetch names the skipped key afterwards. Decode append (width ≤ 6)
  cannot race.
- **Subtractive dither (lever 2), half-cell scale**: d ∈ [-0.5, 0.5)^8 per (head, position, role,
  word) coordinate from a u64 counter hash, subtracted before the E8 nearest-point, added back at
  decode; escalation-consistent (absolute d, decode multiplies (y+d)·2^esc). FULL-cell dither
  [-1,1) was tried and is NET-WORSE: escalation 325→3793 rows (17%), pooled cosine 0.936→0.910 —
  the fixed 512-bit budget cannot afford the wider dynamic range; half-cell keeps SNR 55.4 dB,
  cosine 0.936, escalation ~1.5% (reband 150–600), and HALVES the prefill oracle row error
  (max row rel 0.83%→0.45%).

**Verification (all on a clean rebuild after the pointer-args crash was fixed — see cautions):**
- Standalone gates: test_hq_codec (oracle bit-exact incl. dither mirrors, k==0, 0 fallbacks),
  test_hq_prefill (8 scenarios incl. 2 residual), test_hq_decode (14 scenarios incl. 4 residual:
  sink+recent exact, append dual-write, cleared-bit fallback, 2-slot batch), verify_hq_retrieval
  — ALL PASSED at the final tree.
- Engine probes: the table at the top of this file. The 390k cell (garbled under every factor,
  temperature, and sampling mode since session 10) retrieves verbatim; 592k (Q4's garbled cell)
  retrieves verbatim at yarn:4 with the full 1M auto pool; MTP3 at 390k exercises the
  rollback-invalidate path (46/64 accepted, exact retrieval).
- Perf: tg128 80.56 ± 0.14, pp32768+tg64 decode 62.18 ± 0.12 / prefill 7189 (session 8: 71.7 /
  58.05) — the residual fetch out-paces Rice decode; dither costs nothing measurable.
- int8 envelope 262144→524288 (PageIds 64→128 in the bf16/i8 decode kernels, contract test uses
  the constant symbolically): int8 390k retrieves exactly (Q1a made permanent). Fits ceiling at
  nvfp4full: ~430k keys (524,288 × int8 = 17.1 GiB does NOT fit next to 16 GiB weights).
- `ninfer_gqa_attention_test` builds and PASSES on this box again (targeted build; the four
  pre-existing MSVC-constexpr TUs still block the FULL test build). Two findings on the way: its
  U8-scratch check was STALE since WI-2 (expected the one-shot 2 GiB planes; updated to the
  banded contract), and the bf16/i8 PageIds change is covered by it (A1/A3 oracles at the raised
  envelope). `ninfer_bidirectional_gqa_attention_test` and `ninfer_rope_test` also PASS.

**Files**: `hq_codec.cuh` (dither + side-plane helpers), `gqa_attention_decode.cuh` (GqaTcKVHq
fields), `gqa_attention_decode_bf16.cuh` (tile third source + append dual-write + bit set),
`gqa_attention_prefill_hq.cuh` (fill dual-write + race guard; scratch third source),
`gqa_attention_prefill_common.cuh` (metadata residual_slot), routes, wrapper validation,
`core/paged_kv_cache.h` (view fields), `core/kv_ring_bits.{h,cu}` (NEW: by-value mask apply),
targets decoder_state.{h,cpp} (storage/plan/slot-sliced views/revalidate/invalidate),
program_impl.h (admission revalidate + MTP rollback invalidate), CMake (new TU), docs
(paged-kv-cache.md §12.2, README/cli.md/model.md envelope), tests (6 new scenarios + dither
mirrors + hq_dither_host.h), bench/verify seed plumbing. Fixture `longprompt_500k_needle_d20.json`
(untracked, diagnostic).

### 5o. Session 16 follow-up (same night): REVIEW §8 pulled and addressed

`origin/cometkim/dev` moved by `5682111b "follow ups"` — REVIEW-1m-context.md §8, a design review
of the frozen WI-8 note with D1/D2/D3, one expectation, one nit. Pulled (ROADMAP conflict
resolved: landed-status + review disposition in one section). Disposition:

- **D1 (ring validity under prefix reuse / reactivation / slot reassignment): satisfied by the
  landed bitmap.** Retained bundles keep their lane (`bind_row = sequence.lane`), so planes
  follow ownership with no copy; adoption runs the exact revalidate; a released-then-reused slot
  is a full reset (all bits clear); MTP verify rewinds are "fine by construction" exactly as the
  review predicted (rejected positions are re-appended) plus the explicit invalidate.
- **D3 (MTP layer's own pool): satisfied** — `plan_cache` gives BOTH pools their own side planes.
- **D2 (prompt-phase exactness needs the fresh chunk): REAL GAP, fixed.** The ring holds the
  chunk's last W rows, so early-chunk prefill queries saw none of their own recent window exact.
  Fix: `gqa_attention_prefill_fresh_rotate_kernel` — one warp per (token, kv_head, role), stages
  the CURRENT chunk's bf16 k/v rows rotated straight into each scratch band before the scratch
  kernel (the `from_new` analog, tensors already in hand at the A1 prompt route); the scratch
  kernel then skips fresh rows and moves its ring bound to [positions[0] − W, positions[0]).
  The A3 cached route passes empty tensors and keeps tail coverage (no fresh tensors cross its
  public contract — the MTP bridge keeps sink+ring coverage). Gate: test_hq_prefill's residual
  oracle extended (sink + pre-chunk ring from the planes, fresh chunk host-rotated from bf16);
  ALL PASSED; 304k + 390k engine probes re-verified exact after the change.
- **Nit**: constants comment now states per-row source selection (no ring tile alignment claim).
- The review's expectation ("window restores coherence; far-needle past 390k may still need
  lever 2") was borne out and both were needed: the verified cells needed window AND dither.

Build note: the two per-geometry hq route TUs' instantiation macros had to grow the two new
`const Tensor&` params — hand-editing multi-line backslash continuations in macros is error-
prone; regenerate the whole macro block when the route signature changes.

### 6. Review §15 (session 6 close-out): §13 fully closed, S7 withdrawn

All §13 findings closed; the mutation test was accepted as the evidence. S7 formally withdrawn
(both the review and our session-5 paragraph shared the capacity/kv_capacity misread; the c=1
probe stands as the evidence, ROADMAP WI-2 "Step 0" removed, WI-2 banding unchanged). The one
residual polish landed: the [5] cosine gate now states its nominal (~0.938 at alpha 1.45,
floor 0.93, ~0.008 margin) in the check message so a trip is interpretable.

### 5h. Session 11 (2026-08-23, continued): WI-2 scratch banding landed (M2 unblocked)

- **Design**: the U8 prompt route materializes its rotated scratch in sequential bands of
  `kGqaHqPromptScratchBandKeys = 262,144` keys (public constant). The FA2 prefill kernel
  gained a `Carry` instantiation: per band it covers keys [key_begin, key_end) of the
  band-local scratch, resumes its online-softmax state from carry buffers (m/l fp32,
  un-rotated acc bf16 in `out`'s layout - the decode split-partial precedent), and writes
  the state back instead of normalizing until the last band. Single-band envelopes
  (everything <= 262k) stay on the original bit-identical instantiation. Bands must be
  64-key-tile aligned (the route asserts; 262,144 is).
- **Two bugs found by the oracle gate on the way**: the carried l must split across the
  row's four lanes (the loop runs l as per-lane partials; loading the full l quadrupled it -
  output exactly 1/4, cos 1.0 - the failure signature found it instantly), and the banded
  carry tensors coexist with the scratch inside one call, so the workspace capacity sums
  them (a max-based sizing died with "bad allocation" on the first 524k engine run).
- **Gates**: test_hq_prefill now has six scenarios - two single-band originals, three banded
  (spans 128/192/64), and an ENGINE-SCALE banded case (390k keys at the production band,
  crossing tile 4096): min cos 0.999986, row rel 0.45% - the same error profile as
  single-band. decode/codec gates unchanged and green.
- **Engine**: 1M + `--kv-capacity auto` now resolves the full 1,048,576 pool with 3.37 GiB
  free and 1 GiB headroom (scratch 1.07 GiB banded; was 4 GiB one-shot, explicit-capacity
  only, 569 MiB free). 524k yarn:2 needle runs end-to-end banded at prefill 1277 tok/s /
  decode 18 tok/s (within the +-5-10% noise band of the unbanded 1330-1348 / 20-21.5).
- **390k single-probe note**: the d20 needle that passed pre-banding flipped to garble under
  banding - the engine-scale oracle exonerates the band math (correct to 0.45% row-rel at
  the same scale), so this is the fragile >262k greedy regime responding to the tiny bf16
  carry rounding. The REAL-document signal disagrees with the synthetic-probe fragility:
  **lbv2 short native 0.3833 (n=60) vs lbv2 medium 524k-yarn:2 0.40 (n=15)** - no collapse
  at 100-500k tokens on LongBench v2; single greedy synthetic probes are not the quality
  instrument, the suites are.

## Remaining work (in priority order)

### 0. 1M-context track (M1/M2 quality blocker CLEARED by WI-8, session 16)
- **Verified**: hq needle retrieval clean at 32k/304k/390k/592k (see the table up top); 592k runs
  at yarn:4 on the full 1M auto pool (916 t/s prefill, 15.2 t/s decode); MTP3 multiplies 390k
  decode to 62.9 committed t/s. Remaining for the M2 claim: a true ≥1M-token needle fixture
  (the 592k file is the longest; tile two tangles to ~1,050k tokens), the lbv2_medium 524k /
  lbv2_long 1M real-document cells, and the WI-5 sweep formalization. The int8 envelope raise
  gives a second clean lane to ~430k (int8 390k verified).
- Next decode lever remains Tier 2 (exact V skipping) — unchanged; the residual window made
  decode FASTER at the measured cells, so the hq/int8 gap there is closed or inverted at ≤32k.
- WI-7 (ii) DFlash acceptance trace on llama.cpp: still the remaining zero-code gate.

### 1. Decode performance — next levers (strongest first)

### 1. Decode performance — next levers (strongest first)
Levers (a) and (b) are both closed: (a) landed (session 8, §5e), (b) measured and REFUTED
(the warp-specialization prototypes lose to the production kernel everywhere; the acc-fragment
register wall caps every 8-warp variant at 1 block/SM — see §5e addendum and the refutation
comments in bench_hq_kernel.cu). Priority order now:
- **Tier 2: exact V skipping (BLASST-style, arXiv 2512.12087)** — the kernel holds every
  chunk's scores in `p_sw` before the PV phase; `exp(s − m_running)` is an upper bound on the
  row's final unnormalized weight, so decode V rows only where the running-max-relative weight
  is non-negligible and the skipped mass stays bounded by numbers the kernel already holds.
  Design choices: the threshold (BLASST uses one epsilon per model/phase) and the bound's
  accounting across splits — each split must carry its skipped-mass bound into its partials so
  the reducer can bound the total — or a two-pass variant: scores over the window, then PV over
  selected keys. V is half the rows; attention is peaky at long context → decode work −40–45%
  (BLASST: 1.48× decode at 73% sparsity, accuracy preserved, GQA-native). The only remaining
  path to ≥3 G rows/s.
- **Page-level K selection (Quest-style)** after V skipping, gated on the needle test:
  per-page channel bounds are valid in the rotated frame (the rotation is orthogonal);
  attention sinks and the recent window always on; with GQA take the union over the group's
  q-heads; K decode −60–80% at +5–11% metadata bytes. At 2.25-bit K the bounds are noisier —
  the open question to measure first.
- Asymmetric formats are the cheaper exact intermediate if the combined 3–5× proves hard:
  K in hq, V in INT4-G64 (V is the less sensitive role) — halves the ALU work, KV 2.4×
  smaller than int8 instead of 3.7×.
- short-window split tier (optional, cheap-ish): more hq splits below ~4k to grow the
  16-block tg128 grid and recover the ~6% tg128 give-back — touches the shared
  host/device split policy and the graph capacity math.
- ~~(b) warp specialization~~ refuted; ~~(c) runtime smem~~ obsolete (static smem, 2
  blocks/SM already).

### 2. Prefill polish (optional)
86–95% of bf16. Options unchanged: int8 scratch + IMMA (paper's canonical design), raise
`prefill_chunk` for U8. The group decoder already cut the scratch-decode share.

### 3. Documentation (mandatory before any PR)
- DONE session 16: `docs/maintainer/paged-kv-cache.md` §12.2 (residual window + dither contract);
  README / docs/cli.md / qwen3.6-27b-model.md per-dtype envelope (524,288 linear).
- Still open: paged-kv-cache §1/§4.5/§7.4/§15.1/§16.1/§19.3 reviews;
  `qwen3.6-27b-model.md` §4/§12 decode definition + named criterion (group-decoder profile +
  k==0 invariant, test_hq_decode criterion, batch/column contract) — extend for the residual
  third source and the dithered oracle.
- Op-level U8 tests under `tests/ops` mirroring the bf16/i8 conformance tests (decode + prefill
  oracle routes) — the hq gates currently live only in the standalone `tools/test_kv` suite.
  Same bucket, still open: 35B-geometry scenarios in `test_hq_decode`/`bench_hq_kernel`
  (GroupSize 8 saturates `kGqaHqDecodeMaxRows` at rows=48), and the FP16 row-norm clamp in
  `hq_codec.cuh` (overflows at 65504; Qwen3's k_norm keeps K safe but the clamp is free).

### 4. Real-model quality gate before any quality claim
Run the existing eval scripts + a needle sweep at bf16/int8/hq at 32k/128k/262k; check
attention-sink (position-0) rows specifically; census a dumped real-model cache (heavy-tailed
sink rows are the ones that escalate to α/2) and sweep `kHqAlpha` on those rows — it was
calibrated on synthetic Gaussians. The GPQA accuracy campaign is resumable (session-5 addendum:
EvalScope resume command; confirmation band 89–90% ±2 at n=198; <85% ⇒ second seed).

### 5. Cleanup / optional
- The extraction is done (§5f); before opening the `feat/hyperquant` PR: item 3 (docs +
  op-level tests), item 4 (quality gate), then re-run the four `tools/test_kv` gates on the
  PR branch.
- Evaluate raising the engine envelope 262k → 1M after decode work (hq ceiling ~1.1M) —
  `ROADMAP-1m-context.md` WI-1…WI-6, milestones M1 (524k) → M3 (usable 1M).

### 6. General (non-hq) kernel levers — `ROADMAP-kernel-perf.md` (desk note, 2026-08-23, unmeasured)
Prefill attention format/overlap/wave quantization (WI-K1), decode PDL coverage + small-kernel
fusion (WI-K2), W4A4 TMA GEMM inner loop (WI-K3), A4 quantization fusion (WI-K4), two template
A/Bs (WI-K5); measurement order in its §6. Start with the zero-code `--prefill-chunk` discriminator.

## Cautions (do not reintroduce)

0. **Build-status discipline (session 16, twice-bitten)**: piping `build-ninja.ps1` through
   `grep | head` and appending `; echo OK` masks failures — `head` closes the pipe and the echo
   prints unconditionally. Always capture to a log file, echo `$?`, and grep the log. Worse: a
   header edit that drops inline accessor bodies (PagedKVCache's `pool()/layers()/max_context()`)
   turns into LNK2019 only when a stale TU finally rebuilds — apps silently kept running OLD
   binaries while "probes" produced mixed-evidence garbage. And never pass HOST stack arrays as
   kernel pointer args (the ring-bits apply kernel did; illegal address at admission, surfacing
   in an unrelated downstream CUDA_CHECK). If engine behavior contradicts a fresh code change,
   suspect the binary first: `ls -l build-ninja/apps/*.exe` timestamps, then force the target
   build and check its true exit code.
1. The TC hq decode kernel (session 8) uses ~36.75 KB STATIC shared memory at 128 threads —
   no dynamic-smem opt-in attribute, and the old 97 KB carve-up is gone. If a future change
   pushes static smem past ~48 KB (the per-block static limit) or past ~50 KB/block at the
   2-blocks/SM design point, the static_assert on the append scratch (aliases the qkv tile)
   is the canary — update it, don't delete it. hq inactive splits must keep writing NEUTRAL
   partials (public partials contract; test_hq_decode's fresh cudaMallocs read recycled
   device memory), while the bf16 policy intentionally returns silently (zeroed engine
   partial workspace). `Masked=true` + `valid_columns=nullptr` must mean unmasked — never
   subtract column_begin in that branch (session-8 bug: every split neutralized for chunked
   shapes).
2. **Code-plane bit order**: each 32-bit word is a VALUE accumulated MSB-first and stored
   little-endian; a raw u64 load of two adjacent words reverses the byte order within words.
   Compose windows as `(u64(w32[2l]) << 32) | w32[2l+1]`. (Cost an hour this session.)
3. Meta row format v2: [0..1] FP16 norm; [2] Rice k (always 0 — asserted) | escalation<<4;
   [3] used low 8 bits; [4] used bits 8..9 (10 bits total; `used` is the EXACT bit
   total ≤ 512 — the old writer word-padded to a multiple of 32, the parallel packer
   reports the true count), bits 2..4 = 9th bit of each segment offset; [5..7]
   segment bit-offsets low 8 bits (unused by production decoders; kept for format
   compatibility).
   Load-bearing invariant: code bytes at or past ceil(used/8) are ZERO (the bit writer
   zero-initializes its buffer), so the group decoder reads all 16 words and treats bits
   past `used` as unary-guard padding. A foreign encoder must preserve this.
4. Never delete `Directory.Build.props`.
5. Capacity/sweep measurements only on an otherwise idle GPU; engine cells swing ±5–10% across
   runs (pp8192 varied 6774→8116 across identical binaries one session) — A/B with reruns.
   5b. CHECK `nvidia-smi --query-compute-apps=name --format=csv,noheader` for `.scr`/overlay
   processes before trusting numbers: a 3D screensaver (Mystify.scr) once cost ~5% on every
   cell and masked a +3.1 tok/s tg128 win as "flat"; decode cells under CUDA graph are NOT
   immune. Kill it (`taskkill //IM Mystify.scr //F`) and rerun.
6. Git Bash heredocs eat backslash-n inside embedded CUDA strings — write python patches to a
   FILE and run `python3 file.py`; prefer the Edit tool.
7. Don't pipe build commands through `head`/`tail` — use `build-live.ps1` /
   `build-ninja.ps1 [-Target X]` and check the exit code / log file.
8. The bench must never alias partial_m/partial_l into the meta plane; its fill grid must cover
   tokens × KVHeads × 2 units.
9. Do not retry: bank-conflict smem padding (no effect when decode was 84% of the kernel;
   not retested since — score/PV is now ~50%, re-A/B only as a 10-minute diagnostic before
   lever B), register accumulator (isolated kernel
   faster, engine 8–10% slower from launch_bounds-forced spills in graph replay), whole-row
   prefetch of code words before the segment scan (~9% — the chain, not the loads, was the cost).
