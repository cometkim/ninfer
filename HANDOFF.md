# HANDOFF — cross-session work state

Read this with `AGENTS.md` before planning work; update it before ending a session that changed
anything material. The full pre-rebase history (sessions 2–29, the fork DFlash2 development
diary, and the 2026-08 campaign records) is recoverable from
`refs/backup/pre-sync-20260908/cometkim/dev` and `backup/20260908/all-refs.bundle`.

## Current state (2026-09-09, Port 10)

Ports 1-10 are landed on `cometkim/dev` and `feat/kernel-perf` and pushed (Port 10 = the 1M/YaRN
context track, three commits: dev e648f178/8f79eaa9/4008d8de, kernel-perf
d85cce0a/98058e7d/d3117dcd; cherry-picks verified by content markers + building the new/changed
TUs through the wrapper on feat/kernel-perf).

- **Port 9** = the hq-e8-2b sixth KV profile (codec, sixth closed profile, append/prompt/small-t
  routes, product surface, 18-scenario conformance). Context was capped at 262144 there.
- **Port 10a (WI-8)**: half-cell subtractive dither (hash-derived per head/position/role/word,
  absolute across escalation; codec suite escalation moved to 325/22000 rows, band 150-600) +
  BF16 sink+recent residual window (S=32, W=512): per-(layer, lane-slot) rotated side planes and
  17-word validity rows in the decoder-state layout; dual-write with the chunk-internal
  ring-ownership guard at both append sites; consumers prefer side rows (small-T tile, prompt
  scratch); the prompt route stages the current chunk exact (D2). Ownership follows the address
  space: KVAddressSpaceStore tracks a per-row handover epoch + side-written frontier; trims
  revalidate, rejected MTP drafts invalidate, the perplexity scorer takes over row 0 with a full
  clear, prefix forks inherit the source row's planes when coherent (copy + revalidate) else
  fall back to codec rows.
- **Port 10b**: rope op reworked onto RopeFrequencies tables with a q-side attention temperature
  (factor-1 keeps the bit-stable FP32 angle product; other factors reduce in FP64); YaRN builder
  + plan/EngineOptions/schedule/CLI/serve plumbing (`--rope-scaling none|yarn:F[,t=][,bf=][,bs=]`);
  envelope raised to 1048576 (27B only; linear caches capped at 524288, hq-only to 1M, enforced
  at validation); banded hq prompt scratch (262144-key bands; FA2 Carry instantiation resumes
  online-softmax state between bands; scratch bounded ~1 GiB).
- **Port 10c**: 1M-row needle retrieval gate (tests/ops/test_hq_retrieval.cu, curand corpus).
- Verification (RTX 5090): codec suite green with dither oracle mirrors; 25-scenario hq attention
  conformance green (7 residual scenarios across small-T/chunked/cached/prompt/graph, both
  geometries) using the pre-rebase oracle convention (side-selected keys read the side-plane slot
  row as it stands at attention time); rope suite green incl. YaRN-shaped tables at 1M positions;
  new HF-parity YaRN table test bit-exact; retrieval gate 5/5 needles unique top-1 at 2.250 bps
  (negative control fires, 4/5 missed); engine smokes on nvfp4full: 262144 MTP0 coherent 51.6
  tok/s, MTP3 coherent 88.1 tok/s @50% acceptance (rejection invalidation exercised);
  524288/yarn:2 healthy (52.1 tok/s decode, auto capacity resolved the full 4.53 GiB pool);
  1048576/yarn:4 coherent with explicit `--kv-capacity 1048576` (9.03 GiB pool) — see the caution
  below. Softmax-attention, kv-append, context-materialize, CLI/serve/bench suites green.
- **CAUTION (open)**: at the 1M envelope the decode path runs ~3 tok/s (524k is 52) — the
  graph/workspace/split sizing scales with envelope capacity and something pathological kicks in
  past 524288. Root-cause before advertising the 786k/1m presets. Also 1M does not fit with
  `--kv-capacity auto` (1 GiB headroom + 14.25 GiB reservation > 15.29 GiB free beside weights).
- Presets: `qwen3.8-27b-nvfp4full-hq-e8-2b-262k/524k/786k/1m.bat` all pass `--webui` (Port 11
  pending) so NONE can run as scripts yet; flags are otherwise validated (524k healthy, 1m
  functional-but-slow). Do not advertise larger presets until Port 11 lands AND the 1M perf
  issue is fixed.

## Branch layout correction (this session, follow-up work queued)

The first Port 10 landing wrongly cherry-picked everything onto feat/kernel-perf. Corrected to
the pre-rebase stack shape (all pushed; old lineage tips preserved in
refs/backup/pre-sync-20260908 and backup/20260908/all-refs.bundle):

- `feat/kernel-perf` @ 0f0303e3: TMA Windows blocks, PDL chain, per-dtype TU splits,
  DFlash2 NVFP4 execution + module-format unification. RESET from the Port 9/10 tips.
- `feat/hyperquant` @ 8e96ad92 = kernel-perf + the hq-e8-2b storage profile (Port 9).
- `feat/1m-context` @ cf1e2fb4 = hyperquant + WI-8 dither/window + YaRN/banding/lanes +
  retrieval gate (34a8e036/5fbb5bfa/cf1e2fb4; content-identical to dev's
  e648f178/8f79eaa9/4008d8de).
- `cometkim/dev` @ 434651a9: unchanged integration branch (primary history).

Branch cleanup complete (all pushed):
- `feat/windows-port` @ 723c1290 = master + build layer + MSVC flags + the TMA
  descriptor-block fix (no longer dangling inside the hyperquant stack).
- `feat/hyperquant` @ 5d5ffbfc (rebuilt on aac014c0; codec suite re-verified) = windows-port + the hq-e8-2b profile, self-contained
  (PDL-free plain launches; build + codec/attention suites verified on exactly this tip).
- `feat/kernel-perf` @ aa07cac7 = master + ALL dev perf work fused: TMA fix, PDL
  publish-at-end + full PDL chain, per-dtype TU splits, fused q/k rope, sigmoid-gate
  fusion, i8 fp16-PV + key-split, BF16 prompt key-split, per-request error boundary.
  The DFlash2-NVFP4 work was MOVED OUT (owner directive). NOTE: 9beab516 (TMA) is
  duplicated here and in windows-port - harmless, drop at the next kernel-perf rebase.
- `feat/windows-port` @ aac014c0 additionally defines UTF8PROC_STATIC for ninfer_text
  (master lacks it; MSVC builds fail with dllimport redefinitions without it).
- `feat/dflash2` @ 23d34c77 = windows-port + the four DFlash2-NVFP4 commits (swiglu A16
  fix, replay-order test fix, fork-format NVFP4 draft-module execution, module-format
  unification), SELF-CONTAINED: no kernel-perf ancestry; the one PDL launch site (the
  NVFP4 dflash2 attn-input launcher) converted to plain launches. PR-able directly to
  natpate/ninfer-windows and other forks that treat the nvfp4full profile as first-class
  (old pre-rebase tip 62acfe15 preserved in the backup refs). Verified on this tip:
  dflash2 routes suite passes, ninfer_engine builds. The one-shot unify_module_names.py
  is dropped from the branch and from dev (temporary tool; header backups and the
  upstream-named converters cover it).
- `feat/1m-context` @ cf1e2fb4 still sits on the old kernel-perf-based hq tip; needs the
  self-containment re-port onto 5d5ffbfc (drop fused qk_norm_rope extensions; plain
  rmsnorm->rope with RopeSide; master-shaped schedule/plumbing).

Dev re-stacked AGAIN per owner directives (2026-09-09, final state of this session):
- New fork base 83ddf726 (a9c8e491 amended): the fork README SKELETON lives in the base -
  About-this-fork layout, experimentation cycle, EMPTY feat table and adds-on-top list,
  rebuild block in stack order, parity rules.
- Every feat branch now carries its own docs(readme) tip commit adding exactly its table
  row and (where applicable) its adds-on-top bullet, so each squash(feat/*) contains its
  README change. Verified: every intermediate squash commit README = skeleton + union of
  rows/bullets up to that point (scripted composition, no duplicated bullets).
- feat/qwen3.8-nvfp4full and feat/qwen3.8-nvfp4qat RE-PARENTED onto feat/dflash2
  (bc36e9c2 / c17ccc30): both profiles need the DFlash2 binder unification.
- NEW SQUASH ORDER on cometkim/dev: msvc-test-constexpr, windows-port, webui, mtp7,
  hyperquant, 1m-context, dflash2, qwen3.8-nvfp4full, qwen3.8-nvfp4qat, kernel-perf
  (kernel-perf last = most merge resolutions). Tree parity vs the prior tip exact except
  the intended README changes (row order, nvfp4full/qat stacked-on dflash2).
Dev rebuilt (2026-09-09, this session): cometkim/dev = fork base a9c8e491 + one squash
per feat branch in stack order (msvc-test-constexpr, windows-port, dflash2, kernel-perf,
hyperquant, 1m-context, nvfp4full, qat, mtp7) + the fork layer (AGENTS/CMake state,
PDL-form integration files, splice tool, HANDOFF/README). Tree parity vs the pre-rebuild
tip is EXACT (empty diff; backup ref dev-pre-rebuild-20260909). Verified on the rebuilt
dev: full Ninja build; hq codec + 25-scenario hq attention + rope + YaRN-table +
dflash2-routes + softmax-attention + kv-append suites green; retrieval gate 5/5;
524k/yarn:2 engine smoke coherent at 75.3 tok/s decode. kernel-perf fixed to 32a8cfaf
(the BF16 partials mis-ordering dropped - the i8-split-final state is authoritative; its
008ba92a duplicates the windows-port TMA fix until the next rebase). Squash attribution
note: the kernel-perf squash briefly included the mis-ordered BF16 partials; b7628876
folds the dev-final forms back, so the tree is correct throughout.

Port 11 (webui) landed and re-stacked: `feat/webui` @ e7d12a15 = windows-port + the
in-process llama.cpp WebUI (auto-sync from ggml-org/llama-ui, static mount, SPA fallback,
API-path-only auth, /props stub, server dialect on the OpenAI endpoints; spdlog instead of
the old console_log; upstream renamed the cudart CMake var, resolved). dev rebuilt again
with `squash(feat/webui)` at the BOTTOM of the stack (right after windows-port, before
dflash2) per the owner directive; tree parity vs the pre-rebuild tip exact (empty diff).
Verified on the final tree: serve-options + openai-schema tests green; live acceptance on
the 262k hq preset flags with a local webui dir: / 200 shell, /props 200 n_ctx=262144,
/v1/models 200 status loaded, SPA route 200, bundle 200 text/javascript, unknown /v1 404.
NOTE: --webui auto-download needs network to huggingface.co (error 12005 offline); use
--webui-dir models\webui (66 files, present locally) when offline. All hq presets remain
launcher-blocked on the 1M-envelope decode perf issue only (3 tok/s at capacity 1048576).

Remaining cleanup:
1. Re-port feat/1m-context onto 9d629f67 (see above), verify with rope/rope-scaling/hq
   suites + the retrieval gate.
2. Rebuild cometkim/dev following the pre-rebase structure (fork base 310e-style: one
   squash(feat/*) commit per branch in stack order, then dev-only layers): reset dev to
   a9c8e491, squash-merge msvc-test-constexpr, windows-port, dflash2, kernel-perf,
   hyperquant, 1m-context, nvfp4full, qat, mtp7 (resolve the 1m-context conflicts against the
   self-contained hyperquant forms), then one dev-only commit carrying the splice tool,
   presets/eval/model-cards, and ONE folded docs(handoff) commit. Content docs of each
   feature already live inside their feat commits - keep them there.
3. Verify parity vs 9cee8ff1 (git diff must list fork-only files); the PDL-ful hq small-t
   routes on dev today come from the old landing - after the rebuild dev takes hyperquant
   PDL-free hq; re-adding hq PDL as a kernel-perf commit on top of hyperquant is the
   follow-up perf task.

## Remaining port backlog

1. **Port 11: in-process webui** onto upstream's rewritten serve layer (spdlog/complete
   adapters). Pre-rebase lineage: `feat/windows-port` (webui_update.*, response_store). The 262k
   hq preset already passes `--webui` and fails on it - that is the acceptance test. Machine-
   specific files stay dev-only; update serving docs/schema tests together.
2. **1M-envelope decode perf**: profile the decode chain at capacity 1048576 vs 524288 (graph
   allowance math, split capacities, workspace sizing all scale with envelope).
3. **Idle-GPU paired re-measures** (same-binary alternating A/B only): fusion-vs-PDL split;
   K1-K7 on both v2 artifacts; qat cells; MTP0/MTP7 records; hq-e8-2b cells vs int8/k8v4/nvfp4
   at 8k and 262k (Port 9 landed without clean perf cells - record them).

## Verification & measurement cautions

| branch | base | commits over base | role |
|---|---|---|---|
| master | upstream/master a16b6442 | — | follows upstream; fast-forwarded 151 commits (2190c4a1→a16b6442) |
| feat/msvc-test-constexpr | master | +1 (097ac842) | upstream PR candidate |
| feat/kernel-perf | master | +8 (9beab516 TMA _WIN32, 172a8c24 PDL publish-at-end, 1fa52b9b..5b89e5be PDL chain + TU split, 38017a1f/612f0c88/95c77ece/0f0303e3 port 7 + fixes) | upstream PR candidates |
| feat/qwen3.8-nvfp4full | master | +1 (53d4efcf) | upstream PR candidate: W8-module PR form, recipe v3 |
| feat/qwen3.8-nvfp4qat | feat/qwen3.8-nvfp4full | +1 (2b609d7b) | upstream PR candidate: QUASAR QAT profile, recipe v2 |
| cometkim/dev | fork base a9c8e491 on a16b6442 | squashes (msvc, kernel-perf, nvfp4full, qat) + 2808b656 CMake fix + 42ba5b12 Windows layer + b9d8ee46 fork-format module binding | integration branch |

Superseded (content dropped in the rebase, preserved in `refs/backup/pre-sync-20260908/*` and on
`origin`): `feat/dflash2` (fork DFlash2 implementation — upstream's is the semantic authority),
`feat/kernel-perf` pre-rebase tip 1d70bf95 (items landed or documented below), and the natpate
lineage branches (`origin/feat/hyperquant`, `origin/feat/1m-context`, `feat/windows-port`) whose
content awaits porting (see backlog). `feat/build-speed`/`feat/build-speed-upstream` (natpate PR
forms) untouched — rebase only when opening those PRs.

## What the 2026-09-08 rebase did

**Upstream is the semantic base.** DFlash2 (their stochastic selector walk, K1..15 × B1..8, dual
proposal heads, sparse-draft rejection, W8G32_F16S module), the five closed KV profiles
(BF16/INT8-G64/FP8-row256/NVFP4-G16/K8V4), the completed serving adapters (Responses API,
Anthropic semantics, spdlog), host context cache + value-aware prefix scheduling, perplexity
scoring, and the nvfp4 A4 TMA SwiGLU at every 256-multiple width are all upstream's and replace
the fork's old lineage wholesale.

**Superseded as duplicates (dropped with evidence):** the fork's width-8 int8 verify tile
(upstream `4b0eb36c` routes H24 verify through SmallT width-8 single-pass, plus a 5+4/5 chunking
refinement the fork never had), GDN norm+control fusion semantics (upstream FusedSimt27 covers
cols ≤ 42 vs the fork's ≤ 8), and the fork's old nvfp4 TMA plan (upstream registers every width
its block accepts).

**Kept as fork layers (owner decision — "keep both" module formats):** the nvfp4full/qat profiles
re-enter as clean PR-form branches whose DFlash2 module is the upstream W8G32_F16S suffix; the
existing v2 artifacts (fork NVFP4 module, fork object names) remain loadable through
`bind_fork_format_dflash2` — consumed and validated, Text/Vision/MTP fully usable, DFlash2 lane
selection rejected with a precise error until the NVFP4 draft execution port lands. No artifact
rebuild is required; v2 files stay authoritative.

**Landed ports (verified by full MSVC build + smokes):**
- Windows TMA descriptor blocks (build-blocking on MSVC: by-value `alignas(128)` kernel params)
  with the consuming-stream async free (the 786432-prefill live-lock fix).
- PDL publish-at-end for all upstream PDL sites (q4/q5 GEMV/SIMT, sparse_moe decode + small-t)
  plus the `pdl::sync/publish` helpers; consumer waits unchanged. Upstream's sparse_moe s2 was
  rewritten upstream (one CTA per token) — the fix was re-derived on their new shape.
- The full Windows build layer onto upstream's restructured sources: reader WIN32 mapping +
  direct-read, portable `Uint128`, explicit move-specialization bodies (MSVC 19.44 drops
  out-of-line `= default` explicit specializations — LNK2019), POSIX call sites (getpid/isatty/
  localtime/gmtime/terminal width/Winsock), UTF8PROC_STATIC + ws2_32 wiring, test traps
  (`aligned_alloc`, legacy `near`/`far` macros).

**Verification (this session, RTX 5090):** full Ninja build green (apps, tests, benches link);
converter inventory/recipe validations pass for nvfp4full + nvfp4qat in the torch env
(`~/Workspace/ninfer-src/cvenv`); greedy MTP3 smokes on BOTH v2 artifacts through the rebuilt
engine (nvfp4full 3.29 tok/round 76.2%, coherent output; qat 2.88 tok/round 65.2%); fork-format
DFlash2 selection rejects with the documented message. NOT yet run: the full ctest suite (needs
the qwen3.6 artifacts on box for `ninfer_qwen3_6_27b_load_plan_test`'s mandatory cells), any
serving smoke, any benchmark. All previously recorded token-stream fingerprints and acceptance
fingerprints are STALE — upstream numerics/schedules changed everything; re-record before using
identity gates.

## Owner no-regression directive (2026-09-08, evening)

Fork improvements are never dropped for upstream alignment; a sync that cannot land
regression-free should halt (recorded in AGENTS.md). The pre-sync state stays recoverable from
`refs/backup/pre-sync-20260908/cometkim/dev` and `backup/20260908/all-refs.bundle`.

### DFlash2: ours vs upstream (same corpus, same artifact base, same lane K=7 int8)

The upstream-schema artifact now EXISTS and is verified:
`models/qwen3_8_27b_nvfp4full_w8.ninfer` (v3, 20,550,864,896 B) built by
`tools.convert.qwen3_8_27b.splice_w8_module` — every base object byte-copied from the v2 image,
the 66-object W8G32_F16S module encoded fresh from z-lab rev 50307d4c; full verify pass
(endpoints, 112+135 NVFP4 payloads, divisors, 66 module objects, resources). MTP3 control:
82.41 tok/s @ acc 0.4176470586 — byte-identical behavior to v2, splice is execution-neutral.

| cell (greedy, int8, K=7) | fork engine (idle, recorded) | upstream engine (today, ~30% load) |
|---|---:|---:|
| DFlash2 @8k | 157.6 @ 2.64 tok/round (+30% over fork MTP3) | 119.8 @ 1.90 (full head); 82.5 with --lm-head-draft (shortlist head LOSES here) |
| DFlash2 @262k | 140.8 @ 3.70 | 75.1 @ 1.90 |
| round time @262k | 26.3 ms | 25.3 ms |

Round times are comparable — the throughput gap is ACCEPTANCE (3.70 vs 1.90 tok/round on the
identical bench corpus). CAUTION for the A/B: the fork's 3.70 was recorded with the pre-rebase
selector reading the K16M128x4 scale plane row-major (a real numerics bug, now fixed in the
port) — re-measure the fork-side walk on the current engine before comparing walk semantics.
The upstream side is not contaminated by the context-test race (test-only). Next step:
instrument one walk on both engines (selector temperature handling, sparse-q rejection
strictness, distribution shape) and port whichever semantic wins on-distribution as a thin layer
over upstream's schedule (the authority for K/B/vision/serving surface).

### Reconciliation map (old dev -> new dev, where every change stands)

- RESTORED/REBUILT: fork base layer; msvc test fixes; nvfp4full + nvfp4qat profiles (PR form,
  W8 module; v2 images stay loadable with fork-module validate-only); MTP7; Windows TMA
  descriptor blocks (incl. the live-lock free); PDL publish-at-end; the full MSVC port layer
  (reader/Uint128/move-bodies/POSIX sites).
- UPSTREAM-EQUIVALENT (verified, no action): width-8 int8 verify; GDN norm+control fusion
  semantics; nvfp4 A4 TMA every-width; prompt-threshold/attention refactors; measured
  context-cost scheduling (replaces the fork's context-cost bench prototype).
- REGRESSED — RESTORATION REQUIRED (no-regression contract): DFlash2 on fork v2 images
  (workaround live: the W8 splice artifact; full fix = NVFP4-module execution port + the
  acceptance-semantics A/B above); hq e8-2b KV + 524k-1m lanes (hyperquant/1m-context — the
  largest port, onto the five-profile KV architecture); kernel-perf set (PDL decode chain,
  qk_norm_rope, sigmoid-gate fusion, i8/BF16 prompt key-split + fp16-PV, per-request error
  boundary, GDN MMA split40); uniform width coverage (the K-cell patchwork); build-speed
  per-dtype TU split (re-derive on upstream's launcher tree); in-process webui.

## Port progress (owner directive: port everything, no backlog)

1. **qk_norm_rope — LANDED** (feat/kernel-perf 8c45bde4, dev 0f1e01a2): fused q/k rmsnorm+rope on
   the Text attention path + MTP tail, boundary-parity contract (one BF16 ulp vs the chain; FP64
   oracle semantic bound — the fork's old bit-parity was TU luck, and upstream's rope contract
   assigns storage rounding to the op's criterion). Measured nvfp4full MTP3 @8k int8
   79.93 -> 87.44 tok/s (+9.4%); greedy smoke tokens identical. Full op test passes
   (16 case-runs incl. MRoPE).
2. **Sigmoid-gate fusion — LANDED** (feat/kernel-perf 0e95fe9f, dev 4f8e78b6): both causal ops
   take an optional sigmoid-gated-output input; BF16/INT8 KV routes fuse it into the generic
   small-T reducer epilogue (attention rounded to BF16 first — bit-identical to the
   reducer-then-sigmoid_mul chain); other storages/Prompt keep the separate launch. All three
   runtime sites fused. Paired 5-rep cells: nvfp4full MTP3 @8k int8 87.44 -> 128.00; @262k
   79.62 -> 124.02 (+56%). Tokens byte-identical (76.2% acceptance, same by-pos); op suites
   pass. NOTE: desktop load drifted mid-session (~30% swings) — treat unpaired cross-run cells
   (incl. the K-sweep below) as load-contaminated; only paired cells above are decision-grade.
   Post-fusion sweep (3-rep, loaded): full K1 96.5 / K2 112.5 / K3 104.8 / K5 86.6 / K7 102.2;
   qat K3 95.6; the old per-width cliff ordering reshuffled under load — re-sweep paired when
   idle before drawing width conclusions.
3. **i8 prompt fp16-PV + key-split — LANDED** (feat/kernel-perf bec4f191 + 008ba92a, dev
   03be10b7): fp16-accumulated PV with the 2^-6 range guard (dequant guard + guarded promotion,
   __maxnreg__ 128), then the grid.z key-range split with a prefill-owned reducer and the
   wave-fill split-count policy. One genuine port bug found and fixed: the fork's ceil-based
   ratio metric degenerates at tiny grids (S=4 for a 12-token prompt) and that path overran at
   small verify widths — a short-circuit (items <= SMs -> S=1) keeps short prefills bit-identical
   and fixes the crash. Verified: softmax_attention suite (splits at 512/1024 vs oracle) passes,
   MTP3 smoke unchanged at 76.2%, pp65536 prefill completes (6425 tok/s, load-contaminated).
   Perf deferred to an idle paired measure.
4. **BF16 prompt key-split — LANDED** (feat/kernel-perf 185084b2, dev cherry-picked): same
   grid.z split as INT8, with the two extra traps the fork documented — the empty-segment
   prologue guard (skip K stage, block table past the CTA range holds no page) and the alpha NaN
   guard (segment entirely past a row's causal limit). Split-count policy renamed
   storage-agnostic and gates BF16 too. Verified vs oracle (512/1024-token BF16 prompt cases) +
   eager BF16 smoke; perf deferred to idle paired measure. NOTE: the BF16 CUDA-graph engine path
   fails cudaGraphExecUpdateFailure on this box — pre-existing, reproduces on clean upstream.
5. **Per-request error boundary — LANDED** (feat/kernel-perf de38d41f, dev d3236fca): host
   exceptions scoped to one request now fail only that request (prefill step, output-preview,
   and publication boundaries) via fail_active_request; shared-engine work still fail_all's.
   Verified: engine smoke unchanged (76.2%), admission/context-cost/request-log/resource-manager
   tests pass.
6. **PDL decode chain — COMPLETE** (five increments on feat/kernel-perf, mirrored on dev:
   bb947d1b/4dc6d248/ca001cfc/5ee7afd8/c6a0686d): the fork's WI-K2a surface is fully converted —
   rmsnorm (5 bodies), rope (3 + the fused qk kernel), sigmoid gate, bf16/nvfp4 GEMV + small-T,
   w8 rowsplit + MMA, nvfp4 linear_add + linear_swiglu, all six attn/gdn input projections, the
   four GDN recurrents, and the small-T attention producers + split reduce (the fp8/k8v4/nvfp4
   small-T variants deliberately stay plain-launch: their kernels never wait, the safe
   combination). Two port bugs found and fixed on the way: the bf16/nvfp4 launcher conversions
   need fully explicit template arguments (launch_dependent takes a function pointer; default
   arguments don't exist in pointer types), and the swiglu small_t publish initially landed in a
   host constexpr function. CUMULATIVE same-session cells (desktop-load contaminated but
   sequential): nvfp4full MTP3 @8k int8 79.93 (rebase) -> 87.44 (fusions) -> 128.00 (sigmoid
   fusion) -> **141.2 tok/s** with the full PDL chain (+77% over rebase); MTP7 @262k int8
   **140.07** (above the fork's clean-idle 131.9 record); greedy real-prompt smoke byte-identical
   at every step (76.2% fingerprint); softmax_attention suite green throughout.
7. **Build-speed TU split — RE-DERIVED AND LANDED** (dev 9697fe17): the old per-dtype split
   (d41d617f upstream-form / 195c5dc7 natpate-form, preserved on their pre-rebase bases) targeted
   files upstream deleted, so the work was re-derived on the new tree and extended to every
   family: small_t.cu (169s serial) split into dispatcher + i8/bf16 instantiation TUs;
   w8_small_t.cu (201s) into dispatcher + 7 per-geometry exact TUs + vocabulary TU; fp8/k8v4/nvfp4
   small-T families each split into dispatcher + instantiation TU (the hq-ready pattern, recorded
   in AGENTS.md). 18 split TUs compile in one 140.8s parallel batch vs 370s for the two original
   aggregates alone; suites + smoke unchanged. The old PR branches stay on their bases until the
   upstream PR is opened (then rebase onto master and drop).
8. **NVFP4 DFlash2 module execution — COMPLETE** (dev b9a5c7af, kernel-perf 95c77ece; with
   484201d3 swiglu A16>T16 and 810a6774 format unification): the v2 fork-format images execute
   their 66-object NVFP4 draft module end to end. Op layer: five weight-only A16-only linear
   drafter problems (gemv inline + small-T in a dedicated instantiation TU through the shared
   launch header), the 3-output attn_input NVFP4 route (split-output policy, 32-token chunks),
   NVFP4 kernel_projection/projection branches in the dynamic-conv pair (generic A16 linear +
   format-neutral shared finish), the NVFP4 context_kv_materialize MMA family (raw E2M1 staging,
   FP32 E4M3 scales per 16-group, divisor folded in; common column mapping/key post shared with
   the W8 family), and the NVFP4 selector walk decoding gathered rows from the registered
   K16M128x4 blocked scale plane. Two port-time finds: the pre-rebase selector read that plane
   ROW-MAJOR — every group past the first 512-byte tile was misdecoded (empirically proven against
   the z-lab source; the old 3.70 tok/round @262k ran with that bug), and the conv/finish overlap
   checks needed format-aware weight spans. Verified: new FP64 oracle suite
   (ninfer_dflash2_nvfp4_routes_test: attn_input, conv pair, context materialize, eager + graph
   capture), the fixed context suite, all affected op suites, and greedy smokes on both v2
   artifacts (nvfp4full DFlash2 5.50 tok/round @ 64.3% acceptance, MTP3 coherent; qat 2.50 @
   21.4%) plus the W8 splice artifact (5.75 @ 67.9%).
9. **Pre-existing context_kv test failure — root-caused and fixed** (dev 8632835f, kernel-perf
   612f0c88): ninfer_context_kv_materialize_test failed 45 cells at the rebase base (proven on a
   master+a9c8e491 worktree with only the build-enabler commits) — a stream-ordering race in the
   TEST: replay inputs are copied on the legacy default stream and the recorded executable
   launches on a cudaStreamNonBlocking stream with nothing ordering them, so replays consumed
   capture-time contents deterministically on this box. One cuda_synchronize() before the replay
   launch fixes it; the Op and kernels were correct. First full-suite run post-rebase; the
   splice-artifact DFlash2 numbers were never contaminated (test-only bug).
10. **Module format unification** (dev 810a6774, kernel-perf 0f0303e3): the v2 images were never
    released, so tools/convert/qwen3_8_27b/unify_module_names.py renamed their module directory
    to the upstream sections IN PLACE (attention_conv/base_kernel & co., candidate_selector/*;
    the renamed JSON fits the alignment slack so payload offsets never move; header backups sit
    beside each artifact as *.head.bak). One bind_dflash2 now serves both encodings through
    Binder::declared_format per-object dispatch; the fork-format gate and plan builder are gone.
    The converters already emitted upstream names, so regenerated artifacts match. Renames are
    execution-neutral: acceptance reproduced exactly on every lane.

<!-- Port 10 (WI-8 dither+window, YaRN/banding/lanes, retrieval gate) landed this session;
see "Remaining port backlog" at the top. -->
## Post-rebase perf cells (2026-09-08, RTX 5090, ~30% desktop graphics load — re-measure idle)

nvfp4full v2, greedy, bench tg128/pp, CUDA graphs, seed corpus; old cells are the fork's
recorded baselines (cross-session caveat; the ~30% desktop load class historically depresses
cells 15-38% depending on shape).

| lane | old fork | new engine (today, loaded) | reading |
|---|---:|---:|---|
| MTP0 tg128 @8k int8 | 80.95→83.21 idle | 50.50 | short-context compute chain regression (PDL chain + fusions unported) + load |
| MTP3 tg128 @8k int8 | 121.2 idle | 79.93 (acc 0.4176) | same cause; acceptance fingerprint changed 0.4286→0.4176 (numerics differ) |
| qat MTP3 tg128 @8k int8 | 96.27 idle (fork) | **123.29** (acc 0.3212) | qat is now the FAST lane: +54% over nvfp4full on the same engine — upstream's all-NVFP4 A16 routes beat their BF16 small-t routes (the fork's engine was the opposite); round time 15.9 ms vs nvfp4full's 28.2 ms |
| MTP3 @262k int8 | 76.4 under load (+lm-head-draft) | 79.62; 82.73 with --lm-head-draft | long-context attention parity or better; lm-head-draft wins net despite lower acceptance (0.357 vs 0.418) |
| MTP7 @262k int8 | 131.9 idle | 81.1 (acc 0.278) | RESTORED same session (feat/mtp7 + dev squash 3f106e47): four coordinated bounds — product gate, 27B kMaximumMtpDraftTokens, family decode-frame domain, mtp_prepare_next_round op domain. Best current 262k lane; its 36.3 ms width-8 round still pays the width-route patchwork below |
| MTP3 @262k k8v4 | — | 81.58 (acc 0.4337) | new KV lane ≥ int8 on both axes |
| MTP3 @262k nvfp4-KV | — | 80.52 (acc 0.4654) | lightest KV (288 B/token/head), highest acceptance — promising for the long-context track |
| pp65536 int8 prefill | +14.2% from the unported key-split (absolute unrecorded) | 5,589 ± 245 | new baseline; backlog item 5's split+fp16-PV port is the known win |

Correction after a draft-width sweep + op-level A/B + profiles (same session): the headline
"qat is the fast lane" was a K=3-specific artifact. Per-ROUND times by verify width show a
route-cliff patchwork — nvfp4full: K1 13.0 ms, K2 14.8 (146.8 tok/s, best cell today), K3 28.2,
K5 20.7, K7 36.3; qat: K1 13.0, K2 24.6, K3 15.9, K5 32.2, K7 41.5. Both profiles are equal at K1 (~13 ms) and at
MTP0, so the base chains are equivalent; specific (profile x width) combos land on slow
registrations (the same class as upstream's known "A16 registered only through T=16" gaps; the
fork's old engine covered all widths via its own instantiations — that coverage is what the
rebase dropped). The 9 BF16-exception parents are real but small: op-bench A/B shows BF16
attn-input at T=1/4 runs ~1.3 TB/s (near ceiling) and simply pays the 4x byte ratio vs NVFP4
A16 (112.6 vs 47.2 us) — ~1 ms/round, not the 12 ms gap.

Actionable: pick the draft width per profile today (nvfp4full -> K2 = 146.8 tok/s; qat -> K3 =
123.3); root-cause the slow width cells (nvfp4full K3/K5/K7, qat K2/K5/K7) by diffing width
registrations per route family — likely missing instantiations per (qtype, T) in upstream's
plans; k8v4/nvfp4 KV remain quality-per-byte wins worth needle-validating for the long-context
track.

## Verification & measurement cautions

1. **GPU discipline**: verify the GPU is quiet before trusting a cell; run-to-run noise ±0.5%
   decode / ±6% prefill — same-binary alternating A/B pairs only. Kill strays with
   `taskkill //PID <pid> //T`.
2. **Fingerprint discipline**: ALL pre-rebase fingerprints are stale. Re-record greedy
   token-stream fingerprints and acceptance fingerprints on the rebuilt engine before using them
   as identity gates; remember bench cells seed decode with one-token prefill — always add a
   real-prompt greedy smoke (the PDL entry-trigger bug passed every bench fingerprint).
3. **PDL rule** (contract): producers publish at kernel END after their stores; consumers wait
   before the first dependent read, including early-exit paths. Upstream's remaining PDL sites
   now follow this; keep it true for new conversions.
4. **Windows editing/build**: write large/escaped content with proper tools — Git Bash heredocs
   mangled `\n` escapes twice this session (the load-plan test corruption). Non-ASCII literals
   as hex escapes; never pipe long-running GPU programs through `head`/`tail` (log to files and
   check exit codes — a masked build failure cost one cycle today). `constexpr std::sqrt` is not
   C++20; MSVC drops out-of-line `= default` explicit template specializations (the api_impl
   move bodies must stay explicit).
5. **Engine internals**: positional aggregate init silently nulls trailing members — grep ALL
   construction sites when adding fields. No `cudaStreamSynchronize`/`cudaMemcpy` inside
   captured graph bodies.
6. **Artifacts**: `models/qwen3_8_27b_nvfp4full.ninfer` (v2, weight-only NVFP4 module) and
   `models/qwen3_8_27b_nvfp4qat.ninfer` (v2) are the live fork artifacts — fully executable on
   every lane (Text/Vision/MTP/DFlash2) through the unified module binder; their directories were
   renamed in place to the upstream section names (header backups: `*.head.bak` beside each
   file; payload bytes and offsets untouched). `models/qwen3_8_27b_nvfp4full_w8.ninfer` (v3
   splice) runs the same binder on the W8 encoding. `models/qwen3_8_27b.ninfer` (groupwise-int
   core-only) loads with DFlash2 capability-absent. The official `models/qwen3_8_27b_nvfp4.ninfer`
   is the pre-suffix 1124-object core form. PR-form (W8-module) nvfp4full/qat artifacts have NOT
   been built; converters are ready (recipe v3/v2, splice not needed — full conversion from
   sources, ~19 GB each).

## Build & tools

```bash
powershell -ExecutionPolicy Bypass -File configure-ninja.ps1      # once per build dir
powershell -ExecutionPolicy Bypass -File build-ninja.ps1 [-Target <name>]
~/Workspace/ninfer-src/cvenv/Scripts/python.exe                   # torch env for converters
```

Presets: `qwen3.8-27b-nvfp4full-int8-262k*.bat` (MTP lane valid; the `-dflash2` one errors by
design until backlog 1 — with a W8-module artifact or after the port it runs). All
`hq-e8-2b-*` presets and the 524k/786k/1m variants are NON-FUNCTIONAL until the hq/1m port
(backlog 7). Eval harness unchanged: `eval/.venv` (uv, Python 3.12, EvalScope 1.10.0),
`PYTHONPATH=eval eval/.venv/Scripts/python.exe -m ninfer_eval …`.

Coding-campaign status carried over: reasoning/LBv2/codec phases complete in both cards; the
corrected-lane coding rerun (thinking ON, 32k output, timeouts count as failures) remains
deferred — on the rebuilt engine re-decide the lane first (the A16 T=16 blocker does not apply
to W8-module artifacts; on v2 artifacts only MTP lanes run today).
