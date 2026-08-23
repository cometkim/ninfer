# REVIEW — 1M-context track, M1 landing (`ace0eae0`, `0d33db2d`, `e9328d1e`, `99923bb5`, `b31dbf61`)

Desk review, 2026-08-23, of the commits that land ROADMAP WI-1 (envelope) and WI-3 (YaRN) plus the
first long-context measurements. Nothing run here; measured numbers are quoted from HANDOFF §5g and
the commit messages. Companion to `ROADMAP-1m-context.md`; the former hyperquant review lives in
`HANDOFF.md` since the 2026-08-23 fold.

## 1. Verdict

**WI-1 and WI-3 are implemented correctly, tested the right way, and M1 (524k, yarn:2) is
mechanically verified; the 1M envelope boots.** No defect found in the YaRN math, the angle
precision, the per-dtype envelope, or the option plumbing. What is open is not code: (a) two
contract gaps at option validation, (b) one robustness hole that already bit once, and — the
important one — (c) the first quality data says dense YaRN on this checkpoint is clean to ~200k
and fragile at 390k, so M1's "landed" status depends on the 524k needle suite, and the decisive
ablation (codec vs. YaRN) has not been run yet.

## 2. Checked and correct

- **YaRN table** (`rope_scaling.h`): reproduces HF `_compute_yarn_parameters` — ramp low=14 /
  high=22 at dim 64, base 1e7, original 262,144; `i<low → linear`, `i>high → linear/F`, in between
  `linear·((1−r) + r/F)` with `r=(i−low)/(high−low)`; attention factor `0.1·ln F + 1`. The test's
  reference tables (`test_rope_scaling.cpp`) agree with the independently computed ROADMAP §6 values
  at every printed digit (pairs 15/22/31 at F=4: 4.742398e-04 / 3.849816e-06 / 4.137043e-08; F=2 pair
  15: 4.905929e-04), and the test compares bit-exactly.
- **Where the factor is applied**: into cos and sin (`fixed_sincos`, both kernels), so q and k are
  scaled on the 64 rotated dims only (scores ×F² there) — the correct placement for partial rotary;
  not folded into `kAttentionScale`. Header text states the contract.
- **Angle precision**: scaled tables compute `pos·f` in FP64 with 2π reduction before `sincosf`;
  unscaled keeps the legacy FP32 product, and `check_linear_table`/`check_vision_table` pin the
  float-cast table to the former baked constants (flag-off bit stability). DFlash keeps its FP64
  route; the ~2e-13 constant correction is invisible at bf16 and is declared.
- **Plumbing**: factor resolved once per engine in `build_sequence_candidate` (static YaRN, HF/vLLM
  semantics) → `ProgramImplCore::rope_frequencies` (const) → `ExecutionCore` → `TextContext` → all
  four Text/MTP rope sites; Vision keeps `rope_vision_frequencies`; DFlash keeps linear and is
  rejected with scaling. CLI/serve `--rope-scaling none|yarn:F` parse with full-consumption check;
  (1, 64] validated; `docs/cli.md` updated; `meta.n_ctx` unchanged.
- **Envelope**: `kNativeContext` 1,048,576 for the 27B target; the GQA op bounds envelopes per
  cache dtype (`maximum_visible_keys_for`: U8 → absolute ceiling, BF16/I8 →
  `kGqaAttentionMaximumLinearVisibleKeys = 262144`) in all three validators; the bf16/i8 `PageIds =
  64` comments now say why; the hq decode (global block table) and prompt (linear scratch) routes
  have no staged page table, as §13/§17 established. Scenario J (window 262,208; 4,098 pages;
  16,388 keys/split) and the op-contract test (per-dtype ceilings; 2 GiB U8 scratch for the 35B
  geometry at 1M) pin it. int32 audit: scratch plane ne 1.07e9, MTP pages through `checked_i32`,
  splits clamp at 85.
- **Tests**: rope oracle consumes table + factor with YaRN-shaped cases at position 1,048,575 (the
  FP64 path is exercised, not just the table); rope_scaling bit-exact; gqa envelope contract;
  `test_hq_decode` J. Engine A/B: flag off answers identically (claimed); 524k yarn:2 boots,
  prefills, decodes, MTP3 verifies.

## 3. Findings

**R1 — option validation is not dtype-aware (contract/UX; fix before M1 is "shipped").**
`validate_target_options` checks `max_context ≤ Variant::maximum_context` (now 1M) but not the KV
dtype: `--kv-dtype int8 --max-context 524288` passes and dies later in
`gqa_attention_workspace_capacity_bytes` with "gqa_attention workspace: invalid profile or
interval". Add the per-dtype check at options validation with a message that names the fix
(`bf16/int8 KV supports ≤ 262,144 keys; use --kv-dtype hq-e8-2b`). In the same place: (i) warn
(or reject) `--rope-scaling` when `max_context ≤ original_positions` — static YaRN below the native
range only costs quality, which is exactly what the new `yarn_short_regression` suite measures;
(ii) warn once when `max_context > original_positions` and the factor is 0 (the unscaled OOD
control is a legitimate expert mode, not a default anyone should hit silently); (iii) the 35B target
keeps `kNativeContext` 262,144 but accepts `yarn:F` — either reject or document.

**R2 — `ExecutionCore::rope_frequencies` as a raw pointer with positional aggregate init.**
HANDOFF §5g records the segfault: five positional `ExecutionCore{...}` sites were missed and the
trailing pointer silently became null. That class of bug recurs on the next field. Make it a
reference member (`const ops::RopeFrequencies&`) — every aggregate site then fails to compile
until it supplies the argument — or give `ExecutionCore` a constructor. Cheap, and it turns the
grep-for-`{device, model, work,` caution into a compiler check.

**R3 — kernel parameter table (nit).** `RopeFrequencies` (1,028 B) rides as a by-value kernel
parameter: fine under CUDA 13's 32 KB limit and graph-captured as a static per-engine value, but
per-thread `inv_frequency[pair]` reads hit the constant bank at divergent addresses (serialized).
Only `kHalf` threads per token block do it, so it is not measurable today; if it ever is, move the
table to `__constant__`/global and pass a pointer. Also: `attention_factor == 1.0F` doubles as the
precision-profile selector. Make the profile explicit (a flag/enum in `RopeFrequencies`) so a future
table with factor 1 but scaled frequencies cannot silently take the FP32 product at 1M positions.

**R4 — table-builder guards (nit).** `rope_yarn_frequencies` has no `max(low,0)`/`min(high,dim−1)`
clamps and no `high == low` guard (HF adds 0.001). Irrelevant at dim 64 / 262,144; add them since
the function is generic.

**R5 — measurement framing: compare at equal length, then re-fit.** HANDOFF calls the 390k numbers
"ahead of the ROADMAP fits of ~1245 @512k and ~14 @512k". At 390k the session-4 prefill fit
predicts ~1,590 tok/s; measured 1,330–1,348 is ~15% *under* it — the quadratic term is ~23% larger
than the three-point fit (expected: the one-shot scratch re-decode and the scattered scratch
stores grow with the window). Re-fit with the 390k point: 1M ≈ 31 min, not 26. Decode: 20–21.5
tok/s at 390k against 25.3 at 262k on the session-8 kernel gives ~0.07 ms per 1k keys ⇒ 1M ≈
95 ms/step ≈ 10.5 tok/s — genuinely ahead of the roadmap's 8 (which was the session-4 kernel).
Record the 390k cells as fit points and restate the 1M projections from them.

**R6 — the quality result is the headline, and it needs one ablation before any conclusion.**
Greedy single-needle probes: clean at 32k (factor 1 and 4), 128k and 202k (yarn:2); at 390k
yarn:2 2 of 3 depths (depth 50% answered with an immediate `<|im_end|>`), yarn:4 failed the same
probe, and the *unscaled* OOD control partially retrieved. Three readings, in order of how much
they change the plan:
1. *Factor should track length.* yarn:4 at 1.5× native over-compresses the interpolated pairs;
   Qwen's guidance (factor ≈ target/native) says yarn:2 for anything ≤ 524k and yarn:4 only near
   1M. The eval suite already does this; the probe ladder should too (drop yarn:4 below ~800k).
2. *The `<|im_end|>` is a behavior failure, not a retrieval miss.* That pattern is what the
   HyperQuant paper attributes to per-vector bias compounding at long context (Table 6 / §6.2),
   so the 2.25-bit cache is as plausible a cause as YaRN at 390k keys. **The decisive ablation is
   the same 390k yarn:2 probe with INT8 KV** (12.9 GiB at 390k — fits nvfp4full's ~13.2 GiB only
   with an explicit `--kv-capacity`, so run it at ~350k if auto refuses). INT8 passes ⇒ the codec
   limits, and the BF16 sink/recent residual window (paper Table 13: +42% → +7.4% at 2 bps) is the
   cheapest fix and should precede WI-2; INT8 fails ⇒ YaRN/model limit, and M1 is "boots" rather
   than "landed" until the 524k suite says otherwise.
3. *One needle, one depth is not a gate.* The 32k yarn:4 "clean" probe is a single sample; the
   regression suite's 8k/32k × 5 depths is the check, and its result decides static vs
   per-request scaling (WI-3 policy).
   Until the 524k suite runs, M1 should be described as "mechanically verified, quality unproven
   past ~200k" in HANDOFF and ROADMAP §4.

**R7 — docs drift.** README "Current limits" still says `--max-context` is configurable up to the
native 262,144 limit; make it the per-dtype envelope (`docs/cli.md` already is).

**R8 — lever (b), for the record.** §17 recommended building (b) as warp specialization; HANDOFF
reports the prototypes lose everywhere (the mma accumulator fragments cap every 8-warp variant at
1 block/SM). The prediction was wrong in the way a prediction should be wrong — the experiment
settled it — and it closes the homogeneous-kernel gate: the decode phase is chain-latency-bound at
8 warps/SM with no in-design lever left, so Tier 2 exact V skipping is the next decode item, as
HANDOFF §1 now says.

## 4. Suggested order

1. R1 + R2 (an hour): dtype-aware validation, the scaling warnings, and the reference member.
2. R6.2 — the 390k INT8 ablation (one run) before any further YaRN or codec work; it chooses
   between the residual window and WI-2 as the next 1M item.
3. Run `scaled_524k` and `yarn_short_regression`; restate M1 accordingly; R5 re-fit in the roadmap.
4. WI-2 banding (HANDOFF §0's design is sound: partial-output epilogue + split reducer, per-band
   mask offset, `test_hq_prefill` parity at B±1).
5. R3/R4/R7 whenever the files are next touched.

## 5. Session 11 landing — WI-2 banding (`7a0252c2`), R1/R2/R7 (`7f49a4d5`), UTF-8 (`7337afb9`), LongBench v2 suites

Desk review, 2026-08-23 (later). Nothing run; measured numbers quoted from HANDOFF §5g/§5h and the
commit messages.

### 5.1 Verdict

WI-2 is implemented the way §3/R6 asked and it is correct; M2's memory blocker is gone (1M with
`--kv-capacity auto`, 3.37 GiB free, which matches the roadmap's 10 GiB-with-banding row). R1/R2/R7
are closed as suggested. The UTF-8 change makes the right engine-level call (lossy stream, not a
downed engine) but **ships the replacement character as 6-byte mojibake** — one-line fix, no test
covers it (N1 below). The measurement work finally separates the variables: the hq codec is
cleared through its full native envelope; what remains past 262k is YaRN/model behavior.

### 5.2 WI-2 — checked

- **Band-local staging and masks.** `gqa_prefill_stage_kv` addresses scratch row `k0 − key_base`
  and zero-fills rows past `min(max_query_abs, key_limit)`; the Carry kernel iterates tiles
  `[key_begin/Bc, ceil(min(key_end, max_query_abs+1)/Bc))`, masks `key ≥ key_end`, and drops
  `full_score_tile` for the last tile of a band. Bands are asserted 64-key aligned (the route
  throws otherwise); 262,144 is.
- **Carry state is exact.** The previous band's acc is stored un-normalized and un-rotated (bf16,
  `out`'s layout), re-rotated with the forward FWHT on resume and loaded into the C fragments
  through the 8-row smem window; `m` is row-uniform and `l` is split 0.25 per lane because the
  loop keeps per-lane partials that the epilogue's 4-lane sum recombines — the gate caught the
  ×4 (commit message), and the fix is the right one. The resume runs before K(0) staging so
  `k_s` is dead storage (4 warps × 8 rows × 256 f32 = 32 KB = the K tile, exactly). Index
  conventions for `carry_acc`/`carry_m`/`carry_l` match between store and load.
- **Band count follows the real window, not the envelope maximum.** The engine's prefill passes
  `chunk_envelope{visible, visible}` per chunk (`text_context_impl.h:1147`), so a 100k chunk in a
  1M-envelope engine runs one band, not four. (The decode-time `gqa_attention_cached` prompt route
  bands by the decode envelope maximum — no 27B caller today since widths > 6 don't exist there;
  it becomes a cost the day DFlash-class widths land — see WI-7.)
- **Workspace sizing** now sums scratch + carry inside one call (the "bad allocation" the first
  524k run hit); single-band envelopes keep the original instantiation bit-identical.
- **Gates**: six `test_hq_prefill` scenarios including spans 64/128/192 (final band short) and the
  engine-scale 390k-key case crossing tile 4096: min cos 0.999986 / row rel 0.45% — the same
  profile as single-band (0.999990); ≤ 4 bands at 1M means ≤ 3 bf16 round-trips of the
  accumulator plus two FWHT passes per boundary, which is what that delta is.
- Nit: `visible_keys` is the envelope's `max_visible_keys` under another name; one name, or a
  comment at the launcher that it is exact for prefill chunks and a maximum elsewhere.

### 5.3 R1/R2/R7 — closed as suggested

Dtype-aware rejection at option validation with a fix-naming message; `ExecutionCore::rope_frequencies`
is a reference member (the positional-aggregate class is now a compile error); operator notes for
scaling-below-native and unscaled-past-native flow through `MemorySummary` to the CLI summary and the
serve start log; README states the per-dtype envelope (the continuation lines lost their two-space
indent — renders fine as lazy continuation, cosmetic). The 35B+`yarn:F` split is documented rather
than rejected; acceptable.

### 5.4 Findings

**N1 — the new UTF-8 replacement is mojibake.** `frontend.cpp:576` appends `"ï¿½"` whose bytes
are `C3 AF C2 BF C2 BD` — the UTF-8 encoding of the three Latin-1 characters `ï ¿ ½`, i.e. U+FFFD
double-encoded — while `terminalize` (`:589`) correctly emits `"\xef\xbf\xbd"`. Every invalid
byte now publishes three visible characters instead of one `�`, and the two paths disagree. Fix:
use `"\xef\xbf\xbd"` (or `u8"�"`) at `:576`; this is HANDOFF caution 6 (heredoc/patch tooling
mangling non-ASCII) biting through the editor — keep non-ASCII out of source literals. Add a
frontend test that feeds an invalid lead byte and asserts the emitted bytes are exactly
`EF BF BD`, and one that holds a split 3-byte sequence across two batches; `tests/` has no
UTF-8 coverage today.

**N2 — error segregation (recorded by HANDOFF, endorsed).** The fix removes one fatal path; any
other exception thrown inside the decode round still reaches `fail_all` and downs the engine for
every request. A request-scoped boundary around per-lane output processing (fail that request,
keep the engine) is the structural fix; it should precede any serving of 1M-class requests where
rare degenerate tokens are more likely.

**N3 — reading the quality data.**
- **A2 is the key result:** true-262,070-token needles, hq vs int8, depths 50/80 — all four
  retrieve the exact code. That clears the 2.25-bit codec through its full native envelope (the
  R6 concern) and removes the codec from the >262k explanation.
- A1 (`yarn:1.5` at 390k, factor tracking length) fails with corrupted fragments; yarn:2/4 and
  unscaled all degraded differently at d50; d20 flipped pass→garble under banding whose math is
  oracle-exact to 0.45% row-rel. Together: past 262k the greedy single probe is on a knife-edge —
  a perturbation at the bf16-rounding level flips outputs — which is itself the finding: treat
  ≥ ~300k as "works on average, not per-sample reliable" until a suite quantifies it, and report
  per-sample reliability, not just means.
- lbv2: short native 0.383 (n=60) vs medium 524k yarn:2 0.40 (n=15). "No collapse" is fair as a
  qualitative statement, but the two subsets are different questions at different difficulty and
  n=15 carries ±0.13; the attribution cells are `lbv2_short` on hq vs int8 (same questions) and
  the full `lbv2_medium` run — both are in the suite config now.
- With the codec cleared, the remaining variable is YaRN/model, which is exactly where WI-3b's
  q-side temperature sweep (c ∈ 0.1–0.25) is cheap to run: it is the first experiment that can
  move the >262k cells without touching the cache.

**N4 — speed.** 524k banded: prefill 1,277 tok/s / decode 18 tok/s vs unbanded 1,330–1,348 /
20–21.5 — inside the ±5–10% noise band, as expected for ≤ 2 bands. R5's re-fit stands (1M ≈
~31 min prefill, ~10.5 tok/s decode); the WI-2 commit changed memory, not time.

**N5 — eval suites.** The LongBench v2 split (short → native 262k, medium → 524k yarn:2, long →
1M yarn:4; rule-scored `ANSWER:` line; samples beyond the envelope fail by construction) is the
right instrument; two notes: report the envelope-failed count per cell alongside accuracy (the
README says lower bound — make the runner print it), and pair `lbv2_short` with an int8 launch so
the codec question has a real-document answer, not only the needle one.

### 5.5 Order from here

1. N1 (one line + two tests) — before the next serving campaign; it is user-visible output.
2. Finish the attribution cells: `lbv2_short` hq vs int8, full `lbv2_medium` at 524k.
3. WI-3b items (i)–(ii): expose `t=c`, move the factor to the q side; sweep at 202k/390k.
4. N2 request-scoped error boundary.
5. Tier 2 decode (HANDOFF §1) and WI-7 probes (roadmap §7.2): the "fast enough" track, now that
   the capacity track has no blocker left.

### 5.6 Status after `40cfb666` (N1 re-applied) and the attribution cells

- **N1 — fixed in effect, fragile in form.** `frontend.cpp:576` now holds the bytes `EF BF BD`
  (verified), the new `test_utf8_replacement_bytes` asserts exactly one U+FFFD then continued
  generation, and every earlier fix survived the branch rebuild (banding constant, Carry kernel,
  reference member, dtype-aware validation checked). But the literal is a *raw* `"�"`, not the
  `"\xef\xbf\xbd"` escape `terminalize` uses and HANDOFF §5i's own lesson prescribes (§5i's
  "hex escape (raw bytes in the source)" is the raw form). On the Windows box MSVC reads source
  under CP949 — HANDOFF records the frontend test TU already failing there with C2001 for exactly
  this reason — and the engine was not rebuilt there after this commit. The previous mojibake line
  survived CP949 only because its three 2-byte pairs were valid CP949 double-byte characters;
  `EF BF BD "` has no such guarantee. Fix before the next Windows build: `"\xef\xbf\xbd"` (ASCII-only
  source) plus `add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/utf-8>)`, which also unblocks the
  test TU so the new test runs on the box rather than being review-verified. Habit: run the
  README's post-rebuild parity diff after every branch rebuild — this fix was dropped by one.
- **Codec question closed at native range.** `lbv2_short` at 262k on the same 60 questions: int8
  0.40 vs hq 0.3833 (one question); with needle A2 (all four cells exact) the hq-e8-2b codec is
  indistinguishable from int8 across its native envelope on synthetic retrieval and real documents.
  `lbv2_medium` 524k yarn:2 0.40 (n=15) stands as "no real-document collapse past native"; the
  remaining cells are the full medium (215), `lbv2_long` at 1M, and envelope-failed counts (N5).
- Queue (unchanged, now in HANDOFF §5i too): WI-3b q-side factor move and the temperature/beta grid
  at 202k/390k, N2 request-scoped error boundary, WI-7 MTP tokens/round gates.

## 6. Session 12/13 landing — q-side YaRN temperature + tunable ramp (`209e1357`), `/utf-8` build (`c7f3f5bd`), YaRN ablation cell (`347fbf81`)

Desk review, 2026-08-23 (later still). Nothing run; measured numbers from HANDOFF §5j/§5k and the
commit messages.

### 6.1 Verdict

WI-3b (i)+(ii) landed exactly as specified and the implementation is correct: the attention factor
is a q-side temperature (`f²` on the rotated dims of q, k unscaled, cached K factor-free), the
single-tensor rope form requires an explicit `RopeSide`, every call site carries the right side
(attn/MTP-tail/DFlash-propose/vision pair forms; MTP k-append Key; MTP draft q Query; DFlash
context k Key), factor 1 is a bit-exact no-op, and the parser/validation/summary plumbing is
complete. The measurements then did what the knobs were for: **temperature is ruled out as the
~1.5×-native lever** (t ∈ 0.1–0.25 all clean at 304k and all garbled at 390k), the clean regime is
now known to extend to 304k tokens (1.16× native), and MTP acceptance holds 202k → 304k (d3 2.58 →
2.71 tokens/round) — WI-7 gate (i) is satisfied. The `/utf-8` change is right and the literal is in
escape form; the `configure-ninja` quoting bug it found (`-NoBenchmarks` never worked) is a real
fix.

### 6.2 Checked

- **Kernel**: `rope_q_scale = f²` multiplied into q's `c0,c1,s0,s1` (fixed and split kernels) and
  `head_scale` in the generic kernel; k path untouched; the cos/sin caches no longer carry the
  factor, so the FP64/FP32 angle profiles are unchanged. `(f·q)(f·k) = (f²·q)·k` — one bf16
  rounding difference, as claimed. `rope_generic_kernel` made `inline` to stop the bench/launcher
  ODR clash — fine.
- **API**: `RopeSide` on the single-tensor form (default-less, so a missing side is a compile error
  — the right call given its callers are both sides); the launcher dispatches Query/Key to the
  q/k template slots of the same fixed kernels.
- **Builder**: `rope_yarn_frequencies(theta, dim, original, factor, temperature, beta_fast,
  beta_slow)`; ramp clamps `low ∈ [0, half−2]`, `high ∈ [low+1, half−1]` — covers R4's
  `high == low` hole. Nit: HF clamps `high` to `dim−1` (= 2·half−1), so for exotic `bs` where the
  correction exceeds 31 pairs the tables diverge (HF leaves the top pairs blended; ours fully
  interpolates them). Irrelevant at the defaults; worth one comment line.
- **Parser**: `yarn:F[,t=c][,bf=n][,bs=n]`, `from_chars` with full consumption, duplicate/unknown
  fields rejected, engine-side range validation (`t > 0`, `bf > bs > 0`), knobs only effective
  with factor > 1 (validated). CLI summary and serve start log print non-default knobs.
- **Tests**: oracle takes `q_side` (q gets `f²`, k nothing); new scaled single-q case at
  1,048,570; `test_rope_scaling` checks the temperature changes only the factor, the ramp moves in
  the right direction for `bf=16` / `bs=2` without touching the extrapolated prefix, and all
  rejections. Adequate.
- **Build**: `/utf-8` for C/CXX and `-Xcompiler=/utf-8` for nvcc host passes; frontend.cpp has zero
  non-ASCII bytes now (the literal is `"\xef\xbf\xbd"`). `configure-ninja.ps1`: quoted `-D` args,
  `BUILD_TESTING` default ON, dead `-TuLog` removed. MSVC portability shims in two test TUs.

### 6.3 Findings

**Q1 — the 390k failure is still unattributed; three zero/near-zero-code cells would settle it.**
Temperature is out; the codec is cleared ≤ 262k; `yarn:1.5/2/4` all fail at 390k while the
unscaled control partially retrieves. Remaining candidates and the cheapest discriminator for each:
(a) *KV codec beyond 262k* — the int8 control "isn't runnable" only because of
`kGqaAttentionMaximumLinearVisibleKeys`/`PageIds = 64`; for a one-off experiment raise the linear
ceiling to 524,288 and `PageIds` to 128 (+256 B static smem in the bf16/i8 decode kernels) and run
the 390k probe with int8 (12.9 GiB at 390k fits nvfp4full with an explicit `--kv-capacity`). This
is the single most informative missing datum. (b) *Weight quantization* — the same 390k probe on
the `groupwise-int` artifact + hq (zero code) separates NVFP4 from the rest. (c) *Greedy collapse*
— the same probe at temperature 0.6 / top-p 0.95 (zero code); "garble then `<|im_end|>`" is a
classic greedy degeneration signature. If all three fail the same way, it is the checkpoint under
dense YaRN past ~1.2× native, and the roadmap's M1 claim should say "clean to ~300k".

**Q2 — the lbv2 ablation lowers the urgency of per-request static YaRN.** Same 15 medium samples,
native unscaled 0.333 vs 524k yarn:2 0.40; paired-14 5/14 vs 6/14 with identical correct sets and
one budget-flip differential. At n=14 that is "no measurable YaRN penalty at 62k–242k-token
prompts", not a win — and it means the global `--rope-scaling` flag is not visibly taxing
real-document work below native. Per-request static remains the clean design (WI-3b), but it can
wait behind Q1 and WI-7.

**Q3 — the bf/bs ramp grid is rightly deprioritized.** With no temperature signal at the failing
length, a ramp sweep would be fishing; keep the knob, spend the GPU-hours on Q1.

**Q4 — MTP graph allowance tiers.** 512 MiB at ≤ 524k is measured (427–437 MiB at ~304k); 1024 MiB
at ≤ 1M is an unmeasured extrapolation, and an overflow is a hard engine-construction failure.
Boot MTP3 once at the 1M envelope before anything relies on it, and consider degrading to eager
MTP (or MTP off with a logged note) on allowance overflow instead of failing construction.

**Q5 — the frontend test still cannot execute on the Windows box** (`official_tokenizer()` and the
template fixtures read the upstream maintainer's absolute `/home/neroued/...` paths). Make the
fixture root configurable (`NINFER_TEST_FIXTURES` env or a repo-relative default) so
`test_utf8_replacement_bytes` runs where the bug was found; "Linux-fixture-verified" is one step
removed from the box that ships.

**Q6 — WI-7 gate (i) is met.** MTP at 202k: d2 2.26 / d3 2.58 / d4 2.66 / d5 2.75 tokens/round
with d3 the throughput optimum (63.5 tok/s); 304k d3 2.71 at 57% — acceptance tracks output
coherence, not context. That is ≤ 3 by a margin, so a DFlash-2-class 4–4.8/round would be +50–80%
per round; the remaining zero-code gate is (ii), the llama.cpp acceptance-vs-context trace.

**Q7 — small items.** Fixture labels are nominal (char-ratio) — actual tokens 24.7k / 202k / 304k /
390k are now recorded; rename the files or keep the table next to every result that cites them.
README refresh is accurate; the duplicated intro/table block is gone. The MSVC constexpr-`sqrt`
test failures are upstream and unrelated.

### 6.4 Order

1. Q1's three cells (the int8 one needs the 2-line ceiling/PageIds change for the experiment).
2. WI-7 (ii) llama.cpp trace; then decide the DFlash 2 port.
3. Q4 one-time 1M MTP3 boot; Q5 fixture root; per-request static YaRN when convenient.

## 7. Session 14 landing (`0bfb8580`) — Q1a inverts the attribution; Q4/Q5 closed

Desk review, 2026-08-23. Measured numbers from HANDOFF §5l and the commit message; the one-off
experiment patch is confirmed reverted in HEAD (`kGqaAttentionMaximumLinearVisibleKeys = 262144`,
`PageIds = 64` in both linear decode kernels).

### 7.1 The result

**Q1a — int8 KV at 390,033 tokens retrieves the needle exactly; hq at the same prompt garbles under
every temperature and sampling mode.** Every earlier 390k failure (yarn 1.5/2/4, unscaled, greedy
and sampled) ran on hq. Q1c (temperature 0.6 / top-p 0.95) garbles identically, so it is not greedy
collapse. Q1b (groupwise-int weights) was skipped for lack of the artifact — no longer needed for
the attribution. Together with A2 (hq = int8 through 262k on needles) and the lbv2 cells (hq = int8
on real documents at native): **the hq-e8-2b codec is clean through its native envelope and becomes
the binding constraint past it; YaRN is not the cause of the >262k failure.** §6.3 Q1 was written
as a conditional and this is the branch that fired; §5.4 N3's "codec cleared" was correct only
as stated — through 262k.

Reading: this is the HyperQuant paper's own Table 6 / §6.2 mechanism — deterministic per-vector
reconstruction bias that is harmless at short windows and compounds over hundreds of thousands of
keys (the engine's operating point is 2.25 bps with no rotation and no dither, exactly the "none"
row). The failure is a generation-coherence collapse ("garble then `<|im_end|>`"), i.e. the recent,
locally attended keys lose fidelity first.

Side datum: int8 decodes 49.3 tok/s at 390k vs hq ~20.6 — the hq long-window decode gap widens past
native (0.42 of int8), reinforcing Tier 2 as the decode priority.

### 7.2 What it changes

- **The 1M track now has a codec-side blocker, not a RoPE one.** hq is usable to ≈304k (1.16×
  native, measured clean) and not at 390k. M1 (524k yarn:2) is therefore "boots and prefills; not
  usable end-to-end on hq" until a codec-side fix lands; M2/M3 inherit that.
- **Levers, in order of cost/evidence** (all from the paper and the OSCAR design, none new):
  1. **BF16 sink + recent residual window** — keep the first S (sinks) and last W (e.g. 32–128)
     positions' K/V exact in a small ring and attend over hq pages + ring; paper Table 13 at 2 bps:
     +42% → +7.4% PPL for ≈0.1 bps; OSCAR ships the same BF16 sink/recent layout at 2.28 bpe. In
     ninfer: the TC decode kernel already has two KV-source policies and the fused append has the
     new K/V in bf16 at append time; the prompt route has each chunk's bf16 K/V in hand to fill the
     ring for the tail. Memory is megabytes. This is the cheapest fix and it targets the observed
     symptom directly. Gate: the 390k and 592k probes, then the 524k/1M suites.
  2. **Subtractive dither (K and/or V)** — strict per-vector inner-product unbiasedness (paper
     A.3); regenerate U per (layer, head, position) from a Philox counter at both ends; cost ≈ one
     extra E8 nearest-point per word at decode, i.e. roughly +50–100% of the decode phase on a
     latency-bound kernel. Principled, more expensive; try after 1 if 1 is not enough.
  3. **Raise the bf16/int8 envelope as a product option** — the experiment patch is the whole
     change (linear ceiling 262144 → 524288, `PageIds` 64 → 128, +256 B static smem per kernel) plus
     the R1 contract test/docs; int8 at ~393k is 12.9 GiB and fits nvfp4full only with an explicit
     `--kv-capacity`. Gives a clean ~390k deployment today at int8 quality and speed — not the 1M
     track, but the best long-context configuration currently measured.
  4. **Asymmetric K precision** (K3/V2-style) — a format change; last resort.
- **ROADMAP §7 decision** is unchanged in substance and sharpened in wording: INT8 remains the
  default; hq is the ≥512k opt-in *once* a residual window or dither restores coherence past
  ~300k; until then the practical ceilings are ~300k on hq and ~390k on int8 (with the envelope
  raised).

### 7.3 Q4, Q5, and the rest

- **Q4**: MTP3 boots and decodes at a 592,518-token prompt under the 1M envelope; the >524288
  graph class used 256 KiB against its 1 GiB tier — conservative and safe. Output garbles there
  (hq, consistent with Q1a).
- **Q5**: the frontend suite runs and passes on the Windows box (26 tests) — `NINFER_TEST_FIXTURES`
  + `tools/artifact/extract_frontend_fixtures.py`, the `.gitattributes -text` fix for the jinja
  fixtures under `core.autocrlf=true`, and a diagnostic `catch` in `main`. Two findings of theirs
  worth underlining: the committed replacement test could never have executed anywhere (a raw
  `0xFF` vocab entry cannot survive nlohmann's JSON round-trip — the test now uses the byte-level
  alphabet character `ÿ` → 0xFF, i.e. the production decode path); and the autocrlf hash
  mismatch had silently blocked every template-resolving test on the box. Both are exactly the
  class of "verified elsewhere" gaps §5.6 pointed at; they are closed now.
- Builder comment for the HF `high`-clamp divergence landed (§6.2 nit).

### 7.4 Order from here

1. Residual window (sink + recent) on the hq path; re-run the 390k/592k probes and `lbv2_medium`.
2. In parallel, the product decision on the int8 envelope raise (contract test + docs).
3. If the window is insufficient: subtractive dither on K.
4. WI-7 (ii) llama.cpp acceptance trace; Tier 2 decode.

## 8. WI-8 design note (`f17b5fff`) — review before implementation

Desk review of HANDOFF §5m. The seams are right (the hq branch of the TC decode tile fetch, the
fused append, the A2 append, the scratch kernel's row decode), K stored RHT-rotated matches both
the decode QK frame and the prefill scratch frame, S=32/W=128 keep tiles whole, the per-slot planes
are megabytes, and "empty residual tensors = feature off" keeps the default path bit-compatible.
Three additions before the code is written, one expectation, one nit:

**D1 — ring validity under prefix reuse, retained sequences, and slot reassignment.** The planes are
owned per *sequence slot*, but prefixes outlive slots: a retained prefix adopted by a new request, or
a retained-but-inactive sequence reactivated after its slot served someone else, arrives with hq
pages whose last W keys are no longer in any ring. Without a rule the first decode steps read stale
or zero rows for `key ≥ window − W` — silently wrong attention. Pick one: rebuild the ring from the
hq pages on adoption/reactivation (a tiny kernel: W rows × layers, quality equals hq for those keys
until they roll out), or carry a per-slot "ring valid from key X" watermark and fall back to hq
rows below it. MTP verify rewinds are fine by construction (slot = `key & (W−1)`, rejected positions
are re-appended), but say so.

**D2 — prompt-phase exactness needs the fresh chunk, not just the ring.** The paper's window is
per-query ("implemented per-query, charged exactly"). In the hq prompt route every key in
`[0, visible)` — including the chunk being prefilled — is read back from the scratch, i.e. quantized,
so with only the ring a query at position *p* inside a 1024-token chunk has exact keys only for
`[window−128, window)`: the chunk's last queries get ≤128 exact recent keys and its first queries get
none. The bf16/int8 prompt routes already take current-step keys from `input.k/v` (`from_new`); do
the same here: the scratch kernel's third source copies the current chunk's rows from the call's
bf16 `k`/`v` (K rotated once) and the W keys before the chunk from the ring. Then every prefill
query has ≥ W exact recent keys, and "the window covers both phases" is actually true. Cost: none
(the tensors are in hand).

**D3 — the MTP layer is a separate pool.** `gqa_attention_cached` on `mtp_kv` runs the same kernels,
so the MTP layer family needs its own sink/ring planes (the 17th layer), filled by the MTP k-append
site. DFlash is not on the 27B target.

Expectation: the window targets the observed symptom — generation coherence, which is local
attention over the most recent keys — and the paper's Table 13 is PPL at short context. Far-needle
fidelity at 390k+ still depends on the hq long-window K rows; if the 390k probe becomes coherent
but misses the needle, that is the signal for lever 2 (subtractive dither on K), not a reason to
widen W.

Nit: the ring/hq boundary at `window − W` is not tile-aligned in general (window is arbitrary); the
per-row source selection makes that harmless, so the "no tile straddles a boundary" property only
needs to hold for the sink (keys < 32 = the first tile). State it as per-row selection and drop the
alignment claim for the ring.

Memory check: (32+128) × 4 heads × 256 × 2 B × 2 roles = 640 KiB per layer per slot; ×17 layers ≈
10.6 MiB per slot — megabytes, as stated. Vision-at-1M probe (1.93 GiB free with vision on the 1M hq
envelope) is consistent with the earlier 3.37 GiB minus ~1.3 GiB.
