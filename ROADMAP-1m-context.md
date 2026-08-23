# ROADMAP — 1M context (YaRN) for Qwen3.8-27B on one RTX 5090

Desk note, 2026-08-22. Companion to `HANDOFF.md` (hq-e8-2b state; the former REVIEW/AUDIT
documents were folded into it at the 2026-08-23 extraction). **Nothing here was measured beyond
the tables already in HANDOFF.md and the nvfp4full model card**; every 512k/1M number is a
projection from those points and is labelled as such. Validate the fits at 512k before trusting
the 1M column.

Scope: one request, text, `qwen3_8_27b` / `nvfp4full` (16.03 GiB device weights), `--kv-dtype
hq-e8-2b`. Target context: Qwen's published YaRN figure (factor 4, 1,010,000 tokens) with the
engine envelope at 1,048,576.

## 0. TL;DR

- **Fits only with hq KV.** Payload is 9 KiB/token (16 layers × 2 roles × 4 heads × 72 B) →
  9.0 GiB @1M; int8 is 33 GiB, bf16 64 GiB. Above ~400k tokens hq is the only format that fits.
- **Three engine blockers, in order of cost:** (1) the 262,144 envelope constants, (2) the hq
  prompt route's one-shot rotated-bf16 scratch, sized by the per-sequence envelope (4 KiB/token of
  `--max-context`) → 4 GiB @1M, (3) time:
  projected ~26 min for one 1M prefill and ~8 tok/s decode with today's hq kernels.
- **YaRN itself is small** (a 32-entry frequency table + one scalar into cos/sin + fp64 angle
  reduction), but it must be folded into cos/sin — not into `kAttentionScale` — because rotary is
  partial (64 of 256 dims).
- **Milestone order:** M1 = 524k (factor 2) needs only the envelope + YaRN and fits today;
  M2 = 1M needs scratch banding; M3 = "usable 1M" needs the HANDOFF §1 decode levers.
  Gate each milestone with a needle sweep; the N2 escalation fix and the tightened codec
  oracle gates are landed — finish the accuracy campaign first.
- **Decision (§7):** for the goal as stated (fit + intelligence + fast enough), the default
  profile is nvfp4full + INT8 KV today and INT4-G64 KV next; 2-bit KV is the ≥512k/1M opt-in only.
  The milestones below are the 1M track, gated on 1M being re-affirmed as a goal.

## 1. Baseline (measured — HANDOFF.md, nvfp4full model card)

| item | value |
|---|---|
| device weights (nvfp4full) | 16.03 GiB |
| free after startup with 262,144-token INT8 KV | 4.91 GiB → **~13.2 GiB available for KV + attention scratch** |
| hq KV payload @262k | 2.25 GiB (int8 8.25) |
| hq prefill tok/s @32k / 128k / 262k | 6041 / 3400 / 2159 |
| hq decode tok/s @32k / 128k / 262k | 39.9 / 30.0 / 21.2 (int8 59.2 / 52.2 / 46.8) |
| hq decode kernel @32k, grid.y=85 | 0.31–0.37 ms per KV layer |
| native ceiling | `kNativeContext = 262144` |

## 2. Projections (fits from the three measured points — PROJECTED)

Memory (hq, available ≈ 13.2 GiB):

| context | hq KV (9 KiB/tok) | one-shot prompt scratch (4 KiB/tok) | total | headroom |
|---|---:|---:|---:|---:|
| 262,144 | 2.25 GiB | 1.00 GiB | 3.25 | 9.9 |
| 524,288 | 4.50 | 2.00 | 6.50 | 6.7 |
| 1,010,000 | 8.67 | 3.85 | 12.52 | 0.6 |
| 1,048,576 | 9.00 | 4.00 | 13.00 | ~0 |
| 1,048,576 with scratch banded to ≤262,144 keys | 9.00 | 1.00 | 10.00 | 3.2 |

MTP adds one more hq layer (+1/16) when enabled; the decoder-state spec applies `plan.kv_dtype`
to the MTP layer too (`layouts_impl.h:111-123`).

Time (session-4 points; session 7's parallel packer since moved hq decode to 52.0 @32k / 25.3 @262k
CLI and tg128 to int8 parity, and session 8's tensor-core tile-source kernel (lever a) moved the
same-session hq/int8 ratio to 0.834 MTP0 / 0.852 MTP3 at 32k with tg128 at 0.938 — re-fit before
trusting the 1M column; the decode kernel is latency-bound at ~2.1 G rows/s, not ALU-bound):

- prefill fit: `s/token = 1.23e-4 + 2.6e-9 · (n/2)` (predicts 3412 tok/s @128k vs 3400 measured)
  → 512k ≈ 1245 tok/s avg, **~7 min**; 1M ≈ 675 tok/s avg, **~26 min**. Re-fit with the 390k
  measurement (1,330–1,348 tok/s vs 1,590 predicted: the quadratic term is ~23% larger) → 1M ≈
  **~31 min**; decode at 390k (20–21.5 tok/s) on the session-8 kernel gives ~0.07 ms/1k keys ⇒
  1M ≈ **~10.5 tok/s** (REVIEW-1m-context.md R5). ~90% of the 1M figure is
  the dense attention of the 16 full-attention layers; GDN/GEMM are linear.
- decode: 25.1 / 33.3 / 47.2 ms per step at 32k / 128k / 262k → ~0.096 ms per 1k keys →
  512k ≈ 72 ms (**~14 tok/s**), 1M ≈ 123 ms (**~8 tok/s**) with the current hq kernel. MTP3 roughly
  doubles committed tok/s. int8 cannot be measured at these lengths (does not fit).

## 3. Work items

### WI-1 — Engine envelope 262k → 1M (contract change)

- Constants: `src/targets/qwen3_6_27b/impl/config.h:91` (`kNativeContext`) →
  `variant.h:36` (`maximum_context`, validated at `layouts_impl.h:539`);
  `include/ninfer/ops/gqa_attention.h:13` (`kGqaAttentionMaximumVisibleKeys = 262144`, checked
  in `gqa_attention.cpp:213/290/404`).
- Per-dtype envelope or kernel fix: the bf16/i8 decode kernels stage ≤ `PageIds = 64` physical
  pages per split ("262144-key envelope spans at most 49 pages", `gqa_attention_decode_bf16.cuh:38`,
  `gqa_attention_decode_i8.cuh:82`); the split count saturates for large windows
  (`gqa_attention_decode.cu:20-62`), so keys-per-split — and pages-per-split — grow ~4× at 1M.
  The hq kernel computes row addresses from the global block table (`gqa_attention_decode_hq.cuh:76,
  191`) and has no such table. Simplest: let only U8 accept > 262,144 visible keys; keep BF16/I8 at
  262,144 (they cannot hold it anyway) and assert it.
- Split/partial capacity: `gqa_attention_split_capacity` sizes partial buffers from the envelope
  (`gqa_attention.cpp:322-411`); confirm the split policy above 16,390 keys for 1M windows.
- int32 audit: `Tensor::ne` is int32 (`tensor.h:14`) — the scratch plane is 256×4×1,048,576 =
  1.07e9 elements (fits; 2M would not); page counts go through `checked_i32`
  (`layouts_impl.h:106`); block tables are I32 [logical_pages, rows] — 16,384 pages @1M.
- Surfaces: `--max-context` / `--kv-capacity auto`, `ninfer-serve` `meta.n_ctx`, `docs/cli.md:183`,
  `docs/maintainer/qwen3.6-27b-model.md:37`, `paged-kv-cache.md` envelope statements.
- Fixtures: `longprompt_512k.json` / `longprompt_1m.json` (262k one is 1 MB); a ≥1M-token
  `bench_corpus.ids` for `ninfer_bench -p` (`tools/bench/make_bench_corpus.py` tiles text).
- Acceptance: boots with `--kv-dtype hq-e8-2b --max-context 524288` (M1) / `1048576` (M2, after
  WI-2), `--kv-capacity auto` ≥ the context; hq oracle gates (`test_hq_decode`, `test_hq_prefill`)
  extended with a > 262,144-key scenario; bf16/i8 still reject > 262,144.

### WI-2 — hq prompt-route scratch banding (the 1M memory blocker)

- Today: `gqa_attention_workspace_capacity_bytes` reserves 2 × `max_visible_keys` × 4 × 256 × 2 B
  (`src/ops/wrapper/gqa_attention.cpp:437-451`); the prompt path allocates both planes over the full
  envelope (`:493-498`, `:550-555`), the scratch kernel decodes the whole visible window once, then
  the Rotated FA2 prompt kernel runs over it.
- Sizing note: the envelope `text_envelope{1, plan.capacity}` (`layouts_impl.h:237`) is already
  per-sequence — `plan.capacity` is `options.max_context` (`:719`), the pool is `plan.kv_capacity` —
  so the scratch is a constant 4 KiB/token of `--max-context` (the S7 misread — pool vs
  per-sequence capacity — is withdrawn; the hazard comment at that line is the guard).
  Banding the window is the fix; there is no cheaper sizing fix ahead of it.
- Change: process the visible window in key bands of ≤ B keys (B = 262,144 keeps today's 1 GiB;
  65,536 = 256 MiB), carrying the online-softmax state (m, l, acc) across bands — either the prompt
  kernel takes/emits per-row partials (sequential bands, rescale in place; workspace
1024 × 24 × 256 × 4 B ≈ 24 MiB + m/l) or decode-style split partials + combine. Do **not** move
decode into per-query-tile (that multiplies decode work by chunk/Br).
- Acceptance: `test_hq_prefill` parity across band boundaries (window > B, window = B±1);
  prefill throughput within run-to-run noise of one-shot at 128k/262k; @1M scratch ≤ B × 4 KiB.

### WI-3 — YaRN RoPE

- Replace `theta` with a rope config (inv-freq table[32] + attention factor) in `ops::rope`
  (`include/ninfer/ops/rope.h:36,41`, `src/ops/wrapper/rope.cpp`, `src/ops/launcher/rope.cu:76-196` —
  fixed paths currently keyed on `theta == 1.0e7F` at :80/:92/:144/:154 — and
  `src/ops/kernel/rope.cuh`: `kTextRopeInvFrequency[32]`, `fixed_sincos`).
- Table (HF `_compute_yarn_parameters`, dim 64, base 1e7, original 262,144, β_fast 32, β_slow 1,
  truncate): ramp low=14, high=22 → pairs 0–14 unchanged (extrapolated), 15–21 blended with
  `extrap = 1 − (i−14)/8`, 22–31 divided by the factor. See §6 for the factor-4 table.
  `attention_factor = 0.1·ln(factor) + 1` = 1.0693 (f=2) / 1.1386 (f=4), multiplied into **both**
  cos and sin (q and k both scaled; scores ×1.1434 / ×1.2965 through the 64 rotated dims only;
  the 192 pass-through dims are untouched — never fold this into `kAttentionScale`).
- Precision: the text path computes `float(pos) * freq` in fp32; at pos ≈ 1M pairs 0–1 have ulp
  0.0625 rad (±0.03 rad). HF has the same fp32 product, but `fixed_sincos<DflashText1D>` already
  does fp64 range reduction — reuse it for text.
- Call sites: Text `text_context_impl.h:824` (`attn_mix`), MTP `:385` (`mtp_forward_tail`), `:489` and `:529` (`mtp_prefill_chunk`); Vision rope unchanged.
- Policy: K is cached post-RoPE, so the factor is fixed per sequence. Start with a static engine
  flag (`--rope-scaling yarn:F`, HF/vLLM semantics; Qwen warns it degrades short prompts — use
  factor 2 for a 524k deployment). Keep the table behind a per-slot pointer so a per-request
  factor (1 for ≤262k requests, chosen at admission; prefix reuse keyed by factor) is a follow-up,
  not a redesign.
- Tests: `tests/ops/test_rope.cpp` with YaRN tables at positions up to 1,048,575 vs an FP64
  oracle; bit-level parity of the generated table against HF's `_compute_yarn_parameters`;
  engine A/B: flag off ⇒ greedy output unchanged.

### WI-3b — YaRN quality knobs (assessment of the 2026-08-23 advice, and the plan)

Advice received: make the attention temperature (mscale) and beta_fast/beta_slow configurable,
add randomized position sampling for training, support hybrid/partial scaling and per-layer
factors, and optionally entropy-adaptive temperature. Assessed against what ninfer is — an
inference engine over a frozen checkpoint with a partial-rotary (64 of 256 dims) attention stack:

| advice item | verdict for ninfer | why / how |
|---|---|---|
| 1. tunable attention temperature | **Do it — cheapest knob with the best rationale here.** | `attention_factor` is already a host-side float in `RopeFrequencies`; exposing `--rope-scaling yarn:F[,t=c]` (factor = `c·ln F + 1`, default 0.1) is a parser change, no kernel work. Ninfer-specific reason to expect a different optimum: only 64/256 dims rotate, so the factor scales only the rotary share of the logit — if that share is ρ, the effective logit sharpening is `1 + ρ(f²−1)`, not `f²` (f=1.139 ⇒ ≈1.07–1.15 instead of 1.30 at ρ≈0.25–0.5). The suggested 0.1–0.25 range is exactly where that dilution points. Caveat: HF/vLLM validated Qwen's 1M claims with the same diluted 0.1, so this is an experiment, not a correction. |
| 1b. per-layer / per-head temperature | **Do it as a q-side scale, not in the RoPE tables.** | Today the factor is folded into cos/sin on both q and k, so it is baked into every cached K row and fixed per engine. `(f·q)·(f·k) = (f²·q)·k`: apply `f²` to q's rotary dims only (rope kernel: factor² on the q path, 1.0 on the k path) and the KV cache becomes factor-free. Then the temperature is a runtime knob — per request, per layer, per head — that never invalidates the cache or prefix reuse and is graph-safe as a device-side table `[16 layers × 24 heads]`. Same numerics up to one bf16 rounding. This is the enabling change for items 1/1b/5. |
| 2. tunable beta_fast / beta_slow | **Expose (trivial), search only with evidence.** | Builder arguments; `yarn:F[,bf=32][,bs=1]`. They move the ramp (14/22 today) i.e. which pairs extrapolate vs interpolate. No prior that Qwen's defaults are wrong for this family; each evaluation point at 390k costs ~5 min prefill × 5 depths, so keep the grid small and run it after the codec/YaRN ablation (REVIEW-1m-context.md R6), otherwise a codec failure gets "fixed" by the wrong knob. |
| 3. randomized position sampling for training | **N/A.** | No training path exists or is planned here; the checkpoint is frozen and a 27B fine-tune is out of a single-5090 scope. The only training adjacent to this track is the speculative drafter (WI-7), which is z-lab's, not ours. |
| 4. hybrid / partial scaling | **Already the case; per-layer factors are one table away.** | YaRN's ramp *is* "low frequencies interpolated, high frequencies untouched" (pairs 0–14 unchanged at F=4, 22–31 ÷F). The remaining knob is non-uniform per-pair scaling: `RopeFrequencies` is a free-form 32-entry table, so LongRoPE-style searched tables ([arXiv 2402.13753](https://arxiv.org/abs/2402.13753), [LongRoPE2 2502.20082](https://arxiv.org/abs/2502.20082)) are consumable with zero kernel work; the search itself is offline (HF model, perplexity/needle objective) and expensive. Per-layer *frequency* tables are possible (the table is per rope call) but each layer's K is then position-encoded differently — static per engine, fine; evidence-free without a search. |
| 5. entropy-adaptive temperature | **Skip as a control loop; keep the static per-head scale from 1b.** | Entropy needs a Σp·log p accumulation in the fused decode kernel plus a feedback path that makes outputs run-dependent; the literature's gains come from a *static* temperature vs length, which 1/1b already give. Revisit only if the per-head static sweep shows head-dependent optima. |
| "FP32 for very long context" | **Done better.** | Scaled tables already compute and 2π-reduce the angle in FP64 (`rope.cuh`), which the FP32 product needs at 1M (±0.03 rad on the low pairs). |

Plan (after R1/R2 and the 390k INT8 ablation): (i) `--rope-scaling yarn:F[,t=c][,bf=..][,bs=..]`
parser + builder args + docs; (ii) move the factor to the q side (rope kernel `q_factor = f²`,
`k_factor = 1`; contract note that cached K is factor-free; `test_rope` oracle updated — flag-off
output unchanged because f=1); (iii) per-layer/per-head q-scale table as a follow-up; (iv) grid:
c ∈ {0.1, 0.15, 0.2, 0.25} × (bf, bs) ∈ {(32,1), (16,1), (32,2)} at 202k and 390k under yarn:2,
five depths, greedy — ~2 h per length on the needle fixtures; promote a setting only if it wins at
both lengths and does not regress the 32k factor-4 short-prompt cell. Report alongside R5's re-fit.

**Dynamic YaRN vs per-request static (decision 2026-08-23):** do not build dynamic YaRN. K is cached
post-RoPE (hq: also RHT-rotated and quantized), so a length-following factor is either incoherent
(HF's "dynamic" rope leaves cached K at the old rotation — vLLM ships no dynamic YaRN for this
reason) or needs one of: (a) re-rotating the whole K cache at each threshold (~0.1 s per switch,
16 layers + MTP K, but a cache-rewrite pass, prefix-cache invalidation, and a semantics change —
V/hidden states stay native while K is patched — unvalidated on this checkpoint), or (b)
rope-on-the-fly in the attention kernel (hq: un-RHT → RoPE → RHT per K row per step on a
latency-bound decode). Evidence does not favor it either: the 390k `yarn:1.5` probe (the factor
dynamic would pick) failed like factors 2 and 4. **Per-request static factor chosen at admission**
from `prompt_len + max_tokens` (always known — ninfer bounds max_tokens by the remaining context)
gives the same "don't scale below 262k" property: ≤262k requests run at factor 1, bit-identical to
the flag-off path; longer ones get `ceil(total/262k)` (or a finer step). This is Qwen's own rule —
add `rope_scaling` only when long context is required, with the factor sized to the length —
applied at **request** granularity: one factor per sequence for its whole life, chosen from the
planned length; switching on at token 262,145 of a running sequence is the dynamic variant above,
not "only when exceeded". Cost: a device-side
`[slots × 129]` rope table with a per-lane index instead of the by-value kernel param (graph-safe),
prefix reuse keyed by factor, MTP on the same lane table; the temperature side is free after the
q-side move above. Measure first how much static YaRN costs below native (`yarn_short_regression`,
8k/32k factor 4 vs native): ≈0 ⇒ per-request is optional; measurable ⇒ per-request is the fix.

**WI-3b status 2026-08-23 (`209e1357`):** (i) and (ii) landed — q-side temperature, `RopeSide`,
`yarn:F[,t=][,bf=][,bs=]` — and the temperature grid is a null result: t ∈ {0.1, 0.15, 0.2, 0.25}
all clean at 304k tokens (yarn:2) and all garbled at 390k; controls clean (202k, 24.7k yarn:4).
So the clean regime is ≥ 304k (1.16× native) and temperature is not the ~1.5× lever; the bf/bs
grid is deprioritized. The lbv2 native-vs-yarn ablation (paired 14: 5/14 vs 6/14, identical
correct sets) shows no measurable static-YaRN penalty on 62k–242k real-document prompts, so
per-request static YaRN is the clean design but not urgent. Next for the 390k failure: the three
attribution cells in REVIEW-1m-context.md §6.3 Q1 (int8 KV via a one-off raised linear ceiling,
groupwise-int weights, non-greedy sampling).

### WI-4 — hq decode performance (prerequisite for a usable 1M; HANDOFF §1 levers)

- Status 2026-08-23: (d) parallel packer landed (session 7, short-context parity); (a) landed as
  the TC tile-source kernel with (c)'s 2 blocks/SM coming free (session 8: MTP0 0.834 / MTP3
  0.852 of int8 at 32k, tg128 0.938); (b) warp specialization measured and REFUTED (the
  prototypes lose everywhere — the mma acc fragments cap every 8-warp variant at 1 block/SM).
  The decode phase is bound by the dependent 8-lane chains at 8 warps/SM (2.1 G rows/s ≈ 10%
  of the ALU issue-peak estimate): the homogeneous-kernel gate is closed, and the next decode
  lever is Tier 2 exact V skipping (HANDOFF §1). Target ≤1.3–1.4× int8 cost @262k
  (pre-implementation calibration band); extrapolated that is ≥ ~20 tok/s @1M (vs ~8 at the
  session-4 baseline).
- Add 512k/1M to the length-scaling sweep once WI-1/2 land; the per-step slope (0.096 ms/1k keys)
  is the number to drive down.

### WI-5 — Quality gate (before any long-context claim)

- The N2 escalation rescue is fixed (`dac4d4fe`: encoder halves, both decoders multiply, the
  census asserts zero terminal-fallback rows; ~0.4% of rows now escalate to α/2), and the codec
  oracle gates it motivated are landed (per-coordinate tolerance with a tie-flip class, 2%
  per-row relative-L2, every escalated row oracle-checked with the count banded). Still to do
  before a long-context claim: census a dumped real-model cache — heavy-tailed rows (attention
  sinks, position 0) are the ones that escalate, and a coarser sink row costs more at 1M than
  at 32k.
- Needle sweep (thinking off, greedy): 128k/256k at factor 1 × {int8, hq}; 512k at factor 2 (hq);
  1M at factor 4 (hq); depths 0/25/50/75/100%; plus factor-4 vs factor-1 at 32k (static-YaRN
  short-prompt regression → decides the policy in WI-3). Template:
  `eval/configs/qwen3_6_35b_needle_haystack.yaml` (port to qwen3.8-27b; needs a ≥1M-token corpus).
  Budget ≈ 26 min per 1M prefill, 7 min per 512k.
- HANDOFF §4 items (eval scripts at bf16/int8/hq @32k/128k/262k, sink rows) still apply.

### WI-7 — speculative decoding as the 1M decode multiplier (candidate; see §7.2)

- Why: at 1M the round costs ~95 ms regardless of verify width, so tokens/s ≈ accepted-per-round
  ÷ 0.098 s — AR ≈ 10.5, MTP at 2.4–3.5/round ≈ 24–36, a DFlash-2-class draft at 4–4.8/round ≈
  41–49. No kernel lever left in the engine is that large. Drafts exist for Qwen3.8-27B (DFlash 2
  2B all-SWA-2048, DSpark 1.36B; §7.2 has the card numbers and the long-trace acceptance decay).
- Gate before any port (zero ninfer code): (i) ninfer MTP `--draft-tokens 2/3/4/5`, greedy,
  tokens/round at 262k and 390k under yarn:2 + hq; (ii) DFlash 2 acceptance vs context on
  llama.cpp with the GGUF target + drafter at 32k→128k on the 5090. Port only if (ii) holds
  ≥ 4/round at 128k and (i) stays ≤ 3.
  *Status 2026-08-23:* (i) measured — 202k: d2 2.26 / d3 2.58 / d4 2.66 / d5 2.75 tokens/round,
  d3 the throughput optimum; 304k d3 2.71 at 57% acceptance (holds with context; collapses only
  in the 390k garbled regime). Gate (i) is met (≤ 3); (ii) is the remaining zero-code step.
- Prerequisites in ninfer if it proceeds: verify width 8 on the hq route (today width > 6 with 24
  q-heads resolves to the Prompt route — banded scratch re-decode per round, unusable at long
  context; the TC small-T kernel needs `TokenTile = 8`, 48 rows fit the 64-row tile); a 27B
  `DFlashConfig` + the DFlash 2 selector/conv ops; a W8/NVFP4 drafter (bf16 = 3.9 GB; fits at 524k,
  only ~3 GiB free at 1M after banding); drafter RoPE at >262k positions (relative within the
  2048 window — fine).

### WI-8 — hq long-window bias correction — LANDED AND VERIFIED (2026-08-23 night)

- Finding (REVIEW-1m-context.md §7): hq-e8-2b is indistinguishable from int8 through its native
  envelope (needle A2, lbv2 short) and clean at 304k, but at 390k it garbled under every YaRN
  factor, temperature, and sampling mode while int8 KV retrieved exactly — the HyperQuant paper's
  per-vector-bias-compounding regime (its Table 6 "none" row at ~2 bps).
- Design review (REVIEW-1m-context.md §8) required D1 ring validity under prefix reuse /
  retained-sequence reactivation / slot reassignment, D2 prompt-phase exactness from the fresh
  chunk's bf16 K/V, D3 the MTP pool's own planes. Disposition at land: D1 satisfied by the
  per-slot ring-validity bits (append sets; full reset and every backward frontier move —
  rewrite-checkpoint restore, rejected MTP drafts — clears by position range; retained bundles
  keep their lane, so planes follow ownership); D3 satisfied per-pool (the MTP cache plans its
  own planes); D2 satisfied in the A1 prompt route by the fresh-chunk rotate kernel + the ring
  for the W keys before the chunk (the MTP-bridge A3 route keeps sink+ring coverage — no fresh
  tensors cross its public contract).
- **Landed (both levers, one session): (1) BF16 sink + recent residual window** — S=32, W=512,
  per-(pool, layer, slot) rotated-frame side planes, third KV source in the TC decode tile fetch
  and the prompt scratch kernel, dual-write at both append sites with a chunk-internal
  ring-ownership guard, per-slot ring-validity bits;
  **(2) subtractive dither** on BOTH roles, half-cell scale (full-cell explodes the fixed-budget
  rescue rate 325→3793 rows, cosine 0.936→0.910 — the 512-bit budget cannot afford the wider
  dynamic range), hash-derived per (head, position, role, word), escalation-consistent.
- **Verified (clean build)**: needle retrieval exact at 32k / 304k / 390k / 592k true tokens
  (the 390k cell garbled under every setting pre-WI-8; 592k = Q4's garbled cell); MTP3 at 390k
  exact with 46/64 accepted (rollback-invalidate path exercised); all four `tools/test_kv` gates
  green with dither mirrors and six new residual scenarios; decode PERF improved (tg128
  71.7→80.6, pp32k+tg64 58.1→62.2 vs session 8 — exact 16 B copies beat Rice decode).
- **Product option LANDED too**: bf16/int8 linear envelope 262144 → 524288 (`PageIds` 64 → 128,
  contract test symbolic, docs updated) — int8 390k retrieves exactly (Q1a permanent); int8 fits
  to ~430k keys beside nvfp4full weights.
- **1M cell measured (session 16 close)**: a true 1,029,898-token fixture (d50 AND d20) garbles
  with a depth-independent fluent-token-soup signature at yarn:4 (prefill 517–531 t/s, decode
  8.8 t/s on the full 1M auto pool) — distinct from the pre-WI-8 codec failures (immediate EOS).
  No int8 control fits at 1M; leading hypothesis is dense-YaRN×4-at-1M (§5's named risk — Qwen
  pairs 1M YaRN with DCA + sparse prefill), hq noise floor secondary. Clean/garble cliff:
  (592k, 1.03M]. Next discriminators: YaRN factor/temperature grid at 1M, an ~800k bracket,
  lbv2_long@1M; if dense YaRN binds, the paths are WI-6 sparse prefill or rescoping the 1M goal.
- Remaining for the M2 claim: lbv2_medium@524k and lbv2_long@1M real-document cells, WI-5 sweep
  formalization.

### WI-6 — Prefill time (stretch, only if ~26 min @1M is unacceptable)

- Qwen's own 1M deployment pairs YaRN with DCA + MInference sparse prefill in vLLM/SGLang; dense
  YaRN is what this roadmap assumes. Options if needed: MInference-style vertical-slash sparsity
  for the 16 attention layers; HANDOFF §2 items (int8 scratch + IMMA, larger U8 prefill chunk).

## 4. Milestones

| milestone | needs | projected | gate |
|---|---|---|---|
| **M1 — 524k, factor 2** | WI-1 (envelope ≥ 524,288), WI-3 | fits today (6.5 of 13.2 GiB), ~7 min prefill, ~14 tok/s decode | WI-5 needle @512k |
| *status 2026-08-23 night* | WI-1 + WI-3 + WI-8 landed; hq needle exact at 32k/304k/390k (yarn:2) and 592k (yarn:4); int8 lane raised to a 524,288-key envelope (exact at 390k) | 390k yarn:2: prefill 1,384 tok/s, decode 20.8 (MTP3 62.9 committed) | **quality gate passed through 592k on hq** — the >262k garble (Q1a: the codec, not YaRN) is fixed by WI-8's residual window + dither |
| **M2 — 1M, factor 4** | + WI-2 | 10 GiB with B = 262,144, ~26 min prefill, ~8 tok/s | WI-5 needle @1M |
| *status 2026-08-23 night* | WI-2 + WI-8 landed: banded scratch; residual window (S=32, W=512) + half-cell dither | measured @592k yarn:4: prefill 916 tok/s, decode 15.2 tok/s, full 1M auto pool, 3.33 GiB free; MTP3 @390k 62.9 committed tok/s; re-fit 1M ≈ ~31 min prefill / ~10.5 tok/s | **quality gate cleared through 592k** (needle exact at 32k/304k/390k/592k; the pre-WI-8 390k garble is fixed; MTP3 accepted 46/64 at 390k); remaining: ≥1M-token fixture, lbv2 long@1M, WI-5 sweep |
| **M3 — usable 1M** | + WI-4 (optional WI-6) | ≥ ~20 tok/s decode | length-scaling sweep incl. 1M |

## 5. Risks / open questions

- YaRN ×4 on this checkpoint was validated by Qwen with their stack (incl. sparse attention);
  dense-YaRN quality at 1M is unmeasured here — run WI-5 at 512k before funding WI-4/WI-6.
- hq 2.25-bit K at 1M keys: the softmax noise floor grows with distractor count; only the sweep
  tells. Escalated rows (~0.4% on the synthetic corpus, likely more among real sink rows) are stored
  at α/2 — census a dumped real cache.
- 1M is single-request: the pool holds one such sequence; prefix-reuse retention and
  concurrency > 1 are out of scope at this length.
- Projections come from three points with ±5–10% run-to-run engine noise (HANDOFF caution 5).
- Each 1M run is long on the Windows box; script fixtures and sweeps up front.
- Static YaRN vs per-request factor is a product decision — the sweep's 32k factor-4 vs
  factor-1 row is the evidence.

## 6. Appendix — factor-4 YaRN inv-freq table (dim 64, base 1e7, orig 262,144)

`inv_i = 1e7^(-2i/64)`; `y_i = inv_i` for i ≤ 14; `y_i = inv_i·(e + (1−e)/4)` with
`e = 1 − (i−14)/8` for 15 ≤ i ≤ 21; `y_i = inv_i/4` for i ≥ 22. attention_factor = 1.1386.

| pair | inv_freq | yarn inv_freq | | pair | inv_freq | yarn inv_freq |
|---:|---:|---:|---|---:|---:|---:|
| 0–14 | unchanged | unchanged | | 22 | 1.539927e-05 | 3.849816e-06 |
| 15 | 5.232991e-04 | 4.742398e-04 | | 23 | 9.305720e-06 | 2.326430e-06 |
| 16 | 3.162278e-04 | 2.569351e-04 | | 24 | 5.623413e-06 | 1.405853e-06 |
| 17 | 1.910953e-04 | 1.373497e-04 | | 25 | 3.398208e-06 | 8.495521e-07 |
| 18 | 1.154782e-04 | 7.217387e-05 | | 26 | 2.053525e-06 | 5.133813e-07 |
| 19 | 6.978306e-05 | 3.707225e-05 | | 27 | 1.240938e-06 | 3.102344e-07 |
| 20 | 4.216965e-05 | 1.844922e-05 | | 28 | 7.498942e-07 | 1.874736e-07 |
| 21 | 2.548297e-05 | 8.759770e-06 | | 29 | 4.531584e-07 | 1.132896e-07 |
| | | | | 30 | 2.738420e-07 | 6.846049e-08 |
| | | | | 31 | 1.654817e-07 | 4.137043e-08 |

Factor 2 uses the same ramp (low 14, high 22) with `/2` and attention_factor 1.0693.

## 7. Decision (2026-08-22) — default profile, and where 2-bit KV fits

Goal as stated by the owner: *"make the Qwen3.8-27B dense model fit into a single 5090, keep its
intelligence, and make it fast enough."* Measured against that goal (not against the 1M stretch),
the decision is:

**Weights stay NVFP4 (`nvfp4full`); the default KV path becomes a fixed-rate 4-bit-class format;
2-bit KV (hq-e8-2b or any successor) is strictly the ≥512k/1M opt-in.**

| criterion | state today | what actually moves it |
|---|---|---|
| fits | solved: nvfp4full 16.0 GiB + INT8 KV at native 262k with 4.9 GiB free; hq leaves ~10 GiB free | nothing more at 262k; only >400k needs sub-int8 KV |
| keep intelligence | weights −1 pt GPQA (89.4 vs 90.4, inside n=198 noise); INT8 KV lossless; 2-bit KV is the open risk (HyperQuant's "high-compression regime", Qwen-family instabilities, accuracy campaign unfinished) | INT4-G64 KV is near-lossless across the literature; every 2-bit scheme needs calibration/protection to stay lossless and none is proven on the 3.8 hybrid |
| fast enough | decode is weight-bandwidth-bound (16 GiB/step): INT8 59 tok/s @32k, 47 @262k; hq after sessions 7–8 (parallel packer + TC tile-source kernel): 0.938 of int8 at tg128, 0.834 MTP0 / 0.852 MTP3 at 32k same-session; MTP3 ~150–210 | KV codec is worth ±10% at ≤262k; the big levers are MTP acceptance (3.8: 46–49% vs 68–71% on 3.6 — temp-1.0 profile or draft head?) and not adding ALU to decode |

Ordered plan:

1. **Default = nvfp4full + INT8-G64 KV + MTP3 at 262k.** Zero new code. Re-run GPQA/AIME on that
   exact profile (second seed if < ~88%) so "keep intelligence" rests on a fresh number.
2. **Decode speed that applies at every context:** the MTP-acceptance gap first (it multiplies
   committed tok/s directly), then the prefill polish already listed in HANDOFF §2.
3. **INT4-G64 KV as the next default** (the i8 kernel family with nibbles): bandwidth-bound,
   ~lossless, 2× int8 capacity — 524k fits on nvfp4full with room; days-to-weeks, not the months a
   new 2-bit codec plus its quality campaign costs.
4. **hq-e8-2b stays the ≥512k–1M opt-in — conditional on WI-8** (it is clean to ~304k and garbles
   at 390k today, REVIEW §7); the only format that fits 1M; its kernel/test scaffolding carries over
   to any fixed-rate successor. Until WI-8 lands, the practical ceilings are ~300k on hq and ~390k on
   int8 with the linear envelope raised. Finish the remaining decode levers
   only if 1M remains a real target (the whole-window-decode ceiling is ≈ int8 parity). If a 2-bit *default* is ever
   wanted, switch the codec to an OSCAR-class fixed-rate INT2 (calibrated per-layer/head rotations
   + sink/recent BF16 windows, ≈2.3 bpe; [arXiv 2605.17757](https://arxiv.org/abs/2605.17757)) —
   same bytes, bandwidth-bound decode, the strongest Qwen3 evidence at 2 bits — rather than more
   Rice-decode work. Kitty-style mixed-precision pages
   ([arXiv 2511.18643](https://arxiv.org/abs/2511.18643)) are the calibration-light alternative;
   fixed-rate E8 (NexusQuant K3/V2) the calibration-free one. The full alternatives table is §7.1.
5. **Composable at any rate:** BLASST-style exact V skipping now
   ([arXiv 2512.12087](https://arxiv.org/abs/2512.12087)); Locks/ParisKV-style page selection once
   the needle gate exists ([arXiv 2607.24555](https://arxiv.org/abs/2607.24555),
   [arXiv 2602.07721](https://arxiv.org/abs/2602.07721)).
6. **Don't:** quantize weights below NVFP4 (intelligence and the W4A4 prefill path both lose); add
   any entropy-coded decode path; promise "fast enough" at 1M on one 5090 (26-min prefill,
   ≤~37 tok/s after all levers).

Consequence for this roadmap: M1/M2/M3 above remain the 1M track and are gated on the owner
re-affirming 1M as a goal; the default-profile work (items 1–3) comes first and does not depend on
any of WI-1…WI-6.

### 7.1 Alternatives to HyperQuant for the KV path (surveyed 2026-08-22)

Framing: at ≤ ~524k no 2-bit codec is needed (INT8 today, INT4-G64 next — near-lossless,
bandwidth-bound). The table is for the case where a 2-bit KV *is* required (1M on 32 GB) or where
decode speed at long context matters; "fit" means fit with ninfer's paged, decode-on-read,
CUDA-graph engine on a 5090. HyperQuant's standing advantage is calibration-free rate–distortion
per bit; its weakness here is the ALU/latency-bound Rice decode and the 2-bps quality regime.

| alternative | what it is | bits | evidence | fit / cost |
|---|---|---|---|---|
| **OSCAR** ([arXiv 2605.17757](https://arxiv.org/abs/2605.17757), [code + SGLang paged kernels](https://oscar-quantize.github.io/)) | fixed-rate INT2 after calibrated per-layer/head covariance-aware rotations; BF16 sink + recent window | ≈2.28 | Qwen3-32B −0.02 pts mean pass@1 vs BF16 (8B −1.42, 4B −3.78); RULER-NIAH robust to 128K on Qwen3; AIME25 @32K traces 67–74% vs KIVI-2 52–57%; TurboQuant K3V3 collapses on the same tasks; batch-1 decode 3× vs BF16 | **Best 2-bit fit**: same bytes as hq, scalar dequant ⇒ bandwidth-bound; all layers quantized. Cost: calibration rotations for the 16 attention layers; proven on dense Qwen3, not the 3.8 hybrid |
| **Kitty** ([arXiv 2511.18643](https://arxiv.org/abs/2511.18643), MLSys 2026 oral) | 2-bit + 12.5–25% of K channels boosted to INT4; page-centric layout decomposed into two uniform 2-bit tensors | ≈2.25–2.5 | Qwen3 8B/14B/32B + Llama-3 reasoning "negligible loss", ~8× memory, 2.1–4.1× throughput; validated to 32K | High; paged by design; calibration-light (sensitivity ranking) |
| **NexusQuant** fixed-rate E8 ([repo](https://github.com/jagmarques/nexusquant), self-published) | E8 LUT codebook, Hadamard, per-head fp16 scale, asymmetric K3/V2 | 2.625 (K3V2) / 2.125 (K2V2) | K3V2 within ~1% PPL on 12/14 models; NIAH 30/30 @4K, 24–25/25 @32K | Calibration-free, same lattice as hq but fixed-rate ⇒ LUT decode; immature (CPU encoder, GPU kernel unbenchmarked, not peer-reviewed) |
| OCTOPUS ([arXiv 2605.21226](https://arxiv.org/abs/2605.21226)) | rotation + octahedral triplet Lloyd-Max, data-oblivious, fused Triton decode | ~2–4 + fp32 norm | 128K needle recall 0.81 at 2 bits (PolarQuant 0.04) but +35% PPL; needs outer-layer K protection + 32-token window; slower than bf16 SDPA | Not an upgrade: HyperQuant beats it at matched settings (HQ Table 12) |
| TurboQuant ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874); [llama.cpp forks](https://github.com/ggml-org/llama.cpp/discussions/20969)) | RHT + Lloyd-Max scalar (QJL dropped in practice) | 3.25 | RTX 5090 / Qwen3.5-27B: q8_0 parity @6K, −37% decode @110K, +1.1% PPL; Qwen needs mixed-precision K | Safe quality but only 2.5× vs int8 (13 GiB at 1M — does not fit the 1M goal) |
| watch list: NOVA-KV ([2608.04074](https://arxiv.org/abs/2608.04074)), FibQuant ([2605.11478](https://arxiv.org/abs/2605.11478)), Block-Sphere VQ ([2605.19972](https://arxiv.org/abs/2605.19972)), TCQ-for-KV ([dataset](https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache)) | calibrated attention-preserving VQ; universal fixed-rate spherical VQ; bitshift-trellis 2–3 bit with 1–2 KB LUT | 2–3 | early, small models, or self-published | the trellis one is the "lookup-free fixed-rate" shape the survey named; no NIAH, online encode cost unclear |

Decode-side selection (composable with any codec; attacks the per-step cost no codec fixes):

| alternative | what it is | evidence | fit |
|---|---|---|---|
| **BLASST** ([arXiv 2512.12087](https://arxiv.org/abs/2512.12087), MLSys 2026) | softmax-threshold skipping of V loads + PV, one threshold per model/phase | 1.48× decode at 73% sparsity, accuracy preserved, GQA-native | the exact V skipping queued as Tier 2 in HANDOFF §1; zero-risk first step on the current hq kernel |
| **Locks** ([arXiv 2607.24555](https://arxiv.org/abs/2607.24555)) | page-local spectral key summaries; selection reads no K/V; vLLM plugin in full CUDA graphs | 2048-token budget matches FullKV at 100K+ attending ~2% of tokens; 2.0× at 1M | Same shape as ninfer's decode; summaries ≈10% of a *dense* cache — must be re-budgeted for a 2-bit cache |
| **ParisKV** ([arXiv 2602.07721](https://arxiv.org/abs/2602.07721), ICML 2026) | GPU-native collision candidates + quantized reranking, drift-robust, UVA offload to 1M | full-attention quality at batch 1, up to 2.8× throughput | High |
| HiSparse ([arXiv 2608.07009](https://arxiv.org/abs/2608.07009)) | tiered KV (host RAM + small GPU cache) with top-k decode | up to 4.7× throughput at long context | The other 1M route on 32 GB if KV leaves the GPU |
| evidence on this family | [uncertainty-gated block sparsity on Qwen3.6](https://arxiv.org/abs/2607.07724): RULER-NIAH within 2 pp but 0.89 of dense aggregate at 128K; [The Sparse Frontier](https://arxiv.org/abs/2504.17768): decode page selection generalizes best, longer sequences tolerate more sparsity | | page selection on the 3:1 hybrid is not free — gate it |

Caveats: no unified 128K benchmark across these codecs; several TurboQuant baselines are
reimplementations; NexusQuant/TCQ-for-KV are self-published; every per-vector codec has shown
Qwen-family instabilities (outer K layers, channel outliers). The owner's GPQA/needle campaign is
the gate, not their tables. Full survey context: conversation of 2026-08-22; the
whole-window-decode floor (HANDOFF §5e) is why
hq's decode ceiling is ≈ int8 parity.

### 7.2 Speculative-decoding update (2026-08-23) — the "fast enough" lever has public drafts now

Trigger: the LMSYS/SGLang post on Ling-3.0-flash batch-1 decode on Blackwell
([lmsys.org, 2026-08-21](https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell)):
a 124B-A5B hybrid KDA/MLA MoE on 4× Blackwell, TP4, bf16. Built-in MTP (NEXTN 5/1/6) went 288 →
606 tok/s through host run-ahead, PDL-chained CUDA graphs, fused kernels, bf16 gate/lm_head (+10%),
and "stage, then commit" for the linear-attention state under speculation; a trained DSpark draft
then reached 1,120 tok/s (accept length 9.95 vs 3.25). Datacenter parts; fp8 weights still to do.

What transfers to ninfer (hybrid GDN, batch-1, Blackwell, CUDA graphs + PDL already in place):

- **Linear-attention state under speculation** — the blog's stage-then-commit (per-position
  post-states, commit the accepted one; chain drafts only) is ninfer's ReplaySSM with more
  memory: ninfer records raw inputs and replays the accepted prefix bitwise from the committed
  checkpoint (`docs/maintainer/replayssm-gdn.md`). Nothing to change; the same topk=1 (chain)
  constraint applies.
- **"Verifying more tokens is nearly free at batch 1"** holds only while the chain keeps
  accepting; the blog's own break-even is `Δaccept > 0.05 × accept` with each extra draft step
  costing 4–9% of the step. Qwen3.8's MTP head accepts ~46–49% per draft on long reasoning
  (README), community vLLM/SGLang numbers are accept length ~1.9–3.5 and SGLang's preset is 3/1/4
  — so MTP3 is already at or past the optimum for this head; widening to 5 is not a free win and
  MTP2 may beat it on long traces. Cheap experiment: MTP2/3/4 on the long-reasoning corpus.
- **Draft quality + draft cost is the lever, and drafts for Qwen3.8-27B exist:**
  [incoai/z-lab DFlash 2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (2B, block 8, one
  parallel block forward + top-16 candidates/position + low-rank selector + two-tap dynamic convs;
  [blog](https://inco.ai/blog/dflash2/)) and [RadixArk DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
  (1.36B, 5 layers, block 7, target layers 4/16/28/40/52). Card numbers (SGLang, one H200, bf16,
  temp 1.0 xhigh, concurrency 1, all at 7 drafts): AR 69 tok/s; built-in MTP 135–179 (2.0–2.6×,
  accept 3.7–5.0); DSpark 138–185 (2.0–2.7×); **DFlash 2 184–236 (2.7–3.4×, accept 4.1–5.5)**.
  SGLang reports [206 tok/s on one RTX 5090 with NVFP4 + DSpark](https://x.com/sgl_project/status/2088281320422322413);
  ninfer's MTP3 on the 5090 is 151–220 (README) — already competitive; DFlash 2 is the step up,
  structurally because one parallel draft forward replaces 3–7 sequential MTP forwards (each
  re-reading the draft layer and lm_head).
- **What it would take in ninfer:** the DFlash backend is v1 and 35B-only
  (`DFlashConfig::supported = false` on 27B; v1 = 6 draft layers, 5 sliding + 1 full non-causal,
  8 feature layers, block 16, local window 4096). DFlash 2 adds the candidate/selector path and the
  depthwise convs (+3% params, +1.3% cycle latency per Inco) — new kernels but the same block-verify
  contract; the 2B draft must be quantized (bf16 = 4 GB; z-lab's MLX path uses 4-bit) to fit next to
  16 GiB weights and a long KV. Measure before porting: ninfer's own 35B DFlash-v1-vs-MTP3 campaign
  was mixed (+5% AIME, −6 to −43% on code/story/translation, accept 12–65%), and the DFlash 2 card
  numbers are math/code/MT-Bench at 4k max_new — acceptance decays along long reasoning traces
  (published: 3.7 → 1.5 over a MATH-500 trace; 15 → 1.7 on AIME for Qwen3.6-35B), which is exactly
  ninfer's long-decode regime.
- **Small items worth copying:** bf16 (not fp32) for any per-step bandwidth-only weight read
  (gate/lm_head — ninfer's head is W8G32 and `--lm-head-draft` already shortlists it); the
  metadata-refill "glue graph" pattern if any per-step host refill remains outside the graph.

Effect on §7: item 2 ("decode speed at every context: MTP acceptance first") becomes concrete —
(a) MTP2/3/4 sweep on long reasoning; (b) a quantized DFlash 2 (or DSpark) draft for the 27B target
as the next speed project, ahead of further KV work, because it is the only item with a ≥30%
batch-1 decode upside that does not touch the KV path.
