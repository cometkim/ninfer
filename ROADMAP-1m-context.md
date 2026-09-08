# ROADMAP — the 1M/long-context track (Qwen3.8-27B, one RTX 5090)

Compacted at the 2026-08-27 reorganization; the original desk note with the full work-item and
review detail is recoverable from branch `dev-preorg-20260827` / `backup/20260827/*.bundle`.

**Outcome: 1M dropped (owner, 2026-08-26); 524k held; 768k engineering-ready pending the KL
instrument (open item 1).** Quality through 592k true tokens is verified exact on hq; the 1M cell
itself (1.03M tokens) garbles depth-independently (fluent token soup from token 1) with
dense-YaRN×4-at-1M as the leading hypothesis and the hq noise floor secondary — no int8 control
fits at 1M, so attribution is open; the clean/garble cliff is in (592k, 1.03M]. Measured at the
held extremes: 390k yarn:2 prefill 1,384 / decode 20.8 tok/s (MTP3 62.9 committed); 592k yarn:4
prefill 916–922 / decode 15.2; an honest 1M projection was ~31 min prefill / ~10.5 tok/s decode.

Memory frame (why this track existed): hq payload is 9 KiB/token (9.0 GiB @1M) — the only
format that fits 1M on 32 GB (int8 33 GiB, bf16 64 GiB). Above ~400k tokens hq is the only
option, which is why every >262k quality fix was an hq fix.

Spinoffs: the DFlash2 speculative backend (`ROADMAP-dflash2.md`), the KV-format alternatives
survey and the nvfp4-KV format play (`ROADMAP.md`).

## LANDED

- **WI-1 — engine envelope**: `kNativeContext` 262,144 → 1,048,576 keys on the U8 cache; the
  bf16/i8 linear envelope was later raised to 524,288 (WI-8); int32 addressing audited clean.
- **WI-2 — scratch banding**: the U8 prompt scratch materializes in sequential 262,144-key
  bands carrying the online-softmax state (m/l/acc) — 1M + `--kv-capacity auto` resolves the
  full pool with ~3.3 GiB free (was a 4 GiB one-shot, explicit-capacity only).
- **WI-3/3b — YaRN**: `--rope-scaling yarn:F[,t=c][,bf=n][,bs=n]` — `RopeFrequencies` table,
  q-side temperature (cached K is factor-free), FP64 angle reduction, factor-1 bit-stability.
  Null results recorded: the temperature grid does not rescue the failing regime (0.1–0.25 all
  clean at 304k, all garbled at 390k pre-WI-8); the bf/bs ramp grid was deprioritized; dynamic
  YaRN rejected (cached-K coherence — per-request static factor at admission is the clean
  design if ever taken; prefix identity and park keys must then include the rope config).
- **WI-8 — the >262k quality fix**: BF16 sink(32) + recent(512) residual window with per-slot
  ring-validity bits, dual-write at both append sites, plus half-cell subtractive dither
  (full-cell explodes the fixed-budget escalation rate). Verified: needles exact
  32k/304k/390k/592k; MTP3 rollback-invalidate path exercised at 390k; decode got FASTER
  (exact 16 B row copies beat Rice decode: tg128 71.7→80.6). The bf16/int8 envelope raise to
  524,288 (PageIds 64→128) landed with it.
- **WI-5 — quality gate (executed)**: paired hq-vs-int8 PARITY at 390–400k (the 3-bps risk
  refuted by measurement); GPQA 88.89/88.89 in the 89–90 band; AIME25 100% / AIME26 93.33%;
  LBv2 medium full-suite 58.6% (above the 53.7% human baseline); native-vs-yarn paired
  ablation on real documents indistinguishable (n=14); both MCQ instruments proven blind to
  the factor effect (campaign8: yarn:3 == yarn:4, zero discordants).
- **WI-4 — hq decode levers (closed)**: parallel unary packer + TC tile-source kernel landed;
  the homogeneous-kernel gate is closed (dependent 8-lane chains; warp specialization refuted).
  The current decode surfaces live in `ROADMAP.md` open item 3.
- **WI-7 — speculative decoding**: gate (i) measured (MTP3 the throughput optimum, acceptance
  holds with context); the DFlash2 work it seeded is `ROADMAP-dflash2.md`.

## OPEN

1. **768k decision — the KL instrument (WI-3c tier 1; = ROADMAP.md open item 2).**
   Teacher-forced KL(yarn:F ‖ factor-1) on real text at ≤262k: sensitive by construction to the
   depth-independent from-token-1 damage that MCQ and needles are proven blind to; short
   contexts suffice and logprobs come from any offline stack (HF/vLLM; the NVFP4 checkpoints
   work). Comparing KL(yarn:3‖1) vs KL(yarn:4‖1) is an afternoon of work and the missing
   discriminator for the 768k owner call.
2. **Learned-scaling tiers 2–4 (gated on tier 1).** Tier 2 — per-request factor selection:
   ≤262k requests at factor 1 are bit-identical to today and eliminate the short-context damage
   class; engineering cost is prefix-identity/park-key rope config plus re-prefill on factor
   crossings. Tier 3 — searched frequency table (LongRoPE-style): ~32–64 numbers dropping into
   the existing `RopeFrequencies` plumbing (a `table:<file>` load path is the only engine
   change); the search runs offline against the tier-1 KL objective plus a long-needle
   constraint. Tier 4 — local LoRA distillation-under-yarn:F: feasible on one 5090 (27B QLoRA
   ≈ 22–26 GiB; teacher = the same frozen weights at factor 1; 2–4k training sequences; a
   ~30M-token pilot ≈ overnight at ~1.2k tok/s; embeddings+norms trained; the loop closes
   through the existing NVFP4 encode/convert with the tier-1 KL as both objective and
   acceptance; risk — LoRA capacity for a global positional shift is unproven, the pilot
   decides). Tier 5 — full-model QAT-for-positions is a rental job; any third-party
   long-context-tuned checkpoint enters the WI-M1 re-source pipeline unchanged.
3. **1M attribution (only if 1M is ever re-opened)**: YaRN factor/temperature grid at 1M, an
   ~800k bracket cell, lbv2_long@1M. If dense YaRN binds, the honest paths are
   MInference-style sparse prefill for the 16 attention layers (WI-6 — Qwen's own 1M
   deployment pairs YaRN with DCA + sparse prefill; dense YaRN at 1M was never validated on
   this checkpoint) or rescoping the goal.
4. **Overthinking-style failure at high YaRN factors — candidate attribution + cheap mitigation
   (owner field observation 2026-08-28; review only, not started).**
   [arXiv 2606.00206](https://arxiv.org/html/2606.00206v1) ("Quantized Reasoning Models Think
   They Need to Think Longer, but They Do Not") isolates a failure mode where the correct answer
   already appears in intermediate reasoning but the model never commits to it as the final
   answer — up to 52% of failures in their PTQ setting — traced to high-entropy
   branching/hesitation tokens ("Wait", "But", "Alternatively") whose logits noise inflates
   (token-level KL concentrates exactly there; KL correlates ρ=0.92 with full-precision entropy).
   Their fix is training-free and decoding-time only: subtract a constant λ from a curated set
   of 50 overthinking-marker logits at every step (λ ∈ 0.5–4.0; T=0.6, top-p 0.95 — our exact
   serving profile), giving −12–23% CoT length with accuracy preserved or improved. The paper
   is about quantization, not YaRN — but the owner observes the same
   answer-in-reasoning-but-not-final signature on our YaRN-scaled runs, worst at factor 4
   (1M grade). Transfer checks, when the current task frees up: (a) profile yarn:4 traces for
   the marker signature (marker-token frequency in top-k at high-entropy positions;
   correct-answer-present-in-reasoning rate) — a natural extension of the tier-1 KL instrument;
   note needles passed at 592k/yarn:4, so reasoning traces, not needle probes, are the
   instrument here; (b) if the signature shows, A/B the fixed-set logit penalty — one
   hyperparameter, sampler-level engine tweak, no post-training. Cautions from the paper:
   penalizing the wrong (low-KL math/formatting) tokens is catastrophic (−9.5% accuracy,
   +41% CoT) — use their curated marker set verbatim; negative λ (boosting markers) is
   strictly harmful.

## Decision record (§7, 2026-08-22)

Weights stay NVFP4 (`nvfp4full`); the default KV path is a fixed-rate 4-bit-class format;
2-bit KV is the long-context opt-in. Ordered then: nvfp4full + INT8 + MTP3 default → MTP
acceptance work (later superseded by the DFlash2 project) → INT4-G64 as the next default → hq
as the ≥512k opt-in (this stance was superseded 2026-08-28 by the owner direction that hq is
the general-use KV lane — `ROADMAP.md` open item 3). If a 2-bit *default* is ever wanted,
switch to an OSCAR-class fixed-rate INT2 rather than more Rice-decode work (full survey:
`ROADMAP.md`); BLASST-style exact V skipping composes with any codec. **Don't**: quantize
weights below NVFP4 (intelligence and the W4A4 prefill path both lose); add any new
entropy-coded decode path; promise "fast enough" at 1M on one 5090 (26+ min prefill, ≤~37
tok/s after all levers).
