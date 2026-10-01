# ROADMAP — model-opt: weights-artifact recipes for the fork profiles

Desk note, 2026-08-25. Companion to `HANDOFF.md`, `ROADMAP-kernel-perf.md` (kernels),
`ROADMAP.md` (KV formats) and `ROADMAP-dflash2.md` (drafters), `ROADMAP-host-kv.md` (session
state). Scope: **weights-artifact decisions** — quantization recipes, per-layer format
assignment, converter variants — for `qwen3_8_27b` on this box. Nothing here was measured
locally; sources are the fork's own published numbers, the QUASAR card/config/tensor index and
paper (arXiv 2608.13966), and the gittensor cards, all read 2026-08-25. Every item is gated on
the fork's own instruments (bit-level verify tooling, GPQA n=198 band, needle sweep,
per-profile fingerprints per caution 12) — card numbers are never the acceptance evidence.

## 0. TL;DR

1. **WI-M1 — re-source the text stack from the QUASAR QAT checkpoint** (`nvfp4qat` converter
   variant): the first credible quality upgrade over nvfp4full — QAT recovers ~⅔ of the PTQ gap
   (GPQA 90.91 vs 89.39 at n=396 on their eval; BF16 91.41) while quantizing MORE layers, which
   also lets the fork drop nvfp4full's nine BF16 exception parents (**−0.6 GB/token ≈ −4%
   decode**). One real mechanical crux: fused parents vs per-tensor global scales (three
   options, (a) recommended).
2. **WI-M2 — NVFP4/Q4 lm_head** (moved here from kernel-perf's artifact paragraph): the W8G32
   head is 1.35 GB/token ≈ ~6% of an MTP0 step; fold into the same artifact rebuild as WI-M1.
3. **WI-M3 (conditional) — MTP module to NVFP4**: only if MTP survives the §7.2 Stage-3 purge
   decision; moot otherwise.
4. Reference shelf (WI-M4): the gittensor ModelOpt checkpoint (non-displacing source; the
   multimodal-calibration angle) and the on-policy quantized-target drafter-calibration recipe
   (owned by §7.2).

## WI-M1 — QUASAR QAT re-source (`nvfp4qat` variant)

**Source facts** (`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, card + config.json + safetensors index,
2026-08-25): one epoch of loss-aware NVFP4 quantization-aware distillation against the frozen
BF16 teacher on its own outputs (QUASAR, arXiv 2608.13966 — loss-aware reconstruction in the QAT
loop; real authors, method published); `compressed-tensors` / `nvfp4-pack-quantized`, group 16,
E4M3 scale words, per-tensor `weight_global_scale` **and per-site `input_global_scale`**;
**every text linear quantized** — attention q/k/v/o (separate tensors), MLP gate/up/down
(separate), GDN `in_proj_qkv` (fused) + `in_proj_z` + `in_proj_a`/`in_proj_b` (quantized) +
`out_proj`; `lm_head`/`embed`/MTP/vision/norms/conv unquantized (ignore list). RoPE
theta 1e7, 262,144 positions — same base semantics, no interaction with the YaRN work.
Their eval: GPQA-Diamond 0.9091 (2 runs, n=396) vs BF16 0.9141 vs unsloth-PTQ 0.8939 (the
fork's own nvfp4full number), AIME26 saturated (1.0 for both QAT and BF16 — uninformative).

**Adoption shape**: a converter variant beside nvfp4full — preserve QUASAR's code/scale words
bit-exactly (the same preservation contract `convert_nvfp4full` applies to unsloth's parents),
locally encode everything the source leaves unquantized exactly as today (W8 embedding/head, W8
MTP, Q4 draft head, Q4/Q5/W8 vision). This is a **re-source, not an in-place refinement** — you
do not merge two quantized checkpoints; and running QUASAR QAT locally is out of scope (no
training path on a single 5090).

**The one real crux — fused parents vs per-tensor global scales.** ninfer's parents
(`qkgv [14336,5120]`, GDN `qkvz [16384,5120]`, `gate_up [34816,5120]`) carry ONE trailing `d_w`;
QUASAR's constituent tensors each carry their own `weight_global_scale` (unsloth's fused gate_up
happened to share one, which is why nvfp4full never hit this). Options, best first:
- **(a) per-row-segment divisors in the NVFP4 parent (recommended)**: 2–4 segments per parent,
  a row-range lookup at the single point the kernels apply `inverse_weight_divisor` today (GEMV
  epilogue coefficients and the TMA epilogue `alpha`) — every code and scale word stays
  bit-exact; a storage-layout rev (`blockscale-k16-m128x4-v2` carrying the segment table) plus a
  small epilogue change, oracle-gated like any profile change.
- (b) rescale sub-tensor E4M3 scale words onto a common `d_w`: no format change, but one extra
  E4M3 requantization of scales — breaks bit-exact preservation and adds a small measurable
  error exactly where QAT bought its margin. Fallback only.
- (c) unfuse the parents: touches kernel geometry and the leaf contracts — worst, do not.

**Freebies riding the variant**:
- **Drop the nine BF16 exception parents** (attention qkgv ×6, attention out ×2, GDN out ×1 ≈
  1.07 GB/token, of which ~0.6 GB/token is the NVFP4 delta): QAT trained through exactly those
  sensitivity-motivated exceptions. ≈ **−4% MTP0 decode step** on top of the quality gain.
- `input_global_scale` ships per site → the A4 path's `d_x` needs no local recalibration
  (keep `calibrate_nvfp4full` as the derivation cross-check, like the unsloth ratio check).
- GDN control `a/b`: quantized in the source, but NVFP4 decode is exact — materialize them BF16
  at conversion and keep the existing control route/kernels unchanged.
- Composes with WI-M2 (same rebuild) and with the §7.2 drafter plan (unchanged base semantics).

**Size ledger (why their 19.7 GB > nvfp4full's 18.32 GB, and why it doesn't matter):** the NVFP4
text stacks are equal density (4.5 bpw) — QUASAR is actually 0.77 GB SMALLER there (it quantizes
the nine exception parents). The delta is entirely auxiliary tensors: their embed+lm_head ship
BF16 (5.09 GB vs ninfer's W8 2.70 → +2.38 GB, the dominant term; both are in their quantization
ignore list), and ninfer's 0.36 GB Q4 draft head has no counterpart. The arithmetic closes at
~19.6 GB only if the export is TEXT-ONLY (no MTP, no vision — with those in BF16 it would be
~20.6 GB), which matches the tensor index never surfacing `mtp.*`. Consequence: the `nvfp4qat`
variant takes only the text linears from QUASAR and re-encodes everything else with the existing
formats/sources (the converter is already multi-source), landing at ≈17.3 GB file / ≈15.4 GiB
device — SMALLER than nvfp4full while higher-quality, slightly widening every KV/drafter budget
in the other notes. Kickoff action: verify MTP/vision presence in the source; if absent, keep
sourcing them from the official BF16 checkpoint exactly as `convert_nvfp4full` does today.

**Gates (all fork-local; card numbers are inputs, never evidence)**: exhaustive E2M1/E4M3
bit-level verification through the existing verify tooling extended to the new source fields;
GPQA n=198 under the registered serving profile — acceptance bar ≥ the 89.39 baseline, target
~90.5+; needle at native lengths; a NEW per-profile fingerprint (caution 12 — a new weights
profile gets its own; do not compare against nvfp4full's 0.4285714286); the paused KV-accuracy
campaign cells inherit the profile if it becomes the default. Provenance caution: 57 downloads,
12-day-old org, paper 12 days old — the bit-verify + own-eval discipline is the whole defense.

## WI-M2 — NVFP4/Q4 lm_head (ownership moved here from ROADMAP-kernel-perf)

The W8G32 head reads 1.35 GB/token ≈ 0.8 ms ≈ ~6% of an MTP0 step (amortized over verify width
under MTP3). External evidence a 4-bit head survives is weak (gittensor's n=20 smoke) — treat as
a hypothesis. Run it inside the WI-M1 rebuild (one artifact, one eval campaign covers both);
decision by the same GPQA band, with the head's contribution isolated by an A/B pair of
otherwise-identical artifacts if the combined run lands below target. Q4G64 (the draft head's
format, 0.66 GB) and NVFP4 (0.71 GB) are the candidates; either saves ~0.4 ms/step.

## WI-M3 — MTP module to NVFP4 (conditional)

MTP matrices are W8 (~0.45 GB read per draft step, ×3 per MTP3 round). Converting them to NVFP4
halves that (~0.7 ms/round ≈ +2–3% MTP3). **Blocked on the §7.2 Stage-3 decision**: if DFlash2
validates and MTP is purged, this item is moot — do not spend an eval campaign on it before the
drafter question settles.

## WI-M4 — reference shelf

- `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (ModelOpt PTQ): non-displacing as a source
  (20-item smoke vs n=198), agrees on the sensitive exclusions; its 128-sample image-text
  calibration is the multimodal-calibration reference if vision quality ever gates a recipe.
- On-policy quantized-target drafter calibration (gittensor DSpark card): owned by
  `ROADMAP-dflash2.md` stage 3 — the method reference if any drafter/head tuning ever happens
  against the deployed quantized target.

## WI-M5 — locally feasible training experiments (added 2026-08-25; corrects the blanket
"no training path on one 5090" verdict, which holds only for FULL-model work)

- **(a) MTP-head on-policy retuning — cheapest, best expected ROI, hedges the DFlash2 bet.**
  ~425M-param head, full-FT against the frozen 4-bit target generating on-policy
  features/targets in-process (no-grad forward): ~24 GB total, target-forward-dominated at
  ~2.5–3k tok/s → 50M tokens ≈ 5 h. The gittensor on-policy recipe (WI-M4/§7.2: +14.2% drafter
  acceptance, quantized-target outputs as the distribution) applied to the unpublished case:
  Qwen3.8's own MTP head vs nvfp4full's outputs. Motivation: this model's MTP acceptance is the
  fork's weakest link (45–49% concurrent vs Qwen3.6's 67–82%); half the gap ≈ +20–40% decode.
  Gates: acceptance A/B on the fork's own corpora (thinking-on included, caution from the r/Vllm
  data), fingerprint re-baseline (a retuned head is a new profile), GPQA unchanged by
  construction (target frozen). Interaction: if §7.2 Stage 3 purges MTP, this is moot — but a
  successful retune also RAISES the bar DFlash2 must beat before any purge.
- **(b) QLoRA distillation-under-yarn:F** — owned by `ROADMAP.md` open item 2 (learned scaling, tier 4)
  (feasibility arithmetic there); the model-opt side is the tail of its loop: shard-wise CPU
  merge into BF16, then the existing local NVFP4 encode path (the same
  `NVFP4_MAXABS_DIVISOR_RNE_V1` profile), then the WI-M1 gate set. Note the merge-then-PTQ step
  costs some of a QAT checkpoint's margin — if both WI-M1 and (b) succeed, the ideal source is
  QUASAR-base + LoRA delta, which requires the LoRA to be trained ON the QUASAR base (decide
  the base before the pilot, not after).
- **(c) Drafter-class training (0.9–2B)**: fits memory-wise (frozen target + trainee with 8-bit
  optimizer) but is the most engineering; only if the §7.2 drafter track demands a custom draft.
- Out of local scope, permanently: full-model 27B training (QAT or FT — hundreds of GB of
  state); the rental door (~day-scale on 8×H100 at QUASAR's recipe scale) is noted once and not
  a plan. Practicalities for this box: HF+PEFT+FLA (GDN chunked backward exists in FLA), WSL2
  for the Triton stack, TB4 irrelevant once weights are resident.

## Order

WI-M1 with (a), carrying WI-M2 in the same rebuild → gates → if it clears, it becomes the fork
default and the BF16-exception drop lands the ~4% decode as a side effect. WI-M5(a) (MTP-head
retune) can run in parallel — it needs only the frozen deployed artifact and ~5 h of GPU.
WI-M3 waits on §7.2 Stage 3; WI-M5(b) waits on the WI-3c tier-1 KL instrument. Nothing here
blocks or reorders the kernel-perf, host-kv, or drafter tracks.
