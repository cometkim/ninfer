# ROADMAP — track status and open work

Master index for the fork's work tracks. Each track's detail lives in its own file; this file
carries the status of everything, the completed-work summaries, and the open-work priority order.
Historical session narratives and the former REVIEW-*.md documents were folded here (or deleted
when fully dispositioned) at the 2026-08-27 reorganization; the pre-reorganization state is
recoverable from `backup/20260827/*.bundle` and the `dev-preorg-20260827` branch.

Detail files:

| file | scope | status |
|---|---|---|
| `ROADMAP-dflash2.md` | DFlash2 speculative backend (the active project) | items 1–4 landed; stage-3 decision live |
| `ROADMAP-kernel-perf.md` | non-hq kernel levers (prefill/decode/GEMM) | K1/K2 landed; K1c/K1d/K3/K4 open |
| `ROADMAP-host-kv.md` | host-backed KV (park/restore) research | survey + plan, not started |
| `ROADMAP-model-opt.md` | weights-artifact recipes (QUASAR, lm_head, MTP) | owner plans, not started |

## Completed tracks (summaries)

**hq-e8-2b KV cache** (`feat/hyperquant`, `feat/1m-context`). 2.25-bit E8-lattice + Rice KV:
8-lane cooperative k=0 decoder, parallel unary packer, TC tile-source decode kernel, banded
prompt scratch, BF16 sink(32)+recent(512) residual window with ring-validity bits and half-cell
subtractive dither (the fix for the >262k garble). Verified: needle retrieval exact at
32k/304k/390k/592k true tokens; paired hq-vs-int8 quality PARITY at 390–400k (McNemar p~0.49) —
the 3-bps-quality risk is refuted by measurement; hq/int8 decode ratio 0.83–0.85 at 32k, parity
at short context; LBv2 full-suite 58.6% (above the 53.7% human baseline). The homogeneous-kernel
decode gate is CLOSED (dependent 8-lane chains at 8 warps/SM ≈ 2.1 G rows/s; warp specialization
measured and refuted). Remaining decode levers: the width-8 TC tile-source kernel's serialized
inline decode (open-work item 3a) and Tier-2 selection (item 3b).

**1M context** — DROPPED per owner decision (2026-08-26); 524k held, 768k engineering-ready
(owner call open, gated on the KL instrument below). Landed on the way: envelope to 1,048,576
keys (U8), scratch banding (262,144-key bands, carry state), YaRN rope scaling
(`--rope-scaling yarn:F[,t=][,bf=][,bs=]`, q-side temperature, FP64 angle reduction), bf16/int8
linear envelope to 524,288. The 1M cell itself (1.03M true tokens) garbles depth-independently
(fluent token soup from token 1) with dense-YaRN×4-at-1M as the leading hypothesis — no int8
control fits at 1M; clean/garble cliff is in (592k, 1.03M].

**Kernel performance (landed share)** — see `ROADMAP-kernel-perf.md`: i8 prefill f16-accumulate
PV + 2⁻⁶ range guard, key-range split for all prompt routes (int8 +14.2% / hq +6.0% at pp65536),
PDL across the decode chain with the publish-at-end rule, fused norm+control (Split40), sigmoid
gate folded into the attention op, fused qk_norm_rope (bit-exact), per-request error boundary.

**Windows/natpate role** — frozen 2026-08-27: the fork tracks upstream only. natpate's master is
folded into history via the `feat/windows-port` layer and is not chased; `feat/build-speed` was
rebased once onto natpate's current head (b686696e) purely to keep natpate PR #2 mergeable.

## Open work (priority order)

1. **DFlash2 stage 3 — MTP purge** (owner decision; inputs live). DFlash2 beats MTP3 committed
   t/s at every measured cell (+30% @8k greedy, +52% @258k) at equal-or-better acceptance. Two
   steps per `ROADMAP-dflash2.md`: reversible load-time materialization skip keyed on `--spec`,
   then — only after DFlash2 wins across this fork's workloads including thinking-on traces and
   tool-calling — an artifact variant that drops `mtp/*` + `draft_head`. Preconditions: promote
   the greedy token-stream diff to the primary fingerprint; never purge to make room.
2. **768k decision — the KL instrument** (WI-3c tier 1). Teacher-forced KL(yarn:F ‖ factor-1) on
   real text at ≤262k; MCQ instruments are proven blind to the perceived factor effect
   (campaign8: yarn:3 == yarn:4 at 25%, zero discordants; GPQA method control 91.67% = factor-1).
   An afternoon of work on any offline stack; decides 768k and gates the learned-scaling tiers
   (per-request factor selection → searched LongRoPE table → local LoRA distillation-under-yarn;
   full detail was WI-3c, former `ROADMAP-1m-context.md`).
3. **hq decode for general use** (owner direction 2026-08-28: hq is the general-use KV lane,
   not just the >524k capacity lane — improve it for every context). Two independent surfaces:
   (a) the width-8 verify TC tile-source kernel (`GqaTcKVHq`) serializes its inline tile decode
   with the MMA per 32-key block (cooperative decode → barrier → mma, no cross-tile pipeline),
   where the i8 kernel overlaps loads via cp.async prefetch + warp split — measured idle @258k,
   DFlash2 k=7: hq 84.6 tok/s (45.4 ms/round) vs int8 140.8 (26.2 ms/round), a +19 ms/round
   decode-order gap against a 3.5× byte advantage; the homogeneous-kernel warp-spec refutation
   does not close this surface (different kernel structure — the pipelining candidate here is a
   double-buffered tile decode overlapping the MMA consumer). Attribution without nsys (its
   report converter deterministically hangs at 48% on DFlash2/hq traces — tooling note, both
   MTP-era profiles converted fine): hq's byte floor is 2.25 GiB ≈ 1.25 ms/round (16 µs/layer,
   3% of its 567 µs/layer) so the round is NOT bandwidth-bound; decode ALU alone is
   524,288 row-decodes/layer/round ÷ 2.1 G rows/s = 250 µs/layer (44%) even at the homogeneous
   kernel's tuned full-occupancy rate, and the in-kernel rate is lower (128 threads, 2 CTAs/SM)
   and fully serialized with the MMA — consistent with the +240 µs/layer gap vs int8 (whose own
   327 µs/layer vs a ~58 µs byte floor shows both routes are compute/latency-dominated; the
   lever is decode placement, not bytes). (b) Tier-2 exact V skipping /
   page selection: BLASST-style softmax-threshold V skipping (1.48× decode at 73% sparsity,
   GQA-native) is the zero-risk first step on the current kernel; Locks/ParisKV-style page
   selection is the larger upside (2–2.8×) with a quality gate. HiSparse's merged SGLang fused
   select/LRU/fetch kernel is the in-graph template; its host-fetch half is dead on this TB4
   box. Per-step slope to drive down: ~0.096 ms/1k keys.
4. **kernel-perf open items** (`ROADMAP-kernel-perf.md`): remaining −128-launch fusion block,
   WI-K1c softmax overlap, WI-K1(d) int8/FP4 scratch for the hq prompt route (design frozen,
   nvfp4-KV successor undecided), WI-K3 (blocked: Windows/MSVC host ABI; needs a Linux box),
   WI-K4 A4 epilogue fusion, competitive vLLM base-decode A/B (field: 88.5 vs our 75–76.5 MTP0).
5. **nvfp4 as a KV/scratch format** — the attention-format play: stored blockscale-k16 feeds
   `mma kind::mxf4nvf4` QK directly (SageAttention3 shape; FlashInfer PR #4346 is the working
   SM120 reference). Competes with WI-K1(d)'s int8 scratch; decide between them together. The
   paired quality gate resolved to PARITY, so this is purely a speed/format decision now.
6. **host-backed KV** (`ROADMAP-host-kv.md`) and **model-opt** (`ROADMAP-model-opt.md`) —
   surveyed/planned, unscheduled.

## KV-format alternatives (surveyed 2026-08-22; reference for any future 2-bit default)

Framing: ≤524k no 2-bit codec is needed (int8 today, INT4-G64 next). If a 2-bit default is ever
wanted, the survey's ordering is: **OSCAR** (calibrated rotations + BF16 sink/recent, ≈2.28 bps,
Qwen3-32B −0.02 pts — best fit, scalar dequant ⇒ bandwidth-bound) > **Kitty** (2-bit + boosted
channels, calibration-light) > fixed-rate **E8** (calibration-free, immature) ; TurboQuant (3.25
bps) does not reach the 1M goal; OCTOPUS is dominated by HyperQuant at matched settings. Watch
list: NOVA-KV, FibQuant, Block-Sphere VQ, TCQ-for-KV. Every per-vector codec has shown
Qwen-family instabilities; this fork's own paired campaign is the gate, not their tables.
Decode-side selection (composable with any codec): BLASST (V skipping), Locks/ParisKV (page
selection), HiSparse (tiered KV — host half dead on TB4). Caution: page selection on the 3:1
hybrid is not free — gate it (uncertainty-gated block sparsity on Qwen3.6 measured 0.89 of dense
aggregate at 128K).

## Numbers that stay quotable

- nvfp4full + hq, greedy verified: needle exact 32k–592k; 390k MTP3 62.9 committed tok/s;
  592k yarn:4 prefill 916–922 / decode 15.2 tok/s.
- Accuracy campaign (EvalScope, thinking-on): GPQA int8 88.89% / hq 88.89% (band 89–90 ±2);
  AIME25 100% / AIME26 93.33% (hq); LBv2 medium full 58.6% admitted-only (119/203, SE 3.5pp).
- Engine fingerprints (bit-identity gates): nvfp4full+int8 MTP3 tg128 acceptance 0.4285714286
  (full head; 0.4022988506 `--lm-head-draft`); groupwise-int 0.3080808081. Any decode
  numerics change must re-derive these, not reproduce them.
