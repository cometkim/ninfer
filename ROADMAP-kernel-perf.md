# ROADMAP — kernel-level performance levers (27B targets, one RTX 5090)

Condensed at the 2026-08-27 reorganization: landed items are one-line summaries with their key
measured numbers; open items keep their design detail. Branch: `feat/kernel-perf` (stacks on
`feat/1m-context`); landed on dev as `squash(feat/kernel-perf)`.

Hardware reference (RTX 5090, sm_120a): sustained read 1674.5 GB/s; FP32-acc peaks FP4 1676 /
INT8 838 / FP8+FP16acc 419 / BF16-acc 209.5 TFLOP/s. Decode floor: nvfp4full streams ≈15.85
GB/token ⇒ 9.47 ms/step (105 tok/s) vs 75–76.5 measured (MTP0 tg128); the ~3.7 ms above the
floor is ~630 PDL-chained kernel launches + latency-bound small kernels. Long-context prefill is
attention-format-bound (≥61% of wall at 260k), ≤8k is ~half GEMM.

## LANDED (with reviews closed; review docs deleted 2026-08-27, verdicts folded here)

- **WI-K1b — f16-accumulate PV in the i8 prefill kernel** (+ the review-K1 exact 2⁻⁶ range guard
  and `__maxnreg__` 128): i8 kernel 1.39× at 64k (94.2% of 209.5); engine pp65536 int8 +8.1%,
  pp8192 +1.5% (alternating A/B). Range-profile conformance cases added
  (`large_v_peaked`/`realistic_qk`); the int8 route itself — not the f16 partial — exceeds the
  registered flat-profile criterion at realistic magnitudes (open calibration question, pinned
  per case). Engine-gain model rule recorded: weight per-chunk kernel ratios by chunk context.
- **WI-K1a-split — key-range splitting** for i8, then BF16 and hq single-band prompt routes
  (FP32 pre-normalized partials, prefill-owned reducer, wave-fill S policy; S=1 keeps the
  bit-identical path): kernel 1.66× over pre-K1b at 64k; engine pp65536 int8 **+14.2%**, hq
  **+6.0%** (hq prompt total is scratch-dominated — see WI-K1(d)). Split-kernel trap list
  recorded in HANDOFF cautions (slot aliasing, OOB dead-row writes, q0 offset, capacity SUMs,
  bf16 −inf/−inf NaN guard).
- **WI-K2a — PDL across the decode chain**, with the load-bearing correctness rule: producers
  publish (`pdl::publish`) at kernel END after their stores; consumers wait before the first
  dependent read including early exits. The first landing (entry triggers) raced every edge and
  corrupted real-prompt MTP0 streams while bench fingerprints stayed exact — see HANDOFF
  cautions. MTP0 tg128 +2.4%. Pre-existing q4/q5 + sparse_moe sites converted (REVIEW S1);
  groupwise-int + 35B smokes re-verified.
- **WI-K2b — fused norm+control (MmaCooperativeSplit40)** for the 27B decode route: −48
  launches/MTP0 step; MTP3 fingerprint moves 0.4226… → 0.4285714286 (the current nvfp4full+int8
  full-head reference). MTP3 tg128 +3.9% incl. K2a.
- **Gate fold + qk_norm_rope** (session-21 fusions): sigmoid gate is part of the A1/A3 attention
  contract (reducer epilogue, bit-identical; two registered criteria rescaled ×~1.3 with
  rationale); `ops::qk_norm_rope` replaces rmsnorm(q)+rmsnorm(k)+rope bit-exactly (−32
  launches/step), accepts [T] and [T,3] MRoPE positions (the prefix-reuse bridge regression,
  caught in the field, fixed + covered). Both individually sub-noise at tg128; structural wins.
- **Robustness fixes riding the track**: per-request error boundary (request-scoped host
  exceptions no longer fail_all the engine; verified by fault injection); TMA descriptor blocks
  free on the consuming stream (the 786432 prefill live-lock); U8 banded workspace capacity
  covers sub-band split partials (the 524288 hq bad-allocation).
- **WI-K6 — decode host boundary: sized and deprioritized** (median 91 µs device idle between
  replays; the earlier 0.86 ms was tracer overhead). MTP3 round shape unmeasured.

## REFUTED (do not retry as designed)

- **WI-K5a swiglu StagedRaw**: the apparent −10.5% read uninitialized smem as scales (the kernel
  never staged them); correct staging costs ≈ the gain. `Direct` stays.
- **MlpGateUp TMA Stages 3**: slower than 2. Not pipeline depth.
- **f16-acc PV transfer to the bf16 body**: QK and PV run at the same rate there; the V
  conversion + extra barrier ate the ≤25% ceiling. Regressed; reverted.
- **Warp specialization for the hq decode kernel** (hq track): mma acc fragments cap 8-warp
  variants at 1 CTA/SM; the homogeneous-TC decode gate is closed.

## OPEN (in order)

1. **Remaining fusion block (−128 launches/step)**: attention input rmsnorm + both post-mixer
   rmsnorms into consuming GEMV prologues (−16 −64), `gated_rmsnorm` into the GDN out-proj
   prologue (−48). Each is an observable-Op-output move — per-fusion FP64 oracle gates +
   declared fingerprints apply; keep `hidden` materialized where prefix reuse or MTP reads it.
   The two landed fusions were individually sub-noise; this block is what could clear the tg128
   noise floor.
2. **WI-K1c — softmax/dependency overlap in the i8 prefill kernel** (ncu-attributed: issue/
   latency-bound, no SOL metric above 42%, barrier 16.8% + math_pipe_throttle 18.7% of an 8.78
   warp-cycles/inst stall profile; two q-tiles do NOT fit smem). Design: fold V-dequant into the
   PV warps, two alternating producer groups (4 warps QK+softmax each), named-barrier phase
   splits. Estimated −20–30% kernel ⇒ +5–10% prefill at 64k. External calibration: a hand FA2
   d=128 kernel reaches 94.4% of 209.5 on this GPU (d=256 register pressure is the open
   difference); build the structure format-agnostic — SageAttention2/3 formats are later,
   quality-gated swaps.
3. **WI-K1(d) — int8 scratch for the hq prompt route** (design frozen): quantize the scratch
   kernel's decoded ROTATED rows per group-64 into a page-like i8 scratch behind a synthetic
   identity block table so the existing i8 prefill kernel reads it unchanged (inherits
   K1b + K1a-split); three write paths, U8 capacity re-sum, hq oracle gains the int8 mirror.
   **Competing successor: nvfp4 as the KV/scratch format** (ROADMAP.md item 5) — same-or-fewer
   bytes and the stored blockscale-k16 layout feeds `mxf4nvf4` QK directly; decide between them
   together. The 35B bf16-KV case wants (c)-style overlap instead of the mma swap (refuted).
4. **WI-K3 — W4A4 TMA GEMM (gate_up TMA alone = 17.5% of pp65536 kernel time)**: ncu says
   L2-delivery-bound (L2 77.3% > compute 56.7%): operand re-reads (activations ×272 N-tiles,
   weights ×4 M-tiles) saturate L2. Clusters/TMA-multicast do NOT exist on consumer sm_120;
   the 512-row M tile needs register walls. The CUTLASS ceiling probe is **BLOCKED on the
   Windows/MSVC host ABI** (sm120 collectives pass `alignas(128)` TmaDescriptor by value; MSVC
   cannot lay it out — same class our own TMA kernel documents and sidesteps). Needs a Linux
   box or a pointer-passing CUTLASS fork. Published reference: Colfax ~73% FP4 utilization on
   sm_120; if their schedule hits ~73% here, adopt rather than iterate. One ncu at
   [34816,5120] T=1024 stays owed to a Linux box.
5. **WI-K4 — A4 quantization fused into producer epilogues** (SwiGLU TMA epilogue + rmsnorm own
   full 16-groups; saves the intermediate traffic + one launch per site, ≈ −3%/1024-token
   chunk): re-qualify the fused Ops against the FP64 oracle from public bf16 inputs.
6. **Competitive base-decode A/B**: vLLM 0.27 + FlashInfer SM120 (gittensor NVFP4 checkpoint)
   field-reports 88.5 tok/s no-spec c=1 vs our 75–76.5 MTP0 tg128; run one controlled pair
   (same prompt, greedy, KV dtype noted). If it reproduces, re-base the decode-overhead
   narrative and WI-K2's residual sizing against it. Transferable diagnosis pattern from the
   same report: when engine ≠ bench on identical tensors, suspect stream/L2/scheduling
   interference before kernel quality.

## Checked, not a lever

int8 decode attention at 260k (read-roofline; only fewer bytes help — hq/Tier-2 selection);
BF16 GEMV (89.9–91.8% of sustained read); GDN chunked prefill (FlashInfer's new SM120
`chunk_gated_delta_rule` is the first credible external A/B — reopen if it wins); GDN recurrent
(only the WI-K2 launch/fusion levers); KV append/swiglu decode/GDN snapshot fusions already
fused. Not examined: vision tower kernels, 35B sparse-MoE decode, causal_conv1d prefill,
sampling (hq codec/scratch covered by the hq track).

## Standing caveats

- PDL correctness rule (publish-at-end / wait-before-first-read) is a contract, not a style.
- Private-arithmetic changes (f16 partials, prologue-fused norms) must re-pass the FP64 oracles
  and re-derive the engine fingerprints; pairwise parity is supplementary only.
- A decode-graph schedule change must keep `--prefix-reuse` and Vision prefill on the same
  route (layouts tiering and capture see the launch shape).
