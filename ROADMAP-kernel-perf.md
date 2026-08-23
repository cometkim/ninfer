# ROADMAP — general kernel-level performance levers (27B targets, one RTX 5090)

Desk note, 2026-08-23. Companion to `HANDOFF.md` (hq-e8-2b state) and `ROADMAP-1m-context.md`
(1M track). Scope: the *non-hq* kernel paths every profile runs — GEMM/GEMV, prefill and decode
attention on bf16/int8 KV, GDN, norms, launch structure — for `qwen3_8_27b` / `nvfp4full`
(int8 KV default) and the registered `nvfp4` profiles. **Nothing here was measured.** Every number
is either quoted from `HANDOFF.md`, `README.md`, `docs/performance.md`,
`docs/maintainer/linear-benchmark.md`, `docs/maintainer/qwen3.8-27b-artifact.md` §14.5, or derived
from those quotes plus a code read (file:line cited). Each work item names the measurement that
confirms or kills it; run that first.

Hardware reference used throughout (RTX 5090, sm_120a): sustained pure read 1674.5 GB/s
(`tools/hbm_bandwidth_probe.cu`, linear-benchmark.md §4); dense tensor peaks, FP32-accumulate
unless stated: FP4 1676 TFLOP/s (the 985.2 TFLOP/s = 58.79% row in linear-benchmark.md §9.12 implies
exactly this), INT8 838 TOPS, FP8 419, FP16/BF16 209.5 — and FP16 with **FP16** accumulate 419
(GeForce halves the FP32-accumulate rate).

## 0. TL;DR (ranked by expected end-to-end payoff)

1. **Prefill attention at ≥32k (WI-K1).** At 260k, causal attention FLOPs alone need ≥63.5 s of the
   103.6 s measured prefill even at the BF16/FP32-acc peak; bf16 and int8 prefill measure identical
   in the HANDOFF session-3 matrix although the i8 kernel already runs INT8 QK. Three stacked
   causes in the code: PV is FP32-accumulate f16 MMA (half rate; ≈4× the cycles of its INT8 QK),
   softmax is never overlapped with MMA (4 of 16 warps active in that phase, two CTA barriers per
   tile), and both prefill kernels run 1 CTA/SM with a 384-CTA grid at the 1024 chunk (2.26
   waves → ~75% wave efficiency at long context, any KV dtype). Estimated −30–40% prefill time at
   260k, −15% at 64k. Zero-code discriminator: `--prefill-chunk 2048/4096` at 64k/128k.
2. **Decode launch structure (WI-K2).** nvfp4full streams ≈15.85 GB/token → 9.47 ms floor at the
   sustained read rate vs 13.1–13.3 ms measured (75–76.5 tok/s tg128). A step is ~630 kernels
   (attention layer ≈12, GDN layer 9–10); `src/core/pdl.cuh` is wired only into q4/q5/MoE routes;
   the 27B norm+control projection is two–three kernels (`Composed`) where the 35B has one. PDL
   everywhere + the fusion list below ≈ −1 ms/step (+8–12% decode, more per MTP3 round).
3. **W4A4 TMA GEMM at 52.7–58.8% of FP4 peak (WI-K3)** — ~50% of short-prefill time. Not wave
   quantization (14336 and 16384 rows give the same %); needs ncu before designing.
4. **A4 activation quantization as a separate pass (WI-K4)**; cheap A/Bs (WI-K5).

Already at their roofline / tensor ceiling (checked, no lever): int8 decode attention at 260k,
BF16 GEMV, GDN chunked prefill (bf16/tf32 MMA). The hq decode/prefill kernels are covered by
HANDOFF §1/§2 and not repeated here.

## 1. Quoted baseline (measured elsewhere; exact sources)

| item | value | source |
|---|---|---|
| nvfp4full device weights | 16.03 GiB | qwen3.8-27b-artifact.md §14.5 |
| MTP0 tg128, nvfp4full, int8 KV (same-session) | 75.2 / 76.5 tok/s | HANDOFF §5c / §5e |
| MTP0 pp32768+tg64, nvfp4full int8 | 69.6–70.5 tok/s | HANDOFF §5c/§5e |
| MTP3 pp2048+128 / pp32k+64, nvfp4full int8 | 218.6 / 208.8 tok/s | HANDOFF §5e |
| ninfer_bench prefill bf16 / int8 KV, official nvfp4 (pp512 … pp65536) | 6604/6699 · 8111/7968 · 7636/7508 · 6037/5899 · 4667/4573 tok/s | HANDOFF session-3 matrix |
| Qwen3.6-27B nvfp4 serving, int8 KV: prefill / decode @7,680 and @260,096 | 11,191.5 / 86.4 ; 2,510.6 / 59.9 | README |
| Qwen3.8-27B nvfp4 serving: same cells | 8,340.4 / 71.2 ; 2,203.1 / 52.9 | README |
| Qwen3.6-35B-A3B serving: same cells | 15,544.3 / 271.1 ; 5,157.1 / 188.2 | README |
| BF16 GEMV `[14336,5120]` T=1 | 95.5–97.5 µs = 1505–1537 GB/s = 89.9–91.8% of sustained read | linear-benchmark.md §9.7–9.8 |
| NVFP4 A4 T=1024 `[14336,5120] [16384,5120] [34816,5120] [5120,6144] [5120,17408]` | 58.79 / 58.65 / 55.84 / 52.66 / 55.04% of FP4 peak | linear-benchmark.md §9.12 |

Model facts used (qwen3.6-27b-model.md §2): 64 layers (16 full attention, 48 GDN), hidden 5120,
intermediate 17408, 24 q-heads × 4 kv-heads × d256, GDN 48 V heads × 128 with FP32 state, vocab
248320. Prefill chunk default 1024 (`docs/cli.md`).

## 2. Where the time goes (derived)

### 2.1 Decode step, nvfp4full, tg128

Bytes read per token (nvfp4 = 9 B per 16 values; W8G32 = 34 B per 32; bf16 exceptions per
qwen3.8-27b-artifact.md §14.1):

| stream | GB/token |
|---|---:|
| MLP gate_up + down, 64 layers, NVFP4 | 9.63 |
| GDN in_proj 48 × `[16384,5120]` NVFP4 | 2.27 |
| GDN out_proj 47 NVFP4 + 1 BF16 | 0.89 |
| attention qkgv 10 NVFP4 + 6 BF16 | 1.29 |
| attention out 14 NVFP4 + 2 BF16 | 0.37 |
| lm_head `[248320,5120]` W8G32 | 1.35 |
| GDN control a/b (bf16), conv, norms | 0.05 |
| **total** | **15.85** |

→ 9.47 ms at 1674.5 GB/s (8.85 ms at the 1792 spec) = **105 tok/s ceiling vs 75–76.5 measured**
(13.1–13.3 ms). The ~3.7 ms above the floor decomposes, by construction of the schedule in
`src/targets/qwen3_6/impl/runtime/text_context_impl.h` (`attn_mix` 796–855, `gdn_mix` 857–960,
`mlp_tail` 962–969), into:

- ~200 weight-streaming GEMV launches at some efficiency below the probe (BF16 reaches 90%; the
  nvfp4 decode schedules are "cold-cache winners" but their % is not published);
- ~380 non-streaming kernels: per attention layer `rmsnorm, q-rmsnorm, k-rmsnorm, rope, attention
  partial, split reducer, sigmoid_mul, post rmsnorm` (8); per GDN layer `rmsnorm, control
  projection (1 kernel at T=1, 2 at T=2..8 — `bf16_gdn_gating_proj_plan.cpp:448` Composed route,
  `:214` SmallTSplit10), recurrent, gated_rmsnorm, post rmsnorm` (5–6); each is a few µs of
  latency-bound work;
- ~630 inter-kernel dependencies inside the graph, each a drain/ramp gap unless PDL-chained.

GDN FP32 state is an extra 0.30 GB/token of read+write (48 layers × 3 MiB × 2) ≈ 0.18 ms.

### 2.2 Prefill (27B), by context length

Causal attention FLOPs = 2 matmuls × 2 × 24 heads × 256 × L²/2 = `12288·L²` per layer, × 16.
Linear FLOPs = 2 × 25.6e9 params × L. GEMM time assumes the measured ~950 TFLOP/s W4A4 rate.

| context | measured prefill | attention floor @BF16/FP32-acc peak (209.5) | GEMM @950 TFLOP/s |
|---:|---:|---:|---:|
| 8,192 (pp8192, official nvfp4, bf16 KV: 7636 tok/s) | 1.07 s | 0.06 s | 0.44 s (~41%) |
| 65,536 (pp65536: 4667 tok/s) | 14.0 s | **4.03 s (≥29%)** | 3.5 s |
| 260,096 (Qwen3.6 nvfp4 serving, int8 KV: 2510.6 tok/s) | 103.6 s | **63.5 s (≥61%)** | 14 s |

The 35B cross-check (10 attention layers × 16 heads: `81920·L²`): attention floor 26.5 s of the
50.4 s measured at 260,096 — the same picture. So from ~32k up, prefill is bounded by the attention
kernels' tensor-format ceiling, not by GEMM; and at ≤8k it is roughly half GEMM.

## 3. Work items

### WI-K1 — Prefill attention: format, overlap, and wave quantization

Code facts (`src/ops/kernel/gqa_attention_prefill_i8.cuh`, `..._bf16.cuh`,
`src/ops/launcher/gqa_attention_prefill_{i8,bf16}.cu`):

- Grid is `(ceil(tokens/64), q_heads)` (`gqa_attention_prefill_i8.cu:25`, `_bf16.cu:25`) →
  **384 CTAs at the 1024 chunk**; both kernels are 1 CTA/SM (bf16: `__launch_bounds__(128, 1)` +
  96 KiB dynamic smem, `_bf16.cuh:130`; i8: 92,672 B smem + 512 threads × `__maxnreg__(120)`,
  `_i8.cuh:48,215`). 384/170 = 2.26 waves; at long context CTA durations are uniform (the causal
  diagonal varies work by ≤1024 keys), so the third wave runs 44 CTAs on 170 SMs ≈ 75% wave
  efficiency for every KV dtype, every chunk, for the whole context.
- i8 kernel phase structure per 64-key tile: warps 0–3 (`ProducerWarps`, `:230`) do INT8 QK
  (`mma_s8`, `:423`) **and** the fp32 softmax + P→fp16 smem stores while warps 4–15 only
  dequantize V (`:511–527`); `__syncthreads`; all 16 warps do PV via
  `mma_f16` (`:564`) = `mma.sync...f32.f16.f16.f32` (`src/ops/common/mma.cuh:45`, **FP32
  accumulate**); `__syncthreads`. Per tile and per SM sub-partition: QK = 64 m16n8k32.s8 ≈ 1.0k
  cycles at the INT8 peak; PV = 4 warps × 32 m16n8k16 at the FP32-acc f16 rate ≈ 4.1k cycles; the
  softmax (~32 exp2 + scale shuffles/FMAs + masks + 32 STS.16 per thread) sits on the critical path
  with the tensor pipe idle. The INT8 advantage is spent entirely in the PV phase — which is why
  int8 prefill ≈ bf16 prefill in the session-3 matrix.
- bf16 kernel: FA2 single-buffer layout (`_bf16.cuh:1–12`), 4 warps/SM total, so the softmax is
  also exposed (no second warp per sub-partition to fill the pipe).

Levers, cheapest first:

- (a) **Wave quantization / scheduler.** Zero-code discriminator first: run a 64k and a 128k
  prompt with `--prefill-chunk 2048` and `4096` (768 / 1536 CTAs → 4.5 / 9.0 waves); a ≥10%
  prefill gain at 64k+ confirms. The real fix is a persistent grid (`#SMs` CTAs pulling
  `(q_block, head)` items from an atomic counter, longest key range first — LPT), which also fixes
  the short-context imbalance (q-block 15 sees 2× the keys of q-block 0 at 2k) without touching
  chunk/workspace sizing. Applies to the hq banded route too (it reuses the bf16 kernel body).
  Estimate: +20–25% on the attention kernel at ≥32k.
- (b) **FP16-accumulate PV in the i8 kernel.** P and V are already fp16 in smem; change the PV
  `mma` to `f16.f16.f16.f16` accumulating into per-tile fp16 fragments, then add into the existing
  fp32 `acc` once per tile (P∈[0,1], |V| bounded by the int8 group scale, 64 terms per tile — the
  per-tile fp16 partial then fp32 running sum is the llama.cpp consumer-GPU pattern). Halves the
  dominant phase: tile ≈ 1.0k + 2.0k + softmax. Requires re-qualifying against the prefill oracle
  (`tests/ops` gqa prefill conformance, plus the hq prefill gate if the bf16 body changes).
- (c) **Softmax/MMA overlap.** Give the 12 idle warps of phase A real work: two 64-row q tiles per
  CTA in ping-pong (one tile's softmax while the other's QK/PV runs), or fold the V dequant into the
  PV warps and let two producer groups alternate. FA3's structure on mma.sync. Largest code change
  of the three; do after (a)/(b) are measured.
- (d) bf16/hq route: (a) transfers directly; (b) needs a bf16→fp16 conversion of the staged V (in
  smem after cp.async lands) plus a range caveat (fp16 max 65504) — only worth it if the bf16 route
  remains the hq prompt body; the HANDOFF §2 "int8 scratch + IMMA" option makes the hq route inherit
  the i8 kernel instead.

Confirm with `ninfer_causal_softmax_attention_bench` (bench/README.md §"Softmax Attention Op
benchmarks") at 32k/64k: bf16 vs i8 TFLOP/s against 209.5 (bf16) and the i8 mixed ceiling
(INT8 838 for QK, 209.5 or 419 for PV). Expected end-to-end: −30–40% prefill time at 260k, −15% at
64k, ~0 at 8k.

### WI-K2 — Decode launch structure: PDL coverage and small-kernel fusion

Code facts:

- `src/core/pdl.cuh` (`launch_dependent`, `wait_for_dependencies`, `trigger_dependents`) is used
  only by `linear/q4/*`, `linear/q5/*`, `gdn_input_proj/q4_q5/*`, `sparse_moe/*` (commits
  `7c543611`, `3a3205c2`). Not by the nvfp4 / fp8 / w8 / bf16 GEMV and small-T routes that the
  nvfp4 profiles run, nor by attention, the split reducer, rmsnorm, rope, sigmoid_mul,
  gated_rmsnorm, the GDN recurrent kernel, the control projection, sampling. Stream capture keeps
  the programmatic edges, so the decode graphs inherit it.
- Per-layer launch counts (27B): attention layer 12 (`attn_mix`: rmsnorm, projection, q-norm,
  k-norm, rope, partial, reducer — always launched even at `splits == 1`,
  `gqa_attention_decode.cu:152–160` — sigmoid_mul, out-proj linear_add, rmsnorm, swiglu, down);
  GDN layer 9 at T=1 / 10 at T=2..8 (Composed norm + control: `bf16_gdn_gating_proj_plan.cpp:442–
  461`, split-K partial + reduce `:214`; the 35B has the fused `MmaCooperativeSplit32` norm+control
  kernel, `:386–391`). ≈630 kernels per MTP0 step, ≈680 per MTP3 verify step, plus ~20 per draft
  step × 3.

Levers:

- (a) **PDL on every decode kernel.** Mechanical: `launch_dependent` at every launcher,
  `wait_for_dependencies()` before the first dependent read, `trigger_dependents()` after the last
  weight-stream issue (q4 pattern: `q4_rowsplit_gemv.cuh:424,516`). Typical saving ~1 µs per
  dependency on small kernels → 0.4–0.7 ms/step.
- (b) **Fusions** (each removes launches *and* a latency-bound kernel):
  - q-norm + k-norm + rope → one kernel (−2 × 16/step);
  - `sigmoid_mul` into the split reducer epilogue (−16; the reducer already writes every output
    element) or skip the reducer when `splits == 1` and write the output from the partial kernel;
  - attention input rmsnorm and both post-mixer rmsnorms into the consuming GEMV prologue (each
    CTA recomputes the 5120-element norm from L2, ~10 KB, trivially cheap) (−16 −64);
  - 27B norm + control projection as one kernel (the 35B already has the shape: `MmaCooperativeSplit32`
    with `NormalizeInput`), or fold the 96 control rows into the GDN input-projection GEMV as extra
    rows with the `softplus/exp/sigmoid` epilogue (−48 to −96/step);
  - `gated_rmsnorm` into the GDN out-proj prologue (−48) — per-head 128-wide norm of `o` gated by
    `z`, recomputed per CTA from L2 (24 KB).
  Together ≈ −200 launches/step. With the per-kernel latency these kernels carry, ≈ −0.5 to
  −0.8 ms/step.
- (c) Draft steps (MTP3): the same ~20-kernel chain runs three times serially per round; (a)+(b)
  apply verbatim to `mtp_forward_core`.

Confirm with one nsys of a graph replay (tg128 and pp32k+64): histogram of inter-kernel gaps and
of kernels < 5 µs. Expected: +8–12% MTP0 decode, similar absolute ms per MTP3 round. Contract
note: fusing a norm into a GEMV prologue moves an observable Op output (`h`) inside a fused Op —
allowed by AGENTS.md "Numerical correctness" as long as the fused Op is oracle-qualified on its
public inputs/outputs; keep `hidden` materialized where prefix reuse or MTP reads it.

### WI-K3 — W4A4 TMA GEMM inner-loop efficiency

`src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh`: 256×128×128 per stage, 8 consumer warps (4×2, each
64×64 → 32 `mma_nvfp4_e4m3` per k64) + 128 producer threads, `setmaxnreg` 232/40, 1 CTA/SM,
`Nvfp4W4a4TmaSchedule<256, 3, 1>` for the attention/GDN/residual shapes and `<256, 2, 1>` (two
stages) for `MlpGateUp` (`nvfp4_w4a4_tma.cu:16–17`, `:136`). Measured 52.7–58.8% of the FP4 peak
(§1). Grid is `(N/128, T/256)`.

What the numbers say: `[14336]` (448 CTAs, 2.6 waves) and `[16384]` (512 CTAs, 3.01 waves) reach
the same % → not wave quantization; the inner loop itself runs at roughly 60–68% of peak (after
subtracting pipeline fill/drain and the bf16 epilogue). Candidates to test with ncu
(`sm__pipe_tensor_op_*` active %, `smsp__warp_issue_stalled_{barrier,long_scoreboard,mio_throttle}`,
`lts__t_bytes`): exposed TMA latency with only one stage of lookahead on the 2-stage gate_up
variant (it is the lowest of the three large-N shapes at 55.8%); per-k64 ldmatrix/LDS issue
(4 ldmatrix.x4 + 8 ldmatrix.x2 + 12 LDS per warp per k64 ahead of 32 MMAs) not overlapped across
the stage boundary; L2→SM delivery (CTA intensity 341 FLOP/B needs ≈4.9 TB/s at peak). Establish
the achievable ceiling on this box with CUTLASS's sm120 block-scaled GEMM example at the five
registered shapes before redesigning. Expected if closed to ~75%: +10–15% prefill at ≤8k, a few %
at 64k+.

### WI-K4 — Fuse A4 activation quantization into producer epilogues

`nvfp4_w4a4_quantize_kernel` (`nvfp4_w4a4_mma.cuh:388–407`) is a separate pass per W4A4 site: it
re-reads the bf16 intermediate and writes codes+scales. For the down projection at T=1024 that is
a 35.6 MB write + read of the SwiGLU output per layer (~40 µs round trip against a 198 µs GEMM);
for gate_up/qkgv/in_proj it is the 10 MB normalized hidden. The SwiGLU TMA epilogue owns full
16-groups of its output row (tile columns are 128-aligned) and the rmsnorm kernel owns whole rows,
so both can emit the 16-group E4M3 scale + E2M1 codes directly. Saves one launch and the
intermediate traffic per site (64 × 5 sites per chunk); ≈ −3% per 1024-token chunk. Requires the
fused Ops (`post_mixer`, the projection leaves) to be re-qualified against the FP64 oracle from
their public bf16 inputs.

### WI-K5 — Cheap A/Bs (template parameters only)

- `nvfp4_linear_swiglu_decode.cu:15–16` uses `Nvfp4ScaleAccess::Direct` (per-lane scattered
  1-byte scale loads) while the measured linear-decode winner `Nvfp4LinearDecodeProductionSchedule`
  is `StagedRaw` (`nvfp4_config.h:165–167`, cooperative uint4 staging). gate_up is the largest
  decode stream (6.4 GB/token of the 15.85). One-line A/B on `ninfer_nvfp4_linear_swiglu_bench`.
- `MlpGateUp` TMA at `Stages=2` vs `3` (WI-K3): same shared-memory budget question; one-line A/B on
  `ninfer_linear_bench --qtype nvfp4 --policy a4 --n 34816 --k 5120 --t 1024`.

### Not kernel-level, recorded for completeness

lm_head W8G32 = 1.35 GB/token (≈0.8 ms, ~6% of an MTP0 step; amortized over the verify width under
MTP3); the nine bf16 exception parents of nvfp4full ≈ 1.07 GB/token; MTP draft weights ≈ 0.8 GB per
draft step ×3 per round. These are artifact decisions (`ROADMAP-1m-context.md` §7), not kernels.

## 4. Checked and not a lever

- **int8 decode attention at long context**: Qwen3.6-27B nvfp4 decode 86.4 → 59.9 tok/s from 7,680
  to 260,096 tokens = +5.12 ms/step; the int8 group-64 KV at 260,096 is 16 layers × 2,112 B/token
  = 8.79 GB = 5.25 ms at 1674.5 GB/s. The kernel is at the read roofline; only fewer bytes (hq,
  BLASST/Quest — HANDOFF §1) help.
- **BF16 GEMV** at 89.9–91.8% of sustained read (linear-benchmark.md §9.8); the nvfp4 decode and
  small-T schedules are measured winners (`nvfp4_config.h:165–237`).
- **GDN chunked prefill** (`linear_attention/gated_delta_net/chunked/*`): bf16 and tf32 MMA
  throughout (prepare_wy_wu, state_passing, output); no SIMT-bound stage found by reading.
- **GDN recurrent decode** (`recurrent.cuh:32`): 384 CTAs × 128 threads per layer, state in
  registers across the verify width, ~6 MiB/layer traffic; a few µs per layer — only the launch
  (WI-K2) and an optional `gated_rmsnorm` fold matter.
- Decode split policy, KV append fusion, swiglu decode fusion (`nvfp4_linear_swiglu_decode.cu`,
  one warp per gate/up row pair) and the GDN conv/snapshot fusion into the projection epilogue
  (`nvfp4_gdn_snapshot_decode.cu`) are already the fused forms.

## 5. Not examined

Vision tower kernels (`bidirectional_gqa_attention.cuh`, `vision_attention.cuh`, vision Q4/Q5
small-T), the 35B sparse-MoE decode/prefill kernels and DFlash path, `causal_conv1d.cuh` prefill,
sampling, the hq codec/prefill scratch (HANDOFF §2 F2 and §1 Tier 2 cover those).

## 6. Measurement plan (in this order; each is ≤ an hour on the bench box)

1. `--prefill-chunk 1024 / 2048 / 4096` on a 64k and a 128k prompt, int8 KV (WI-K1a discriminator).
2. `ninfer_causal_softmax_attention_bench` bf16 vs i8 at 8k/32k/64k → TFLOP/s vs the 209.5 / 419 /
   838 ceilings (WI-K1b/c sizing).
3. nsys of one decode graph replay at tg128 and pp32k+64: gap histogram + sub-5 µs kernel share
   (WI-K2 sizing).
4. ncu on `ninfer_linear_bench --qtype nvfp4 --policy a4` at `[34816,5120]` and `[14336,5120]`,
   T=1024, plus the CUTLASS sm120 example at the same shapes (WI-K3).
5. WI-K5 A/Bs.

## 7. Risks / caveats

- All estimates assume the spec dense-peak ratios (INT8 4×, FP16-acc 2× the BF16/FP32-acc rate)
  hold for `mma.sync` on sm_120; item 2 of §6 measures that directly.
- FP16-accumulate PV and prologue-fused norms change private arithmetic, not Op semantics; both
  must re-pass the existing FP64 oracles (gqa prefill conformance; linear/post_mixer suites) and the
  hq `tools/test_kv` gates where the bf16 kernel body is shared.
- PDL correctness rule (`pdl.cuh`): every consumer control path must `wait_for_dependencies()`
  before its first producer-dependent read, including early-exit paths; every producer CTA must
  trigger or exit.
- A persistent attention scheduler changes the launch shape the layouts tiering and graph capture
  see (`layouts_impl.h` split capacities are decode-side and unaffected; prefill is eager) — verify
  `--prefix-reuse` and Vision prefill still take the same route.
