# ROADMAP — DFlash2 speculative backend (Qwen3.8-27B, one RTX 5090)

The active speed project. DFlash 2 (incoai/z-lab, 2B all-SWA-2048 drafter: one parallel block
forward + top-16 candidates/position + low-rank selector + two-tap dynamic convs) replaces MTP's
sequential draft forwards with one parallel pass. It is a SEPARATE `SpeculativeBackend::DFlash2`
on the same family block-verify schedule and ReplaySSM transactions — not an extension of the
35B-only v1 DFlash (whose rope-scaling reject is substantive and stays).

Branch: `feat/dflash2` (stacks on `feat/kernel-perf`; merges `feat/qwen3.8-nvfp4full` for the
artifact module). Landed as `squash(feat/dflash2)` on dev.

## Gate (ii) verdict — acceptance survives long context (llama.cpp, 2026-08-26)

Built llama.cpp PR #27342 (f7aadef) locally — REQUIRED: the b10629 prebuilt rejects both hub
drafter GGUFs ("wrong number of tensors"). Target `ggml-org/Qwen3.8-27B-Q4_K_M`, KV q8_0, -fa,
one slot, seed 42; prompts = the longcontext soup fixtures tiled to 1,489/29,976/124,075/258,025
true tokens + one identical long-reasoning question; 1,024 tokens/request; metric = committed
tok/round from per-request verify counters.

| true context | DFlash2 greedy | DFlash2 serving | DSpark greedy | DSpark serving |
|---:|---:|---:|---:|---:|
| 1,489 | 3.73 | 3.43 | 2.86 | 2.75 |
| 29,976 | 3.47 | 3.23 | 2.64 | 2.44 |
| 124,075 | 3.76 | 3.22 | 3.15 | 2.81 |
| 258,025 | 3.86 | 3.11 | — | — |

- **No steep decay**: serving gives up ~9% from 2k to 262k; greedy rises. The
  SWA-drafter/target mismatch hq cannot fix costs single digits, not collapse.
- Every cell was thinking-ON by construction (all 1,024 tokens in `reasoning_content`; verified
  by full-response capture). Thinking split at 2k serving: OFF 4.65 vs ON 3.43 tok/round —
  reasoning content costs ~1.2 tok/round; the margin over MTP3-class is the narrower +12–25%
  thinking-on, far larger thinking-off. Stage-3 must be judged on thinking-ON traces.
- Quantized drafter: uniform Q4_K_M costs −0.06…+0.05 tok/round (nothing) vs BF16, and
  syvai/W4A16 independently verified dead-even at 258k — two 4-bit-class drafters hold at both
  extremes. BF16 stays the plan because it fits, not because it buys acceptance; a quantized
  drafter is the acceptance-free VRAM fallback at 786k.
- DSpark ≈ built-in MTP (confirmed on a third stack); DFlash2 is the only step-up candidate.
- Sizing rule: use the MEASURED ratio over MTP3-class (3.1–3.9 tok/round here), never the card's
  temp-1.0 4.1–5.5 band (confirmed inflated; temp-0.6 costs 0.2–0.6 tok/round).
- llama.cpp wall-clock is meaningless for fork comparisons (Qwen3.8 GDN decode collapses to ~1
  t/s at 258k — ggml-org/llama.cpp#27623; non-BF16 draft prefills crawl ~4× slower) — the
  acceptance counters are per-round and unaffected.

## Port checklist — items 1–4 LANDED

1. **Width-8 hq block verify + MTP K≤7 plumbing.** `gqa_small_t_chunk_tokens(DType)` (U8→8,
   else 6), hq instantiation TokenTile=8, `kMtpDecodeMaximumDrafts` 5→7. Conformance 18/18; hq
   MTP3 greedy stream byte-identical old-vs-new; int8 fingerprint EXACT; MTP7-on-hq 3.68
   tok/round / 159.1 committed t/s; BF16/I8 keep their tuned 6-token tiles.
2. **The artifact module (nvfp4full v2).** The drafter rides the 27B nvfp4full artifact v1-style
   (same-artifact appended objects, binder-enumerated; converter + binder land atomically). 66
   module objects; weight matrices quantized with the fork's own NVFP4 encoder (9.48–9.59% rel
   err, the target's own band; norms + conv base stay BF16); RECIPE_ID
   `qwen3_8_27b_nvfp4full-v2`; artifact 19,406,942,468 bytes (18.07 GiB; the module ~1.3 GiB,
   freeing ~2.55 GiB at every held envelope). Verified: all non-reformatted objects
   byte-identical to v1, module tensors code-exact against reference re-encodes, MTP3-hq stream
   byte-identical, load-plan green. Fixed facts: block 8, taps [5,19,33,47,61] (0-based),
   conv kernel 2/group 16, selector rank 256/top-k 16, SWA 2048 all layers, rope base 1e7
   unscaled with SWA-local positions (target YaRN never applies — no reject exists).
3. **Draft ops.** `dflash2_dynamic_conv`, `dflash2_selector_scores` (NVFP4 codebook decode
   inline), `dflash2_topk`, `dflash2_selector_walk` (device-side gather so the proposal composes
   inside the decode graph) + `cast_bf16_to_fp32`, `dflash2_selector_predecessors`; `ops::swa`
   window parameterized (2048 | 4096); five NVFP4 A16 linear problems. All conformance suites
   pass. Layer wiring pinned from llama.cpp models/dflash.cpp 670–745.
4. **Engine integration + validation.** `DFlash2PersistentState` (cyclic SWA KV ×5), sinks,
   append/propose/decode, layouts/schedule/program wiring, staged reject removed. The one
   execution defect (session 27's 1.3 tok/round): the context K/V injection applied
   `context_norm` with `unit_offset=true` — the fork TARGET's (1+w) convention — while the incoai
   module uses plain `w` everywhere (llama.cpp, source config, v1 DFlash). ~2.2× median
   per-channel distortion of the cached context K/V; one-line fix, acceptance → the reference
   band. Native cells (nvfp4full + hq, 1,024 tokens, thinking on; llama.cpp BF16 in parens):

   | cell | greedy | serving | committed t/s greedy | serving |
   |---|---|---|---:|---:|
   | 2k | 3.63 (3.73) | 3.13 (3.43) | 176.0 | 152.1 |
   | 8k | 3.43 (–) | 3.32 (–) | 157.6 | 149.1 |
   | 32k | 3.67 (3.47) | 2.98 (3.23) | 153.3 | 121.6 |

   Same-stack 8k greedy controls: MTP3 2.64 @ 121.2, MTP7 3.52 @ 137.4 — **+30% over MTP3, +15%
   over MTP7 committed**; the one-parallel-draft-forward structural advantage measured in full.
   Long context: greedy 3.82 @124k / 3.83 @258k (llama.cpp 3.76/3.86 — flat-to-up like the
   reference); serving 3.25/3.23; decode 115.2/82.1 tok/s greedy; same-cell MTP3 @258k 2.60 @
   54.0 — **+52% committed t/s at the native envelope**; a 258k raw-output run is coherent.
   Vision prompt 3.17 (sink tap verified); caution-repro: prefix reuse clean (fresh 4.91,
   continuing 4.68, zero errors); swa suite gained real 2048-window fixtures.

## Stage 3 — the MTP purge (owner decision; inputs live)

If DFlash2 validates as strictly dominant: reclaim ≈1.2–1.4 GB (MTP W8 ~0.45 + Q4 draft head
~0.36 + the +1 hq KV layer ~0.30 @524k + round state/graphs) + ~0.85 GB off the artifact.
Two steps: (1) a load-time materialization skip keyed on `--spec` (reversible, contract-neutral)
— rides any near-term landing; (2) only after DFlash2 beats MTP3 across THIS fork's workloads
(long context, tool-calling, temp 0.6, thinking-on), an artifact variant dropping `mtp/*` +
`draft_head`. Preconditions: never purge to make room (hq headroom covers the drafter); promote
the greedy token-stream diff to the primary fingerprint first. If the fork ever tunes a drafter
or the MTP head, calibrate against nvfp4full's OWN outputs (the gittenser on-policy NVFP4-target
recipe is the method reference).
