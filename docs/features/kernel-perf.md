# Runtime inference kernel performance

`feat/kernel-perf` owns runtime execution changes, not compilation speed. It is
stacked on `feat/dflash2`; splitting translation units and parallelizing compilation
belong to the separate build-speed features. These changes target the existing
single-GPU RTX 5090 (`sm_120a`) inference routes.

## Runtime changes

- **Fused Q/K normalization and RoPE.** Text attention and MTP use one Q/K
  preparation Op instead of separate Q RMSNorm, K RMSNorm and RoPE launches. It
  consumes the supplied frequency table and attention factor, including startup
  YaRN scaling, rather than baking in unscaled frequencies. The supported
  head-dimension-256, rotary-dimension-64 geometries accept both 1-D Text positions
  and three-axis MRoPE positions. Normalization and rotation form one operation:
  each output rounds to BF16 once, so the unrotated dimensions equal RMSNorm's
  output exactly while the rotated ones can differ from the separate chain by about
  one BF16 ulp.
- **Attention sigmoid gating.** The gated attention entry points apply the output
  gate in the output epilogue of every route that merges split partial state
  (grouped and parallel decode/verify/prefill routes of all KV types, the split
  prompt routes, and the hq-e8-2b small-T reducer), with the attention value still
  in FP32. Routes that write their output directly apply the gate as a separate
  sigmoid-multiply pass.
- **Programmatic dependent launch (PDL).** Projection, normalization, activation
  and attention kernels participate in producer/consumer launch chains: the NVFP4
  A16 GEMV, SIMT and sliced-K templates (Linear, LinearAdd, LinearSwiGLU and the
  attention/GDN input projections), RMSNorm, RoPE, sigmoid gating, the BF16/INT8
  grouped attention kernels and the attention merges. Consumers wait before their
  first dependent read; producers publish after their output stores. Upstream's
  own early-trigger sites (Q4/Q5 GEMV pairs, MoE decode) are unchanged. This
  changes launch scheduling, not the model's dependency ordering.
- **GDN recurrence.** Direct recurrent updates, batched updates, speculative
  record generation and replay folding join the PDL chain, including waits and
  publication around FP32 recurrent-state accesses. This is not a new recurrence
  formula or a change to the persistent-state dtype.
- **INT8 prompt split reduction.** The tiled INT8 prompt route (query width above
  256) can partition each query tile's key range across CTAs, write FP32 partial
  outputs and softmax statistics, and merge them with a final reduction. The split
  count (at most four) is chosen from the query-tile count and the device's SM
  count; workspace planning covers every reachable width. Causally empty
  partitions carry neutral statistics. This is runtime parallelism, distinct from
  the later build-speed feature's compilation split.

The implementation entry points are the [Q/K preparation contract](../../include/ninfer/ops/qk_norm_rope.h),
[gated attention contract](../../include/ninfer/ops/softmax_attention.h),
[Text execution](../../src/models/qwen3_5/execution/text.cpp),
[PDL helpers](../../src/core/pdl.cuh),
[GDN recurrence](../../src/ops/linear_attention/gated_delta_net/recurrent.cu),
and [causal attention dispatcher](../../src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp).

## Measurement and attribution

Use the [benchmark guide](../../bench/README.md) for same-binary controls and the
[public Engine product benchmark](../../bench/README.md#product-benchmark) for
prefill/decode measurements against an explicit `.ninfer` artifact.

- Defaults enable PDL and the available fusions. Set `NINFER_BENCH_PDL=0` to remove
  programmatic serialization from dependent launches, or `NINFER_BENCH_FUSIONS=0`
  to use separate Q/K RMSNorm/RoPE and separate attention sigmoid gating.
- PDL on/off leaves outputs unchanged. Fusions on/off is not byte-identical: the
  fused forms round each output once, so affected elements move by about one BF16
  ulp and greedy token streams can diverge.
- These controls are process-fixed: set them before Engine construction and use
  a fresh process for each comparison so eager execution and graph capture agree.
  Hold artifact, prompt/decode lengths, KV format, speculation, CUDA Graph mode,
  warmup and repetitions constant; vary one control at a time for attribution.
- Op microbenchmarks characterize individual operators, not end-to-end speedups.
  Confirm request-phase or complete-inference claims through the public Engine;
  use the guide's measured-only profiling boundary when launch attribution is
  needed. Disabling PDL/fusions does not undo the INT8 prompt split and is not a
  complete pre-feature baseline.

This guide describes implemented mechanisms, not a newly measured speedup. No
new GPU qualification or universal throughput/latency gain is asserted here.
A follow-up may re-measure FP16 accumulation for the INT8 prompt PV product
against the FP32 accumulation used now.
