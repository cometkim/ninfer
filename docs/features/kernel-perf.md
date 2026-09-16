# Runtime inference kernel performance

`feat/kernel-perf` owns runtime execution changes, not compilation speed. It is
stacked on `feat/dflash2`; splitting translation units and parallelizing compilation
belong to the separate build-speed features. These changes target the existing
single-GPU RTX 5090 (`sm_120a`) inference routes.

## Runtime changes

- **Fused Q/K normalization and RoPE.** Text attention and MTP use one Q/K
  preparation Op instead of separate Q RMSNorm, K RMSNorm and RoPE launches. It
  retains the BF16 normalization boundary and consumes the supplied frequency
  table and attention factor, including startup YaRN scaling, rather than baking
  in unscaled frequencies. The supported head-dimension-256, rotary-dimension-64
  geometries accept both 1-D Text positions and three-axis MRoPE positions.
- **Attention sigmoid gating.** Eligible Small-T attention routes apply the
  output gate in their final output/reduction kernel, avoiding a separate
  sigmoid-multiply pass. Prompt and other non-fused routes still apply the gate
  separately; fusion is not claimed for every attention invocation.
- **Programmatic dependent launch (PDL).** Projection, normalization, activation
  and attention-related kernels participate in producer/consumer launch chains.
  Consumers wait before dependent reads; producers publish after their output
  stores. This changes launch scheduling, not the model's dependency ordering.
- **GDN recurrence.** Direct recurrent updates, batched updates, speculative
  record generation and replay folding join the PDL chain, including waits and
  publication around FP32 recurrent-state accesses. This is not a new recurrence
  formula or a change to the persistent-state dtype.
- **Small-T attention and cache routes.** Tensor-core paths and route/split
  selection cover short query batches with BF16, INT8, FP8, NVFP4 and K8V4 KV;
  the existing HQ route also receives dispatch/gating changes. Selection depends
  on head geometry, query width, cache representation and visible-key envelope,
  rather than forcing one kernel across all decode and speculative workloads.
- **INT8 prompt split reduction.** The prompt route can partition the key range
  across CTAs, write FP32 partial outputs and softmax statistics, and merge them
  with a final reduction. Workspace planning accounts for those partials;
  causally empty partitions carry neutral statistics. This is runtime parallelism,
  distinct from the later build-speed feature's compilation split.

The implementation entry points are the [Q/K preparation contract](../../include/ninfer/ops/qk_norm_rope.h),
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
- These controls are process-fixed: set them before Engine construction and use
  a fresh process for each comparison so eager execution and graph capture agree.
  Hold artifact, prompt/decode lengths, KV format, speculation, CUDA Graph mode,
  warmup and repetitions constant; vary one control at a time for attribution.
- Op microbenchmarks characterize individual operators, not end-to-end speedups.
  Confirm request-phase or complete-inference claims through the public Engine;
  use the guide's measured-only profiling boundary when launch attribution is
  needed. Disabling PDL/fusions does not undo every attention-route change and is
  not a complete pre-feature baseline.

This guide describes implemented mechanisms, not a newly measured speedup. No
new GPU qualification or universal throughput/latency gain is asserted here.
