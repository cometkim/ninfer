---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - w4a4
  - dflash2
  - speculative-decoding
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.8-27B-nvfp4full-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 87.88
            name: Accuracy (3-round mean, thinking, rule)
        source:
          url: https://github.com/cometkim/ninfer
          name: NInfer EvalScope (fork validation)
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime_2026
        metrics:
          - type: accuracy
            value: 93.33
            name: Accuracy (3-round mean, thinking, rule)
        source:
          url: https://github.com/cometkim/ninfer
          name: NInfer EvalScope (fork validation)
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: LongBench v2
          type: longbench_v2
        metrics:
          - type: accuracy
            value: 67.41
            name: Accuracy (short subset, 3-round mean, full-capability, rule)
        source:
          url: https://github.com/cometkim/ninfer
          name: NInfer EvalScope (fork validation)
---

# Qwen3.8-27B fuller NVFP4 for NInfer

This model card is the version-controlled source for [cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer).

The repository contains a fuller-NVFP4 weight profile of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the native [NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format, with the [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) block-diffusion speculative drafter embedded in the same image. The artifact is intended only for NInfer engines with [cometkim/ninfer](https://github.com/cometkim/ninfer) patches; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Weight profile

This is a third weight profile for the existing `qwen3_8_27b` target.

Compared with the official [`nvfp4`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) profile it extends NVFP4 from the layers 0–55 MLP to nearly the whole Text backbone:
- every Text `mlp/gate_up` and `mlp/down` is NVFP4 (the layers 0–55 words are copied bit-exactly from [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); layers 56–63 are quantized locally from the official BF16 checkpoint);
- every GDN `query_key_value_z` is NVFP4, and GDN `output` is NVFP4 on 47 of 48 layers;
- full-attention `query_key_gate_value` is NVFP4 on the ten deepest layers and `attention/output` on 14/16, with 9 BF16 exception parents retaining the registered Qwen3.6-27B NVFP4 exception pattern;
- the token embedding and full output head use groupwise W8G32_F16S instead of row-scaled FP8.

This yields 247 NVFP4 parents with 247 site-level input divisors. Locally quantized parents use the documented encoder profile `NVFP4_MAXABS_DIVISOR_RNE_V1` (per-tensor FP32 divisor, per-16 E4M3FN block scales, RNE E2M1 codes) with site divisors calibrated by streaming the official BF16 checkpoint. The complete contract, encoder, calibration corpus, and validation evidence are [`docs/maintainer/qwen3.8-27b-artifact.md` §14 in the fork](https://github.com/cometkim/ninfer/blob/feat/qwen3.8-nvfp4full/docs/maintainer/qwen3.8-27b-artifact.md#14-fork-artifact-nvfp4full).

## DFlash2 speculative drafter

The artifact carries the fork's DFlash2 companion module — the 2B-parameter masked block-diffusion drafter of [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), five layers, sliding window 2048, selector rank 256 / top-16 — as 66 further objects, so `--spec dflash2` needs no second file.

Upstream's registered schema stores the module's matrices as W8G32_F16S. This image stores the 34 drafter matrices weight-only NVFP4 instead (norms and conv base kernels stay BF16), shrinking the module payload from 2,226,805,248 to 1,082,882,820 bytes (−1.07 GiB); Text weights are byte-identical between the two encodings.

On the conversion-verification workload the NVFP4 module drafted 5.50 tokens/round at 64.3% acceptance, against 5.75 at 67.9% for the `W8G32_F16S` encoding of the same drafter — the ≈1 GiB saving trades a few points of drafter acceptance.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4full.ninfer` |
| Size | 19,406,942,468 bytes (18.07 GiB) |
| SHA-256 | `abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4full` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,325 (1,319 tensors and 6 resources) |
| NVFP4 tensors | 281 (247 Text parents + 34 DFlash2 module matrices) |
| BF16 exception tensors | 9 |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af' \
  'qwen3_8_27b_nvfp4full.ninfer' | sha256sum --check
```

## Engine support

Upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) rejects this artifact as-is.

Some patches from [cometkim/ninfer](https://github.com/cometkim/ninfer) required to make it run on other NInfer engines, for example [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows).

**1. Weights-profile registration.** NInfer accepts an artifact only when its `model_id`/`weights_id` pair resolves in the target's compile-time registered set (`Package::resolve_weights` in `src/targets/qwen3_6_27b/impl/package.cpp`); upstream registers only `groupwise-int` and `nvfp4` for `qwen3.8-27b` and throws on every other identity.
**2. NVFP4-encoded DFlash2 module execution.** Upstream DFlash2 executes its companion module with `W8G32_F16S` matrices; this image's module is NVFP4-encoded (see above), which needs the NVFP4 execution routes.

The minimal patch set is therefore:

| layer | branch | carries |
|---|---|---|
| module execution | [`feat/dflash2`](https://github.com/cometkim/ninfer/tree/feat/dflash2) | NVFP4 DFlash2 module execution behind one binder contract, self-contained on the Windows build layer |
| profile registration | [`feat/qwen3.8-nvfp4full`](https://github.com/cometkim/ninfer/tree/feat/qwen3.8-nvfp4full) | the `nvfp4full` identity, converter/verifier, artifact contract, this card |

`feat/qwen3.8-nvfp4full` is stacked directly on `feat/dflash2`, which itself carries only the Windows/MSVC build layer besides upstream master. The sibling [`feat/qwen3.8-nvfp4qat`](https://github.com/cometkim/ninfer/tree/feat/qwen3.8-nvfp4qat) branch registers the QUASAR QAT profile the same way.

The long-context and KV-codec cells in this card additionally use two fork capabilities that are **not** part of the minimal patch set — the `hq-e8-2b` KV codec (~9× denser than INT8 group-64) and YaRN rope scaling — which ride the fork's `feat/hyperquant` and `feat/1m-context` branches or its `cometkim/dev` integration branch.

| capability | minimal patch set | full fork |
|---|---|---|
| Text / Vision / MTP / DFlash2, CLI and OpenAI/Anthropic serving | ✓ | ✓ |
| KV storage | bf16, int8, fp8, nvfp4, k8v4 | + hq-e8-2b |
| context envelope | native 262,144 | + YaRN 524,288 / 786,432 / 1,048,576 |

You can get full capability by using `cometkim/dev` branch that integrated all fork's own experiments.

## Run with NInfer

Any engine carrying the two patches above runs this artifact; the reference engine is the [cometkim/ninfer](https://github.com/cometkim/ninfer) fork (integration branch `cometkim/dev`). Windows (MSVC + CUDA 13.1+) or 64-bit Linux, NVIDIA GeForce RTX 5090 (`sm_120a`).

```bash
hf download cometkim/Qwen3.8-27B-nvfp4full-NInfer qwen3_8_27b_nvfp4full.ninfer \
  --local-dir models

# greedy text generation with MTP speculative decoding
./build-ninja/apps/ninfer.exe models/qwen3_8_27b_nvfp4full.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 --max-new 256 \
  --spec mtp --draft-tokens 3

# the DFlash2 drafter (single parallel draft pass; the recommended lane)
./build-ninja/apps/ninfer.exe models/qwen3_8_27b_nvfp4full.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 --max-new 256 \
  --spec dflash2 --draft-tokens 7

# OpenAI/Anthropic-compatible serving, full 262,144-token context on INT8 KV
./build-ninja/apps/ninfer-serve.exe models/qwen3_8_27b_nvfp4full.ninfer \
  --model-id qwen3.8-27b-nvfp4full --vision \
  --spec dflash2 --draft-tokens 7 \
  --host 0.0.0.0 --port 8080 --cors \
  --kv-dtype int8 --max-context 262144
```

## Size and quality

### Device weights

The artifact is 18.07 GiB on disk.

Device weights depend on the startup option, which is optional and fixed for the process lifetime.

The MTP lane adds 0.42 GiB and the DFlash2 lane 1.01 GiB. `--vision` adds the 333-object Vision tower (295,711,648 bytes = 0.275 GiB) identically in every lane; the KV cache (`--kv-dtype`) and workspace are allocated on top.

| Device weights | no Vision | with `--vision` |
|---|---:|---:|
| no speculation | 16.02 GiB | 16.30 GiB |
| `--spec mtp` | 16.44 GiB | 16.72 GiB |
| `--spec dflash2` | 17.03 GiB | 17.30 GiB |

KV pool bytes at the native 262,144-token capacity (per-token geometry is shared by every weights profile of this target):

| KV dtype | pool @ 262,144 tokens | per token |
|---|---:|---:|
| `bf16` | 16.00 GiB | 65,536 B |
| `int8` | 8.25 GiB | 33,792 B |
| `fp8` | 8.06 GiB | 33,024 B |
| `k8v4` | 6.28 GiB | 25,728 B |
| `nvfp4` | 4.50 GiB | 18,432 B |
| `hq-e8-2b` | 2.28 GiB | 9,352 B |

> **What `hq-e8-2b` is and why it exists.** [HyperQuant](https://arxiv.org/abs/2606.23406): A ~2.25-bit E8-lattice + Rice KV cache codec with a BF16 sink-plus-recent residual window, built into a separated [fork branch](https://github.com/cometkim/ninfer/tree/feat/hyperquant) (no vLLM/llama.cpp equivalent)
> It exists because of a hard memory constraint: on the 32 GB RTX 5090 the ≈16 GiB of device weights leave room for at most ~262k tokens of INT8 group-64 KV, so the 524k, 786k, 1M envelopes only fit at hq's ~9× smaller per-key footprint. 
> Quality is not the compromise the bit-width suggests: paired same-prompt campaigns measured hq-vs-INT8 parity at 390–400k contexts (McNemar p ≈ 0.49) with exact needle retrieval out to 592k tokens.

### Model baseline — INT8 group-64 KV, native 262,144 context

Measured on one NVIDIA GeForce RTX 5090 through the registered serving profile (thinking, 0-shot, rule scoring; the reasoning suites run MTP3 at the full output head on INT8 KV with the native 262,144-token context, temperature 0.6, top_p 0.95, top_k 20; three independent seed rounds).

LongBench v2 runs full-capability mode — thinking on under the same sampling and seed rounds, rule-scored on the final `ANSWER: [LETTER]` line with a 16,384-token output budget, under its RoPE profiles: short at native 262,144, medium at 524,288 (YaRN factor 2), long at 786,432 (YaRN factor 3).

| Benchmark | Score |
|---|---:|
| GPQA-Diamond (3-round mean) | 87.88 ± 2.62 |
| AIME 2026 (3-round mean) | 93.33 ± 3.34 |
| LongBench v2 short (3-round mean) | 67.41 ± 1.95 |

### Engine-specific `hq-e8-2b` cells — long-context envelope and KV-codec A/B

| Benchmark | KV cache | Context / RoPE | Score |
|---|---|---|---:|
| GPQA-Diamond (seed 42) | hq-e8-2b | native 262,144 | 89.90 |
| AIME 2026 (seed 42) | hq-e8-2b | native 262,144 | 96.67 |
| LongBench v2 short | hq-e8-2b | native 262,144 | 66.67 |
| LongBench v2 medium | hq-e8-2b | 524,288 · YaRN 2 | 59.07 ± 2.13 |
| LongBench v2 long | hq-e8-2b | 786,432 · YaRN 3 | 38.89 ± 2.45 |

The A/B rows are paired runs: identical prompts, seed, and envelope as the INT8 baseline rows above, with only the KV dtype changed, isolating the codec's effect per benchmark.

The LBv2 medium and long rows cannot pair as INT8 KV does not fit beside the weights at those envelopes.

## Reproduce

Conversion and bit-level verification (sources: the official BF16 checkpoint, the unsloth NVFP4 checkpoint, the z-lab DFlash2 BF16 drafter):

```bash
python3 -m tools.convert.qwen3_8_27b.calibrate_nvfp4full \
  --model /path/to/Qwen3.8-27B --quantized-model /path/to/Qwen3.8-27B-NVFP4 \
  --out out/qwen3_8_27b_nvfp4full_calibration.json
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4full \
  --model /path/to/Qwen3.8-27B --quantized-model /path/to/Qwen3.8-27B-NVFP4 \
  --calibration out/qwen3_8_27b_nvfp4full_calibration.json \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b_nvfp4full.ninfer
python3 -m tools.convert.qwen3_8_27b.verify_nvfp4full out/qwen3_8_27b_nvfp4full.ninfer \
  --model /path/to/Qwen3.8-27B --quantized-model /path/to/Qwen3.8-27B-NVFP4 \
  --calibration out/qwen3_8_27b_nvfp4full_calibration.json \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2
```

Evaluation — GPQA-Diamond + AIME26 on the INT8 baseline lane, the same suites on the `hq-e8-2b` KV lane (the codec A/B cells), and the LongBench v2 RoPE cells:

```bash
eval/run_card_quality.sh models/qwen3_8_27b_nvfp4full.ninfer nvfp4full int8 42 43 44
eval/run_card_quality.sh models/qwen3_8_27b_nvfp4full.ninfer nvfp4full hq-e8-2b 42 43 44
eval/run_card_lbv2.sh models/qwen3_8_27b_nvfp4full.ninfer nvfp4full 42 43 44
```

## Provenance

| Source | Revision | Role |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | every locally quantized parent, BF16 exceptions, direct tensors, W8 endpoints, MTP, Vision, frontend |
| [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | the 112 layers 0–55 MLP NVFP4 parents and their divisors, copied bit-exactly |
| [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | the embedded DFlash2 drafter module |

## Cites

```bibtex
@article{qwen38,
  title  = {Qwen3.8-27B},
  author = {Qwen Team},
  note   = {https://huggingface.co/Qwen/Qwen3.8-27B}
}
@misc{ninfer,
  title  = {NInfer: a from-scratch single-GPU inference engine},
  author = {Neroued and contributors},
  howpublished = {https://github.com/Neroued/ninfer}
}
@article{dflash2,
  title  = {DFlash2: block-diffusion speculative decoding},
  author = {z-lab},
  note   = {2B all-SWA drafter checkpoint, z-lab/Qwen3.8-27B-DFlash2}
}
@article{hyperquant,
  title   = {HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline for Large Language and Diffusion Models},
  author  = {Domb, Yuval and Sackstein, Hadar and Solberg, Tomer},
  journal = {arXiv 2606.23406},
  note    = {https://arxiv.org/abs/2606.23406}
}
```

## Limits

Multi-seed benchmark results under the stated profiles, not pass@k.

Quantization quality is gated against the official artifact on GPQA-Diamond; the nine BF16 exception layers reuse the Qwen3.6-27B NVFP4 exception set without per-weight-version tuning.
