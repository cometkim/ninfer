---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - w4a4
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
            value: 89.39
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/cometkim/ninfer/tree/feat/qwen3.8-nvfp4full
          name: NInfer EvalScope 1.9.0 (fork validation)
---

# Qwen3.8-27B fuller NVFP4 for NInfer

This model card is the version-controlled source for
[cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer). The repository contains a fuller-NVFP4 weight profile of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format, produced by the
[cometkim/ninfer](https://github.com/cometkim/ninfer) fork on its `feat/qwen3.8-nvfp4full` branch,
which is based on the [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows)
Windows-build fork. The artifact is intended
only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

This is a third weight profile for the existing `qwen3_8_27b` target — a peer of the official
`groupwise-int` and `nvfp4` profiles — not a separate model target. Compared with the official
[`nvfp4`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) profile it extends NVFP4 from
the layers 0–55 MLP to nearly the whole Text backbone, freeing about 3 GiB of device memory while
keeping NVFP4-class speed:

- every Text `mlp/gate_up` and `mlp/down` is NVFP4 (the layers 0–55 words are copied bit-exactly
  from [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); layers 56–63
  are quantized locally from the official BF16 checkpoint);
- every GDN `query_key_value_z` is NVFP4, and GDN `output` is NVFP4 on 47 of 48 layers;
- full-attention `query_key_gate_value` is NVFP4 on the ten deepest layers and `attention/output`
  on fourteen of sixteen, with nine BF16 exception parents retaining the registered Qwen3.6-27B
  NVFP4 exception pattern;
- the token embedding and full output head use groupwise `W8G32_F16S` instead of row-scaled FP8.

This yields 247 NVFP4 parents with 247 site-level input divisors. Locally quantized parents use the
documented encoder profile `NVFP4_MAXABS_DIVISOR_RNE_V1` (per-tensor FP32 divisor, per-16 E4M3FN
block scales, RNE E2M1 codes) with site divisors calibrated by streaming the official BF16
checkpoint. The complete contract, encoder, calibration corpus, and validation evidence are defined
in the fork's `docs/maintainer/qwen3.8-27b-artifact.md` §14.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4full.ninfer` |
| Size | 18,324,059,648 bytes (17.07 GiB) |
| SHA-256 | `2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4full` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,259 (1,253 tensors and 6 resources) |
| NVFP4 tensors | 247 |
| BF16 exception tensors | 9 |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7' \
  'qwen3_8_27b_nvfp4full.ninfer' | sha256sum --check
```

## Provenance

| Source | Revision | Role |
|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | every locally quantized parent, BF16 exceptions, direct tensors, W8 endpoints, MTP, Vision, frontend |
| `unsloth/Qwen3.8-27B-NVFP4` | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | the 112 layers 0–55 MLP NVFP4 parents and their divisors, copied bit-exactly |

The upstream pinned revision `60e813d4…` of the quantized source was force-pushed out of the
repository; the reachable `main` revision above carries the identical mixed-precision allocation,
which the converter validates structurally before copying a word.

## Requirements

- the [cometkim/ninfer](https://github.com/cometkim/ninfer/tree/feat/qwen3.8-nvfp4full) fork
  on the `feat/qwen3.8-nvfp4full` branch or later, which registers the `nvfp4full` weights
  profile on top of the natpate/ninfer-windows Windows build (upstream Neroued/ninfer does not
  yet know this identity);
- Windows (MSVC + CUDA 13.1+, vcpkg for FFmpeg/curl) or 64-bit Linux (WSL2 validated);
- NVIDIA GeForce RTX 5090 (`sm_120a`).

## Download and run

```bash
hf download cometkim/Qwen3.8-27B-nvfp4full-NInfer qwen3_8_27b_nvfp4full.ninfer \n  --local-dir models

./build-win/apps/Release/ninfer.exe models/qwen3_8_27b_nvfp4full.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3
```

## Supported use

Identical to the official `qwen3_8_27b` profiles: text generation in thinking and non-thinking
modes; image, multi-image, video, and mixed multimodal messages; MTP speculative decoding; BF16 and
INT8 group-64 KV cache; CUDA Graph decode and compatible-prefix reuse; bounded concurrent serving;
the NInfer CLI; and OpenAI/Anthropic-compatible HTTP serving.

## Measured results (RTX 5090)

Quality gate — GPQA-Diamond under the registered serving profile (thinking, MTP=3, INT8 KV,
262,144-token context; EvalScope 1.9.0, 0-shot, rule scoring, one sample, temperature 0.6,
seed 42):

| Benchmark | This artifact | Official `nvfp4` (published) |
|---|---:|---:|
| GPQA-Diamond | 89.39% (177 / 198) | 90.40% (179 / 198) |

Memory and speed at the same settings (INT8 group-64 KV, CUDA Graphs):

| Measurement | `nvfp4full` | official `nvfp4` |
|---|---:|---:|
| Artifact size | 17.07 GiB | 20.02 GiB |
| Device weights | 16.03 GiB | 18.98 GiB |
| Free after startup, 262,144-token INT8 KV | 4.91 GiB | 2.22 GiB |
| MTP3 decode tok/s (8,192-token context, greedy) | 134.7 | 118.9 |
| Prefill tok/s (same run) | 913 | 703 |

Full 262,144-token context now boots with INT8 KV and nearly 5 GiB to spare, where the official
profile's published evaluation ceiling was 252,928 tokens.

## Limits

Single-sample benchmark results under the stated profiles, not pass@k. Quantization quality is
gated against the official artifact on GPQA-Diamond; no broader independent quality study is
claimed. The nine BF16 exception layers reuse the Qwen3.6-27B NVFP4 exception set; per-weight-version
tuning of that set was not performed.
