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
  - dflash2
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
---

# Qwen3.8-27B fuller NVFP4 for NInfer

This model card is the version-controlled source for
[cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer).
The repository contains a fuller-NVFP4 weight profile of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format, produced by the
[cometkim/ninfer](https://github.com/cometkim/ninfer) fork. The artifact is intended only
for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Source

The artifact is built by the `cometkim/ninfer` fork (the `feat/qwen3.8-nvfp4full` lineage,
carried on the fork's integration branch). The fork tracks
[Neroued/ninfer](https://github.com/Neroued/ninfer) upstream and adds this profile, the
DFlash2 speculative drafter, and Windows/MSVC support. The full conversion contract is
`docs/maintainer/qwen3.8-27b-artifact.md` §14 in the fork.

Module-format note (2026-09 rebase): the DFlash2 companion bundle exists in two encodings —
the upstream-schema build stores its matrices as `W8G32_F16S` (the registered profiles'
suffix contract), while the fork image measured throughout this card carries them as NVFP4
(weight-only drafter parents, ~0.7 GiB smaller module). Text weights are byte-identical
between the two; only DFlash2-lane cells are affected by the choice.

This is a third weight profile for the existing `qwen3_8_27b` target — a peer of the official
`groupwise-int` and `nvfp4` profiles — not a separate model target. Compared with the official
[`nvfp4`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) profile it extends NVFP4 from
the layers 0–55 MLP to nearly the whole Text backbone while carrying the fork's DFlash2
block-diffusion drafter in the same image:

- every Text `mlp/gate_up` and `mlp/down` is NVFP4 (the layers 0–55 words are copied bit-exactly
  from [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); layers 56–63
  are quantized locally from the official BF16 checkpoint);
- every GDN `query_key_value_z` is NVFP4, and GDN `output` is NVFP4 on 47 of 48 layers;
- full-attention `query_key_gate_value` is NVFP4 on the ten deepest layers and
  `attention/output` on fourteen of sixteen, with nine BF16 exception parents retaining the
  registered Qwen3.6-27B NVFP4 exception pattern;
- the token embedding and full output head use groupwise `W8G32_F16S` instead of row-scaled FP8;
- the DFlash2 drafter module (2B parameters, in-place masked block denoising) rides the same
  artifact as 66 further objects — matrices in weight-only NVFP4, norms and conv kernels BF16 —
  enabling `--spec dflash2` without a second file.

Locally quantized parents use the documented encoder profile `NVFP4_MAXABS_DIVISOR_RNE_V1`
with site divisors calibrated by streaming the official BF16 checkpoint.

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

## Run with NInfer

Requires the [cometkim/ninfer](https://github.com/cometkim/ninfer) fork, Windows (MSVC +
CUDA 13.1+) or 64-bit Linux, and an NVIDIA GeForce RTX 5090 (`sm_120a`).

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
  --host 0.0.0.0 --port 8080 --cors --webui \
  --kv-dtype int8 --max-context 262144
```

Supported use is identical to the official `qwen3_8_27b` profiles: text generation in thinking
and non-thinking modes; image, multi-image, video, and mixed multimodal messages; MTP and
DFlash2 speculative decoding; BF16 and INT8 group-64 KV cache plus the fork's hq-e8-2b KV with
YaRN rope scaling to 786,144 tokens; CUDA Graph decode and compatible-prefix reuse; bounded
concurrent serving; the NInfer CLI; and OpenAI/Anthropic-compatible HTTP serving.

## Quality and size

Measured on one NVIDIA GeForce RTX 5090 through the registered serving profile (thinking,
0-shot, rule scoring; the reasoning suites run MTP3 at the full output head on INT8 KV with the
native 262,144-token context, temperature 0.6, top_p 0.95, top_k 20; three independent seed
rounds). LongBench v2 runs full-capability mode — thinking on under the same sampling and seed
rounds, rule-scored on the final `ANSWER: [LETTER]` line with a 16,384-token output budget —
under its RoPE profiles: short at native 262,144 (INT8 KV), medium at 524,288 (YaRN factor 2,
hq KV), long at 786,144 (YaRN factor 3, hq KV). Terminal-Bench 2.1 runs the Terminus-2 agent
and SWE-Bench Pro the mini-swe-agent scaffold in full-capability mode against the
OpenAI-compatible endpoint on the DFlash2 lane. Agentic cells state their own round counts.

### Model baseline — INT8 group-64 KV, native 262,144 context

| Benchmark | Score |
|---|---:|
| GPQA-Diamond (3-round mean) | 87.88 ± 2.62 |
| AIME 2026 (3-round mean) | 93.33 ± 3.34 |
| LongBench v2 short | 67.41 ± 1.95 |
| Terminal-Bench 2.1 (1 round) | deferred to the next campaign session |
| SWE-Bench Pro (1 round) | deferred to the next campaign session |

These cells run the standard KV lane of this engine and represent the profile's model
quality without engine-specific codecs.

### Engine-specific `hq-e8-2b` cells — long-context envelope and KV-codec A/B

> **What `hq-e8-2b` is and why it exists.** A ~2.25-bit E8-lattice + Rice KV cache codec with
> a BF16 sink-plus-recent residual window, built into NInfer's fork (no vLLM/llama.cpp
> equivalent). It exists because of a hard memory constraint: on the 32 GB RTX 5090 the
> 16.03 GiB of device weights leave room for at most ~262k tokens of INT8 group-64 KV, so
> the 524k and 786k envelopes only fit at hq's ~9× smaller per-key footprint. Quality is not
> the compromise the bit-width suggests: paired same-prompt campaigns measured hq-vs-INT8
> parity at 390–400k contexts (McNemar p ≈ 0.49) with exact needle retrieval out to 592k
> tokens.

| Benchmark | KV cache | Context / RoPE | Score |
|---|---|---|---:|
| GPQA-Diamond (seed 42) | hq-e8-2b | native 262,144 | 89.90 |
| AIME 2026 (seed 42) | hq-e8-2b | native 262,144 | 96.67 |
| LongBench v2 short | hq-e8-2b | native 262,144 | 66.67 |
| LongBench v2 medium | hq-e8-2b | 524,288 · YaRN 2 | 59.07 ± 2.13 |
| LongBench v2 long | hq-e8-2b | 786,432 · YaRN 3 | 38.89 ± 2.45 |

The A/B rows are paired runs: identical prompts, seed, and envelope as the INT8 baseline rows
above, with only the KV dtype changed, isolating the codec's effect per benchmark. The medium
and long rows cannot pair — INT8 KV does not fit beside the weights at those envelopes.

Every cell is measured fresh on this box under the profile above; no number is carried over
from earlier sessions or from other artifacts' published results. Per-round values: GPQA-Diamond
89.39 / 84.85 / 89.39 and AIME 2026 96.67 / 90.00 / 93.33 over seeds 42/43/44; LongBench v2
short 69.44 / 67.22 / 65.56, medium 58.60 / 57.21 / 61.40, and long 36.11 / 39.81 / 40.74 over
three process rounds.

| Measurement | Value |
|---|---:|
| Artifact size | 18.07 GiB |
| Device weights | 16.03 GiB |
| Free after startup, 262,144-token INT8 KV | 5.10 GiB |
| MTP3 decode tok/s (8,192-token context, greedy) | 99.05 |
| DFlash2 decode tok/s (same cell) | 161.49 |
| Prefill tok/s (same run) | 5,895 |

## Reproduce

Conversion and bit-level verification (sources: the official BF16 checkpoint, the unsloth
NVFP4 checkpoint, the incoai DFlash2 BF16 drafter):

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

Evaluation: `eval/run_card_quality.sh models/qwen3_8_27b_nvfp4full.ninfer nvfp4full` runs the
three GPQA-Diamond + AIME26 seed rounds; `eval/run_card_lbv2.sh … 3` runs the LongBench v2
RoPE cells. The agentic suites run through their upstream harnesses against the engine's
OpenAI-compatible endpoint — Terminal-Bench 2.1 via Harbor with the Terminus-2 agent (the
model is an `openai/` LiteLLM name with `api_base` pointing at the server), SWE-Bench Pro via
the mini-swe-agent scaffold; both need Docker.

## Cites

```bibtex
@misc{ninfer,
  title  = {NInfer: a from-scratch single-GPU inference engine},
  author = {Neroued and contributors},
  howpublished = {https://github.com/Neroued/ninfer}
}
@article{dflash2,
  title  = {DFlash2: block-diffusion speculative decoding},
  author = {incoai, z-lab},
  note   = {2B all-SWA drafter checkpoint, incoai/Qwen3.8-27B-DFlash2}
}
@article{qwen38,
  title  = {Qwen3.8-27B},
  author = {Qwen Team},
  note   = {https://huggingface.co/Qwen/Qwen3.8-27B}
}
```

## Limits

Multi-seed benchmark results under the stated profiles, not pass@k. Quantization quality is
gated against the official artifact on GPQA-Diamond; the nine BF16 exception layers reuse the
Qwen3.6-27B NVFP4 exception set without per-weight-version tuning. Terminal-Bench 2.1 and
SWE-Bench Pro numbers depend on their agent scaffolds as much as on the model; they are
reported for this fork's registered lane, not as scaffold-independent capability.
