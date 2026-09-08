---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - w4a4
  - qat
  - quasar
  - dflash2
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.8-27B-nvfp4qat-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 89.22
            name: Accuracy (3-round mean, thinking, rule)
        source:
          url: https://github.com/cometkim/ninfer
          name: NInfer EvalScope (fork validation)
---

# Qwen3.8-27B QUASAR QAT NVFP4 for NInfer

This model card is the version-controlled source for
[cometkim/Qwen3.8-27B-nvfp4qat-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer).
The repository contains a QAT-sourced NVFP4 weight profile of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format, produced by the
[cometkim/ninfer](https://github.com/cometkim/ninfer) fork. The artifact is intended only for
NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Source

The artifact is built by the `cometkim/ninfer` fork; the full conversion contract is
`docs/maintainer/qwen3.8-27b-artifact.md` §15 in the fork.

Module-format note (2026-09 rebase): the DFlash2 companion bundle exists in two encodings —
the upstream-schema build stores its matrices as `W8G32_F16S`, while the fork image measured
throughout this card carries them as NVFP4 (weight-only drafter parents, ~0.7 GiB smaller
module). Text weights are byte-identical between the two; only DFlash2-lane cells are
affected by the choice.

The Text weight stack is copied **word-for-word** from
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
— a quantization-aware-trained checkpoint (QUASAR loss-aware NVFP4 distillation against the
frozen BF16 teacher, arXiv 2608.13966) where **every one of the 496 text linear layers is
NVFP4** (W4A4): attention, gated-delta-net, and MLP alike, with no high-precision exceptions.
The QAT factory quantizes per site, so every constituent tensor of a fused parent shares one
weight and input global scale; the converter enforces that sharing before copying a word, and
the site input divisors ride the artifact directly. The GDN control projections are decoded
from their QAT NVFP4 words to the BF16 control parent the engine consumes.

Everything else comes from the official BF16 base exactly as the fork's
[`nvfp4full`](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer) profile builds it:
the W8 embedding and output head, the optimized draft head, MTP, Vision, and the frontend.
The converter proves the routing complete by byte-comparing every unquantized QAT tensor
against the official source (703 tensors, bit-identical). The DFlash2 drafter module rides
the same image, enabling `--spec dflash2` without a second file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4qat.ninfer` |
| Size | 18,638,209,796 bytes (17.35 GiB) |
| SHA-256 | `3bd37e032f1984250458ad6527d874913a96a26f9673537512c29726d3033e72` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4qat` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,343 (1,337 tensors and 6 resources) |
| NVFP4 tensors | 290 (256 Text parents + 34 DFlash2 module matrices) |
| BF16 exception tensors | 0 |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '3bd37e032f1984250458ad6527d874913a96a26f9673537512c29726d3033e72' \
  'qwen3_8_27b_nvfp4qat.ninfer' | sha256sum --check
```

## Run with NInfer

Requires the [cometkim/ninfer](https://github.com/cometkim/ninfer) fork, Windows (MSVC +
CUDA 13.1+) or 64-bit Linux, and an NVIDIA GeForce RTX 5090 (`sm_120a`).

```bash
hf download cometkim/Qwen3.8-27B-nvfp4qat-NInfer qwen3_8_27b_nvfp4qat.ninfer \
  --local-dir models

# greedy text generation with MTP speculative decoding
./build-ninja/apps/ninfer.exe models/qwen3_8_27b_nvfp4qat.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 --max-new 256 \
  --spec mtp --draft-tokens 3

# the DFlash2 drafter
./build-ninja/apps/ninfer.exe models/qwen3_8_27b_nvfp4qat.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 --max-new 256 \
  --spec dflash2 --draft-tokens 7

# OpenAI/Anthropic-compatible serving, full 262,144-token context on INT8 KV
./build-ninja/apps/ninfer-serve.exe models/qwen3_8_27b_nvfp4qat.ninfer \
  --model-id qwen3.8-27b-nvfp4qat --vision \
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
| GPQA-Diamond (3-round mean) | 89.22 ± 2.49 |
| AIME 2026 (3-round mean) | 91.11 ± 3.85 |
| LongBench v2 short | 66.30 ± 0.85 |
| Terminal-Bench 2.1 (1 round) | deferred to the next campaign session |
| SWE-Bench Pro (1 round) | deferred to the next campaign session |

These cells run the standard KV lane of this engine and represent the profiles' model
quality without engine-specific codecs.

### Engine-specific `hq-e8-2b` cells — long-context envelope and KV-codec A/B

> **What `hq-e8-2b` is and why it exists.** A ~2.25-bit E8-lattice + Rice KV cache codec with
> a BF16 sink-plus-recent residual window, built into NInfer's fork (no vLLM/llama.cpp
> equivalent). It exists because of a hard memory constraint: on the 32 GB RTX 5090 the
> 15.31 GiB of device weights leave room for at most ~262k tokens of INT8 group-64 KV, so
> the 524k and 786k envelopes only fit at hq's ~9× smaller per-key footprint. Quality is not
> the compromise the bit-width suggests: paired same-prompt campaigns measured hq-vs-INT8
> parity at 390–400k contexts (McNemar p ≈ 0.49) with exact needle retrieval out to 592k
> tokens.

| Benchmark | KV cache | Context / RoPE | Score |
|---|---|---|---:|
| GPQA-Diamond (seed 42) | hq-e8-2b | native 262,144 | 90.91 |
| AIME 2026 (seed 42) | hq-e8-2b | native 262,144 | 96.67 |
| LongBench v2 short | hq-e8-2b | native 262,144 | 66.67 |
| LongBench v2 medium | hq-e8-2b | 524,288 · YaRN 2 | 56.90 ± 2.56 |
| LongBench v2 long | hq-e8-2b | 786,432 · YaRN 3 | 39.50 ± 0.53 |

The A/B rows are paired runs: identical prompts, seed, and envelope as the INT8 baseline rows
above, with only the KV dtype changed, isolating the codec's effect per benchmark. The medium
and long rows cannot pair — INT8 KV does not fit beside the weights at those envelopes. Both
profiles use the identical codec at identical envelopes, so the cross-profile comparison stays
controlled.

Every cell is measured fresh on this box under the profile above, both artifacts in one
campaign; no number is carried over from earlier sessions or from published results of other
artifacts. Per-round values: GPQA-Diamond 90.91 / 86.36 / 90.40 and AIME 2026 93.33 / 93.33 /
86.67 over seeds 42/43/44; LongBench v2 short 66.11 / 65.56 / 67.22, medium 53.95 / 58.60 /
58.14, and long 39.81 / 38.89 / 39.81 over three process rounds.

| Measurement | Value |
|---|---:|
| Artifact size | 17.35 GiB |
| Device weights | 15.31 GiB |
| Free after startup, 262,144-token INT8 KV | 5.73 GiB |
| MTP3 decode tok/s (8,192-token context, greedy) | 96.27 |
| DFlash2 decode tok/s (same cell) | 119.12 |
| Prefill tok/s (same run) | 6,147 |

## Reproduce

Conversion and bit-level verification (sources: the official BF16 checkpoint, the QUASAR QAT
checkpoint, the incoai DFlash2 BF16 drafter):

```bash
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4qat \
  --model /path/to/Qwen3.8-27B \
  --quantized-model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out out/qwen3_8_27b_nvfp4qat.ninfer
python3 -m tools.convert.qwen3_8_27b.verify_nvfp4qat out/qwen3_8_27b_nvfp4qat.ninfer \
  --model /path/to/Qwen3.8-27B \
  --quantized-model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2
```

`verify_nvfp4qat` checks all 256 QAT payloads word-for-word, the 256 site divisors, the 48
decoded control parents against the independent decode oracle, the DFlash2 module against a
reference re-encode, and the weight-divisor derivation cross-check `d_w = binary32(2688/amax)`
(single-tensor site families match exactly).

Evaluation: `eval/run_card_quality.sh models/qwen3_8_27b_nvfp4qat.ninfer nvfp4qat` runs the
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
@article{quasar,
  title   = {QUASAR: loss-aware quantization-aware training for NVFP4},
  author  = {QUASAR-QAT},
  journal = {arXiv 2608.13966},
  note    = {https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4}
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

Multi-seed benchmark results under the stated profiles, not pass@k. The profile is a re-source
of the QUASAR QAT checkpoint, not an independent QAT run; its quality ceiling is theirs. The
GDN control parents are the BF16 materialization of the QAT NVFP4 words (one rounding). The
QUASAR organization is new (first published checkpoint August 2026); every quality claim here
is re-measured locally rather than taken from their card. Terminal-Bench 2.1 and SWE-Bench Pro
numbers depend on their agent scaffolds as much as on the model; they are reported for this
fork's registered lane, not as scaffold-independent capability.
