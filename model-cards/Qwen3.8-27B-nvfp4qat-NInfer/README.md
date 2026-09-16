---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - w4a4
  - qat
  - quasar
  - dflash2
  - speculative-decoding
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
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime_2026
        metrics:
          - type: accuracy
            value: 91.11
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
            value: 66.30
            name: Accuracy (short subset, 3-round mean, full-capability, rule)
        source:
          url: https://github.com/cometkim/ninfer
          name: NInfer EvalScope (fork validation)
---

# Qwen3.8-27B QUASAR QAT NVFP4 for NInfer

This model card is the version-controlled source for [cometkim/Qwen3.8-27B-nvfp4qat-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4qat-NInfer).

The repository contains a QAT-sourced NVFP4 weight profile of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the native [NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format, with the [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) block-diffusion speculative drafter embedded in the same image. It runs on NInfer **v3** engines — Text and MTP components through format/shape routes the upstream runtime already implements, the optional NVFP4-encoded DFlash2 companion through the fork's runtime support. It is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Weight profile

This is a fourth weight profile for the existing `qwen3_8_27b` target — a peer of the official `groupwise-int` and `nvfp4` profiles and of the fork's fuller-requant [`nvfp4full`](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer) profile.

The Text weight stack is copied **word-for-word** from [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) — a quantization-aware-trained checkpoint (QUASAR loss-aware NVFP4 distillation against the frozen BF16 teacher, arXiv 2608.13966):
- every one of the 496 text linear layers is NVFP4 (W4A4) — attention, gated-delta-net, and MLP alike, with no high-precision exceptions;
- the QAT factory quantizes per site, so every constituent tensor of a fused parent shares one weight and input global scale; the converter enforces that sharing before copying a word, and the site input divisors ride the artifact directly;
- the GDN control projections are decoded from their QAT NVFP4 words to the BF16 control parent the engine consumes.

Everything else comes from the official BF16 base exactly as the fork's `nvfp4full` profile builds it: the W8 embedding and output head, the optimized draft head, MTP, Vision, and the frontend. The converter proves the routing complete by byte-comparing every unquantized QAT tensor against the official source (703 tensors, bit-identical). The complete contract is [`docs/maintainer/qwen3.8-27b-artifact.md` §15 in the fork](https://github.com/cometkim/ninfer/blob/91f8a447e3d53f764c30116d7df53c5b4e213511/docs/maintainer/qwen3.8-27b-artifact.md#15-fork-artifact-nvfp4qat).

## DFlash2 speculative drafter

The artifact carries the fork's DFlash2 companion module — the 2B-parameter masked block-diffusion drafter of [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), five layers, sliding window 2048, selector rank 256 / top-16 — as 66 further objects, so `--spec dflash2` needs no second file.

Upstream's registered schema stores the module's matrices as W8G32_F16S. This image stores the 34 drafter matrices weight-only NVFP4 instead (norms and conv base kernels stay BF16), shrinking the module payload from 2,226,805,248 to 1,082,882,820 bytes (−1.07 GiB); Text weights are byte-identical between the two encodings.

On the conversion-verification workload the NVFP4 module drafted 2.50 tokens/round at 21.4% acceptance for this profile, against 5.50 tokens/round at 64.3% for `nvfp4full` in the same workload.

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

### v3 artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4qat.v3.ninfer` |
| Size | 18,638,510,576 bytes (17.35 GiB) |
| SHA-256 | `8b86901a8cd2a297a3d737e470c793b67e5ce65b49131c48c2f2f0b346fd943c` |
| Container version | 3 |
| Stored objects | 1,334 (1,328 tensors and six resources under the v3 schema) |
| Components | Text, Vision, MTP, DFlash2 (NVFP4-encoded module), optimized draft head |

This file is the v2 release upgraded offline with the standard `tools/upgrade_ninfer_v2_to_v3.py`: weight bytes are preserved, the directory and bindings move to the v3 schema, and the maintained chat template is installed. The 290 NVFP4 tensors — 256 Text parents and the 34 drafter matrices — are unchanged. Verified on one RTX 5090 through the fork engine (Windows, INT8 group-64 KV, 8,192-token context): greedy text, MTP3, DFlash2 draft-window 7 and Vision image input all execute end to end. The v2 evaluation tables above continue to describe these byte-preserved weights; acceptance observed on a 128-token greedy smoke is not a new quality claim.

## Engine support

NInfer v3 selects execution from the artifact's configuration, bindings, weights and frontend resources; there is no compile-time `(model_id, weights_id)` registration to patch. The Text and MTP components of this profile use format/shape combinations the upstream v3 runtime already implements — NVFP4 parent matrices with `AllowA4` activation policies and Q8 vocabulary endpoints — so a converted v3 artifact runs on upstream-based v3 builds.

The optional companion differs by encoding: a W8 DFlash2 module converted from the BF16 drafter uses the upstream schema, while an NVFP4-encoded module like the released v2 image needs the fork's NVFP4 DFlash2 execution support.

| capability | branch | carries |
|---|---|---|
| this profile's recipe | [`feat/qwen3.8-nvfp4qat`](https://github.com/cometkim/ninfer/tree/feat/qwen3.8-nvfp4qat) | the v3 recipe, its QAT source imports, tests and this card, over [`feat/qwen3.8-profile-base`](https://github.com/cometkim/ninfer/tree/feat/qwen3.8-profile-base) |
| NVFP4 DFlash2 module execution | [`feat/nvfp4-dflash2`](https://github.com/cometkim/ninfer/tree/feat/nvfp4-dflash2) | execution, binding and Op support for NVFP4-encoded companion modules |

Full and QAT are sibling branches over the shared base helper; neither depends on DFlash execution, the Windows port, or the other.

The long-context and KV-codec cells in this card additionally use two fork capabilities — the `hq-e8-2b` KV codec (~9× denser than INT8 group-64) and YaRN rope scaling — which ride the fork's `feat/hyperquant` and `feat/1m-context` branches or its `cometkim/dev` integration branch.

| capability | v3 base recipe | full fork |
|---|---|---|
| Text / Vision / MTP / DFlash2, CLI and OpenAI/Anthropic serving | ✓ | ✓ |
| KV storage | bf16, int8, fp8, nvfp4, k8v4 | + hq-e8-2b |
| context envelope | native 262,144 | + YaRN 524,288 / 786,432 / 1,048,576 |

You can get full capability by using `cometkim/dev` branch that integrated all fork's own experiments.

## Run with NInfer

Current NInfer builds accept **v3** `.ninfer` artifacts; download the v3 file below, or upgrade an existing v2 copy offline with `tools/upgrade_ninfer_v2_to_v3.py`. Windows (MSVC + CUDA 13.1+) or 64-bit Linux, NVIDIA GeForce RTX 5090 (`sm_120a`).

```bash
hf download cometkim/Qwen3.8-27B-nvfp4qat-NInfer qwen3_8_27b_nvfp4qat.v3.ninfer \
  --local-dir models
# or upgrade an existing v2 copy offline:
# python3 tools/upgrade_ninfer_v2_to_v3.py \
#   models/qwen3_8_27b_nvfp4qat.ninfer models/qwen3_8_27b_nvfp4qat.v3.ninfer

# greedy text generation with MTP speculative decoding
./build/apps/ninfer models/qwen3_8_27b_nvfp4qat.v3.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 --max-new 256 \
  --spec mtp --draft-tokens 3

# the DFlash2 drafter (single parallel draft pass; the recommended lane).
# The released NVFP4 module needs the fork's NVFP4 DFlash2 runtime;
# a freshly converted W8 companion runs on the upstream schema.
./build/apps/ninfer models/qwen3_8_27b_nvfp4qat.v3.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 --max-new 256 \
  --spec dflash2 --draft-tokens 7

# OpenAI/Anthropic-compatible serving, 262,144-token context on INT8 KV
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4qat.v3.ninfer \
  --model-id qwen3.8-27b-nvfp4qat --vision \
  --spec dflash2 --draft-tokens 7 \
  --kv-dtype int8 --max-context 262144
```

## Size and quality

### Device weights

The artifact is 17.35 GiB on disk.

Device weights depend on the startup option, which is optional and fixed for the process lifetime.

The MTP lane adds 0.42 GiB and the DFlash2 lane 1.01 GiB. `--vision` adds the 333-object Vision tower (295,711,648 bytes = 0.275 GiB) identically in every lane; the KV cache (`--kv-dtype`) and workspace are allocated on top.

| Device weights | no Vision | with `--vision` |
|---|---:|---:|
| no speculation | 15.31 GiB | 15.59 GiB |
| `--spec mtp` | 15.73 GiB | 16.01 GiB |
| `--spec dflash2` | 16.32 GiB | 16.60 GiB |

KV pool bytes at the native 262,144-token capacity (per-token geometry is shared by every weights profile of this target):

| KV dtype | pool @ 262,144 tokens | per token |
|---|---:|---:|
| `bf16` | 16.00 GiB (derived) | 65,536 B |
| `int8` | 8.25 GiB | 33,792 B |
| `fp8` | 8.06 GiB | 33,024 B |
| `k8v4` | 6.28 GiB | 25,728 B |
| `nvfp4` | 4.50 GiB | 18,432 B |
| `hq-e8-2b` | 2.28 GiB | 9,352 B |

The bf16 row is derived from the exact 8.00 GiB pool measured at 131,072 tokens (65,536 B/token); a full 262,144-token bf16 pool does not fit beside the weights.

### Model baseline — INT8 group-64 KV, native 262,144 context

Measured on one NVIDIA GeForce RTX 5090 through the registered serving profile (thinking, 0-shot, rule scoring; the reasoning suites run MTP3 at the full output head on INT8 KV with the native 262,144-token context, temperature 0.6, top_p 0.95, top_k 20; three independent seed rounds).

LongBench v2 runs full-capability mode — thinking on under the same sampling and seed rounds, rule-scored on the final `ANSWER: [LETTER]` line with a 16,384-token output budget, under its RoPE profiles: short at native 262,144, medium at 524,288 (YaRN factor 2), long at 786,432 (YaRN factor 3).

| Benchmark | Score |
|---|---:|
| GPQA-Diamond (3-round mean) | 89.22 ± 2.49 |
| AIME 2026 (3-round mean) | 91.11 ± 3.85 |
| LongBench v2 short (3-round mean) | 66.30 ± 0.85 |

### Engine-specific `hq-e8-2b` cells — long-context envelope and KV-codec A/B

> **What `hq-e8-2b` is and why it exists.** [HyperQuant](https://arxiv.org/abs/2606.23406): A ~2.25-bit E8-lattice + Rice KV cache codec with a BF16 sink-plus-recent residual window, built into NInfer's fork (no vLLM/llama.cpp equivalent)
> It exists because of a hard memory constraint: on the 32 GB RTX 5090 the ≈15.3 GiB of device weights leave room for at most ~262k tokens of INT8 group-64 KV, so the 524k and 786k envelopes only fit at hq's ~9× smaller per-key footprint.
> Quality is not the compromise the bit-width suggests: paired same-prompt campaigns measured hq-vs-INT8 parity at 390–400k contexts (McNemar p ≈ 0.49) with exact needle retrieval out to 592k tokens.

| Benchmark | KV cache | Context / RoPE | Score |
|---|---|---|---:|
| GPQA-Diamond (seed 42) | hq-e8-2b | native 262,144 | 90.91 |
| AIME 2026 (seed 42) | hq-e8-2b | native 262,144 | 96.67 |
| LongBench v2 short | hq-e8-2b | native 262,144 | 66.67 |
| LongBench v2 medium | hq-e8-2b | 524,288 · YaRN 2 | 56.90 ± 2.56 |
| LongBench v2 long | hq-e8-2b | 786,432 · YaRN 3 | 39.50 ± 0.53 |

The A/B rows are paired runs: identical prompts, seed, and envelope as the INT8 baseline rows above, with only the KV dtype changed, isolating the codec's effect per benchmark.

The LBv2 medium and long rows cannot pair as INT8 KV does not fit beside the weights at those envelopes.

## Reproduce

Conversion uses the native v3 logical converter with this profile's recipe, `tools/convert/recipes/qwen3_8_27b_nvfp4qat.py`.

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --source quantized=/path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --recipe tools/convert/recipes/qwen3_8_27b_nvfp4qat.py \
  --components text,mtp \
  --out out/qwen3_8_27b_nvfp4qat.v3.ninfer
```

Sources are the official BF16 checkpoint and the QUASAR QAT checkpoint; packed
words and stored weight/input divisors are imported, so no calibration file is
needed (see [Weight profile](#weight-profile)).

`--components text,mtp,vision` adds Vision; append `dflash2` with `--source dflash2=/path/to/Qwen3.8-27B-DFlash2` for a W8 companion from the BF16 drafter. Allocation, import and encoder checks run with `python3 -m tests.convert.test_nvfp4qat_profile`.

Evaluation — GPQA-Diamond + AIME26 on the INT8 baseline lane, the same suites on the `hq-e8-2b` KV lane, and the LongBench v2 RoPE cells — ran through the fork's `eval/` configurations on the integration branch; those campaigns are the historical evidence for the release tables below, not part of conversion.

## Provenance

| Source | Revision | Role |
|---|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | every unquantized tensor (703 byte-compared, bit-identical), MTP, Vision, frontend, W8 endpoints |
| [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | `d8e6fbfa3e3a78899b440222b827430045a05b44` | all 256 NVFP4 text parents and their site divisors, copied word-for-word |
| [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | the embedded DFlash2 drafter module |

## Cites

```bibtex
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
@article{quasar,
  title   = {QUASAR: loss-aware quantization-aware training for NVFP4},
  author  = {QUASAR-QAT},
  journal = {arXiv 2608.13966},
  note    = {https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4}
}
@article{qwen38,
  title  = {Qwen3.8-27B},
  author = {Qwen Team},
  note   = {https://huggingface.co/Qwen/Qwen3.8-27B}
}
```

## Limits

Multi-seed benchmark results under the stated profiles, not pass@k.

The profile is a re-source of the QUASAR QAT checkpoint, not an independent QAT run; its quality ceiling is theirs. The GDN control parents are the BF16 materialization of the QAT NVFP4 words (one rounding).

The QUASAR organization is new (first published checkpoint August 2026); every quality claim here is re-measured locally rather than taken from their card.
