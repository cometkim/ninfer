# NInfer — cometkim fork

## About this fork

This repository (`cometkim/ninfer`, branch `cometkim/dev`) is a personal integration workspace.
Work happens on stacked feature branches and is re-applied here; nothing from this branch is
pushed upstream directly.

| Remote | Repository | Role |
|---|---|---|
| `upstream` | `Neroued/ninfer` | the original engine; feat/* branches must stay PR-able against it |
| `natpate` | `natpate/ninfer-windows` | the Windows port this workspace started from (environment base) |
| `origin` | `cometkim/ninfer` | personal fork; all branches below are pushed here |

Branch layout:

- `master` tracks `origin/master` and follows `upstream/master` (rebase; force-push own fork only).
- `cometkim/dev` is the primary experimentation workspace AND the integration branch: the
  fork-base commit (Windows toolchain pins, local build wrapper, bench prompts, handoff
  notes) plus **one squashed commit per verified feature branch**, applied in stack order.
  Anything may be committed here at any time — no discipline required.
- `feat/*` are the curated feature branches containing **no machine-specific files** (no
  absolute toolchain paths, local prompts, or agent workspaces) so any of them can become an
  upstream PR at any time. They are rewritten freely
  (`git push --force-with-lease origin feat/...`). Feature branches stack on `master`
  directly or on each other; the historical `feat/windows-port` layer (from
  `natpate/ninfer-windows`) is folded into the fork base — natpate's line is tracked no
  further (only `feat/build-speed` still PRs there, kept mergeable against their master).

The experimentation cycle:

1. **Commit anything on `cometkim/dev`** — it is the sandbox.
2. **Extract / cherry-pick the meaningful pieces into `feat/*`**, keeping those branches
   upstream-PR-able.
3. **Squash verified feat branches back into `cometkim/dev` and dedup**: reset dev to the
   fork base, re-apply each feat branch as one squashed commit in stack order — this both
   integrates the verified state and drops the superseded experimental commits.
   Squashed feature commits carry the prefix `squash(feat/<branch>):` — they are re-applied
   snapshots, not primary history, and are the only commits on this branch that a future
   rebuild may drop or reorder freely (`git log --grep '^squash(feat/'`).
   Exception: a feat branch that upstream has **merged** is never squashed into dev — sync
   `master` to upstream, rebase the remaining stack onto it, and the merged branch's content
   arrives through the lineage; drop its squash from the rebuild.

Ongoing feature experimentation:

| feat branch | stacked on | status | squashed on dev as |
|---|---|---|---|
| `feat/windows-port` | `natpate/master` | webui + `meta.n_ctx` (already absorbed by natpate's dev; not PR-active) | — (dev's lineage base, never squashed) |
| `feat/build-speed` | `natpate/master` | per-dtype GQA launcher TU split — head of **natpate PR #2** (one commit) | `squash(feat/build-speed)` |
| `feat/qwen3.8-nvfp4full` | `upstream/master` | verified; upstream-PR candidate (0 behind upstream) | `squash(feat/qwen3.8-nvfp4full)` |
| `feat/hyperquant` | `feat/build-speed` | verified (TC tile-source decode, oracle-gated); natpate-lineage PR candidate | `squash(feat/hyperquant)` |
| `feat/1m-context` | `feat/hyperquant` | active — 1M envelope + YaRN + scratch banding + WI-8 residual window/dither (hq needle-retrieval clean through 592k true tokens); upstream-PR candidate | `squash(feat/1m-context)` |

When a row reaches *merged upstream*, remove its squash from the next dev rebuild and fold
its row into the upstream lineage note.

Fork-local on `cometkim/dev` only (never in a feat branch): the `/utf-8` source/execution
charset flags — a CP949-locale build workaround owned by this environment, not engine work.

```
git switch cometkim/dev
git reset --hard <base-commit>          # the chore(dev) fork base, before any squash
git merge --squash feat/build-speed     && git commit   # re-apply in stack order
git merge --squash feat/qwen3.8-nvfp4full && git commit
git merge --squash feat/hyperquant     && git commit
git merge --squash feat/1m-context     && git commit
```

After every re-apply, verify content parity — the integration tree must differ from the feature
tip only by fork-only files (environment and session docs) and content owned by sibling feature
branches or upstream commits the feature lineage predates:

```
git diff feat/dflash2 cometkim/dev --name-only      # expect fork files (.gitignore,
                                                    # Directory.Build.props, HANDOFF.md,
                                                    # ROADMAP*.md, build scripts,
                                                    # longprompt_*.json, README) plus
                                                    # sibling-branch files — never a
                                                    # feature-owned file
```

Machine-specific files live only in the environment commit (`Directory.Build.props` pins the
CUDA 13.3 toolkit path for MSBuild; `configure-ninja.ps1`/`build-ninja.ps1` are the Ninja
fast-path build wrappers and `build-live.ps1` the live-progress Visual Studio wrapper, with
machine-local fallbacks for `VCPKG_ROOT` and `CUDA_PATH`; `longprompt_*.json` are fixed-context
bench inputs; `HANDOFF.md` and `ROADMAP.md` track active work).

> Selected checkpoints. Maximum single-GPU inference performance.

This fork carries [Neroued/ninfer](https://github.com/Neroued/ninfer), a from-scratch C++/CUDA
inference engine for explicitly registered Qwen checkpoints on a single NVIDIA GeForce RTX 5090,
and runs text, image, and video prompts through a local CLI, OpenAI-/Anthropic-compatible HTTP
APIs, or the included llama.cpp webui. Development and measurement happen on Windows 11 x64
today (native builds via the port lineage); the Linux build path is unchanged from upstream and
is the intended next home for this environment.

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | Weights | NInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |
| [Qwen3.8-27B fuller NVFP4](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer) | `nvfp4full` | `qwen3_8_27b_nvfp4full.ninfer` | 18,324,059,648 bytes (17.07 GiB) | `2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7` |

Qwen3.6-27B and Qwen3.8-27B each expose two registered weight profiles. The version-2 artifact
identity selects the profile without a separate runtime flag; Qwen3.8 uses target key
`qwen3_8_27b` while sharing the 27B execution package. The Qwen3.6 `nvfp4` profile uses W4A4 Tensor
Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4` profile preserves its
source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and row-scaled FP8 for the token
embedding, attention input/output projections, GDN Q/K/V/Z and output projections, output head, and
remaining MLP weights. All four 27B artifacts retain the same Text, Vision, MTP, prefix-reuse, CLI,
and serving routes.

## Upstream lineage

NInfer is [Neroued](https://github.com/Neroued)'s project
([Neroued/ninfer](https://github.com/Neroued/ninfer)); that repository remains the reference
implementation, and `master` here follows `upstream/master` directly. This workspace builds on
[natpate/ninfer-windows](https://github.com/natpate/ninfer-windows), the native Windows port of
that engine — the port core below is inherited from it (the compatibility layer itself ported
from [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090), without its RTX 3090
retargeting). The WebUI and context-reporting items live on this fork's `feat/windows-port`
lineage — natpate's dev branch has already absorbed them, and [PR #2](https://github.com/natpate/ninfer-windows/pull/2)
delivers them to its master.

Windows-port lineage:

- **Native Windows 11 x64 build and run** — CMake with Visual Studio 2022 (MSVC), with
  [vcpkg](https://github.com/microsoft/vcpkg) resolving FFmpeg, libcurl, and zlib via the
  `vcpkg.json` manifest; the CUDA runtime is statically linked, so the CUDA Toolkit is only
  needed at build time.
- **Windows porting of the runtime** — memory-mapped artifact reading with unbuffered
  overlapped I/O (the Windows counterpart of POSIX `O_DIRECT`/`pread`, with the same 4096-byte
  alignment contract), portable console logging and load progress, and portable media
  acquisition for image and video input.
- **MSVC/TMA kernel compatibility** — fixes that let the upstream Blackwell kernels compile
  under MSVC: device-pointer NVFP4 TMA descriptors, the pair-row SwiGLU TMA epilogue, and
  MSVC move-construction details in the target runtime.
- **Stock llama.cpp WebUI** — the HTTP server additionally accepts the stock llama.cpp WebUI's
  API dialect, and `ninfer-serve` can serve the unmodified WebUI in-process: `--webui`
  downloads the latest build from the
  [ggml-org/llama-ui](https://huggingface.co/ggml-org/llama-ui) bucket on first start, or
  `--webui-dir DIR` serves an existing local copy.
- **Context window reporting** — `ninfer-serve` advertises the served context ceiling in the
  OpenAI dialect: the objects returned by `/v1/models` and `/v1/models/{id}` carry
  `meta.n_ctx` = the `--max-context` value in force.
- **Portable Windows release** — a self-contained zip containing the executables and all
  runtime DLLs; see [Prebuilt Windows release](#prebuilt-windows-release).

What this fork adds on top (one squashed commit per feature branch, per
[About this fork](#about-this-fork)):

- **`feat/qwen3.8-nvfp4full`** — the Qwen3.8-27B NVFP4 full-precision-requant artifact:
  unified activation scale space, conversion tooling, model card, and the published gpqa
  comparison.
- **`feat/hyperquant`** — the `hq-e8-2b` KV cache: E8-lattice + Rice entropy coding at
  2.25 bits/scalar (3.7× smaller payload than INT8 at 262k context), prompt attention over
  one-shot decoded rotated bf16 scratch, an 8-lane cooperative k=0 Rice decoder with a
  parallel unary packer, and a tensor-core tile-source decode kernel (the hq route is a
  KV-source policy of the bf16 TC kernel). After the residual-window + dither work below,
  hq decode matches or beats same-session INT8 at the measured cells (tg128 80.6 vs 71.7,
  pp32k+tg64 62.2 vs 58.1 tok/s).
- **`feat/1m-context`** — the 1M-context track: engine envelope raised to 1,048,576 keys on
  the U8 cache (banded rotated scratch, 1M + `--kv-capacity auto` resolves the full pool with
  ~3.3 GiB free), YaRN rope scaling (`--rope-scaling yarn:F[,t=][,bf=][,bs=]`, q-side
  temperature, FP64 angle reduction), the bf16/int8 linear envelope raised to 524,288 keys,
  and the WI-8 quality fix — a BF16 sink+recent residual window (S=32/W=512 exact side rows
  with per-slot ring validity) plus half-cell subtractive dither, which fixed the >262k hq
  garble: greedy needle retrieval is exact at 32k/304k/390k/592k true tokens (592k runs
  yarn:4 at 916–922 prefill / 15.2 decode tok/s; MTP3 at 390k commits 62.9 tok/s). Beyond
  ~592k the 1M cell still garbles (depth-independent; dense-YaRN-at-1M is the leading
  suspect); the 1M goal itself is dropped — 524k is the held envelope, 786k is
  engineering-ready pending a decision.
- **`feat/kernel-perf`** — the general kernel-performance track on the non-hq paths: f16-
  accumulate PV with an exact range guard and key-range splitting in the prompt attention
  kernels (pp65536 int8 +14.2%, hq +6.0%), programmatic dependent launch across the decode
  chain with the publish-at-end rule, a fused norm+control projection for the 27B decode
  route, the sigmoid gate folded into the attention op, a bit-exact fused q/k rmsnorm+rope
  kernel, and a per-request error boundary in the engine.
- **`feat/dflash2`** — the DFlash2 speculative backend: width-8 block verify on the hq route
  with MTP K≤7 plumbing, the 2B drafter embedded in the nvfp4full artifact (module matrices
  in NVFP4), four new draft ops (two-tap dynamic conv, selector scores/top-k/walk), and the
  full engine integration — one parallel draft forward replaces MTP's sequential drafts and
  commits +30% over MTP3 at 8k and +52% at the 258k native envelope at equal-or-better
  acceptance.

Everything else — the Linux build path, the RTX 5090 (`sm_120a`) target, the CUDA 13.1
requirement, and the NVFP4/W4A4 Blackwell execution paths — is unchanged from upstream.

## Performance

The published measurements cover the three Qwen3.6 artifact profiles and the Qwen3.8-27B NVFP4
profile. The Qwen3.8-27B `groupwise-int` profile is supported by current NInfer builds but is not
yet included in a published benchmark campaign.

### Concurrent MTP3 decode

Saturated decode was measured on an RTX 5090 with INT8 group-64 KV cache, CUDA Graphs, MTP3, and
one 8,192-token generation per active request. The values below are aggregate committed decode
throughput from complete one-second intervals in which the actual decode batch remained equal to
the configured concurrency. MTP acceptance is aggregated over the complete request wave. Each
concurrency cell reports `decode tok/s / MTP acceptance`; profiles should be read independently.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

At C=8, Qwen3.6-35B-A3B reaches **1,313.8 aggregate decode tok/s**. Qwen3.6-27B NVFP4 reaches
**1,146.9 tok/s** and **5.67×** its C=1 throughput. Qwen3.8-27B NVFP4 has **45.8–48.9%** MTP
acceptance, versus **67.2–71.4%** across the other measured profiles, so aggregate committed
throughput reflects both execution performance and speculative acceptance.

### Single-request serving

The single-request corpus was measured on the same GPU with INT8 group-64 KV cache, CUDA Graphs,
and a 1,024-token prefill chunk. Each reported fixture uses five fixed seeds after server warm-up.
Targets and weight profiles are reported independently rather than as cross-target comparisons.
Requests were submitted serially to a persistent server. The Qwen3.8-27B NVFP4 MTP0 results use the
same dedicated serial corpus runner as the Qwen3.6 profiles; its MTP3 results come from the C=1 point
of the fixed concurrent-corpus campaign documented in [Performance](docs/performance.md).

**Qwen3.6-35B-A3B**

- MTP0 at a 7,680-token prompt: **15,544.3 prefill tok/s** and **271.1 decode tok/s**.
- MTP0 at a 260,096-token prompt: **5,157.1 prefill tok/s** and **188.2 decode tok/s**.
- MTP3 long reasoning: **620.3–726.2 decode tok/s** with **72.7–82.8% acceptance**.
- MTP3 structured output: **770.9 decode tok/s**, **89.1% acceptance**, and **3.67 tokens/round**.

**Qwen3.6-27B (`groupwise-int`)**

- MTP0 at a 7,680-token prompt: **3,218.1 prefill tok/s** and **77.6 decode tok/s**.
- MTP0 at a 260,096-token prompt: **1,614.8 prefill tok/s** and **54.8 decode tok/s**.
- MTP3 long reasoning: **161.9–175.4 decode tok/s** with **73.4–78.8% acceptance**.
- MTP3 structured output: **193.0 decode tok/s**, **88.7% acceptance**, and **3.66 tokens/round**.

**Qwen3.6-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **11,191.5 prefill tok/s** and **86.4 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,510.6 prefill tok/s** and **59.9 decode tok/s**.
- MTP3 long reasoning: **213.1–231.0 decode tok/s** with **76.3–81.1% acceptance**.
- MTP3 structured output: **252.2 decode tok/s**, **89.8% acceptance**, and **3.69 tokens/round**.
- Against groupwise-int on the same corpus and runtime options: **3.48× the 7,680-token prefill
  throughput**, **1.55× the 260,096-token prefill throughput**, and **30–32% higher MTP3 decode
  throughput**.

**Qwen3.8-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **8,340.4 prefill tok/s** and **71.2 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,203.1 prefill tok/s** and **52.9 decode tok/s**.
- MTP3 long reasoning: **151.4–195.2 decode tok/s** with **56.2–76.0% acceptance**.
- MTP3 structured output: **219.8 decode tok/s**, **90.8% acceptance**, and **3.72 tokens/round**.

See [Performance](docs/performance.md) for the full methodology, variability, reproduction command,
and per-fixture results.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8-27B rows used
temperature 1.0 and presence penalty 0.0. The multimodal columns (ERQA and RealWorldQA) ran with
`--vision` at a 81,920-token context limit; the text columns used a 262,144-token limit except
Qwen3.8-27B NVFP4, which needs 252,928 to fit the RTX 5090 after weights.

These are single-sample results under that NInfer evaluation profile, not pass@k. See the model
cards and [full performance document](docs/performance.md) for correct/total counts and evaluation
notes.

## Requirements

NInfer currently requires:

- 64-bit Linux or Windows 11 x64;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler (GCC or Clang on Linux, MSVC from
  Visual Studio 2022 on Windows);
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- `pkg-config` on Linux, or [vcpkg](https://github.com/microsoft/vcpkg) on Windows (the
  repository pins the dependency baseline in `vcpkg.json`);
- Ninja, when using the commands below.

The build rejects CUDA architectures other than `120a`. On Linux, NInfer is run from its
source build tree; on Windows, the [prebuilt portable release](#prebuilt-windows-release)
provides the same binaries without a toolchain.

## Prebuilt Windows release

Windows users who would rather not build can use the portable release instead of the build
steps below. The zip is self-contained — executables, all runtime DLLs (FFmpeg,
libcurl, zlib, and the VC++ runtime; the CUDA runtime is statically linked), launcher scripts,
a `models\` folder, a `README.txt`, and `SHA256SUMS`:

1. Download the latest `ninfer-windows-<version>-win64-cuda131.zip` from
   [GitHub Releases](https://github.com/natpate/ninfer-windows/releases). Verify files against
   `SHA256SUMS`, e.g. `Get-FileHash ninfer-serve.exe -Algorithm SHA256`.
2. Extract it anywhere — the launcher scripts use relative paths and work from any location.
3. Download a model into `models\` as in [Download a model](#download-a-model).
4. Run the matching launcher, e.g. `.\qwen3_8_27b.bat`. This starts `ninfer-serve` on
   `http://127.0.0.1:8080` (API at `/v1`) and serves the WebUI at the root URL; `--webui`
   downloads the WebUI on first start, so the first run needs an internet connection (later
   runs reuse the local copy).
5. Or run `.\ninfer-serve.exe models\<model>.ninfer [flags]` directly — the options are
   identical to a source build (see [Run the HTTP server](#run-the-http-server)).

The launchers default to a 150,000-token context (`--max-context` / `--default-max-tokens`) to
leave VRAM headroom for the Windows desktop. On the 32 GB RTX 5090, the smaller models
(`qwen3_6_27b`, `qwen3_6_27b_nvfp4`, and `qwen3_8_27b`) can be safely raised to 200,000 when
VRAM is completely free at startup; the two larger models (`qwen3_8_27b_nvfp4` and
`qwen3_6_35b_a3b`) do not fit at 200,000 and should stay at 150,000. Hardware requirements are
unchanged: Windows 11 x64, RTX 5090, and an NVIDIA driver supporting CUDA 13.1.

## Build

### Linux

```bash
git clone https://github.com/cometkim/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

### Windows

The fork's build entry point is a pair of plain-PowerShell Ninja wrappers — no developer
prompt needed. `configure-ninja.ps1` (run once per build directory) imports the MSVC
(vcvars64) environment itself, resolves `cmake`/`ninja` from PATH with VS-bundled and
winget fallbacks, points the vcpkg toolchain at `VCPKG_ROOT` (falling back to this
machine's local checkout) and `nvcc` at `CUDA_PATH`, and enables tests and benchmarks by
default (`-NoTests` / `-NoBenchmarks` exclude them). `build-ninja.ps1` builds everything
or one target:

```powershell
git clone https://github.com/cometkim/ninfer.git
cd ninfer

powershell -ExecutionPolicy Bypass -File configure-ninja.ps1
powershell -ExecutionPolicy Bypass -File build-ninja.ps1
powershell -ExecutionPolicy Bypass -File build-ninja.ps1 -Target ninfer_bench
```

The default configuration builds:

```text
build-ninja/apps/ninfer.exe
build-ninja/apps/ninfer-serve.exe
```

with tests under `build-ninja/tests/` and benchmarks under `build-ninja/bench/`. The GQA
launcher TUs are split per dtype (and per geometry for the hq codec kernels), so a cold
build takes ~15 minutes on the RTX 5090 box versus ~50 minutes through the Visual Studio
generator, and touching a launcher dispatcher rebuilds in ~90 seconds. MSVC compiles with
`/utf-8` for source and execution charset, matching the Linux toolchain default.

The same configuration works through the Visual Studio generator without the wrappers —
Visual Studio 2022 (MSVC) and vcpkg; the manifest in the repository root pins `curl`,
`ffmpeg`, and `pkgconf`:

```powershell
cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-windows --config Release --parallel
```

building `build-windows/apps/Release/ninfer.exe` and
`build-windows/apps/Release/ninfer-serve.exe` (tests and benchmarks require
`-DBUILD_TESTING=ON` / `-DNINFER_BUILD_BENCHMARKS=ON`; the Ninja wrapper turns both on by
default). `build-live.ps1` wraps the Visual Studio flow with live progress output and
remains supported.

See [the Windows guide](docs/windows.md) for complete setup instructions, vcpkg installation, and
notes on the resulting DLL layout.

## Docker

Build the runtime image on a 64-bit Linux host with an RTX 5090, a CUDA 13.1-compatible NVIDIA
driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build --tag ninfer:local .
```

Download a model into `models/` as described below, then run the HTTP server:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_6_27b.ninfer \
  --host 0.0.0.0
```

Run the CLI from the same image:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer /models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 256
```

## Download a model

Use the Hugging Face CLI to download one of the registered artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# Or Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

Current NInfer builds accept only the version-2 artifact container, and all five downloads above
are version 2. Migration applies only to Qwen3.6 artifacts downloaded before their version-2
publication; both Qwen3.8-27B profiles were published directly as version 2. Migrate an older exact
local file in place:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with `qwen3_6_27b_nvfp4.ninfer` or `qwen3_6_35b_a3b.ninfer` for those
artifacts. The migration updates only container metadata; it does not rewrite the weight payload.
Alternatively, download the current version-2 file again from its Hugging Face repository.

Each `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

Each artifact is complete, while GPU residency is fixed at process startup. Speculative decoding is
disabled by default, so MTP/DFlash state and the optimized proposal head are not uploaded.
Vision is also disabled by default, so its weights, Vision scratch phase, and frozen
request-transient allocation are omitted. Add `--vision` to the CLI or server process that must
accept image or video input. Disabled capabilities cannot be enabled by a later request. DFlash is
available only for the 35B-A3B target and is text-only.

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Capabilities

All three registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen.

## Current limits

- Only the five `(model_id, weights_id)` artifact identities listed above are accepted product
  identities.
- Execution is specialized for one RTX 5090 and one CUDA device.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' per-dtype context envelope: 524,288 tokens with `bf16`/`int8` KV and 1,048,576
with `hq-e8-2b` (YaRN `--rope-scaling yarn:F` past the checkpoint's trained 262,144
positions). `--kv-capacity N` explicitly sizes the shared Main Text KV
  pool for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Windows](docs/windows.md)
- [CLI examples](examples/cli/)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
