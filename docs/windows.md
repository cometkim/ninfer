# Building and running NInfer on Windows

This guide covers native Windows 11 x64 source builds of NInfer for an NVIDIA GeForce RTX 5090
(`sm_120a`). There is no prebuilt Windows distribution: the same `.ninfer` artifacts, CLI, and
HTTP server options apply as on Linux. See the [project README](../README.md), the
[CLI guide](cli.md), and [HTTP serving](serving.md) for model downloads, CLI options, and the
serving API.

## Requirements

- Windows 11 x64;
- NVIDIA GeForce RTX 5090 with a driver supporting CUDA 13.1;
- [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-downloads) or newer;
- Visual Studio 2022 with the **Desktop development with C++** workload;
- CMake 3.28 or newer;
- [vcpkg](https://github.com/microsoft/vcpkg); the repository pins the dependency baseline in
  `vcpkg.json`.

The build rejects CUDA architectures other than `120a`, matching the upstream RTX 5090 target.
On Windows, FFmpeg and libcurl come from vcpkg during configure; no system package installation
is required. CUDA 13.1 uses MSVC's conforming preprocessor automatically.

## Installing vcpkg

```powershell
git clone https://github.com/microsoft/vcpkg C:\src\vcpkg
C:\src\vcpkg\bootstrap-vcpkg.bat
```

With the vcpkg toolchain file passed to CMake (below), vcpkg installs `curl`, `ffmpeg` (with the
`zlib` feature), and `pkgconf` into the build directory's `vcpkg_installed/` tree, following the
`vcpkg.json` manifest. `vcpkg_installed/` is git-ignored.

## Building from source

From the **x64 Native Tools Command Prompt for VS 2022** (or any shell with the MSVC toolchain
and the CUDA toolkit in `PATH`):

```powershell
git clone https://github.com/natpate/ninfer-windows.git
cd ninfer-windows

cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-windows --config Release --parallel
```

The default configuration builds:

```text
build-windows/apps/Release/ninfer.exe
build-windows/apps/Release/ninfer-serve.exe
```

Tests, benchmarks, and maintainer tools are excluded from the default build, as on Linux.
The release binaries use the FFmpeg, libcurl, zlib, and Winsock DLLs from the
`build-windows/` vcpkg output tree; keep that tree next to the executables or copy the required
DLLs beside them. The CUDA runtime is linked statically, so no `cudart*.dll` is required from the
toolkit.

## Running the CLI

Download an artifact as described in the [project README](../README.md), then:

```powershell
.\build-windows\apps\Release\ninfer.exe models\qwen3_6_27b.ninfer `
  --prompt "Explain prefill and decode in three sentences." `
  --max-context 16384 `
  --max-new 256 `
  --spec mtp --draft-tokens 3 `
  --lm-head-draft
```

Answer content is written to stdout; loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr, exactly as on Linux.

## Running the HTTP server

```powershell
.\build-windows\apps\Release\ninfer-serve.exe models\qwen3_6_27b.ninfer `
  --max-context 16384 `
  --kv-capacity auto `
  --max-concurrency 2 `
  --spec mtp --draft-tokens 3 `
  --lm-head-draft
```

The API is then available at `http://127.0.0.1:8080/v1`. To listen on the network instead of
localhost, pass `--host 0.0.0.0` and allow TCP 8080 through Windows Firewall for the
`ninfer-serve.exe` process.

## Notes and differences from Linux

- Windows uses the Visual Studio generator in the examples above; Ninja Multi-Conf is also
  supported if installed. The build tree layout for multi-config generators is
  `build-windows/apps/Release/`.
- On Windows the CUDA runtime is linked statically (`CUDA::cudart_static`) and the project forces
  a single MSVC runtime library across the CUDA static runtime and the vcpkg dependencies.
- The artifact reader uses memory-mapped files plus unbuffered overlapped reads on Windows and
  `O_DIRECT`/`pread` on POSIX; the 4096-byte alignment contract is identical.
- `ninfer.exe` and `ninfer-serve.exe` are the only required outputs; the Docker path in the
  [project README](../README.md) remains Linux-only.
