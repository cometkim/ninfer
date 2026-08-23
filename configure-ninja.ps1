# configure-ninja.ps1 - configure the fast Ninja build directory for NInfer.
#
#   powershell -ExecutionPolicy Bypass -File configure-ninja.ps1
#   powershell -ExecutionPolicy Bypass -File configure-ninja.ps1 -NoBenchmarks
#   powershell -ExecutionPolicy Bypass -File configure-ninja.ps1 -NoTests
#
# Runs in a plain PowerShell with no developer environment: the script imports
# the MSVC (vcvars64) environment itself, and resolves ninja and cmake from
# PATH, the VS-bundled copies, or winget's Links directory - whichever is
# available first. Tests and benchmarks are ON by default; -NoTests/
# -NoBenchmarks exclude them. VCPKG_ROOT is respected when set and otherwise
# derived from vcpkg on PATH; CUDA_PATH must already be set in the environment.

param(
    [string]$BuildDir = "build-ninja",
    [switch]$NoBenchmarks,
    [switch]$NoTests
)

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Repo $BuildDir }

function Import-Vcvars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { Write-Error "vswhere not found at $vswhere"; exit 2 }
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $install) { Write-Error "no VS instance with C++ tools found"; exit 2 }
    $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] }
    }
    Write-Host "[configure-ninja] imported MSVC environment from $vcvars"
}

function Resolve-Ninja {
    $ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($ninja) { Write-Host "[configure-ninja] ninja: PATH ($($ninja.Source))"; return $null }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ninja.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) {
            $dir = Split-Path -Parent $c
            $env:PATH = "$dir;$env:PATH"
            Write-Host "[configure-ninja] ninja: fallback ($c)"
            return $null
        }
    }
    Write-Error "ninja not found in PATH, VS, or winget Links"; exit 2
}

function Resolve-CMake {
    $cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmake) { Write-Host "[configure-ninja] cmake: PATH ($($cmake.Source))"; return $cmake.Source }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cmake.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) {
            Write-Host "[configure-ninja] cmake: fallback ($c)"
            return $c
        }
    }
    Write-Error "cmake not found in PATH, VS, or winget Links"; exit 2
}

Import-Vcvars
Resolve-Ninja

$CMake = Resolve-CMake

if (-not $env:VCPKG_ROOT) {
    $vcpkg = Get-Command vcpkg.exe -ErrorAction SilentlyContinue
    if ($vcpkg) { $env:VCPKG_ROOT = Split-Path -Parent $vcpkg.Source }
}
if (-not $env:VCPKG_ROOT) { Write-Error "VCPKG_ROOT not set and vcpkg not found in PATH"; exit 2 }
if (-not $env:CUDA_PATH) { Write-Error "CUDA_PATH not set in the environment"; exit 2 }

# Explicitly empty compiler launchers clear any launcher a previous configuration
# may have cached into the build directory.
$launcherArgs = @(
    '-DCMAKE_C_COMPILER_LAUNCHER=',
    '-DCMAKE_CXX_COMPILER_LAUNCHER=',
    '-DCMAKE_CUDA_COMPILER_LAUNCHER='
)

$bench = if ($NoBenchmarks) { "OFF" } else { "ON" }
$tests = if ($NoTests) { "OFF" } else { "ON" }

& $CMake -S $Repo -B $BuildPath -G Ninja `
    "-DNINFER_BUILD_BENCHMARKS=$bench" `
    "-DBUILD_TESTING=$tests" `
    "-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH/bin/nvcc.exe" `
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
    "-DVCPKG_TARGET_TRIPLET=x64-windows" `
    "-DCMAKE_BUILD_TYPE=Release" `
    @launcherArgs
exit $LASTEXITCODE
