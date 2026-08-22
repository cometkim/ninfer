# build-ninja-tk.ps1 - configure + build the standalone tools/test_kv suite with
# Ninja and the imported MSVC environment (mirrors configure-ninja.ps1; no engine).
param(
    [string]$BuildDir = "build-ninja"
)
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Repo $BuildDir }

function Import-Vcvars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] }
    }
    Write-Host "[tk] imported MSVC environment from $vcvars"
}

function Resolve-Ninja {
    $ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($ninja) { Write-Host "[tk] ninja: PATH ($($ninja.Source))"; return }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ninja.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) {
            $env:PATH = "$(Split-Path -Parent $c);$env:PATH"
            Write-Host "[tk] ninja: fallback ($c)"
            return
        }
    }
    Write-Error "ninja not found"; exit 2
}

Import-Vcvars
Resolve-Ninja
if (-not $env:CUDA_PATH) { $env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }

$cmake = 'C:\Program Files\CMake\bin\cmake.exe'
& $cmake -S $Repo -B $BuildPath -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmake --build $BuildPath -j
exit $LASTEXITCODE
