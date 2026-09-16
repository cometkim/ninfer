# build-ninja.ps1 - build the Ninja build directory (default build-ninja).
#
#   powershell -ExecutionPolicy Bypass -File build-ninja.ps1
#   powershell -ExecutionPolicy Bypass -File build-ninja.ps1 -Target ninfer_bench
#
# Plain PowerShell, no developer environment needed: the script imports the
# MSVC (vcvars64) environment itself. Output streams straight to the console
# (never pipe builds through head/tail - SIGPIPE kills them and masks exit
# codes); the script exits with cmake's own exit code.

param(
    [string]$BuildDir = "build-ninja",
    [string]$Target = ""
)

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Repo $BuildDir }

if (-not (Test-Path (Join-Path $BuildPath "build.ninja"))) {
    Write-Error "no build.ninja under $BuildPath - run configure-ninja.ps1 first"; exit 2
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Write-Error "vswhere not found at $vswhere"; exit 2 }
$install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] }
}

$CMake = Get-Command cmake.exe -ErrorAction SilentlyContinue
if ($CMake) {
    $CMake = $CMake.Source
} else {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cmake.exe"
    )
    $CMake = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $CMake) { Write-Error "cmake not found in PATH, VS, or winget Links"; exit 2 }

$argList = @('--build', "`"$BuildPath`"", '-j')
if ($Target) { $argList += @('--target', $Target) }

& $CMake @argList
exit $LASTEXITCODE
