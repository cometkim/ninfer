# build-live.ps1 - cmake build wrapper with live progress reporting.
#
#   powershell -ExecutionPolicy Bypass -File build-live.ps1
#   powershell -ExecutionPolicy Bypass -File build-live.ps1 -Target ninfer_bench
#   powershell -ExecutionPolicy Bypass -File build-live.ps1 -BuildDir tools/test_kv/build
#
# Live reporting, three channels:
#   * console: one line per completed project, errors immediately, and a
#     heartbeat line (elapsed | projects done/total | current file) that
#     updates every second even while a monster .cu compiles silently.
#   * status file: $Repo\build-live.status - one line, overwritten every
#     second: state=running|ok|failed elapsed=.. done=.. total=.. errors=..
#     current=.. - pollable from anywhere at any time.
#   * raw log: kept next to the status file on failure (build-live.log) so a
#     silent failure can always be autopsied.
#
# The cmd /c argument line is wrapped in an extra pair of quotes (the
# documented cmd pattern); without it cmd strips the first and last quote of
# the line and the mangled redirect makes the build die with an EMPTY log.
# Exit code is cmake's own exit code, never masked.

param(
    [string]$BuildDir = "build-windows",
    [string]$Config = "Release",
    [string]$Target = "",
    [switch]$Quiet
)

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Repo $BuildDir }
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

$StatusFile = Join-Path $Repo 'build-live.status'
$LogFile = Join-Path $Repo 'build-live.log'

$total = @(Get-ChildItem -Path $BuildPath -Recurse -Filter *.vcxproj -ErrorAction SilentlyContinue).Count
if ($total -eq 0) { Write-Host "[build-live] no .vcxproj under $BuildPath - wrong build dir?"; exit 2 }

$argList = @('--build', "`"$BuildPath`"", '--config', $Config, '-j')
if ($Target) { $argList += @('--target', $Target) }
$cmdLine = '"' + $CMake + '" ' + ($argList -join ' ') + ' > "' + $LogFile + '" 2>&1'

function Write-Status([string]$state, [int]$doneN, [int]$errsN, [string]$cur) {
    $line = "state=$state elapsed={0:mm\:ss} done=$doneN total=$total errors=$errsN current=$cur" -f $sw.Elapsed
    Set-Content -Path $StatusFile -Value $line -NoNewline
}

Write-Host "[build-live] $BuildPath ($total projects, config $Config$(if ($Target) { ", target $Target" }))"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = 'cmd.exe'
# Extra outer quotes: the documented fix so cmd parses the inner quoted
# executable + redirect correctly instead of stripping the first/last quote.
$psi.Arguments = '/c "' + $cmdLine + '"'
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
# Capture cmd's own complaints (parse errors) instead of losing them.
$psi.RedirectStandardError = $true
$proc = [System.Diagnostics.Process]::Start($psi)
$sw = [Diagnostics.Stopwatch]::StartNew()

$done = 0; $errs = 0; $cur = 'starting...'; $lastPaint = -1; $read = 0

try {
    while (-not $proc.HasExited) {
        if (Test-Path $LogFile) {
            $fresh = @(Get-Content $LogFile -ErrorAction SilentlyContinue | Select-Object -Skip $read)
            foreach ($line in $fresh) {
                if ($null -eq $line) { continue }
                $read++
                if ($line -match '\.vcxproj ->') {
                    $done++
                    $name = ($line -replace '^\s*', '' -split ' -> ')[0]
                    Write-Host ("[{0:mm\:ss}] done {1}" -f $sw.Elapsed, $name)
                } elseif ($line -match '(?i)\berror\b' -and $line -notmatch '(?i)warning') {
                    $errs++
                    Write-Host ("[{0:mm\:ss}] ERROR {1}" -f $sw.Elapsed, $line.Trim()) -ForegroundColor Red
                } elseif ($line -match '^\s+([\w.\-]+\.(cu|cpp|c|cxx|h|hpp|rc))\s*$') {
                    $cur = $Matches[1]
                }
            }
        }
        if ([int]$sw.Elapsed.TotalSeconds -ne $lastPaint) {
            $lastPaint = [int]$sw.Elapsed.TotalSeconds
            Write-Status 'running' $done $errs $cur
            if (-not $Quiet) {
                $tag = $cur; if ($tag.Length -gt 46) { $tag = $tag.Substring(0, 46) }
                Write-Host -NoNewline ("`r[{0,4:mm\:ss}] projects {1,2}/{2,2} | errors {3} | {4}" -f $sw.Elapsed, $done, $total, $errs, $tag.PadRight(46))
            }
        }
        Start-Sleep -Milliseconds 500
    }
    if (Test-Path $LogFile) {
        @(Get-Content $LogFile -ErrorAction SilentlyContinue | Select-Object -Skip $read) | ForEach-Object {
            if ($_ -match '\.vcxproj ->') { $done++ }
            elseif ($_ -match '(?i)\berror\b' -and $_ -notmatch '(?i)warning') {
                $errs++
                Write-Host ("ERROR {0}" -f $_.Trim()) -ForegroundColor Red
            }
        }
    }
} finally {
    if (-not $proc.HasExited) { $proc.Kill() }
}

$proc.WaitForExit()
$code = $proc.ExitCode
$cmdErr = $proc.StandardError.ReadToEnd()
if (-not $Quiet) { Write-Host "" }
if ($code -eq 0) {
    Write-Host ("[build-live] OK in {0:mm\:ss} - {1}/{2} projects" -f $sw.Elapsed, $done, $total) -ForegroundColor Green
    Write-Status 'ok' $done $errs ''
    Remove-Item $LogFile -ErrorAction SilentlyContinue
} else {
    Write-Host ("[build-live] FAILED (exit {0}) after {1:mm\:ss} - errors: {2}" -f $code, $sw.Elapsed, $errs) -ForegroundColor Red
    if ($cmdErr) { Write-Host "[build-live] cmd stderr: $cmdErr" -ForegroundColor Red }
    if ((Test-Path $LogFile) -and (Get-Item $LogFile).Length -gt 0) {
        Get-Content $LogFile | Select-Object -Last 15 | ForEach-Object { Write-Host "  $($_.Trim())" -ForegroundColor Yellow }
        Write-Host "[build-live] raw log kept at $LogFile"
    } else {
        Write-Host "[build-live] cmake produced NO output - environment or quoting failure" -ForegroundColor Red
    }
    Write-Status 'failed' $done $errs $cur
}
exit $code
