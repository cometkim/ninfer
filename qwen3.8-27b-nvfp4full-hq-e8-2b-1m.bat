@echo off
setlocal
rem 1048576-key hq envelope with YaRN factor 4, Vision and no speculation.
rem Banded prompt scratch bounds workspace; the old 3 tok/s unbanded-build cell is stale.
rem True long-context model quality is separate from envelope startup/decode validation.
rem Host retention: 2 pinned state slots + 1 GiB KV on this 32 GiB Windows host.
rem Extra flags pass through after the preset (later duplicates override earlier ones).
set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full.ninfer"
set "WEBUI=--webui"
if exist "%ROOT%models\webui\index.html" set WEBUI=--webui-dir "%ROOT%models\webui"
if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-1m] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)
if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-1m] missing %WEIGHTS%
    exit /b 2
)
echo [qwen3.8-27b-nvfp4full-hq-e8-2b-1m] vision, NO MTP (memory at 1M), yarn:4 1048576 context, auto KV capacity, 0.0.0.0:8080 + webui
"%SERVER%" "%WEIGHTS%" --model-id qwen3.8-27b-nvfp4full-hq-e8-2b-1m --vision --host 0.0.0.0 --port 8080 --cors --preserve-thinking %WEBUI% --max-pending-requests 50 --pending-timeout-ms 3000000 --host-state-slots 2 --host-kv-mib 1024 --kv-dtype hq-e8-2b --max-context 1048576 --kv-capacity auto --rope-scaling yarn:4 %*
exit /b %ERRORLEVEL%
