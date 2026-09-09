@echo off
setlocal
rem 786432-key hq envelope with YaRN factor 3, Vision and DFlash2 K7.
rem Post-rebase launch validation and measurements are recorded in HANDOFF.md.
rem True long-context model quality is separate from envelope startup/decode validation.
rem Host retention: 2 pinned state slots + 1 GiB KV on this 32 GiB Windows host.
rem Extra flags pass through after the preset (later duplicates override earlier ones).
set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full.ninfer"
set "WEBUI=--webui"
if exist "%ROOT%models\webui\index.html" set WEBUI=--webui-dir "%ROOT%models\webui"
if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-786k] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)
if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-786k] missing %WEIGHTS%
    exit /b 2
)
echo [qwen3.8-27b-nvfp4full-hq-e8-2b-786k] vision + DFlash2, yarn:3 786432 context, auto KV capacity, 0.0.0.0:8080 + webui
"%SERVER%" "%WEIGHTS%" --model-id qwen3.8-27b-nvfp4full-hq-e8-2b-786k --vision --spec dflash2 --draft-tokens 7 --host 0.0.0.0 --port 8080 --cors --preserve-thinking %WEBUI% --max-pending-requests 50 --pending-timeout-ms 3000000 --host-state-slots 2 --host-kv-mib 1024 --kv-dtype hq-e8-2b --max-context 786432 --kv-capacity auto --rope-scaling yarn:3 %*
exit /b %ERRORLEVEL%
