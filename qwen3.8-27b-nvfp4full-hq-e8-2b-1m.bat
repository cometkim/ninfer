@echo off
setlocal
rem Quickstart: ninfer-serve, model id qwen3.8-27b-nvfp4full-hq-e8-2b-1m (EXPERIMENTAL:
rem 1,048,576 via YaRN factor 4, auto KV capacity - the M2 profile). Vision serving preset
rem WITHOUT speculation: at the full 1M pool, MTP3's graph tier + draft pool and DFlash2's
rem cyclic draft pools both fail to fit beside vision after the weights; text+DFlash2 at 1M
rem DOES fit and remains available by passing "--spec dflash2 --draft-tokens 7" (and dropping
rem --vision).
rem Quality past ~600k true tokens is not yet verified end-to-end; see HANDOFF "1M boundary".
rem Extra flags pass through after the preset (later duplicates override earlier ones).
set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full.ninfer"
if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-1m] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)
if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-1m] missing %WEIGHTS%
    exit /b 2
)
echo [qwen3.8-27b-nvfp4full-hq-e8-2b-1m] vision, NO MTP (memory at 1M), yarn:4 1048576 context, auto KV capacity, 0.0.0.0:8080 + webui
"%SERVER%" "%WEIGHTS%" --model-id qwen3.8-27b-nvfp4full-hq-e8-2b-1m --vision --host 0.0.0.0 --port 8080 --cors --preserve-thinking --webui --max-pending-requests 50 --pending-timeout-ms 3000000 --kv-dtype hq-e8-2b --max-context 1048576 --kv-capacity auto --rope-scaling yarn:4 %*
exit /b %ERRORLEVEL%
