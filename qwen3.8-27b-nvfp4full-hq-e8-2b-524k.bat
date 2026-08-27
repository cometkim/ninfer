@echo off
setlocal
rem Quickstart: ninfer-serve, model id qwen3.8-27b-nvfp4full-hq-e8-2b-524k (long context:
rem 524,288 via YaRN factor 2, auto KV capacity - the M1 profile). Vision + MTP3 serving preset.
rem Extra flags pass through after the preset (later duplicates override earlier ones).
set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full.ninfer"
if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-524k] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)
if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-524k] missing %WEIGHTS%
    exit /b 2
)
echo [qwen3.8-27b-nvfp4full-hq-e8-2b-524k] vision + MTP3, yarn:2 524288 context, auto KV capacity, 0.0.0.0:8080 + webui
"%SERVER%" "%WEIGHTS%" --model-id qwen3.8-27b-nvfp4full-hq-e8-2b-524k --vision --spec mtp --draft-tokens 3 --lm-head-draft --host 0.0.0.0 --port 8080 --cors --preserve-thinking --webui --max-pending-requests 50 --pending-timeout-ms 3000000 --kv-dtype hq-e8-2b --max-context 524288 --kv-capacity auto --rope-scaling yarn:2 %*
exit /b %ERRORLEVEL%
