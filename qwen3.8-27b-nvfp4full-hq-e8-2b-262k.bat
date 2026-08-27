@echo off
setlocal
rem Quickstart: ninfer-serve, model id qwen3.8-27b-nvfp4full-hq-e8-2b-262k (the spec: native
rem 262,144 context, no scaling). Vision + MTP3 serving preset.
rem Extra flags pass through after the preset (later duplicates override earlier ones).
set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full.ninfer"
if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-262k] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)
if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-262k] missing %WEIGHTS%
    exit /b 2
)
echo [qwen3.8-27b-nvfp4full-hq-e8-2b-262k] vision + MTP3, native 262144 context, 0.0.0.0:8080 + webui
"%SERVER%" "%WEIGHTS%" --model-id qwen3.8-27b-nvfp4full-hq-e8-2b-262k --vision --spec mtp --draft-tokens 3 --lm-head-draft --host 0.0.0.0 --port 8080 --cors --preserve-thinking --webui --max-pending-requests 50 --pending-timeout-ms 3000000 --kv-dtype hq-e8-2b --max-context 262144 %*
exit /b %ERRORLEVEL%
