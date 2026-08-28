@echo off
setlocal
rem Quickstart: ninfer-serve, model id qwen3.8-27b-nvfp4full-int8-262k (the spec: native 262,144
rem context, no scaling; int8 KV - the accuracy-reference lane). Vision + DFlash2 serving preset.
rem Extra flags pass through after the preset (later duplicates override earlier ones).
set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full.ninfer"
if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-int8-262k] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)
if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-int8-262k] missing %WEIGHTS%
    exit /b 2
)
echo [qwen3.8-27b-nvfp4full-int8-262k] vision + DFlash2, int8 KV, native 262144 context, 0.0.0.0:8080 + webui
"%SERVER%" "%WEIGHTS%" --model-id qwen3.8-27b-nvfp4full-int8-262k --vision --spec dflash2 --draft-tokens 7 --host 0.0.0.0 --port 8080 --cors --preserve-thinking --webui --max-pending-requests 50 --pending-timeout-ms 3000000 --kv-dtype int8 --max-context 262144 %*
exit /b %ERRORLEVEL%
