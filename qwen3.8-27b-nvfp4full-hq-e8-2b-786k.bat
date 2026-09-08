@echo off
setlocal
rem Quickstart: ninfer-serve, model id qwen3.8-27b-nvfp4full-hq-e8-2b-786k (long context:
rem 786,432 = exactly 3x native via YaRN factor 3, auto KV capacity). Vision + DFlash2 serving
rem preset; unlike the 1M profile, vision AND DFlash2 fit at this envelope (verified boot).
rem Quality status 2026-08-25: the envelope is engineering-verified (12/12 long docs x 2
rem cells, zero failures, after the 786432 arena-overflow and TMA-descriptor-live-lock
rem fixes), and yarn:3 measured EQUAL to yarn:4 on every MCQ instrument tried (GPQA short,
rem lbv2 long; both controls blind to the factor effect) - the quality read past ~600k true
rem tokens is the owner's qualitative call. See HANDOFF item 0 ("786k").
rem Extra flags pass through after the preset (later duplicates override earlier ones).
set "ROOT=%~dp0"
set "SERVER=%ROOT%build-ninja\apps\ninfer-serve.exe"
set "WEIGHTS=%ROOT%models\qwen3_8_27b_nvfp4full.ninfer"
if not exist "%SERVER%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-786k] missing %SERVER% - run configure-ninja.ps1 then build-ninja.ps1 first
    exit /b 2
)
if not exist "%WEIGHTS%" (
    echo [qwen3.8-27b-nvfp4full-hq-e8-2b-786k] missing %WEIGHTS%
    exit /b 2
)
echo [qwen3.8-27b-nvfp4full-hq-e8-2b-786k] vision + DFlash2, yarn:3 786432 context, auto KV capacity, 0.0.0.0:8080 + webui
"%SERVER%" "%WEIGHTS%" --model-id qwen3.8-27b-nvfp4full-hq-e8-2b-786k --vision --spec dflash2 --draft-tokens 7 --host 0.0.0.0 --port 8080 --cors --preserve-thinking --webui --max-pending-requests 50 --pending-timeout-ms 3000000 --kv-dtype hq-e8-2b --max-context 786432 --kv-capacity auto --rope-scaling yarn:3 %*
exit /b %ERRORLEVEL%
