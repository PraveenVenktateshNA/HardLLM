@echo off
REM Step 5: the final model (Step 4 "selected", Step 2 data) made phone-ready.
REM Put this Step_5 folder next to the Step_4 folder that holds the bundle
REM (win_python, llama-3.2-3b, models\lora-selected). It trains nothing.
REM Installs nothing, needs no internet, writes only inside Step_5.
REM
REM   0. code tests    about 1 min, the real code on a tiny model
REM   1. merge         adapter folded into Llama, about 3 min
REM   2. export        GGUF 16-bit, then 8-bit and 4-bit, about 5 min
REM   3. score         same test as Step 4 for each version, about 10 min
REM   4. results check prints the table
REM Safe to stop and re-run: finished parts are skipped.

setlocal
cd /d "%~dp0"
set "PORT=%~dp0"
set "STEP5=%~dp0.."
set "STEP4=%~dp0..\..\Step_4"
set "PY=%STEP4%\portable\win_python\python.exe"
set "LLAMA=%STEP4%\portable\llama-3.2-3b"

REM keep every cache and temp file inside this folder, never in C:\Users
set "HF_HOME=%PORT%hf_home"
set "TMP=%PORT%tmp"
set "TEMP=%PORT%tmp"
if not exist "%TMP%" mkdir "%TMP%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

if not exist "%PY%" (
    echo [error] %PY% not found. Put Step_5 next to the Step_4 folder that has the bundle.
    pause & exit /b 1
)
for %%F in ("%LLAMA%\config.json" "%STEP4%\models\lora-selected\adapter_model.safetensors" "%STEP4%\out\eval_selected.json" "%STEP4%\out\eval_base.json" "%STEP4%\data\test_private.jsonl" "%PORT%llama.cpp\bin\llama-quantize.exe") do (
    if not exist %%F (
        echo [error] %%F is missing.
        pause & exit /b 1
    )
)

echo == checking the GPU and free disk space
"%PY%" -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO NVIDIA GPU FOUND'); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
    echo [error] PyTorch can't see the NVIDIA GPU. Update the NVIDIA driver and retry.
    pause & exit /b 1
)
"%PY%" -c "import shutil; g=shutil.disk_usage(r'%STEP5%').free/2**30; print(f'free disk: {g:.0f} GB (needs about 20)'); raise SystemExit(0 if g > 20 else 1)"
if errorlevel 1 ( echo [error] not enough free disk space & pause & exit /b 1 )

echo.
echo == 0/4 code tests
"%PY%" "%STEP5%\tests\test_step5.py"
if errorlevel 1 ( echo [error] code tests failed, nothing was exported & pause & exit /b 1 )

echo.
echo == 1/4 merging the selected adapter into Llama
"%PY%" "%STEP5%\merge_model.py" --model "%LLAMA%"
if errorlevel 1 ( echo [error] merge failed, see above & pause & exit /b 1 )

echo.
echo == 2/4 writing GGUF files (16-bit, 8-bit, 4-bit)
"%PY%" "%STEP5%\export_gguf.py"
if errorlevel 1 ( echo [error] export failed, see above & pause & exit /b 1 )

echo.
echo == 3/4 scoring every version on the Step 4 test
"%PY%" "%STEP5%\score_gguf.py"
if errorlevel 1 ( echo [error] scoring failed, see above & pause & exit /b 1 )

echo.
echo == 4/4 checking the results
"%PY%" "%STEP5%\tests\check_results_step5.py"
if errorlevel 1 ( echo [error] the results failed their checks, see above & pause & exit /b 1 )

echo.
echo == done. Bring back from Step_5:
echo    out\                          (results, about 1 MB)
echo    models\selected-q4_k_m.gguf   (the phone model, about 2 GB)
echo    models\selected-q8_0.gguf     (optional, about 3.4 GB)
pause
