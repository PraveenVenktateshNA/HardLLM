@echo off
REM Step 4, EVALUATION ONLY. Use this when the 4 models are already trained.
REM It never trains: if a trained model is missing it stops and says so.
REM Installs nothing, needs no internet, writes only inside this folder.
REM
REM   0. code tests      about 1 min, the real code on a tiny model
REM   1. models check    the 4 trained adapters are in Step_4\models\
REM   2. evaluation      base + the 4 fine-tuned models, about 20 to 40 min
REM   3. results check   everything is there and finite, prints the table
REM Safe to stop and re-run: models already evaluated are skipped.

setlocal
cd /d "%~dp0"
set "PORT=%~dp0"
set "STEP4=%~dp0.."
set "PY=%PORT%win_python\python.exe"
set "LLAMA=%PORT%llama-3.2-3b"

REM keep every cache and temp file inside this folder, never in C:\Users
set "HF_HOME=%PORT%hf_home"
set "HF_DATASETS_CACHE=%PORT%hf_home\datasets"
set "TMP=%PORT%tmp"
set "TEMP=%PORT%tmp"
if not exist "%TMP%" mkdir "%TMP%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

if not exist "%PY%" (
    echo [error] %PY% not found. The bundle is incomplete.
    pause & exit /b 1
)

echo == checking the GPU
"%PY%" -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO NVIDIA GPU FOUND'); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
    echo [error] PyTorch can't see the NVIDIA GPU. Update the NVIDIA driver and retry.
    pause & exit /b 1
)

echo.
echo == 0/3 code tests
"%PY%" "%STEP4%\tests\test_step4.py"
if errorlevel 1 ( echo [error] code tests failed, nothing was evaluated & pause & exit /b 1 )

echo.
echo == 1/3 checking the 4 trained models
for %%A in (hardllm selected random private) do (
    if not exist "%STEP4%\models\lora-%%A\adapter_model.safetensors" (
        echo [error] Step_4\models\lora-%%A is missing. This file does not train. Copy the trained models back first.
        pause & exit /b 1
    )
    if not exist "%STEP4%\out\train_%%A.json" (
        echo [error] Step_4\out\train_%%A.json is missing. Copy Step_4\out back as well.
        pause & exit /b 1
    )
    echo   lora-%%A found
)

echo.
echo == 2/3 evaluating base + 4 fine-tuned models
"%PY%" "%STEP4%\evaluate_models.py" --model "%LLAMA%"
if errorlevel 1 ( echo [error] evaluation failed, see above & pause & exit /b 1 )

echo.
echo == 3/3 checking the results
"%PY%" "%STEP4%\tests\check_results_step4.py"
if errorlevel 1 ( echo [error] the results failed their checks, see above & pause & exit /b 1 )

echo.
echo == done. Bring back Step_4\out\ from this folder:
echo    results_step4.md, demo_answers.md and all the json files
pause
