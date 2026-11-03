@echo off
REM Step 4 on a Windows PC with an NVIDIA GPU, straight from this folder.
REM Uses the Python + libraries + Llama-3.2-3B bundled next to this file.
REM Installs nothing, needs no internet, writes only inside this folder.
REM
REM Six stages, each stops everything if it fails:
REM   0. code tests      about 1 min, the real code on a tiny model
REM   1. pre-flight      about 8 min, picks the fastest memory setting on this GPU
REM   2. training        Llama-3.2-3B with LoRA, 4 times: hardllm, selected, random, private
REM   3. evaluation      base + the 4 fine-tuned models, same tests for all
REM   4. results check   everything is there and finite, prints the table
REM Safe to stop and re-run: finished models are skipped, training resumes.

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
echo == 0/4 code tests
"%PY%" "%STEP4%\tests\test_step4.py"
if errorlevel 1 ( echo [error] code tests failed, nothing was trained & pause & exit /b 1 )

echo.
echo == 1/4 pre-flight: memory and speed on this GPU
"%PY%" "%PORT%preflight_step4.py" --model "%LLAMA%"
if errorlevel 1 ( echo [error] pre-flight failed, the real run was not started & pause & exit /b 1 )
set /p MICRO=<"%STEP4%\out\train_micro.txt"
set /p CKPT=<"%STEP4%\out\train_ckpt.txt"

echo.
echo == 2/4 fine-tuning Llama-3.2-3B, 4 models
for %%A in (hardllm selected random private) do (
    echo.
    echo ---- training %%A
    "%PY%" "%STEP4%\train_lora.py" --arm %%A --model "%LLAMA%" --micro_batch %MICRO% --checkpointing %CKPT%
    if errorlevel 1 ( echo [error] training %%A failed, see above & pause & exit /b 1 )
)

echo.
echo == 3/4 evaluating base + 4 fine-tuned models
"%PY%" "%STEP4%\evaluate_models.py" --model "%LLAMA%"
if errorlevel 1 ( echo [error] evaluation failed, see above & pause & exit /b 1 )

echo.
echo == 4/4 checking the results
"%PY%" "%STEP4%\tests\check_results_step4.py"
if errorlevel 1 ( echo [error] the results failed their checks, see above & pause & exit /b 1 )

echo.
echo == done. Bring back these from this folder:
echo    Step_4\out\       results_step4.md, demo_answers.md, all the json files
echo    Step_4\models\    the 4 LoRA adapters, about 100 MB each
pause
