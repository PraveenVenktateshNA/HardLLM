@echo off
REM Step 4, extra model: hardllm-filtered. The first 5 models are NOT retrained
REM and their results are NOT changed: this adds one model next to them.
REM Installs nothing, needs no internet, writes only inside this folder.
REM
REM   0. code tests      about 1 min
REM   1. first results   the 4 trained models and 5 evaluations are present
REM   2. filter          scores the 90,155 Step 3 pairs, keeps 16,000, about 2 min
REM   3. training        hardllm-filtered only, about 35 min
REM   4. evaluation      hardllm-filtered only, the other 5 are read from out, about 3 min
REM   5. results check   prints the table with all 6 models
REM Safe to stop and re-run: finished stages are skipped, training resumes.

setlocal
cd /d "%~dp0"
set "PORT=%~dp0"
set "STEP4=%~dp0.."
set "PY=%PORT%win_python\python.exe"
set "LLAMA=%PORT%llama-3.2-3b"
set "MINILM=%PORT%minilm"

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
echo == 0/5 code tests
"%PY%" "%STEP4%\tests\test_step4.py"
if errorlevel 1 ( echo [error] code tests failed, nothing was run & pause & exit /b 1 )

echo.
echo == 1/5 checking the first results are here
for %%A in (base hardllm selected random private) do (
    if not exist "%STEP4%\out\eval_%%A.json" (
        echo [error] Step_4\out\eval_%%A.json is missing. Copy the evaluated out folder back first.
        pause & exit /b 1
    )
)
for %%A in (hardllm selected random private) do (
    if not exist "%STEP4%\models\lora-%%A\adapter_model.safetensors" (
        echo [error] Step_4\models\lora-%%A is missing. Copy the trained models back first.
        pause & exit /b 1
    )
)
echo   all found

echo.
echo == 2/5 filtering the Step 3 pairs
"%PY%" "%STEP4%\filter_hardllm.py" --embedder "%MINILM%"
if errorlevel 1 ( echo [error] filtering failed, see above & pause & exit /b 1 )

echo.
echo == 3/5 fine-tuning Llama-3.2-3B on hardllm-filtered
REM same memory setting the pre-flight picked for the first 4 models
"%PY%" "%STEP4%\train_lora.py" --arm hardllm-filtered --model "%LLAMA%" --micro_batch 2 --checkpointing 1
if errorlevel 1 ( echo [error] training failed, see above & pause & exit /b 1 )

echo.
echo == 4/5 evaluating hardllm-filtered
"%PY%" "%STEP4%\evaluate_models.py" --model "%LLAMA%" --with_extra
if errorlevel 1 ( echo [error] evaluation failed, see above & pause & exit /b 1 )

echo.
echo == 5/5 checking the results
"%PY%" "%STEP4%\tests\check_results_filtered.py"
if errorlevel 1 ( echo [error] the results failed their checks, see above & pause & exit /b 1 )

echo.
echo == done. Bring back Step_4\out\ from this folder.
echo    The new files: results_step4_filtered.md, demo_answers_filtered.md,
echo    eval_hardllm-filtered.json, train_hardllm-filtered.json, filter_hardllm.json,
echo    filter_hardllm_examples.txt. And Step_4\models\lora-hardllm-filtered\
pause
