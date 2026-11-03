@echo off
REM Step 4, extra model: hardllm-fullsample. Step 3 writes new synthetic pairs
REM with the SAME trained GPT-2, sampled without the top-k and top-p cut, then
REM Llama is trained on them. Nothing earlier is retrained or changed.
REM Installs nothing, needs no internet, writes only inside this folder.
REM
REM   0. code tests      about 2 min
REM   1. first results   the 5 trained models, 6 evaluations and GPT-2 are present
REM   2. Step 3 writing  40,000 pairs with GPT-2, about 25 min
REM   3. Step 3 cleaning same cleaner as before, seconds
REM   4. training set    16,000 pairs plus a vocabulary check, seconds
REM   5. training        hardllm-fullsample only, about 35 min
REM   6. evaluation      hardllm-fullsample only, about 3 min
REM   7. results check   prints the table with all 7 models
REM Safe to stop and re-run: writing and training resume, finished stages are skipped.

setlocal
cd /d "%~dp0"
set "PORT=%~dp0"
set "STEP4=%~dp0.."
set "STEP3=%~dp0..\..\Step_3"
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
echo == 0/7 code tests
"%PY%" "%STEP4%\tests\test_step4.py"
if errorlevel 1 ( echo [error] code tests failed, nothing was run & pause & exit /b 1 )

echo.
echo == 1/7 checking the earlier results are here
for %%A in (base hardllm selected random private hardllm-filtered) do (
    if not exist "%STEP4%\out\eval_%%A.json" (
        echo [error] Step_4\out\eval_%%A.json is missing. Copy the whole folder from the SSD.
        pause & exit /b 1
    )
)
for %%A in (hardllm selected random private hardllm-filtered) do (
    if not exist "%STEP4%\models\lora-%%A\adapter_model.safetensors" (
        echo [error] Step_4\models\lora-%%A is missing. Copy the whole folder from the SSD.
        pause & exit /b 1
    )
)
if not exist "%STEP3%\models\gpt2-qa\model.safetensors" (
    echo [error] Step_3\models\gpt2-qa is missing. Copy the whole folder from the SSD.
    pause & exit /b 1
)
echo   all found

echo.
echo == 2/7 Step 3: GPT-2 writes 40,000 pairs, no top-k or top-p cut
"%PY%" "%STEP3%\generate_synthetic.py" --qa --total 40000 --batch_size 128 --top_k 0 --top_p 1.0 --temperature 1.0 --output "%STEP3%\out\synthetic_qa_fullsample.jsonl"
if errorlevel 1 ( echo [error] writing failed, see above. Bring back Step_3\out & pause & exit /b 1 )

echo.
echo == 3/7 Step 3: cleaning, same rules as before
"%PY%" "%STEP3%\clean_synthetic_qa.py" --input "%STEP3%\out\synthetic_qa_fullsample.jsonl" --output "%STEP3%\out\synthetic_qa_fullsample_clean.jsonl" --report "%STEP3%\out\clean_qa_fullsample_report.json" --examples "%STEP3%\out\clean_qa_fullsample_examples.txt"
if errorlevel 1 ( echo [error] cleaning failed, see above & pause & exit /b 1 )

echo.
echo == 4/7 building the training set
"%PY%" "%STEP4%\build_fullsample_arm.py"
if errorlevel 1 ( echo [error] building the training set failed, see above & pause & exit /b 1 )

echo.
echo == 5/7 fine-tuning Llama-3.2-3B on hardllm-fullsample
REM same memory setting the pre-flight picked for the first 4 models
"%PY%" "%STEP4%\train_lora.py" --arm hardllm-fullsample --model "%LLAMA%" --micro_batch 2 --checkpointing 1
if errorlevel 1 ( echo [error] training failed, see above & pause & exit /b 1 )

echo.
echo == 6/7 evaluating hardllm-fullsample
"%PY%" "%STEP4%\evaluate_models.py" --model "%LLAMA%" --with_extra --suffix _fullsample
if errorlevel 1 ( echo [error] evaluation failed, see above & pause & exit /b 1 )

echo.
echo == 7/7 checking the results
"%PY%" "%STEP4%\tests\check_results_fullsample.py"
if errorlevel 1 ( echo [error] the results failed their checks, see above & pause & exit /b 1 )

echo.
echo == done. Bring back from this folder:
echo    Step_4\out\        and   Step_4\models\lora-hardllm-fullsample\
echo    Step_3\out\
pause
