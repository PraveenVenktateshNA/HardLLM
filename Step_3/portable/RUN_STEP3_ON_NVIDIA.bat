@echo off
REM Step 3 on a Windows PC with an NVIDIA GPU, straight from the project folder.
REM Uses the Python + libraries + GPT-2 bundled next to this file
REM (built on the Mac by build_windows_bundle.sh). Installs nothing, needs no
REM internet, and writes only inside the project folder.
REM
REM Question-answer mode (--qa). Five stages, each stops everything if it fails:
REM   0. code tests        about 1 min
REM   1. pre-flight        about 3 min, real memory and speed test on this GPU,
REM                        picks the generation batch size that fits
REM   2. training          GPT-2 on the 16,788 question-answer pairs Step 2 picked
REM   3. generation        100,000 synthetic question-answer pairs
REM   4. results check     tests the generated data and writes a report
REM Safe to stop and re-run: training and generation pick up where they left off.
REM The earlier question-only outputs (gpt2-generator, synthetic.jsonl) are not touched.

setlocal
cd /d "%~dp0"
set "PORT=%~dp0"
set "STEP3=%~dp0.."
set "PY=%PORT%win_python\python.exe"

REM keep every cache and temp file inside the project folder, never in C:\Users
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
    echo [error] %PY% not found. Run build_windows_bundle.sh on the Mac first.
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
"%PY%" "%STEP3%\tests\test_step3_qa.py"
if errorlevel 1 ( echo [error] code tests failed, nothing was trained & pause & exit /b 1 )

echo.
echo == 1/4 pre-flight: memory and speed test on this GPU
"%PY%" "%PORT%preflight_qa.py" --model "%PORT%gpt2"
if errorlevel 1 ( echo [error] pre-flight failed, the real run was not started & pause & exit /b 1 )
set /p GENBS=<"%STEP3%\out\gen_batch_qa.txt"

echo.
echo == 2/4 fine-tuning GPT-2 on question-answer pairs
"%PY%" "%STEP3%\train_generator.py" --qa --model "%PORT%gpt2" %*
if errorlevel 1 ( echo [error] training failed, see above & pause & exit /b 1 )

echo.
echo == 3/4 generating synthetic question-answer pairs
"%PY%" "%STEP3%\generate_synthetic.py" --qa --batch_size %GENBS%
if errorlevel 1 ( echo [error] generation failed, see above & pause & exit /b 1 )

echo.
echo == 4/4 checking the generated data
"%PY%" "%STEP3%\tests\check_results_qa.py"
if errorlevel 1 ( echo [error] the generated data failed its checks, see above & pause & exit /b 1 )

echo.
echo == done. Bring back these from the project folder:
echo    Step_3\models\gpt2-qa\     the trained model
echo    Step_3\out\                summaries, synthetic_qa.jsonl, results_qa.json, samples_qa.txt
pause
