#!/usr/bin/env bash
# Build a self-contained Windows bundle for running Step 3 on an NVIDIA PC.
#
# Run this ONCE on the Mac (needs internet, ~3.5 GB download). It fills
# Step_3/portable/ with:
#   win_python/   Windows Python 3.13 (embeddable zip) + every library Step 3
#                 needs, including PyTorch for CUDA 12.8, inside win_python/Lib
#   gpt2/         the GPT-2 base weights, so the PC never touches the internet
#
# Then copy the project to the drive and double-click RUN_STEP3_ON_NVIDIA.bat
# on the PC. Nothing is installed there and nothing is written outside the drive.
#
#     bash HardLLM/Step_3/portable/build_windows_bundle.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY_VER=3.13.14
TORCH="torch==2.11.0+cu128"   # cu128 builds run on every RTX 50-series driver
WIN="$HERE/win_python"
SITE="$WIN/Lib/site-packages"

command -v uv >/dev/null || { echo "needs uv: brew install uv"; exit 1; }

echo "== 1/3 Windows Python $PY_VER"
if [ ! -f "$WIN/python.exe" ]; then
    mkdir -p "$WIN"
    curl -fL --progress-bar -o "$HERE/py.zip" \
        "https://www.python.org/ftp/python/$PY_VER/python-$PY_VER-embed-amd64.zip"
    unzip -q -o "$HERE/py.zip" -d "$WIN"
    rm "$HERE/py.zip"
fi
# The embeddable Python ignores site-packages unless its ._pth file lists it.
PTH="$(ls "$WIN"/python3*._pth)"
printf 'python313.zip\n.\nLib\\site-packages\nimport site\n' > "$PTH"

echo "== 2/3 libraries for Windows (PyTorch CUDA build is ~3 GB)"
mkdir -p "$SITE"
uv pip install \
    --target "$SITE" \
    --python-platform x86_64-pc-windows-msvc \
    --python-version 3.13 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "$TORCH" \
    "transformers==5.16.1" "datasets==5.0.1" "accelerate==1.14.0" \
    "tokenizers==0.23.2" "safetensors==0.8.0" "huggingface_hub==1.30.0" \
    msvc-runtime

# msvc-runtime ships the Visual C++ DLLs PyTorch needs, so the PC doesn't have
# to have the redistributable installed. Put them next to python.exe.
find "$SITE" -maxdepth 3 -iname 'msvcp140*.dll' -o -maxdepth 3 -iname 'vcruntime140*.dll' \
    -o -maxdepth 3 -iname 'concrt140.dll' | while read -r dll; do
    cp -n "$dll" "$WIN/" 2>/dev/null || true
done

echo "== 3/3 GPT-2 base weights"
"$HERE/../../../.venv/bin/python" - "$HERE/gpt2" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("gpt2", local_dir=sys.argv[1], allow_patterns=[
    "config.json", "generation_config.json", "model.safetensors",
    "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json"])
EOF

du -sh "$WIN" "$HERE/gpt2"
echo "done. Copy the project to the drive and run RUN_STEP3_ON_NVIDIA.bat on the PC."
