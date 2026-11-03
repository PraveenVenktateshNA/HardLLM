#!/data/data/com.termux/files/usr/bin/bash
# One-time setup on the phone, in Termux, WITH internet. About 5 min (20-30 if llama.cpp has to be built).
#
#     bash ~/HardLLM_phone/HardLLM/Step_5/phone/setup_termux.sh
#
# Installs Python and llama.cpp's llama-server. After this the demo needs no internet.
set -euo pipefail

pkg update -y
pkg install -y python llama-cpp || pkg install -y python

if ! command -v llama-server >/dev/null; then
  # the Termux package is missing or has no llama-server: build the release the Mac uses
  echo "building llama-server from source (one time, 20-30 min)..."
  pkg install -y git cmake clang
  rm -rf ~/llama.cpp-src
  git clone --depth 1 --branch b11552 https://github.com/ggml-org/llama.cpp ~/llama.cpp-src
  cmake -S ~/llama.cpp-src -B ~/llama.cpp-src/build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build ~/llama.cpp-src/build --target llama-server -j "$(nproc)"
  cp ~/llama.cpp-src/build/bin/llama-server "$PREFIX/bin/"
fi

echo
echo "llama-server: $(command -v llama-server)"
echo "python:       $(python --version)"
echo "setup done. Start the demo with:"
echo "    bash ~/HardLLM_phone/HardLLM/Step_5/phone/run_demo.sh"
