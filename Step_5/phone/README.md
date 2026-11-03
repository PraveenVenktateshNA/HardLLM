# The demo on an Android phone, offline

The same page as on the Mac (Ask, Intake, Hospital queue, Results), with every
safety rule, running on the phone itself: Python runs `server.py`, llama.cpp
runs the 4-bit model on the phone's CPU. Once set up it works in airplane mode.

Needs: Android phone, about 3 GB free, 6 GB+ RAM (the OnePlus 11 is fine).

## 1. On the Mac: build the bundle

    bash HardLLM/Step_5/phone/build_phone_bundle.sh

Copy `HardLLM/Step_5/phone/dist/HardLLM_phone.zip` (1.9 GB) to the phone's
**Download** folder (USB cable in "File transfer" mode, or Google Drive).

## 2. On the phone: install Termux

Install **Termux from F-Droid** (f-droid.org) or its GitHub releases page.
The Play Store version is outdated and its packages may not install.

## 3. In Termux, once, with internet

    termux-setup-storage            # tap Allow: lets Termux read Download
    pkg install -y unzip
    unzip ~/storage/downloads/HardLLM_phone.zip -d ~
    bash ~/HardLLM_phone/HardLLM/Step_5/phone/setup_termux.sh

## 4. Every time: start the demo

    bash ~/HardLLM_phone/HardLLM/Step_5/phone/run_demo.sh

It loads the model (20-60 s) and opens `http://127.0.0.1:8080/#intake` in the
browser. Switch on airplane mode to show it needs no internet. Ctrl-C in
Termux stops it. Answers take longer than on the Mac (the phone uses its CPU).

Check it before showing it (with the demo running, in a second Termux
session: swipe from the left edge, "New session"):

    python ~/HardLLM_phone/HardLLM/Step_5/tests/check_demo_live.py

## If something goes wrong

- `no llama-server found`: run `setup_termux.sh` again with internet.
- Stops while loading: see `~/HardLLM_phone/HardLLM/Step_5/phone/demo.log`;
  usually not enough free RAM, so close other apps.
- Slow answers: try `HARDLLM_THREADS=4 bash .../run_demo.sh`.
- The page without the model: `bash .../run_demo.sh --mock`.
