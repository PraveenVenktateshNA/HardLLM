"""Step 5, part 2: turn the merged model into GGUF files, the format phone apps
(llama.cpp, PocketPal, ...) run.

    python Step_5/export_gguf.py [--arm selected]

  models/{arm}-f16.gguf     ~6.4 GB  16-bit, the exact model; only an input for the next two
  models/{arm}-q8_0.gguf    ~3.4 GB  8-bit
  models/{arm}-q4_k_m.gguf  ~2.0 GB  4-bit (llama.cpp's usual phone setting)

Uses llama.cpp b11552 from portable/llama.cpp/ (see VERSION.txt): the
converter through run_convert.py, then llama-quantize.exe. Each file is
written under a temporary name and renamed when complete, and skipped if it
already exists, so a stopped run resumes.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "portable", "llama.cpp", "bin")
QUANTS = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}


def gguf_path(arm, variant):
    return os.path.join(HERE, "models", f"{arm}-{variant}.gguf")


def exe(name):
    return os.path.join(BIN, name + (".exe" if os.name == "nt" else ""))


def run(cmd, target):
    if os.path.exists(target):
        print(f"[skip] {os.path.basename(target)} already exists")
        return
    tmp = target + ".partial"
    t0 = time.time()
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run([c.replace(target, tmp) for c in cmd], check=True)
    os.replace(tmp, target)
    print(f"wrote {os.path.basename(target)}: {os.path.getsize(target) / 2**30:.2f} GB, "
          f"{(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="selected")
    args = ap.parse_args()
    merged = os.path.join(HERE, "models", f"{args.arm}-merged")
    if not os.path.exists(os.path.join(merged, "config.json")):
        raise SystemExit(f"[stopped] {merged} is missing, run merge_model.py first")
    f16 = gguf_path(args.arm, "f16")
    run([sys.executable, os.path.join(HERE, "run_convert.py"), merged,
         "--outfile", f16, "--outtype", "f16"], f16)
    for variant, qtype in QUANTS.items():
        out = gguf_path(args.arm, variant)
        run([exe("llama-quantize"), f16, out, qtype], out)


if __name__ == "__main__":
    main()
