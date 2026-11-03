"""Ask the final model questions by hand, on the Mac.

Type a patient question, the model writes the doctor's answer as it goes.
Runs the GGUF files from export_gguf.py with llama.cpp's Mac build
(portable/llama.cpp-mac/, release b11552, Apple GPU). Plain Python, no torch.

    python3 HardLLM/Step_5/ask_model.py                          # interactive
    python3 HardLLM/Step_5/ask_model.py -q "my knee hurts"       # one shot
    python3 HardLLM/Step_5/ask_model.py --version f16            # the full 16-bit model

--version q4 (default) is the 1.9 GB file the phone runs, q8 the 3.2 GB one,
f16 the 6 GB exact model. The prompt is the training format,
"Question: ...\\nAnswer:", and decoding is Step 4's demo setting (greedy,
repetition penalty 1.1, at most 256 tokens), so answers match
Step_5/out/demo_answers_step5.md. --temp above 0 samples instead.

Starts llama-server on a free local port and stops it on exit. Which
llama-server: $HARDLLM_LLAMA_SERVER if set, else the bundled Mac build on a
Mac, else the one on PATH (on an Android phone in Termux: pkg install
llama-cpp; see phone/README.md). $HARDLLM_THREADS sets its CPU threads.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "portable", "llama.cpp-mac", "llama-server")
VERSIONS = {"q4": "q4_k_m", "q8": "q8_0", "f16": "f16"}
MAX_TOKENS = 256


def prompt_text(question):
    # the same as Step_4/train_lora.prompt_text; copied so this needs no torch
    return f"Question: {' '.join(question.split())}\nAnswer:"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_server():
    """The llama-server binary to run (see the module notes)."""
    env = os.environ.get("HARDLLM_LLAMA_SERVER")
    if env:
        return env
    if sys.platform == "darwin" and os.path.exists(SERVER):
        return SERVER
    found = shutil.which("llama-server")
    if found:
        return found
    raise SystemExit("no llama-server found: on a phone run phone/setup_termux.sh, "
                     "or set HARDLLM_LLAMA_SERVER to its path")


def start_server(model, port, log_path):
    """llama-server with every layer on the GPU where there is one; returns once it answers."""
    log = open(log_path, "w")
    cmd = [find_server(), "-m", model, "--port", str(port), "--host", "127.0.0.1", "-ngl", "99", "-c", "2048"]
    if os.environ.get("HARDLLM_THREADS"):
        cmd += ["-t", os.environ["HARDLLM_THREADS"]]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(600):  # 5 min: a phone reads the 1.9 GB file far slower than a Mac
        if proc.poll() is not None:
            raise SystemExit(f"llama-server stopped while loading, see {log_path}")
        try:
            if json.load(urllib.request.urlopen(url, timeout=1)).get("status") == "ok":
                return proc
        except OSError:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit(f"llama-server did not start in 5 min, see {log_path}")


def ask(port, question, temp=0.0, out=sys.stdout):
    """Streams the answer to out; returns (answer, timings)."""
    body = {"prompt": prompt_text(question), "n_predict": MAX_TOKENS, "temperature": temp,
            "repeat_penalty": 1.1, "stop": ["\nQuestion:"], "stream": True, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    text, timings = "", {}
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            piece = chunk.get("content", "")
            if not text:
                piece = piece.lstrip()
            text += piece
            out.write(piece)
            out.flush()
            if chunk.get("stop"):
                timings = chunk.get("timings", {})
                break
    return text.strip(), timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", help="one question, then exit")
    ap.add_argument("--version", choices=list(VERSIONS), default="q4")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy, like Step 4")
    args = ap.parse_args()
    model = os.path.join(HERE, "models", f"selected-{VERSIONS[args.version]}.gguf")
    if not os.path.exists(model):
        raise SystemExit(f"{model} is missing (bring it back from the PC run, Step_5/models/)")

    os.makedirs(os.path.join(HERE, "portable", "tmp"), exist_ok=True)
    log = os.path.join(HERE, "portable", "tmp", "llama_server_mac.log")
    port = free_port()
    print(f"loading {os.path.basename(model)} ...", flush=True)
    server = start_server(model, port, log)
    def answer(q):
        print("Model: ", end="", flush=True)
        _, t = ask(port, q, args.temp)
        print(f"\n  [{t.get('predicted_n', 0)} tokens, {t.get('predicted_per_second', 0):.1f} tokens/s]")

    try:
        if args.question:
            answer(args.question)
        else:
            print("ready. Type a patient question (empty line or Ctrl-C to quit).")
            while q := input("\nYou: ").strip():
                answer(q)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
