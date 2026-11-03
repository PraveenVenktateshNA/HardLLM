"""Pre-flight check for Step 3 --qa on the NVIDIA PC. Runs before the real job.

Takes ~2 minutes and answers, on the actual GPU, the questions the real run
would otherwise answer an hour in:

  1. input    the Step 2 file is there and holds the expected Q&A pairs
  2. generate one batch at the --qa settings, forced to the full 400 tokens
              (the worst case), each in a fresh process so nothing else is in
              GPU memory. Tries batch 256, then 128, then 64, and keeps the
              biggest that fits; the .bat passes it to the real run through
              Step_3/out/gen_batch_qa.txt.
  3. train    10 real optimizer steps with the exact --qa settings: does it fit
              in GPU memory, and how fast is it?

Prints a time estimate for the full run. Exits non-zero (and the .bat stops)
if anything fails or GPU memory peaks above 90%, so a doomed run never starts.
Writes Step_3/out/preflight_qa.json.

    python Step_3/portable/preflight_qa.py --model <path to base gpt2>
"""
import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STEP3 = os.path.dirname(HERE)
sys.path.insert(0, STEP3)
import generate_synthetic as gen  # noqa: E402
import train_generator as tg      # noqa: E402

EXPECTED_PAIRS = 16_788   # selected_centroid_eps0.5_t3.jsonl, train_generator.is_qa_pair
TRAIN_STEPS = 10
MEM_LIMIT = 0.90
GEN_BATCHES = (gen.QA_BATCH_CUDA, 128, 64)   # biggest first
BATCH_FILE = os.path.join(STEP3, "out", "gen_batch_qa.txt")


def gib(b):
    return b / 2**30


def gen_probe(model_path, bs):
    """Child process: one worst-case generation batch, on an otherwise empty GPU.
    Prints one line: PROBE {json}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    total = torch.cuda.get_device_properties(0).total_memory
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    start = tok("Question:")["input_ids"]
    prompt = torch.tensor([start] * bs, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        model.generate(prompt, attention_mask=torch.ones_like(prompt),
                       do_sample=True, max_new_tokens=gen.QA_LENGTH,
                       min_new_tokens=gen.QA_LENGTH,   # worst case: nobody stops early
                       temperature=1.0, top_k=50, top_p=0.9,
                       no_repeat_ngram_size=gen.NOREPEAT_QA,
                       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    print("PROBE " + json.dumps({
        "batch": bs, "seconds": round(time.time() - t0, 1),
        "peak_reserved_frac": round(torch.cuda.max_memory_reserved() / total, 3),
        "peak_allocated_frac": round(torch.cuda.max_memory_allocated() / total, 3)}))


def run_probe(model_path, bs):
    """Run gen_probe in a fresh Python process; None if it crashed."""
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", model_path,
                        "--gen_probe", str(bs)], stdout=subprocess.PIPE, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
    if r.returncode or not lines:
        return None
    return json.loads(lines[-1][len("PROBE "):])


def choose_batch(probe):
    """Biggest batch in GEN_BATCHES whose worst case fits under MEM_LIMIT.
    Returns (chosen probe result or None, every result tried)."""
    tried = []
    for bs in GEN_BATCHES:
        res = probe(bs) or {"batch": bs, "crashed": True}
        tried.append(res)
        if not res.get("crashed") and res["peak_reserved_frac"] <= MEM_LIMIT:
            return res, tried
    return None, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base GPT-2 folder")
    ap.add_argument("--gen_probe", type=int, help=argparse.SUPPRESS)  # internal
    args = ap.parse_args()
    if args.gen_probe:
        return gen_probe(args.model, args.gen_probe)
    report, failures = {}, []

    def fail(msg):
        failures.append(msg)
        print(f"  [FAIL] {msg}")

    # ------------------------------------------------------------------ GPU
    if not torch.cuda.is_available():
        sys.exit("[FAIL] no CUDA GPU visible to PyTorch")
    dev = torch.cuda.get_device_properties(0)
    total_mem = dev.total_memory
    print(f"GPU: {dev.name}, {gib(total_mem):.1f} GiB")
    report["gpu"] = {"name": dev.name, "gib": round(gib(total_mem), 1)}

    # ---------------------------------------------------------------- input
    print("\n1/3 input")
    if not os.path.exists(tg.DEFAULT_DATA):
        sys.exit(f"[FAIL] missing {tg.DEFAULT_DATA}\n"
                 "       Step_2 and Step_3 must sit side by side in the same folder.")
    rows = [json.loads(l) for l in open(tg.DEFAULT_DATA, encoding="utf-8")]
    pairs = [r for r in rows if tg.is_qa_pair(r)]
    print(f"  {len(rows):,} selected rows, {len(pairs):,} question-answer pairs")
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS:,} pairs, found {len(pairs):,}: wrong input file?")
    report["pairs"] = len(pairs)

    # ------------------------------------------------------------- generate
    # Each try runs in its own process, before training, so only generation's
    # own memory is measured (the first version counted training's leftovers).
    print(f"\n2/3 generation: one batch forced to the full {gen.QA_LENGTH} tokens, "
          f"fresh process each, batch {' -> '.join(map(str, GEN_BATCHES))} until it fits")
    if os.path.exists(BATCH_FILE):
        os.remove(BATCH_FILE)

    def probe(bs):
        res = run_probe(args.model, bs)
        if res is None:
            print(f"  batch {bs}: crashed (see above)")
        else:
            print(f"  batch {bs}: {res['seconds']:.1f} s, peak GPU memory "
                  f"{res['peak_reserved_frac']:.0%} (in use {res['peak_allocated_frac']:.0%})")
        return res

    chosen, tried = choose_batch(probe)
    report["generate"] = {"tokens": gen.QA_LENGTH, "tried": tried, "chosen": chosen}
    if chosen is None:
        fail(f"generation does not fit under {MEM_LIMIT:.0%} of GPU memory even at batch "
             f"{GEN_BATCHES[-1]}")
        bs, batch_s = GEN_BATCHES[-1], float("nan")
    else:
        bs, batch_s = chosen["batch"], chosen["seconds"]
        print(f"  using batch {bs} for the real run")

    # ---------------------------------------------------------------- train
    print(f"\n3/3 training: {TRAIN_STEPS} real steps at the --qa settings")
    tmp = os.path.join(STEP3, "out", "_preflight_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    torch.cuda.reset_peak_memory_stats()
    argv = sys.argv
    sys.argv = ["train_generator.py", "--qa", "--model", args.model,
                "--max_steps", str(TRAIN_STEPS), "--output_dir", tmp]
    result = None
    try:
        result = tg.main()
    except torch.cuda.OutOfMemoryError:
        fail("training ran out of GPU memory")
    except SystemExit as e:
        if e.code:
            fail(f"training exited with {e.code}")
    except Exception as e:  # noqa: BLE001 -- report anything, then stop
        fail(f"training crashed: {type(e).__name__}: {e}")
    finally:
        sys.argv = argv
        shutil.rmtree(tmp, ignore_errors=True)
    peak = torch.cuda.max_memory_reserved() / total_mem
    print(f"  peak GPU memory {peak:.0%} of {gib(total_mem):.1f} GiB")
    if peak > MEM_LIMIT:
        fail(f"training peaked at {peak:.0%} of GPU memory (limit {MEM_LIMIT:.0%})")
    per_step = result["seconds"] / result["steps"] if result else float("nan")
    if result:
        print(f"  {per_step:.2f} s per step")
    report["train"] = {"steps": TRAIN_STEPS, "seconds_per_step": round(per_step, 3),
                       "peak_mem_frac": round(peak, 3)}
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------- estimate
    # Training: 3 epochs of batch 16 over the 95% train split.
    steps_full = math.ceil((len(pairs) - int(len(pairs) * 0.05)) / 16) * 3
    train_min = steps_full * per_step / 60
    # Generation: assume 85% of samples come out usable (unfinished/malformed
    # are dropped); the first-batch check in the real run catches far worse.
    gen_min = math.ceil(100_000 / (bs * 0.85)) * batch_s / 60
    print(f"\nestimate: training ~{train_min:.0f} min ({steps_full:,} steps), "
          f"generation ~{gen_min:.0f} min (upper bound: every sample at full length)")
    report["estimate_minutes"] = {k: (round(v) if math.isfinite(v) else None)
                                  for k, v in [("train", train_min), ("generate", gen_min)]}

    report["failures"] = failures
    os.makedirs(os.path.join(STEP3, "out"), exist_ok=True)
    with open(os.path.join(STEP3, "out", "preflight_qa.json"), "w") as f:
        json.dump(report, f, indent=2)

    if failures:
        print("\n[PREFLIGHT FAILED] the real run was NOT started:")
        for m in failures:
            print(f"  - {m}")
        print("Bring back Step_3/out/preflight_qa.json.")
        sys.exit(1)
    with open(BATCH_FILE, "w") as f:
        f.write(str(bs))
    print(f"\n[PREFLIGHT OK] starting the real run (generation batch {bs})")


if __name__ == "__main__":
    main()
