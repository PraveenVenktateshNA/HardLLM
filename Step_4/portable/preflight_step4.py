"""Pre-flight for Step 4 on the NVIDIA PC. Runs before the real job, ~8 minutes.

  1. input   every data file is there and matches data/manifest.json
  2. train   6 real LoRA steps on Llama-3.2-3B with every row stretched to the
             full 512 tokens (the worst case), in a fresh process per setting:
                 checkpointing off: 4, then 2 rows per pass
                 checkpointing on:  8, 4, then 2 rows per pass
             The fastest setting that stays under 90% of GPU memory is used for
             the real run (the .bat reads out/train_micro.txt, out/train_ckpt.txt).
             Every setting trains on batch 16; only speed and memory change.
  3. time    the estimate for all 4 trainings and the evaluation

Exits non-zero (and the .bat stops) if nothing fits. Writes out/preflight_step4.json.

    python Step_4/portable/preflight_step4.py --model <path to Llama-3.2-3B>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
sys.path.insert(0, STEP4)
import train_lora as tl  # noqa: E402

MEM_LIMIT = 0.90
PROBE_STEPS = 6
SETTINGS = [(0, 4), (0, 2), (1, 8), (1, 4), (1, 2)]   # (checkpointing, rows per pass)
OUT = tl.OUT
EVAL_TOKENS = 1_089 * 150 + 1_000 * 330   # MMLU prompts + test pairs, forward only


def run_probe(model, ckpt, micro):
    tmp = os.path.join(OUT, "_preflight_tmp")
    r = subprocess.run([sys.executable, os.path.join(STEP4, "train_lora.py"), "--arm", "private",
                        "--model", model, "--max_steps", str(PROBE_STEPS), "--worst_case",
                        "--micro_batch", str(micro), "--checkpointing", str(ckpt), "--rows", "200",
                        "--output_dir", tmp],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding="utf-8", errors="replace")
    lines = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
    if r.returncode or not lines:
        tail = "\n".join(r.stdout.splitlines()[-4:])
        oom = "out of memory" in r.stdout.lower() or "outofmemory" in r.stdout.lower()
        return {"ok": False, "why": "out of memory" if oom else f"crashed (exit {r.returncode})",
                "tail": tail}
    res = json.loads(lines[-1][len("PROBE "):])
    res["ok"] = res.get("peak_reserved_frac", 1.0) <= MEM_LIMIT
    if not res["ok"]:
        res["why"] = f"peak memory {res['peak_reserved_frac']:.0%} > {MEM_LIMIT:.0%}"
    return res


def choose(results):
    """Fastest setting that fits, or None."""
    fits = [(r["tokens_per_second"], s) for s, r in results.items() if r["ok"]]
    return max(fits)[1] if fits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    import torch
    report, failures = {}, []

    def fail(msg):
        failures.append(msg)
        print(f"  [FAIL] {msg}")

    if not torch.cuda.is_available():
        sys.exit("[FAIL] no CUDA GPU visible to PyTorch")
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name}, {dev.total_memory / 2**30:.1f} GiB")
    report["gpu"] = dev.name
    for f in ("train_micro.txt", "train_ckpt.txt"):   # never reuse an old run's choice
        if os.path.exists(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))

    # ---------------------------------------------------------------- input
    print("\n1/3 input")
    manifest = json.load(open(os.path.join(tl.DATA, "manifest.json")))
    for name, m in manifest["sets"].items():
        p = os.path.join(tl.DATA, m["file"])
        if not os.path.exists(p):
            fail(f"missing data/{m['file']}")
        elif hashlib.sha256(open(p, "rb").read()).hexdigest() != m["sha256"]:
            fail(f"data/{m['file']} is not the file prepare_data.py wrote")
    tok = tl.setup_tokenizer(args.model)
    total_tokens = 0
    for arm in tl.ARMS:
        rows = [json.loads(l) for l in open(os.path.join(tl.DATA, f"train_{arm}.jsonl"), encoding="utf-8")]
        n = sum(len(tl.encode(tok, r["question"], r["answer"], 512)["input_ids"]) for r in rows)
        print(f"  {arm:9} {len(rows):,} rows, {n:,} tokens")
        total_tokens += n
    report["train_tokens_all_arms"] = total_tokens
    if failures:
        return finish(report, failures)

    # ---------------------------------------------------------------- train
    print(f"\n2/3 training: {PROBE_STEPS} steps per setting, every row at 512 tokens")
    results = {}
    for ckpt, micro in SETTINGS:
        r = run_probe(args.model, ckpt, micro)
        key = f"checkpointing={ckpt} rows_per_pass={micro}"
        results[key] = r
        if r["ok"]:
            print(f"  {key}: {r['tokens_per_second']:,.0f} tokens/s, peak memory {r['peak_reserved_frac']:.0%}")
        else:
            print(f"  {key}: does not fit ({r['why']})")
            if "crashed" in r["why"]:
                print("    " + r["tail"].replace("\n", "\n    "))
    report["probes"] = results
    best = choose(results)
    if best is None:
        fail("no setting fits in GPU memory")
        return finish(report, failures)
    ckpt = int(best.split()[0].split("=")[1])
    micro = int(best.split()[1].split("=")[1])
    tps = results[best]["tokens_per_second"]
    print(f"  using {best}")

    # ------------------------------------------------------------- estimate
    train_min = total_tokens * 1.05 / tps / 60 + 4 * 1.5     # + padding, + model loads
    eval_min = 5 * (1.0 + EVAL_TOKENS / (3 * tps) / 60 + 1.0)  # load, forward, 20 answers
    print(f"\n3/3 estimate: training ~{train_min:.0f} min (4 models), evaluation ~{eval_min:.0f} min "
          f"(5 models), total ~{(train_min + eval_min) / 60:.1f} h")
    report["chosen"] = {"checkpointing": ckpt, "rows_per_pass": micro, "tokens_per_second": round(tps)}
    report["estimate_minutes"] = {"train": round(train_min), "evaluate": round(eval_min)}
    with open(os.path.join(OUT, "train_micro.txt"), "w") as f:
        f.write(str(micro))
    with open(os.path.join(OUT, "train_ckpt.txt"), "w") as f:
        f.write(str(ckpt))
    return finish(report, failures)


def finish(report, failures):
    report["failures"] = failures
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "preflight_step4.json"), "w") as f:
        json.dump(report, f, indent=2)
    import shutil
    shutil.rmtree(os.path.join(OUT, "_preflight_tmp"), ignore_errors=True)
    if failures:
        print("\n[PREFLIGHT FAILED] the real run was NOT started:")
        for m in failures:
            print(f"  - {m}")
        print("Bring back Step_4/out/.")
        sys.exit(1)
    print("\n[PREFLIGHT OK] starting the real run")


if __name__ == "__main__":
    main()
