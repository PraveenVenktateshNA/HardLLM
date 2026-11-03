"""Check the Step 5 run. RUN_STEP5_EXPORT.bat runs this last. Seconds.

FAIL (exit 1) if a file is missing or the wrong size, the merged model does
not score what Step 4 measured for the same model, the f16 file does not
score what the merged model does (the reading back would be wrong), or a
score is not finite. It reports how much compression cost; it does not judge
whether that is acceptable.

    python Step_5/tests/check_results_step5.py [--arm selected]
"""
import argparse
import json
import math
import os
import sys

STEP5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP4 = os.path.join(os.path.dirname(STEP5), "Step_4")
OUT = os.path.join(STEP5, "out")
# expected file sizes, GB, for Llama-3.2-3B
SIZES = {"f16": (5.5, 7.0), "q8_0": (3.0, 3.8), "q4_k_m": (1.7, 2.4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="selected")
    args = ap.parse_args()
    fails = []
    r = {}
    for v in ["merged", "f16", "q8_0", "q4_k_m"]:
        p = os.path.join(OUT, f"eval_{v}.json")
        if not os.path.exists(p):
            fails.append(f"{v}: not scored")
            continue
        r[v] = json.load(open(p))
        if not math.isfinite(r[v]["answer_perplexity"]):
            fails.append(f"{v}: answer perplexity {r[v]['answer_perplexity']}")
    for v, (lo, hi) in SIZES.items():
        f = os.path.join(STEP5, "models", f"{args.arm}-{v}.gguf")
        gb = os.path.getsize(f) / 2**30 if os.path.exists(f) else 0
        if not lo <= gb <= hi:
            fails.append(f"{os.path.basename(f)}: {gb:.2f} GB, expected {lo}-{hi}")
        lc = os.path.join(OUT, f"llamacpp_{v}.json")
        if not os.path.exists(lc) or not math.isfinite(json.load(open(lc))["ppl"]):
            fails.append(f"{v}: no llama.cpp perplexity")

    s4 = json.load(open(os.path.join(STEP4, "out", f"eval_{args.arm}.json")))
    if "merged" in r:
        if abs(r["merged"]["answer_perplexity"] - s4["answer_perplexity"]) > 0.02:
            fails.append(f"merged model scores {r['merged']['answer_perplexity']}, "
                         f"Step 4 measured {s4['answer_perplexity']} for the same model")
        if r["merged"]["answer_tokens_scored"] != s4["answer_tokens_scored"]:
            fails.append("scored on different tokens from Step 4")
    if "merged" in r and "f16" in r:
        if abs(r["f16"]["answer_perplexity"] - r["merged"]["answer_perplexity"]) > 0.02:
            fails.append(f"f16 file scores {r['f16']['answer_perplexity']}, merged "
                         f"{r['merged']['answer_perplexity']}: the GGUF reading back is wrong")
    for v in ["q8_0", "q4_k_m"]:
        if v in r and len(r[v].get("demo_answers", [])) != 20:
            fails.append(f"{v}: {len(r[v].get('demo_answers', []))} demo answers")
    for name in ["results_step5.md", "results_step5.json", "demo_answers_step5.md"]:
        if not os.path.exists(os.path.join(OUT, name)):
            fails.append(f"out/{name} missing")

    if fails:
        for x in fails:
            print(f"  [FAIL] {x}")
        print("[RESULTS CHECK FAILED] bring back Step_5/out/")
        sys.exit(1)
    print(open(os.path.join(OUT, "results_step5.md"), encoding="utf-8").read())
    print("[RESULTS CHECK OK]")


if __name__ == "__main__":
    main()
