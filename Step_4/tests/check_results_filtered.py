"""Check the hardllm-filtered run. RUN_STEP4_FILTERED.bat runs this last. Seconds.

FAIL (exit 1) if the filtered training set is wrong (size, repeats, a pair
that is not a Step 3 candidate), the model is not fully trained or not
evaluated, its score is not finite, it was scored on different tokens from the
other models, or the first results were changed. It never judges who won.

    python Step_4/tests/check_results_filtered.py
"""
import json
import math
import os
import sys

STEP4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(STEP4, "out")
DATA = os.path.join(STEP4, "data")
ARM = "hardllm-filtered"
FIRST = ["base", "hardllm", "selected", "random", "private"]


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    fails = []
    rows = read(os.path.join(DATA, f"train_{ARM}.jsonl"))
    cand = {json.dumps(r, sort_keys=True) for r in read(os.path.join(DATA, "hardllm_candidates.jsonl"))}
    keys = [json.dumps(r, sort_keys=True) for r in rows]
    if len(rows) != 16_000 or len(set(keys)) != 16_000:
        fails.append(f"training set: {len(rows)} rows, {len(set(keys))} distinct (expected 16,000 each)")
    if not all(k in cand for k in keys):
        fails.append("training set holds pairs that are not Step 3 candidates")
    f = json.load(open(os.path.join(OUT, "filter_hardllm.json")))
    if f["median_similarity"]["hardllm_filtered_arm"] <= f["median_similarity"]["hardllm_arm_as_trained"]:
        fails.append("the filtered set does not match its questions better than the original hardllm set")

    t = json.load(open(os.path.join(OUT, f"train_{ARM}.json")))
    if t["steps"] != 1000 or not math.isfinite(t["train_loss"]):
        fails.append(f"training: {t['steps']} steps, loss {t['train_loss']}")
    if not os.path.exists(os.path.join(STEP4, "models", f"lora-{ARM}", "adapter_model.safetensors")):
        fails.append("adapter missing")

    scored = {}
    for m in FIRST + [ARM]:
        p = os.path.join(OUT, f"eval_{m}.json")
        if not os.path.exists(p):
            fails.append(f"{m}: not evaluated")
            continue
        e = json.load(open(p))
        scored[m] = e["answer_tokens_scored"]
        if not math.isfinite(e["answer_perplexity"]) or len(e["demo_answers"]) != 20:
            fails.append(f"{m}: perplexity {e['answer_perplexity']}, {len(e['demo_answers'])} demo answers")
    if len(set(scored.values())) > 1:
        fails.append(f"models were scored on different token counts: {scored}")

    # the first report must be untouched: it lists exactly the first 5 models
    first = json.load(open(os.path.join(OUT, "results_step4.json")))
    if sorted(first) != sorted(FIRST):
        fails.append(f"results_step4.json was changed: it now lists {sorted(first)}")
    for name in ["results_step4_filtered.md", "results_step4_filtered.json", "demo_answers_filtered.md"]:
        if not os.path.exists(os.path.join(OUT, name)):
            fails.append(f"out/{name} missing")

    if fails:
        for x in fails:
            print(f"  [FAIL] {x}")
        print("[RESULTS CHECK FAILED] bring back Step_4/out/")
        sys.exit(1)
    print(open(os.path.join(OUT, "results_step4_filtered.md"), encoding="utf-8").read())
    print("[RESULTS CHECK OK]")


if __name__ == "__main__":
    main()
