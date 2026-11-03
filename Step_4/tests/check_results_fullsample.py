"""Check the hardllm-fullsample run. RUN_STEP4_FULLSAMPLE.bat runs this last. Seconds.

FAIL (exit 1) if the synthetic pairs were not sampled without the cut, the
training set is wrong (size, repeats, a test question, a pair that is not from
the new Step 3 run), the model is not fully trained or not evaluated, it was
scored on different tokens from the others, or an earlier report was changed.
It never judges who won.

    python Step_4/tests/check_results_fullsample.py
"""
import json
import math
import os
import sys

STEP4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP3_OUT = os.path.join(os.path.dirname(STEP4), "Step_3", "out")
OUT = os.path.join(STEP4, "out")
DATA = os.path.join(STEP4, "data")
ARM = "hardllm-fullsample"
FIRST = ["base", "hardllm", "selected", "random", "private"]


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def norm(s):
    return " ".join(s.lower().split())


def main():
    fails = []
    g = json.load(open(os.path.join(STEP3_OUT, "synthetic_qa_fullsample_stats.json")))["args"]
    if (g["top_k"], g["top_p"], g["temperature"]) != (0, 1.0, 1.0) or "gpt2-qa" not in g["model"]:
        fails.append(f"generation settings: model {g['model']}, top_k {g['top_k']}, "
                     f"top_p {g['top_p']}, temperature {g['temperature']}")

    rows = read(os.path.join(DATA, f"train_{ARM}.jsonl"))
    src = {json.dumps(r, sort_keys=True) for r in read(os.path.join(STEP3_OUT, "synthetic_qa_fullsample_clean.jsonl"))}
    keys = [json.dumps(r, sort_keys=True) for r in rows]
    test_q = {norm(r["question"]) for r in read(os.path.join(DATA, "test_private.jsonl"))}
    if len(rows) != 16_000 or len({norm(r["question"]) for r in rows}) != 16_000:
        fails.append(f"training set: {len(rows)} rows, expected 16,000 different questions")
    if not all(k in src for k in keys):
        fails.append("training set holds pairs that are not from the new Step 3 run")
    if {norm(r["question"]) for r in rows} & test_q:
        fails.append("training set contains test questions")

    t = json.load(open(os.path.join(OUT, f"train_{ARM}.json")))
    if t["steps"] != 1000 or not math.isfinite(t["train_loss"]):
        fails.append(f"training: {t['steps']} steps, loss {t['train_loss']}")
    if not os.path.exists(os.path.join(STEP4, "models", f"lora-{ARM}", "adapter_model.safetensors")):
        fails.append("adapter missing")

    scored = {}
    for m in FIRST + ["hardllm-filtered", ARM]:
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

    # earlier reports must be untouched
    for name, models in [("results_step4.json", FIRST),
                         ("results_step4_filtered.json", FIRST + ["hardllm-filtered"])]:
        have = sorted(json.load(open(os.path.join(OUT, name))))
        if have != sorted(models):
            fails.append(f"{name} was changed: it now lists {have}")
    for name in ["results_step4_fullsample.md", "results_step4_fullsample.json",
                 "demo_answers_fullsample.md", "fullsample_arm.json"]:
        if not os.path.exists(os.path.join(OUT, name)):
            fails.append(f"out/{name} missing")

    if fails:
        for x in fails:
            print(f"  [FAIL] {x}")
        print("[RESULTS CHECK FAILED] bring back Step_4/out/ and Step_3/out/")
        sys.exit(1)
    print(open(os.path.join(OUT, "results_step4_fullsample.md"), encoding="utf-8").read())
    print("[RESULTS CHECK OK]")


if __name__ == "__main__":
    main()
