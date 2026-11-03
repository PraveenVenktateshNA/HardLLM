"""Check Step 4's outputs. RUN_STEP4_ON_NVIDIA.bat runs this last. Seconds.

FAIL (exit 1) if anything is missing or broken: a model not trained or not
evaluated, a non-finite score, models scored on different numbers of tokens,
missing demo answers. It never judges which model won; that is the result.

    python Step_4/tests/check_results_step4.py
"""
import json
import math
import os
import sys

STEP4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(STEP4, "out")
ARMS = ["hardllm", "selected", "random", "private"]


def main():
    fails = []
    for arm in ARMS:
        p = os.path.join(OUT, f"train_{arm}.json")
        if not os.path.exists(p):
            fails.append(f"{arm}: not trained (no out/train_{arm}.json)")
            continue
        t = json.load(open(p))
        if t["steps"] != 1000 or not math.isfinite(t["train_loss"]):
            fails.append(f"{arm}: {t['steps']} steps, loss {t['train_loss']}")
        if not os.path.exists(os.path.join(STEP4, "models", f"lora-{arm}", "adapter_model.safetensors")):
            fails.append(f"{arm}: adapter missing")
    scored = {}
    for m in ["base"] + ARMS:
        p = os.path.join(OUT, f"eval_{m}.json")
        if not os.path.exists(p):
            fails.append(f"{m}: not evaluated")
            continue
        e = json.load(open(p))
        scored[m] = e["answer_tokens_scored"]
        if not (math.isfinite(e["answer_perplexity"]) and 0.2 <= e["mmlu_medical"] <= 1):
            fails.append(f"{m}: perplexity {e['answer_perplexity']}, MMLU {e['mmlu_medical']}")
        if len(e["demo_answers"]) != 20:
            fails.append(f"{m}: {len(e['demo_answers'])} demo answers, expected 20")
    if len(set(scored.values())) > 1:
        fails.append(f"models were scored on different token counts: {scored}")
    for f in ["results_step4.md", "results_step4.json", "demo_answers.md"]:
        if not os.path.exists(os.path.join(OUT, f)):
            fails.append(f"out/{f} missing")
    if fails:
        for f in fails:
            print(f"  [FAIL] {f}")
        print("[RESULTS CHECK FAILED] bring back Step_4/out/")
        sys.exit(1)
    print(open(os.path.join(OUT, "results_step4.md"), encoding="utf-8").read())
    print("[RESULTS CHECK OK]")


if __name__ == "__main__":
    main()
