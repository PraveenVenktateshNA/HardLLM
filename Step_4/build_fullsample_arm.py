"""Step 4, extra arm: hardllm-fullsample = the HardLLM pipeline with Step 3's
synthetic pairs written by the SAME fine-tuned GPT-2, but sampled from its full
distribution (top_k off, top_p 1.0, temperature 1.0) instead of the paper's
top_k 50 / top_p 0.9. Plain Python, seconds.

    python Step_4/build_fullsample_arm.py

Why: the hardllm training answers use about half the vocabulary of real doctor
answers (3,687 distinct words per 100k against ~7,000-7,700), and 3.3% of the
words in the real test answers never appear in them (real sets: ~0.65%). The
paper's top_k/top_p cut the rare words at every step; a model trained on that
text learns to never expect them, and is scored on real answers full of them.
hardllm-filtered (relevance filter) did not help: every pair has the cut.

Reads ../Step_3/out/synthetic_qa_fullsample_clean.jsonl (written by the .bat:
generate_synthetic.py, then clean_synthetic_qa.py, unchanged rules). Drops test
questions and repeats, draws 16,000 at random, like the hardllm arm.
Writes data/train_hardllm-fullsample.jsonl and out/fullsample_arm.json with the
vocabulary measures for every training set, so the effect of the sampling is
visible before any training. Does nothing if the training file already exists.
"""
import collections
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
SOURCE = os.path.join(os.path.dirname(HERE), "Step_3", "out", "synthetic_qa_fullsample_clean.jsonl")
ARM = "hardllm-fullsample"
N_TRAIN = 16_000
SEED = 313


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def norm(s):
    return " ".join(s.lower().split())


def answer_words(rows):
    return [w for r in rows for w in re.findall(r"[a-z']+", r["answer"].lower())]


def vocab(rows, test_counts):
    """distinct words per 100k answer words; share of real test-answer words
    the set never uses; share taken by its 100 most common words."""
    w = answer_words(rows)
    c = collections.Counter(w)
    total = sum(test_counts.values())
    return {"answer_words": len(w),
            "distinct_per_100k": len(set(w[:100_000])),
            "test_words_never_seen": round(sum(n for t, n in test_counts.items() if t not in c) / total, 4),
            "top100_share": round(sum(n for _, n in c.most_common(100)) / len(w), 4)}


def main():
    target = os.path.join(DATA, f"train_{ARM}.jsonl")
    if os.path.exists(target):
        print(f"[skip] {target} already exists")
        return
    test = read(os.path.join(DATA, "test_private.jsonl"))
    test_q = {norm(r["question"]) for r in test}
    rows, seen = [], set()
    for r in read(SOURCE):
        k = norm(r["question"])
        if k in test_q or k in seen:
            continue
        seen.add(k)
        rows.append({"question": r["question"], "answer": r["answer"]})
    if len(rows) < N_TRAIN:
        sys.exit(f"[stopped] only {len(rows):,} usable pairs after cleaning, need {N_TRAIN:,}. "
                 f"Bring back Step_3/out/.")
    picked = random.Random(SEED).sample(rows, N_TRAIN)
    with open(target, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")

    test_counts = collections.Counter(answer_words(test))
    report = {"usable_pairs": len(rows), "kept_for_training": N_TRAIN,
              "sampling": "same GPT-2 as hardllm; top_k off, top_p 1.0, temperature 1.0",
              "vocabulary": {}}
    for name in ["hardllm", "hardllm-fullsample", "selected", "random", "private"]:
        report["vocabulary"][name] = vocab(read(os.path.join(DATA, f"train_{name}.jsonl")), test_counts)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fullsample_arm.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"{len(rows):,} usable pairs, {N_TRAIN:,} drawn for training")
    print(f"{'set':20}{'distinct/100k':>14}{'test words never seen':>23}{'top-100 share':>15}")
    for name, v in report["vocabulary"].items():
        print(f"{name:20}{v['distinct_per_100k']:>14,}{v['test_words_never_seen']:>23.2%}{v['top100_share']:>15.1%}")
    print(f"wrote {target}\n      out/fullsample_arm.json")


if __name__ == "__main__":
    main()
