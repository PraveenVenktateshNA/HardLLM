"""Step 4, part 1: build the training sets for the comparison. Plain Python, no
model, ~1 minute. Runs on the Mac; the PC gets the finished files in data/.

Every model is Llama-3.2-3B with the same LoRA settings and the same number of
training rows (N_TRAIN). Only the data differs:

  hardllm   the full HardLLM pipeline: synthetic pairs from Step 3 (cleaned)
  selected  the public pairs Step 2 selected with ε = 0.5, used directly
            (no GPT-2 synthesis): shows what Step 3 adds or loses
  random    public pairs picked at random from the same pool, same count as
            `selected`: shows what the private selection is worth
  private   the real private records: the ceiling. The point of HardLLM is to
            never do this; it is here only to measure the cost of privacy.
  (base)    Llama-3.2-3B with no fine-tuning, evaluated as-is

Test sets (no model trains on them):
  test_private.jsonl   1,000 held-out private pairs: answer perplexity
  demo_questions.jsonl 20 of those questions: side-by-side answers to read
  mmlu_medical.jsonl   1,089 MMLU medical multiple-choice questions: did
                       fine-tuning damage medical knowledge?

Every set goes through the same Step 3 cleaner (clean_synthetic_qa.py), so a
model trained on cleaned text is never scored against greetings and sign-offs.
When a set has fewer than N_TRAIN usable pairs it is repeated to N_TRAIN, so
every model gets the same number of training steps.

    python Step_4/prepare_data.py
"""
import collections
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # HardLLM/HardLLM
POOL = os.path.join(os.path.dirname(ROOT), "pool")  # HardLLM/pool
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(ROOT, "Step_3"))
import clean_synthetic_qa as cleaner  # noqa: E402

SEED = 313
N_TRAIN = 16_000   # rows per model: 1,000 optimizer steps at batch 16
N_TEST = 1_000
N_DEMO = 20
ARMS = ["hardllm", "selected", "random", "private"]

# Same rule as Step_3/train_generator.is_qa_pair (DialogSum's target is a
# summary, not an answer).
QA_SOURCES = {"lavita/ChatDoctor-HealthCareMagic-100k", "lavita/MedQuAD", "LLukas22/fiqa"}
MIN_RAW_ANSWER_WORDS = 3


def is_qa_pair(row):
    return (row.get("source") in QA_SOURCES and bool(row.get("target"))
            and len(str(row["target"]).split()) >= MIN_RAW_ANSWER_WORDS)


def norm(s):
    return " ".join(s.lower().split())


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def clean_pairs(pairs, drops):
    """(question, answer) -> cleaned, deduplicated {"question", "answer"} rows."""
    out, seen = [], set()
    for q, a in pairs:
        q2 = cleaner.clean_question(q)
        a2 = cleaner.clean_answer(a)
        why = cleaner.drop_reason(q2, a, a2)
        if why is None and norm(q2) in seen:
            why = "duplicate_question"
        if why:
            drops[why] += 1
            continue
        seen.add(norm(q2))
        out.append({"question": q2, "answer": a2})
    return out


def fill(rows, n, rng):
    """Exactly n rows: shuffled, repeated whole when there are fewer than n."""
    rows = rows[:]
    rng.shuffle(rows)
    out = []
    while len(out) < n:
        out.extend(rows[: n - len(out)])
    return out


def write(name, rows):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    os.makedirs(DATA, exist_ok=True)
    rng = random.Random(SEED)
    manifest = {"seed": SEED, "n_train": N_TRAIN, "sets": {}}

    # ------------------------------------------------- private: test + ceiling
    drops = collections.Counter()
    private = clean_pairs([(r["text"], r["target"]) for r in read(os.path.join(POOL, "private_medical.jsonl"))
                           if r.get("target")], drops)
    rng.shuffle(private)
    test, private_train = private[:N_TEST], private[N_TEST:N_TEST + N_TRAIN]
    test_q = {norm(r["question"]) for r in test}
    manifest["private_cleaning_drops"] = dict(drops)

    # --------------------------------------------------------------- the arms
    unique = {}
    unique["hardllm"] = read(os.path.join(ROOT, "Step_3", "out", "synthetic_qa_clean.jsonl"))

    selected_raw = [r for r in read(os.path.join(ROOT, "Step_2", "out", "selected_centroid_eps0.5_t3.jsonl"))
                    if is_qa_pair(r)]
    drops = collections.Counter()
    unique["selected"] = clean_pairs([(r["text"], str(r["target"])) for r in selected_raw], drops)
    manifest["selected_cleaning_drops"] = dict(drops)

    # Random: the same number of raw pairs as `selected`, drawn uniformly from
    # every Q&A pair in the public pool, then cleaned the same way.
    pool_pairs = [r for r in read(os.path.join(POOL, "pool_with_clusters.jsonl")) if is_qa_pair(r)]
    random_raw = random.Random(SEED + 1).sample(pool_pairs, len(selected_raw))
    drops = collections.Counter()
    unique["random"] = clean_pairs([(r["text"], str(r["target"])) for r in random_raw], drops)
    manifest["random_cleaning_drops"] = dict(drops)
    manifest["random_sources"] = dict(collections.Counter(r["source"] for r in random_raw))
    manifest["selected_sources"] = dict(collections.Counter(r["source"] for r in selected_raw))

    unique["private"] = private_train

    # No test question may appear in any training set.
    for arm in ARMS:
        before = len(unique[arm])
        unique[arm] = [r for r in unique[arm] if norm(r["question"]) not in test_q]
        manifest.setdefault("test_overlap_removed", {})[arm] = before - len(unique[arm])

    for arm in ARMS:
        rows = fill(unique[arm], N_TRAIN, random.Random(SEED + 7))
        manifest["sets"][arm] = {
            "file": f"train_{arm}.jsonl", "rows": len(rows), "unique_pairs": len(unique[arm]),
            "times_each_pair_is_seen": round(N_TRAIN / len(unique[arm]), 2),
            "sha256": write(f"train_{arm}.jsonl", rows)}

    manifest["sets"]["test_private"] = {"file": "test_private.jsonl", "rows": len(test),
                                       "sha256": write("test_private.jsonl", test)}
    demo = [{"question": r["question"]} for r in test[:N_DEMO]]
    manifest["sets"]["demo_questions"] = {"file": "demo_questions.jsonl", "rows": len(demo),
                                         "sha256": write("demo_questions.jsonl", demo)}
    mmlu = os.path.join(DATA, "mmlu_medical.jsonl")
    manifest["sets"]["mmlu_medical"] = {"file": "mmlu_medical.jsonl",
                                       "rows": sum(1 for _ in open(mmlu)),
                                       "sha256": hashlib.sha256(open(mmlu, "rb").read()).hexdigest()}

    with open(os.path.join(DATA, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for k, v in manifest["sets"].items():
        extra = (f"  {v['unique_pairs']:,} unique, each seen {v['times_each_pair_is_seen']}x"
                 if "unique_pairs" in v else "")
        print(f"{k:16} {v['rows']:>7,} rows{extra}")
    print("random sources:  ", manifest["random_sources"])
    print("selected sources:", manifest["selected_sources"])
    print("test overlap removed:", manifest["test_overlap_removed"])
    print(f"wrote {DATA}/ and manifest.json")


if __name__ == "__main__":
    main()
