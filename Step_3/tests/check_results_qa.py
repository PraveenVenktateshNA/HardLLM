"""Check the generated question -> answer data. RUN_STEP3_ON_NVIDIA.bat runs this
last, on the PC, right after generation. CPU only, under a minute.

FAIL (exit 1) on broken output:
  - not exactly the target number of rows, bad JSON, missing/empty fields
  - answers under the minimum length, or format markers leaking into the text
    (more than 1% of rows still containing "Question:" / "Answer:")

REPORT (printed, saved, never fails the run) on quality:
  - how many generations were dropped, and why
  - duplicates, and word-for-word copies of the training data
  - lengths and medical-word share, next to the same numbers for the real
    training pairs, so the two can be compared directly
  - 20 random pairs to read

Writes Step_3/out/results_qa.json and Step_3/out/samples_qa.txt.

    python Step_3/tests/check_results_qa.py
"""
import collections
import json
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP3 = os.path.dirname(HERE)
sys.path.insert(0, STEP3)
import generate_synthetic as gen  # noqa: E402
import train_generator as tg  # noqa: E402

TARGET = 100_000
OUT = os.path.join(STEP3, "out")

# A rough topical signal, not a classifier: share of texts with at least one of
# these words. Only meaningful next to the same number for the training data.
MEDICAL = re.compile(r"\b(" + "|".join([
    "pain", "doctor", "fever", "infection", "blood", "pressure", "medicine",
    "medication", "tablet", "dose", "symptoms?", "treatment", "disease", "skin",
    "rash", "cough", "headache", "stomach", "diabetes", "sugar", "heart",
    "allerg\\w*", "antibiotics?", "pregnan\\w*", "period", "surgery", "x-?ray",
    "scan", "test", "swelling", "vomiting", "diarrh\\w*", "anxiety", "sleep",
    "injury", "fracture", "cancer", "tumou?r", "liver", "kidney", "lungs?",
    "thyroid", "hormone", "vitamin", "consult", "physician", "hospital",
]) + r")\b", re.I)


def norm(s):
    return " ".join(str(s).split()).lower()


def describe(qs, ans):
    qw = [len(q.split()) for q in qs]
    aw = [len(a.split()) for a in ans]
    return {
        "question_words_median": statistics.median(qw),
        "answer_words_median": statistics.median(aw),
        "answer_words_p90": sorted(aw)[int(0.9 * len(aw))],
        "medical_word_share_questions": round(sum(bool(MEDICAL.search(q)) for q in qs) / len(qs), 3),
        "answers_starting_hi_hello": round(
            sum(a.lower().startswith(("hi", "hello", "dear")) for a in ans) / len(ans), 3),
    }


def main():
    path = os.path.join(OUT, "synthetic_qa.jsonl")
    fails, report = [], {}

    def fail(msg):
        fails.append(msg)
        print(f"  [FAIL] {msg}")

    if not os.path.exists(path):
        sys.exit(f"[FAIL] {path} does not exist: generation did not run")

    # ------------------------------------------------------------ structure
    rows, bad_json = [], 0
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad_json += 1
    print(f"rows: {len(rows):,}   unreadable lines: {bad_json}")
    if len(rows) != TARGET:
        fail(f"expected {TARGET:,} rows, found {len(rows):,}")
    if bad_json:
        fail(f"{bad_json} lines are not valid JSON")
    shape = [r for r in rows if isinstance(r, dict) and set(r) == {"question", "answer"}
             and str(r["question"]).strip() and str(r["answer"]).strip()]
    if len(shape) != len(rows):
        fail(f"{len(rows) - len(shape)} rows lack a non-empty question and answer")
    rows = shape
    short = sum(len(r["answer"].split()) < gen.MIN_ANSWER_WORDS for r in rows)
    if short:
        fail(f"{short} answers under {gen.MIN_ANSWER_WORDS} words")
    leak = sum(("Question:" in r["answer"] or "Answer:" in r["question"]
                or "Answer:" in r["answer"]) for r in rows)
    print(f"format markers left inside the text: {leak:,} ({leak / max(len(rows), 1):.2%})")
    if leak > 0.01 * len(rows):
        fail(f"{leak:,} rows still contain 'Question:'/'Answer:' inside the text")
    report["rows"] = len(rows)
    report["format_marker_leaks"] = leak

    # ---------------------------------------------------------------- drops
    stats_path = os.path.join(OUT, "synthetic_qa_stats.json")
    if os.path.exists(stats_path):
        st = json.load(open(stats_path))
        made = st["written_this_session"] + st["dropped_unfinished"] + st["dropped_malformed"]
        if made:
            print(f"dropped while generating (last session): "
                  f"{st['dropped_unfinished'] / made:.1%} unfinished at the length cap, "
                  f"{st['dropped_malformed'] / made:.1%} malformed")
        report["generation_stats"] = st

    # -------------------------------------------------- copies, duplicates
    train = [r for r in (json.loads(l) for l in open(tg.DEFAULT_DATA, encoding="utf-8"))
             if tg.is_qa_pair(r)]
    t_q = {norm(r["text"]) for r in train}
    t_a = {norm(r["target"]) for r in train}
    pairs = collections.Counter((norm(r["question"]), norm(r["answer"])) for r in rows)
    dup_pairs = sum(c - 1 for c in pairs.values() if c > 1)
    dup_q = len(rows) - len({norm(r["question"]) for r in rows})
    copy_q = sum(norm(r["question"]) in t_q for r in rows)
    copy_a = sum(norm(r["answer"]) in t_a for r in rows)
    n = max(len(rows), 1)
    print(f"duplicate pairs: {dup_pairs:,} ({dup_pairs / n:.2%})   "
          f"duplicate questions: {dup_q:,} ({dup_q / n:.2%})")
    print(f"copied word-for-word from training: questions {copy_q:,} ({copy_q / n:.2%}), "
          f"answers {copy_a:,} ({copy_a / n:.2%})")
    report.update(duplicate_pairs=dup_pairs, duplicate_questions=dup_q,
                  copied_questions=copy_q, copied_answers=copy_a)

    # ------------------------------------------------- compare to training
    syn = describe([r["question"] for r in rows], [r["answer"] for r in rows]) if rows else {}
    real = describe([r["text"] for r in train], [str(r["target"]) for r in train])
    print(f"\n{'':32}{'synthetic':>12}{'real training':>15}")
    for k in real:
        print(f"  {k:30}{syn.get(k, float('nan')):>12}{real[k]:>15}")
    report["synthetic"], report["real_training"] = syn, real

    # -------------------------------------------------------------- samples
    picks = random.Random(313).sample(rows, min(20, len(rows)))
    with open(os.path.join(OUT, "samples_qa.txt"), "w", encoding="utf-8") as f:
        for i, r in enumerate(picks, 1):
            f.write(f"--- {i}\nQ: {r['question']}\nA: {r['answer']}\n\n")
    print("\n3 random pairs (20 in out/samples_qa.txt):")
    for r in picks[:3]:
        print(f"  Q: {r['question'][:160]}\n  A: {r['answer'][:240]}\n")

    report["failures"] = fails
    with open(os.path.join(OUT, "results_qa.json"), "w") as f:
        json.dump(report, f, indent=2)
    if fails:
        print("[RESULTS CHECK FAILED] see above; bring back Step_3/out/")
        sys.exit(1)
    print("[RESULTS CHECK OK] wrote out/results_qa.json and out/samples_qa.txt")


if __name__ == "__main__":
    main()
