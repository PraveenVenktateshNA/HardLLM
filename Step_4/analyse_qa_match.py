"""Step 4, analysis: does each training answer talk about its own question?
Plain Python, no model, seconds. Runs on the Mac.

    python Step_4/analyse_qa_match.py

For every pair, the question's specific words are the ones that are not common
across all questions (a word in more than 2% of questions, like "pain" or
"doctor", says nothing about the topic). A pair "matches" when the answer
contains at least one of them.

Some matches happen by luck, so every set is also scored with its answers
shuffled onto other questions. The useful number is matched minus shuffled:
how much the answer depends on the question it was given.

Reads data/train_*.jsonl and data/test_private.jsonl; changes nothing.
Writes out/qa_match.json.
"""
import collections
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
SETS = ["hardllm", "selected", "random", "private", "test_private"]
COMMON_SHARE = 0.02
STEM = 6     # "infection"/"infections"/"infected" -> "infect"
STOP = set("""
about above after again against also although always among another anything
around because been before being below between both cannot could does doing
done down during each either else even ever every from further have having
here hers herself himself however into itself just know like made make many
might more most much must myself near need neither never only other otherwise
ours ourselves over please quite rather really same shall should since some
something still such than thank thanks that their theirs them themselves then
there these they thing things this those though through thus till under until
upon very want were what whatever when where whether which while whom whose
will with within without would your yours yourself yourselves hello dear
""".split())


def read(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name), encoding="utf-8")]


def terms(text):
    return {w[:STEM] for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def match_rate(questions, answers):
    hits = [bool(q & a) for q, a in zip(questions, answers) if q]
    return sum(hits) / len(hits)


def main():
    sets = {s: read(f"train_{s}.jsonl" if s != "test_private" else "test_private.jsonl") for s in SETS}
    # what counts as common is decided once, over every question, so all sets
    # use the same word list
    all_q = [terms(r["question"]) for rows in sets.values() for r in rows]
    df = collections.Counter(t for q in all_q for t in q)
    common = {t for t, n in df.items() if n / len(all_q) > COMMON_SHARE}

    rng = random.Random(313)
    report = {"common_words_ignored": len(common), "sets": {}}
    for name, rows in sets.items():
        qs = [terms(r["question"]) - common for r in rows]
        ans = [terms(r["answer"]) for r in rows]
        shuffled = ans[:]
        rng.shuffle(shuffled)
        m, s = match_rate(qs, ans), match_rate(qs, shuffled)
        overlap = [len(q & a) / len(q) for q, a in zip(qs, ans) if q]
        report["sets"][name] = {
            "pairs": len(rows),
            "questions_with_specific_words": sum(bool(q) for q in qs),
            "answer_uses_its_question_words": round(m, 4),
            "same_with_answers_shuffled": round(s, 4),
            "depends_on_question": round(m - s, 4),
            "mean_share_of_question_words_in_answer": round(sum(overlap) / len(overlap), 4)}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "qa_match.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"{'set':13} {'matched':>8} {'shuffled':>9} {'depends':>8} {'share of q words':>17}")
    for name, r in report["sets"].items():
        print(f"{name:13} {r['answer_uses_its_question_words']:>8.1%} {r['same_with_answers_shuffled']:>9.1%} "
              f"{r['depends_on_question']:>8.1%} {r['mean_share_of_question_words_in_answer']:>17.1%}")
    print("wrote out/qa_match.json")


if __name__ == "__main__":
    main()
