"""Step 3: clean the synthetic question -> answer pairs before Step 4.

Plain text rules, no model: a few seconds on any machine.

    python Step_3/clean_synthetic_qa.py

Reads out/synthetic_qa.jsonl, writes:
  out/synthetic_qa_clean.jsonl    the kept pairs, cleaned
  out/clean_qa_report.json        how many rows each rule touched or dropped
  out/clean_qa_examples.txt       before/after examples for every rule, to check by eye

What it does NOT do: check medical facts. A fluent wrong answer passes.

Cleaning only reads synthetic data, never the private data, so it is
post-processing in the differential-privacy sense and costs no privacy.

Step 4 must run clean_answer()/clean_question() over EVERY set the same way:
the HardLLM data, the random baseline, the private ceiling and the test set.
Otherwise a model trained on cleaned text is scored on uncleaned text and loses
points for not saying "Hi, welcome to Chat Doctor".

The rules were fixed from the synthetic data alone, before any Step 4 score.
"""
import argparse
import collections
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MIN_ANSWER_WORDS = 15     # after cleaning; shorter is a greeting or a dangling stub
MIN_QUESTION_WORDS = 3
REPEAT_NGRAM, REPEAT_MAX = 6, 2   # a 6-word phrase 3+ times = a generation loop

# ----------------------------------------------------------------- patterns

# Greeting words at the start of an answer ("Hi, Welcome to Chat Doctor,") are
# cut and the rest of that sentence is kept: real content often follows on.
SALUTE = re.compile(
    r"^(?:(?:hi(?: there)?|hello(?: there)?|hey|hai|dear|sir|madam|ma'?am|friend|"
    r"greetings|good (?:morning|afternoon|evening|day)|welcome(?: to chat ?doctor)?|"
    r"com|forum)\b[\s,!.:;-]*)+",
    re.I)
# A clause that is only a pleasantry. If a comma follows, what comes after it
# is kept ("Thanks for query, In my opinion..."); otherwise the sentence goes.
PLEASANTRY = re.compile(
    r"^(?:and )?(thanks?|thank you|welcome|i (have|had) (gone|read)|i read|read your|"
    r"i (can )?understand|i went through|i have carefully|degree understand|"
    r"dairy have|come have gone|in brief|brief answer|i will try to help|"
    r"i would like to help|understanding your concern|i (have )?studied your|"
    r"i appreciate your|i (have )?(read|reviewed|noted) (your|the) (query|question|history)|"
    r"i (have )?(gone|went) through|i understood)\b", re.I)
# Sentences that are sign-offs, not content. Only stripped from the END.
SIGNOFF = re.compile(
    r"\b(hope (this|that|i|my|it|you|your|to|so|the information|the answer)|"
    r"wish(ing)? (you|your|him|her|them|the)|available for further|additional information|"
    r"take care|regards|best (wishes|regards|of luck)|get well|let me know|feel free|"
    r"thanks|thank you|good luck|chat doctor|jay in|bye|kindly|welcome for|cvts|"
    r"i will be happy|happy to help|do not hesitate|don'?t hesitate|rate (the|my) answer|"
    r"writing to us|has been resolved|don'?t worry|do not worry|will be alright|"
    r"any (further|more|other) (query|queries|questions?|doubts?|clarifications?|help)|"
    r"further (query|queries|clarification)|contact (me|us)|direct question|"
    r"this url|below[- ]mentioned link|bit\.ly|https?:|follow ?up)\b", re.I)
# Anonymisation artefacts in ChatDoctor: doctor names and some drug names were
# replaced by "Chat Doctor", so a sentence holding it has lost its meaning.
CHATDOCTOR = re.compile(r"chat\s*doctor|chatdoctor|genl-cvts", re.I)
URL = re.compile(r"(https?:\S*|www\.\S+|bit\.ly\S*)", re.I)
# MedQuAD answers that are link lists or point at a table that isn't there.
MEDQUAD_SHELL = re.compile(
    r"these resources (address|from medlineplus)|genetic testing registry|"
    r"human phenotype ontology|medlineplus encyclopedia|gene ?review:|"
    r"the table below includes", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")


def words(s):
    return s.split()


def sentences(text):
    return [s.strip() for s in SENT.split(text) if s.strip()]


def repeats(text, n=REPEAT_NGRAM):
    w = text.lower().split()
    grams = collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(grams.values()) if grams else 0


def is_stub(s):
    """Leftover signature bits like "N." or "S. M.": one word, or no real word."""
    return len(words(s)) == 1 or not re.search(r"[a-z]{2}", s)


def strip_greeting(s):
    """Cut greeting words and pleasantry clauses off the front of a sentence.
    Returns what is left ("" if it was all pleasantry)."""
    for _ in range(5):
        t = SALUTE.sub("", s)
        if PLEASANTRY.match(t):
            if "," not in t:
                return ""
            t = t.split(",", 1)[1]
        t = t.lstrip(" ,.;:-")
        if t == s:
            break
        s = t
    return s[:1].upper() + s[1:]


def clean_question(q):
    """Whitespace and URLs only: patients' questions are kept as they are."""
    q = URL.sub("", q)
    return " ".join(q.split())


def clean_answer(a, hits=None):
    """Return the cleaned answer, or "" if nothing usable is left. `hits` (a
    Counter) records which rules touched this answer."""
    hits = hits if hits is not None else collections.Counter()
    a = " ".join(a.split())
    # glued words from the scraped source: "HelloThanks", "RegardsJay"
    a = re.sub(r"\bHiT hank", "Hi. Thank", a)
    a = re.sub(r"\b(Hi|Hello|Hey|Dear|Regards|Thanks)(?=[A-Z])", r"\1. ", a)
    # list numbers glued to the previous sentence ("foods.2. Avoid") hide the
    # sentence breaks; drop the numbers and keep the break
    a = re.sub(r"(?<=[.:!?\s])\d{1,2}[.)]\s*(?=[A-Z]|$)", " ", a)
    a = " ".join(a.split())
    sents = sentences(a)
    n0 = len(sents)

    while sents:
        # a short opening line naming "Chat Doctor" is an introduction
        # ("I am Chat Doctor.", "My name is Chat Doctor.")
        intro = CHATDOCTOR.search(sents[0]) and len(words(sents[0])) < 15
        first = "" if intro or is_stub(sents[0]) else strip_greeting(sents[0])
        if first == sents[0]:
            break
        hits["greeting_removed"] += 1
        if first:
            sents[0] = first
        else:
            sents.pop(0)
    while sents and ((SIGNOFF.search(sents[-1]) and len(words(sents[-1])) < 30)
                     or is_stub(sents[-1])):
        sents.pop()
        hits["signoff_removed"] += 1
    # an unfinished last sentence: cut it
    if sents and not sents[-1].endswith((".", "!", "?", ")", '"')):
        sents.pop()
        hits["unfinished_tail_cut"] += 1
    # a last sentence that announces a list which never came
    while sents and sents[-1].endswith(":"):
        sents.pop()
        hits["dangling_list_cut"] += 1
    kept = []
    for s in sents:
        if CHATDOCTOR.search(s) or URL.search(s):
            hits["chatdoctor_or_url_sentence_removed"] += 1
            continue
        kept.append(s)
    if len(kept) < n0:
        hits["changed"] += 1
    return " ".join(kept)


def drop_reason(q, a_raw, a):
    if MEDQUAD_SHELL.search(a_raw):
        return "medquad_link_list_or_missing_table"
    if repeats(a_raw) > REPEAT_MAX:
        return "repetition_loop"
    if repeats(q) > REPEAT_MAX:
        return "question_repetition_loop"
    if len(words(q)) < MIN_QUESTION_WORDS:
        return "question_too_short"
    if len(words(a)) < MIN_ANSWER_WORDS:
        return "answer_too_short_after_cleaning"
    return None


def norm(s):
    return " ".join(s.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(OUT, "synthetic_qa.jsonl"))
    ap.add_argument("--output", default=os.path.join(OUT, "synthetic_qa_clean.jsonl"))
    ap.add_argument("--report", default=os.path.join(OUT, "clean_qa_report.json"))
    ap.add_argument("--examples", default=os.path.join(OUT, "clean_qa_examples.txt"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    touched, dropped = collections.Counter(), collections.Counter()
    ex_touch, ex_drop = collections.defaultdict(list), collections.defaultdict(list)
    seen_q, seen_pair = set(), set()
    kept = []
    for r in rows:
        q = clean_question(r["question"])
        hits = collections.Counter()
        a = clean_answer(r["answer"], hits)
        why = drop_reason(q, r["answer"], a)
        if why is None:
            if (norm(q), norm(a)) in seen_pair:
                why = "duplicate_pair"
            elif norm(q) in seen_q:
                why = "duplicate_question"
        if why:
            dropped[why] += 1
            if len(ex_drop[why]) < 200:
                ex_drop[why].append((r, a))
            continue
        seen_q.add(norm(q))
        seen_pair.add((norm(q), norm(a)))
        for k in hits:
            touched[k] += 1
            if len(ex_touch[k]) < 200:
                ex_touch[k].append((r, a))
        kept.append({"question": q, "answer": a})

    with open(args.output, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    aw = sorted(len(words(r["answer"])) for r in kept)
    report = {
        "input_rows": len(rows), "kept_rows": len(kept),
        "kept_share": round(len(kept) / len(rows), 3),
        "dropped": dict(dropped.most_common()),
        "rows_changed_but_kept": dict(touched.most_common()),
        "kept_answer_words_median": aw[len(aw) // 2] if aw else 0,
        "kept_answers_still_mentioning_chat_doctor":
            sum(bool(CHATDOCTOR.search(r["answer"])) for r in kept),
        "kept_answers_starting_hi_hello":
            sum(r["answer"].lower().startswith(("hi", "hello", "dear")) for r in kept),
        "kept_answers_with_url": sum(bool(URL.search(r["answer"])) for r in kept),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    rnd = random.Random(313)
    with open(args.examples, "w", encoding="utf-8") as f:
        for title, group in [("DROPPED", ex_drop), ("CLEANED AND KEPT", ex_touch)]:
            for rule, exs in group.items():
                f.write(f"\n{'=' * 70}\n{title}: {rule}\n{'=' * 70}\n")
                for r, a in rnd.sample(exs, min(4, len(exs))):
                    f.write(f"Q:      {r['question'][:300]}\nBEFORE: {r['answer']}\n"
                            f"AFTER:  {a}\n\n")

    print(f"kept {len(kept):,} of {len(rows):,} ({report['kept_share']:.1%})")
    print("dropped:")
    for k, v in dropped.most_common():
        print(f"  {v:>7,}  {k}")
    print("cleaned and kept:")
    for k, v in touched.most_common():
        print(f"  {v:>7,}  {k}")
    for k in ["kept_answer_words_median", "kept_answers_still_mentioning_chat_doctor",
              "kept_answers_starting_hi_hello", "kept_answers_with_url"]:
        print(f"{k}: {report[k]:,}")
    print(f"wrote {args.output}\n      {args.report}\n      {args.examples}")


if __name__ == "__main__":
    main()
