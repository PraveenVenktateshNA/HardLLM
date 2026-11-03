"""Step 4, extra arm: hardllm-filtered = the HardLLM synthetic pairs (Step 3)
with the answers that do not fit their question removed.

    python Step_4/filter_hardllm.py --embedder <path to all-MiniLM-L6-v2>

Why: analyse_qa_match.py found ~40% of the synthetic answers share no specific
word with their own question (real doctor answers: ~25%), and the demo answers
of the hardllm model ignore the question. This arm asks whether HardLLM works
once those pairs are filtered out.

How: every pair gets a score, the similarity in meaning between its question
and its answer (all-MiniLM-L6-v2, the embedder Step 1 clustered with). The
worse-matching half is dropped and 16,000 pairs are drawn at random from the
better half. Not "the top 16,000": that would favour answers that only repeat
the question. Same size as every other arm, each pair seen once, like hardllm.

Filtering synthetic data is post-processing: it reads no private data and
costs no privacy.

Reads data/hardllm_candidates.jsonl (all 90,155 cleaned Step 3 pairs, none
with a test question). Writes data/train_hardllm-filtered.jsonl,
out/filter_hardllm.json and out/filter_hardllm_examples.txt. Does nothing if
the training file already exists, so a stopped run trains on the same rows.
"""
import argparse
import json
import os
import random
import statistics
import time

import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
ARM = "hardllm-filtered"
N_TRAIN = 16_000
KEEP_SHARE = 0.5
SEED = 313


def read(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name), encoding="utf-8")]


@torch.no_grad()
def embed(model, tok, texts, batch=256):
    """Unit vectors, the way sentence-transformers builds them for this model:
    mean of the token vectors (padding excluded), then normalised."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=256,
                return_tensors="pt").to(dev)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu())
    return torch.cat(out)


def pair_scores(model, tok, rows):
    q = embed(model, tok, [r["question"] for r in rows])
    a = embed(model, tok, [r["answer"] for r in rows])
    return (q * a).sum(-1).tolist(), q, a


def select(rows, scores, n=N_TRAIN, keep_share=KEEP_SHARE, seed=SEED):
    """Drop the worse-matching share, then n at random from the rest."""
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    better = order[:int(len(order) * keep_share)]
    if len(better) < n:
        raise SystemExit(f"only {len(better)} pairs in the better half, need {n}")
    picked = random.Random(seed).sample(better, n)
    return picked, scores[better[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True, help="all-MiniLM-L6-v2 folder")
    args = ap.parse_args()
    target = os.path.join(DATA, f"train_{ARM}.jsonl")
    if os.path.exists(target):
        print(f"[skip] {target} already exists")
        return

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.embedder)
    model = AutoModel.from_pretrained(args.embedder).to(dev).eval()

    rows = read("hardllm_candidates.jsonl")
    print(f"scoring {len(rows):,} synthetic pairs on {dev}")
    scores, q, a = pair_scores(model, tok, rows)
    # references: real doctor answers to the test questions, and synthetic
    # answers put next to the wrong question (what "no fit at all" scores)
    test_scores, _, _ = pair_scores(model, tok, read("test_private.jsonl"))
    perm = list(range(len(rows)))
    random.Random(SEED).shuffle(perm)
    shuffled = (q * a[perm]).sum(-1).tolist()

    picked, cutoff = select(rows, scores)
    with open(target, "w", encoding="utf-8") as f:
        for i in picked:
            f.write(json.dumps(rows[i]) + "\n")

    med = statistics.median
    old = {json.dumps(r, sort_keys=True) for r in read("train_hardllm.jsonl")}
    idx = {json.dumps(r, sort_keys=True): i for i, r in enumerate(rows)}
    old_scores = [scores[idx[k]] for k in old if k in idx]
    report = {
        "candidates": len(rows), "kept_for_training": len(picked),
        "rule": f"drop the worse-matching {1 - KEEP_SHARE:.0%} by question-answer similarity, "
                f"then {N_TRAIN:,} at random from the rest",
        "embedder": "sentence-transformers/all-MiniLM-L6-v2 (mean pooling, normalised)",
        "similarity_cutoff": round(cutoff, 4),
        "median_similarity": {
            "all_synthetic": round(med(scores), 4),
            "hardllm_arm_as_trained": round(med(old_scores), 4),
            "hardllm_filtered_arm": round(med(scores[i] for i in picked), 4),
            "real_test_pairs": round(med(test_scores), 4),
            "synthetic_shuffled_no_fit": round(med(shuffled), 4)},
        "share_below_cutoff": {
            "real_test_pairs": round(sum(s < cutoff for s in test_scores) / len(test_scores), 4),
            "hardllm_arm_as_trained": round(sum(s < cutoff for s in old_scores) / len(old_scores), 4)},
        "answer_words_median": {
            "hardllm_arm_as_trained": med(len(json.loads(k)["answer"].split()) for k in old),
            "hardllm_filtered_arm": med(len(rows[i]["answer"].split()) for i in picked)},
        "seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "filter_hardllm.json"), "w") as f:
        json.dump(report, f, indent=2)

    order = sorted(range(len(rows)), key=lambda i: scores[i])
    rng = random.Random(SEED)
    with open(os.path.join(OUT, "filter_hardllm_examples.txt"), "w", encoding="utf-8") as f:
        for title, ids in [("WORST-MATCHING (dropped)", order[:15]),
                           ("JUST BELOW THE CUTOFF (dropped)", order[len(order) // 2 - 15:len(order) // 2]),
                           ("RANDOM KEPT FOR TRAINING", rng.sample(picked, 15))]:
            f.write(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n")
            for i in ids:
                f.write(f"[{scores[i]:.3f}] Q: {rows[i]['question'][:300]}\n"
                        f"        A: {rows[i]['answer'][:400]}\n\n")

    print(json.dumps(report, indent=2))
    print(f"wrote {target}\n      out/filter_hardllm.json, out/filter_hardllm_examples.txt")


if __name__ == "__main__":
    main()
