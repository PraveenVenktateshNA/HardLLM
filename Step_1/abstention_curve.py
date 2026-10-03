"""Measure the abstention operating curve and the latency envelope.

Why this exists
---------------
eval_unseen.py says the model is right 56.4% of the time on the top pick and
97.6% of the time about "is this medical". Neither number is a specification.
A critical-system component is specified by what it does when it is UNSURE.

Beam search returns a length-normalised log-probability per sequence. Exp'd,
that is a confidence in [0,1]. Thresholding it gives a refusal policy:

    confidence >= tau  ->  emit the cluster id
    confidence <  tau  ->  abstain, contribute NOTHING to the histogram

Abstaining is fail-closed. An uncertain record is simply not counted, so
uncertainty costs histogram accuracy and never costs privacy. This script
sweeps tau and reports, at each coverage level, the accuracy conditional on
having answered -- the curve you quote as the component's spec.

It also times both paths on this machine: the DSI decode that produces the
id, and the nearest-centroid computation that defines the ground truth.

    python HardLLM/Step_1/abstention_curve.py --n 500
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from eval_unseen import (HERE, PRIVATE, SEED, build_int_token_ids,
                         load_centroids)

DEFAULT_CKPT = os.path.join(HERE, "models", "checkpoint-4000-cooldown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ipca, centroids, cluster_ids, medical_frac = load_centroids()
    rng = np.random.default_rng(SEED)
    private = [json.loads(l)["text"] for l in open(PRIVATE)]
    idx = rng.choice(len(private), size=min(args.n, len(private)), replace=False)
    texts = [private[i] for i in idx]
    print(f"{len(texts):,} unseen private medical records")

    # ---- baseline path: encode -> reduce -> nearest centroid ---------------
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    t0 = time.perf_counter()
    q_emb = st.encode(texts, batch_size=256, convert_to_numpy=True)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_red = ipca.transform(q_emb)
    d = ((q_red[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    truth = cluster_ids[d.argmin(axis=1)]
    t_assign = time.perf_counter() - t0

    # ---- DSI path ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(device).eval()
    int_token_ids = build_int_token_ids(tokenizer)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return int_token_ids

    valid = set(int(c) for c in cluster_ids)
    conf, correct, med_ok = [], [], []
    batch_times = []

    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation="only_first", max_length=args.max_length)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                enc.input_ids.to(device),
                attention_mask=enc.attention_mask.to(device),
                max_length=20,
                num_beams=args.num_beams,
                prefix_allowed_tokens_fn=restrict_decode_vocab,
                num_return_sequences=1,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        if device == "mps":
            torch.mps.synchronize()
        batch_times.append((time.perf_counter() - t0) / len(batch))

        top = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        # length-normalised log-prob of the returned beam -> confidence in [0,1]
        scores = out.sequences_scores.detach().float().cpu().numpy()
        for j, pred in enumerate(top):
            pred = pred.strip()
            t = str(truth[start + j])
            conf.append(float(np.exp(scores[j])))
            correct.append(pred == t)
            ok = False
            if pred.isdigit() and int(pred) in valid:
                k = int(np.where(cluster_ids == int(pred))[0][0])
                ok = bool(medical_frac[k] > 0.5)
            med_ok.append(ok)
        done = min(start + args.batch_size, len(texts))
        print(f"\r  {done}/{len(texts)}", end="")

    conf = np.array(conf)
    correct = np.array(correct)
    med_ok = np.array(med_ok)
    n = len(conf)

    # ---- the curve ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("ABSTENTION OPERATING CURVE  (fail-closed: below tau, emit nothing)")
    print(f"{'tau':>8}{'coverage':>11}{'Hits@1|ans':>13}{'medical|ans':>14}"
          f"{'abstained':>12}")
    rows = []
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = conf >= tau
        cov = keep.mean()
        if keep.sum() == 0:
            break
        acc = correct[keep].mean()
        med = med_ok[keep].mean()
        rows.append({"tau": tau, "coverage": float(cov),
                     "hits1_given_answered": float(acc),
                     "medical_given_answered": float(med)})
        print(f"{tau:>8.2f}{cov:>11.3f}{acc:>13.3f}{med:>14.3f}"
              f"{1-cov:>12.3f}")

    # ---- latency -----------------------------------------------------------
    bt = np.array(batch_times) * 1000.0
    print("\n" + "=" * 70)
    print(f"LATENCY on {device}  (per query, ms)")
    print(f"{'':28}{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}")
    print(f"{'DSI  mT5 beam-'+str(args.num_beams):28}"
          f"{np.percentile(bt,50):>10.1f}{np.percentile(bt,95):>10.1f}"
          f"{np.percentile(bt,99):>10.1f}{bt.max():>10.1f}")
    base_ms = (t_encode + t_assign) / n * 1000
    print(f"{'baseline  MiniLM+centroid':28}{base_ms:>10.2f}"
          f"{'--':>10}{'--':>10}{'--':>10}")
    print(f"\n  encode {t_encode:.2f}s + assign {t_assign:.2f}s for {n} queries")
    print(f"  speedup: {np.percentile(bt,50)/base_ms:.0f}x")

    out_path = os.path.join(HERE, "models", "abstention_curve.json")
    with open(out_path, "w") as f:
        json.dump({"n": n, "curve": rows,
                   "latency_ms_dsi": {"p50": float(np.percentile(bt, 50)),
                                      "p95": float(np.percentile(bt, 95)),
                                      "p99": float(np.percentile(bt, 99)),
                                      "max": float(bt.max())},
                   "latency_ms_baseline": base_ms,
                   "confidence": conf.tolist(),
                   "correct": correct.tolist()}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
