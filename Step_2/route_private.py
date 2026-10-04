"""Step 2a: route every private medical record to its top-5 public clusters.

This is the only part of Step 2 that touches private text. What leaves this
script is a per-record list of 5 cluster ids; route_private's output never
contains any private text, and noisy_select.py only ever sees vote counts.

Two routers, same output format, so the rest of Step 2 doesn't care which ran:

  centroid  MiniLM embedding -> PCA(48) -> 5 nearest cluster centroids.
            The exact rule k-means used to build the clusters. ~3 min for 60k.
            This is the dry run.

  dsi       The Step 1 retrieval model, configured the way the paper's
            query_retrival_model.py does it: num_beams=20, top-5 returned.
            ~7 h for 60k on an M1 (measured 428 ms/record, MPS, batch 8).
            Writes as it goes and resumes if interrupted.

    python HardLLM/Step_2/route_private.py --router centroid
    python HardLLM/Step_2/route_private.py --router dsi
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Step_1"))
from eval_unseen import (DEFAULT_CKPT, PRIVATE, build_int_token_ids,  # noqa: E402
                         load_centroids)

OUT = os.path.join(HERE, "out")
TOP_K = 5   # paper: num_return_sequences=5, i.e. each record casts 5 votes


def load_private(limit):
    texts = [json.loads(l)["text"] for l in open(PRIVATE)]
    return texts[:limit] if limit else texts


def route_centroid(texts, device):
    from sentence_transformers import SentenceTransformer
    ipca, centroids, cluster_ids, _ = load_centroids()
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    emb = st.encode(texts, batch_size=256, convert_to_numpy=True,
                    show_progress_bar=True)
    red = ipca.transform(emb)
    routes = []
    for i in range(0, len(red), 4096):
        d = ((red[i:i + 4096, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        top = np.argsort(d, axis=1)[:, :TOP_K]
        routes.extend(cluster_ids[top].tolist())
    return [[int(c) for c in r] for r in routes]


def route_dsi(texts, device, args, partial_path):
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    done = []
    if os.path.exists(partial_path):
        done = [json.loads(l) for l in open(partial_path)]
        print(f"resuming: {len(done):,} records already routed")

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(device).eval()
    int_token_ids = build_int_token_ids(tokenizer)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return int_token_ids

    t0 = time.time()
    start_n = len(done)
    with open(partial_path, "a") as f:
        for start in range(len(done), len(texts), args.batch_size):
            batch = texts[start:start + args.batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation="only_first", max_length=256)
            with torch.no_grad():
                beams = model.generate(
                    enc.input_ids.to(device),
                    attention_mask=enc.attention_mask.to(device),
                    max_length=20,
                    num_beams=20,
                    prefix_allowed_tokens_fn=restrict_decode_vocab,
                    num_return_sequences=TOP_K,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(beams, skip_special_tokens=True)
            for j in range(len(batch)):
                # Keep the raw strings: an invalid id is a measurement, not an error.
                r = [s.strip() for s in decoded[j * TOP_K:(j + 1) * TOP_K]]
                f.write(json.dumps(r) + "\n")
                done.append(r)
            f.flush()
            n = len(done)
            rate = (time.time() - t0) / max(n - start_n, 1)
            eta = rate * (len(texts) - n) / 3600
            print(f"\r  {n:,}/{len(texts):,}  {rate*1000:.0f} ms/record  "
                  f"ETA {eta:.1f} h", end="", flush=True)
    print()

    routes, invalid = [], 0
    for r in done:
        ids = []
        for s in r:
            if s.isdigit():
                ids.append(int(s))
            else:
                invalid += 1
        routes.append(ids)
    print(f"non-numeric beam outputs dropped: {invalid}")
    return routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", choices=["centroid", "dsi"], required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="route only the first N records (0 = all 60,000)")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="dsi only; 8 is fastest on a 16 GB M1, 16 is slower")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    tag = args.router + (f"_n{args.limit}" if args.limit else "")

    texts = load_private(args.limit)
    print(f"router: {args.router}   device: {device}   records: {len(texts):,}")

    t0 = time.time()
    if args.router == "centroid":
        routes = route_centroid(texts, device)
    else:
        routes = route_dsi(texts, device, args,
                           os.path.join(OUT, f"routes_{tag}.partial.jsonl"))
    elapsed = time.time() - t0

    _, _, cluster_ids, medical_frac = load_centroids()
    is_med = {int(c): bool(m > 0.5) for c, m in zip(cluster_ids, medical_frac)}
    valid = set(is_med)
    top1 = [r[0] for r in routes if r]
    out = {
        "router": args.router,
        "records": len(routes),
        "votes_per_record": TOP_K,
        "seconds": round(elapsed, 1),
        "top1_in_medical_cluster": sum(is_med.get(c, False) for c in top1) / max(len(top1), 1),
        "invalid_cluster_ids": sum(c not in valid for r in routes for c in r),
        "routes": routes,
    }
    path = os.path.join(OUT, f"routes_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f)

    votes = np.zeros(len(cluster_ids), dtype=int)
    pos = {int(c): i for i, c in enumerate(cluster_ids)}
    for r in routes:
        for c in r:
            if c in pos:
                votes[pos[c]] += 1
    print("=" * 60)
    print(f"routed {len(routes):,} records in {elapsed/60:.1f} min")
    print(f"top-1 lands in a majority-medical cluster: "
          f"{100*out['top1_in_medical_cluster']:.1f}%")
    print(f"invalid cluster ids: {out['invalid_cluster_ids']}")
    print(f"clusters receiving any vote: {(votes > 0).sum():,} / {len(votes):,}")
    print(f"busiest cluster: {votes.max():,} votes")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
