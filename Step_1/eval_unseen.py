"""Measure the retrieval model on records it has never seen.

Why this exists
---------------
Training reported Hits@1 0.775 / Hits@10 0.973, but that eval set was
train_dataset[:1000] -- text the model was trained on. DSI indexing IS
memorisation, so that number is the right one for indexing. It says nothing
about Step 2, which feeds the index 60,000 private medical records the model
has never seen.

The gap between the two is the whole question. If unseen performance holds up,
the model learned the clustering function. If it collapses, it memorised, and
QuDP's selection quality rests on something weaker than the paper implies.

Ground truth for unseen text
----------------------------
Private records were never clustered, so they carry no text_id. We reconstruct
the assignment function cluster_pool.py used: IncrementalPCA(48) fitted on the
cached pool embeddings -- deterministic, no random state, so refitting
reproduces it -- then nearest cluster centroid in that reduced space, where a
centroid is the mean of its members. That is what k-means would assign, so it
is the correct target for "did the model learn the clustering rule".

    python HardLLM/Step_1/eval_unseen.py --n 500
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOL = os.path.join(ROOT, "pool", "pool_with_clusters.jsonl")
EMB_CACHE = os.path.join(ROOT, "pool", "embeddings.npy")
PRIVATE = os.path.join(ROOT, "pool", "private_medical.jsonl")
DEFAULT_CKPT = os.path.join(HERE, "models", "checkpoint-4000-cooldown")
CENTROID_CACHE = os.path.join(HERE, "models", "centroids_pca48.npz")

PCA_DIMS = 48          # must match cluster_pool.py
SEED = 313
SPIECE_UNDERLINE = "▁"


def build_int_token_ids(tokenizer):
    ids = []
    for token, tid in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                ids.append(tid)
        if token == SPIECE_UNDERLINE:
            ids.append(tid)
        elif token.isdigit():
            ids.append(tid)
    ids.append(tokenizer.eos_token_id)
    return ids


def load_centroids():
    """(ipca, centroids, cluster_ids, medical_frac) -- cached after first run."""
    from sklearn.decomposition import IncrementalPCA

    labels = np.array([json.loads(l)["text_id"] for l in open(POOL)])
    med = np.array([bool(json.loads(l).get("is_medical")) for l in open(POOL)])

    if os.path.exists(CENTROID_CACHE):
        z = np.load(CENTROID_CACHE, allow_pickle=True)
        ipca = IncrementalPCA(n_components=PCA_DIMS)
        ipca.components_ = z["components"]
        ipca.mean_ = z["mean"]
        # sklearn's transform() reaches for these even when whiten=False.
        ipca.explained_variance_ = z["explained_variance"]
        ipca.explained_variance_ratio_ = z["explained_variance_ratio"]
        ipca.n_samples_seen_ = int(z["n_samples_seen"])
        return ipca, z["centroids"], z["cluster_ids"], z["medical_frac"]

    print("fitting IncrementalPCA(48) on pool embeddings (~1 min)...")
    emb = np.load(EMB_CACHE)
    assert len(emb) == len(labels), "embeddings.npy is stale for this pool"
    ipca = IncrementalPCA(n_components=PCA_DIMS, batch_size=2000)
    for i in range(0, len(emb), 2000):
        ipca.partial_fit(emb[i:i + 2000])
    reduced = ipca.transform(emb)
    print(f"  explained variance: {ipca.explained_variance_ratio_.sum():.3f}")

    cluster_ids = np.unique(labels)
    centroids = np.stack([reduced[labels == c].mean(axis=0) for c in cluster_ids])
    medical_frac = np.array([med[labels == c].mean() for c in cluster_ids])

    os.makedirs(os.path.dirname(CENTROID_CACHE), exist_ok=True)
    np.savez(CENTROID_CACHE, components=ipca.components_, mean=ipca.mean_,
             explained_variance=ipca.explained_variance_,
             explained_variance_ratio=ipca.explained_variance_ratio_,
             n_samples_seen=ipca.n_samples_seen_,
             centroids=centroids, cluster_ids=cluster_ids,
             medical_frac=medical_frac)
    return ipca, centroids, cluster_ids, medical_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="private records to score")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ipca, centroids, cluster_ids, medical_frac = load_centroids()
    print(f"{len(cluster_ids):,} clusters, "
          f"{(medical_frac > 0.5).sum()} majority-medical")

    rng = np.random.default_rng(SEED)
    private = [json.loads(l)["text"] for l in open(PRIVATE)]
    idx = rng.choice(len(private), size=min(args.n, len(private)), replace=False)
    texts = [private[i] for i in idx]
    print(f"scoring {len(texts):,} unseen private medical records")

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    q_emb = st.encode(texts, batch_size=256, convert_to_numpy=True,
                      show_progress_bar=True)
    q_red = ipca.transform(q_emb)
    # Nearest centroid == what k-means would have assigned.
    d = ((q_red[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    truth = cluster_ids[d.argmin(axis=1)]

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(device).eval()
    int_token_ids = build_int_token_ids(tokenizer)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return int_token_ids

    valid = set(int(c) for c in cluster_ids)
    hits1 = hits5 = hits10 = invalid = 0
    med_top1 = 0
    preds_all = []

    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation="only_first", max_length=args.max_length)
        with torch.no_grad():
            beams = model.generate(
                enc.input_ids.to(device),
                attention_mask=enc.attention_mask.to(device),
                max_length=20,
                num_beams=args.num_beams,
                prefix_allowed_tokens_fn=restrict_decode_vocab,
                num_return_sequences=10,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(beams, skip_special_tokens=True)
        for j in range(len(batch)):
            ranked = [s.strip() for s in decoded[j * 10:(j + 1) * 10]]
            t = str(truth[start + j])
            if ranked and not (ranked[0].isdigit() and int(ranked[0]) in valid):
                invalid += 1
            if ranked[:1] == [t]:
                hits1 += 1
            if t in ranked[:5]:
                hits5 += 1
            if t in ranked[:10]:
                hits10 += 1
            if ranked and ranked[0].isdigit() and int(ranked[0]) in valid:
                k = int(np.where(cluster_ids == int(ranked[0]))[0][0])
                med_top1 += medical_frac[k] > 0.5
            preds_all.append((t, ranked))
        done = min(start + args.batch_size, len(texts))
        print(f"\r  {done}/{len(texts)}  "
              f"Hits@1 {hits1/done:.3f}  Hits@10 {hits10/done:.3f}", end="")

    n = len(texts)
    print("\n" + "=" * 66)
    print("UNSEEN private medical records vs held-in training eval")
    print(f"{'':22}{'unseen':>10}{'held-in':>10}")
    print(f"{'Hits@1':22}{hits1/n:>10.3f}{0.775:>10.3f}")
    print(f"{'Hits@5 (Step 2 uses)':22}{hits5/n:>10.3f}{'--':>10}")
    print(f"{'Hits@10':22}{hits10/n:>10.3f}{0.973:>10.3f}")
    print(f"\nrandom baseline: {1/len(cluster_ids):.5f}")
    print(f"top-1 not a real cluster id: {invalid}/{n}")
    print(f"top-1 lands in a majority-MEDICAL cluster: {med_top1}/{n} "
          f"({100*med_top1/n:.1f}%)")
    print("\nThat last line is what Step 2 actually depends on: a private")
    print("medical record should route to medical clusters even when the")
    print("exact cluster id is wrong.")

    out = os.path.join(HERE, "models", "unseen_eval.json")
    with open(out, "w") as f:
        json.dump({"n": n, "hits1": hits1 / n, "hits5": hits5 / n,
                   "hits10": hits10 / n, "invalid": invalid,
                   "medical_top1": med_top1 / n,
                   "preds": preds_all[:100]}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
