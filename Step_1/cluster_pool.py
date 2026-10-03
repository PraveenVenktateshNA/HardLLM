"""Cluster the public pool and assign each row a text_id.

Adapted from HardLLM/Step_1/cluster_sst2.py for the heterogeneous pool.
Changes from the original, and why:

  * reads ./pool/pool_public.jsonl with field 'text' (was ./sst2/... 'sentence')
  * embeddings are cached to ./pool/embeddings.npy so the is_medical labeller
    and any re-clustering reuse them instead of re-encoding 440k rows
  * CLUSTER_DIVISOR is 200, not 100 -- //100 gives 4,417 clusters, a large
    output space for mT5 to learn as digit sequences. Start coarse.
  * MiniBatchKMeans batch_size is 10,000, not 50. A batch smaller than
    n_clusters means most centroids get no update on most batches, so the
    original setting would not converge at this scale.
  * init_size set explicitly (sklearn requires init_size >= n_clusters)
  * prints a cluster-quality report before you spend A100 hours on Step 1
"""
import json
import os
import collections

import numpy as np

# Data lives at the repo root, not in Step_1 -- Steps 2/3/4 read the same
# files. Resolving from __file__ means this runs from any working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POOL = os.path.join(ROOT, "pool", "pool_public.jsonl")
OUT = os.path.join(ROOT, "pool", "pool_with_clusters.jsonl")
EMB_CACHE = os.path.join(ROOT, "pool", "embeddings.npy")
CLUSTER_DIVISOR = 200
PCA_DIMS = 48
SEED = 313

# ------------------------------------------------------------------- load

print("loading pool...")
rows = [json.loads(l) for l in open(POOL)]
texts = [r["text"] for r in rows]
print(f"  {len(rows):,} rows")

# -------------------------------------------------------------- embed

if os.path.exists(EMB_CACHE):
    print(f"loading cached embeddings from {EMB_CACHE}")
    embeddings = np.load(EMB_CACHE)
    assert len(embeddings) == len(rows), "cache is stale -- delete it and re-run"
else:
    from sentence_transformers import SentenceTransformer
    import torch

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"embedding on {device} (this is the slow part, ~10-20 min on M1)")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device=device)
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                              convert_to_numpy=True)
    np.save(EMB_CACHE, embeddings)
    print(f"  cached to {EMB_CACHE}  shape={embeddings.shape}")

# ---------------------------------------------------------------- reduce

from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans

print(f"reducing {embeddings.shape[1]} -> {PCA_DIMS} dims...")
ipca = IncrementalPCA(n_components=PCA_DIMS, batch_size=2000)
for i in range(0, len(embeddings), 2000):
    ipca.partial_fit(embeddings[i:i + 2000])
reduced = ipca.transform(embeddings)
print(f"  explained variance: {ipca.explained_variance_ratio_.sum():.3f}")

# --------------------------------------------------------------- cluster

n_clusters = len(rows) // CLUSTER_DIVISOR
print(f"clustering into {n_clusters:,} clusters...")

kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    init="k-means++",
    max_iter=100,
    batch_size=10_000,            # must be >= n_clusters to converge
    init_size=3 * n_clusters,     # sklearn requires init_size >= n_clusters
    n_init=3,
    max_no_improvement=20,
    reassignment_ratio=0.01,
    random_state=SEED,
    verbose=0,
)
labels = kmeans.fit_predict(reduced)

for r, lab in zip(rows, labels):
    r["text_id"] = int(lab)

# ----------------------------------------------------------------- write

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {OUT}")

# ---------------------------------------------------------------- report

print("\n" + "=" * 62)
print("CLUSTER QUALITY")
sizes = collections.Counter(labels)
s = np.array(sorted(sizes.values()))
print(f"  clusters      : {len(sizes):,}")
print(f"  size min/med/max : {s.min()} / {int(np.median(s))} / {s.max()}")
print(f"  singletons    : {(s == 1).sum():,}")
print(f"  clusters < 10 : {(s < 10).sum():,}")

# role purity -- how single-domain is a typical cluster?
by_cluster = collections.defaultdict(collections.Counter)
for r in rows:
    by_cluster[r["text_id"]][r["role"]] += 1
purity = np.array([c.most_common(1)[0][1] / sum(c.values())
                   for c in by_cluster.values()])
print(f"\n  mean role purity : {purity.mean():.3f}")
print("  (1.0 = every cluster is one domain; ~0.4 = domains are smeared)")

# the diagnostic that matters: do medical rows concentrate?
med_clusters = {cid for cid, c in by_cluster.items()
                if c["medical"] / sum(c.values()) > 0.5}
med_rows = [r for r in rows if r["role"] == "medical"]
in_med = sum(1 for r in med_rows if r["text_id"] in med_clusters)
print(f"\n  majority-medical clusters : {len(med_clusters):,} / {len(sizes):,}")
print(f"  medical rows landing in them: {in_med:,}/{len(med_rows):,} "
      f"({100 * in_med / len(med_rows):.1f}%)")
print("\n  ^ THIS is the number that decides whether Step 1 is worth running.")
print("    High (>70%) = medical content is separable, selection can work.")
print("    Low  (<40%) = medical is smeared across clusters; raise")
print("                  CLUSTER_DIVISOR or revisit the pool before training.")
