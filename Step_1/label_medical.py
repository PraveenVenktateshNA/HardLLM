"""Re-label is_medical from embeddings instead of a keyword list.

Why this exists
---------------
The original label in build_pool.py counted words from a hand-written list and
called a row medical if two or more distinct terms appeared. Measured against
the pool it catches only 28% of medical-source rows: patients write "I have a
rash on my left leg at the ankle", which contains none of the listed terms,
while a Yelp review saying "the doctor was so nice and the nurse checked in"
contains two and gets flagged. The list is the problem, not the data.

What replaces it
----------------
A logistic regression over the cached MiniLM embeddings. Positives are rows
from the medical sources, negatives are rows from the distractor sources, and
the classifier is fit on half the pool and calibrated on the other half so the
reported numbers are held-out, not fitted.

This is a *content* label even though it is trained on *source*: two rows that
read alike land near each other in embedding space regardless of which file
they came from, so a genuinely medical Yelp review or SQuAD passage scores
high. That is the intended behaviour -- selection precision in Steps 2/3 must
be scored against what a row is about.

Nothing in the training path reads this field (data.py uses only 'text' and
'text_id'), so re-running this does not invalidate the clustering.

    python HardLLM/Step_1/label_medical.py
"""
import collections
import json
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POOL = os.path.join(ROOT, "pool", "pool_with_clusters.jsonl")
EMB_CACHE = os.path.join(ROOT, "pool", "embeddings.npy")
SEED = 313

MEDICAL_SOURCES = {
    "lavita/ChatDoctor-HealthCareMagic-100k",
    "lavita/MedQuAD",
    "lewtun/drug-reviews",
}

random.seed(SEED)
rng = np.random.default_rng(SEED)

# -------------------------------------------------------------------- load

print("loading pool...")
rows = [json.loads(l) for l in open(POOL)]
emb = np.load(EMB_CACHE)
assert len(rows) == len(emb), (
    f"{len(rows):,} rows but {len(emb):,} embeddings -- embeddings.npy is stale "
    "for this pool file. Re-run cluster_pool.py."
)
print(f"  {len(rows):,} rows, embeddings {emb.shape}")

# Cosine geometry: unit-normalise so the linear boundary is an angle, which is
# the space MiniLM was actually trained to be meaningful in.
X = normalize(emb.astype(np.float32))
y = np.array([r["source"] in MEDICAL_SOURCES for r in rows])
print(f"  medical-source rows: {y.sum():,} ({100 * y.mean():.1f}%)")

# ----------------------------------------------------------------- fit

# Half to fit, half to calibrate and report on. Splitting by a permutation
# rather than by source keeps both halves representative of the whole pool.
perm = rng.permutation(len(rows))
fit_idx, held_idx = perm[: len(rows) // 2], perm[len(rows) // 2:]

print("fitting classifier (roughly a minute)...")
clf = LogisticRegression(
    max_iter=2000,
    C=1.0,
    class_weight="balanced",   # medical is 9% of the pool; without this the
                               # boundary collapses toward "never medical"
)
clf.fit(X[fit_idx], y[fit_idx])

scores = clf.predict_proba(X)[:, 1]

# ------------------------------------------------------- pick a threshold

# Chosen on the held-out half only. F1 against the source label, which is the
# best proxy available -- it is noisy in the direction of calling genuinely
# medical distractor rows "wrong", so the true precision is a little higher
# than reported here.
yh, sh = y[held_idx], scores[held_idx]
print("\n  thr   recall   prec     F1   flagged")
best = (0, 0.5)
for thr in np.arange(0.30, 0.96, 0.05):
    pred = sh >= thr
    tp = (pred & yh).sum()
    rec = tp / yh.sum()
    prec = tp / max(pred.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mark = ""
    if f1 > best[0]:
        best, mark = (f1, thr), " <-"
    print(f"  {thr:.2f}  {rec:6.3f}  {prec:6.3f}  {f1:6.3f}  {pred.sum():>7,}{mark}")
THR = best[1]
print(f"\nchosen threshold: {THR:.2f}  (best held-out F1 = {best[0]:.3f})")

# ---------------------------------------------------------------- apply

# On a re-run the keyword answer already lives under its own key.
old_kw = np.array([bool(r.get("is_medical_kw", r.get("is_medical", False)))
                   for r in rows])
new = scores >= THR

for r, s, n in zip(rows, scores, new):
    # Preserve the ORIGINAL keyword answer across re-runs. Without this guard a
    # second run would overwrite it with the first run's embedding label and the
    # before/after comparison would silently become before/before.
    if "is_medical_kw" not in r:
        r["is_medical_kw"] = bool(r.get("is_medical", False))
    r.pop("is_medical", None)
    r["med_score"] = round(float(s), 4)
    r["is_medical"] = bool(n)

tmp = POOL + ".tmp"
with open(tmp, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
os.replace(tmp, POOL)      # atomic; a crash mid-write cannot truncate the pool
print(f"rewrote {POOL}")

# ---------------------------------------------------------------- report

print("\n" + "=" * 72)
print("LABEL QUALITY  (recall = fraction of that source's rows flagged medical)")
per = collections.defaultdict(lambda: [0, 0, 0])
for r, o, n in zip(rows, old_kw, new):
    p = per[(r["role"], r["source"])]
    p[0] += 1
    p[1] += int(o)
    p[2] += int(n)
print(f"\n{'role':<9}{'source':<44}{'n':>8}{'keyword':>9}{'embed':>8}")
for (role, src), (n, o, m) in sorted(per.items()):
    print(f"{role:<9}{src:<44}{n:>8,}{100*o/n:>8.1f}%{100*m/n:>7.1f}%")

med = y
print(f"\nmedical-source recall : keyword {100*old_kw[med].mean():5.1f}%"
      f"   ->  embedding {100*new[med].mean():5.1f}%")
print(f"total flagged medical : keyword {old_kw.sum():>7,}"
      f"      ->  embedding {new.sum():>7,}")

# Rows the embedding label promotes out of the distractor slices. These are the
# interesting ones -- if they read as medical, the label is doing its job.
promoted = [r for r in rows
            if r["is_medical"] and r["source"] not in MEDICAL_SOURCES]
print(f"\nnon-medical-source rows now flagged: {len(promoted):,} "
      f"({100*len(promoted)/len(rows):.1f}% of pool)")
print("  by role:", dict(collections.Counter(r["role"] for r in promoted)))
print("\n--- sample: distractor rows the embedding label calls medical ---")
for r in random.sample(promoted, min(8, len(promoted))):
    print(f"[{r['role']:<8} {r['med_score']:.2f}] {r['text'][:120]!r}")

missed = [r for r in rows
          if not r["is_medical"] and r["source"] in MEDICAL_SOURCES]
print(f"\n--- sample: medical-source rows still missed ({len(missed):,}) ---")
for r in random.sample(missed, min(6, len(missed))):
    print(f"[{r['source'].split('/')[-1]:<12} {r['med_score']:.2f}] {r['text'][:120]!r}")

print("\nfields now on every pool row: is_medical (embedding), "
      "is_medical_kw (old), med_score")
print("Steps 2/3 should score selection precision against is_medical.")
