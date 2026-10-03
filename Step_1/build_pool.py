"""Build the public pool, the private sets, and the drug RAG index.

Design notes
------------
* Every row is normalised to {text, target, source, role, is_medical}.
    text       - what the retrieval model embeds and clusters (Step 1)
    target     - downstream supervision where the source provides it, else None
    source     - the HF dataset it came from (provenance)
    role       - the pool slice it was sampled for (reviews/news/medical/...)
    is_medical - CONTENT label, computed over every row regardless of source

* is_medical is deliberately NOT the same as role == "medical". Medical text
  occurs naturally in Yelp, AG News, SQuAD and DialogSum, and selection
  precision must be scored against what a row is ABOUT, not which file it
  came from. Both numbers get reported so the difference is visible.

* The medical slice is held to ~9% of the pool so selection has real work to
  do. The pool composition is fixed and identical for every private domain.
"""
import json
import os
import random
import re

from datasets import load_dataset

# Data lives at the repo root, not in Step_1 -- Steps 2/3/4 read the same files.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "hf_cache")
OUT = os.path.join(ROOT, "pool")
SEED = 313                       # same seed the paper's code uses
random.seed(SEED)

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- cleaning

HTML_ENT = re.compile(r"&#\d+;|&amp;|&quot;|&lt;|&gt;|&nbsp;")
WS = re.compile(r"\s+")


def clean(s):
    """Strip the artifacts we found during inspection."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\b", " ")            # ag_news escape junk
    s = HTML_ENT.sub(" ", s)             # drug-reviews html entities
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]                      # drug reviews are quote-wrapped
    return WS.sub(" ", s).strip()


# ------------------------------------------------- content-based labelling

MEDICAL_TERMS = re.compile(
    r"\b(doctor|physician|nurse|patient|hospital|clinic|dentist|surgeon|"
    r"symptom|diagnos\w*|prescri\w*|medication|medicine|drug|dose|dosage|"
    r"tablet|pill|therapy|treatment|disease|illness|infection|fever|pain|"
    r"nausea|vomit\w*|antibiotic|vaccin\w*|pharmac\w*|mg\b|surgery|"
    r"chronic|acute|allerg\w*|blood pressure|cholesterol)\b",
    re.I,
)


def is_medical(text):
    """Two or more distinct medical terms -- one alone is too noisy
    ('pain' appears in plenty of restaurant reviews)."""
    return len(set(m.lower() for m in MEDICAL_TERMS.findall(text))) >= 2


# ------------------------------------------------------------ row builders

def rows_from(name, config, role, text_col, target_col=None, split="train",
              limit=None):
    """Subsample BEFORE materialising, and read columns in bulk.

    Row-by-row iteration over a HF dataset builds a Python dict per row and is
    ~50x slower than columnar access, so we shuffle/select first and only touch
    the rows we actually keep.
    """
    ds = load_dataset(name, config, cache_dir=CACHE)[split]
    if limit is not None and limit < len(ds):
        # oversample 1.3x so short-text drops don't leave us under target
        ds = ds.shuffle(seed=SEED).select(range(min(len(ds), int(limit * 1.3))))
    texts = ds[text_col]
    targets = ds[target_col] if target_col else [None] * len(texts)
    out = []
    for t, g in zip(texts, targets):
        t = clean(t)
        if len(t) < 20:                  # drop empties and stubs
            continue
        out.append({
            "text": t,
            "target": clean(g) if target_col else None,
            "source": name,
            "role": role,
        })
        if limit is not None and len(out) >= limit:
            break
    print(f"  {name:44s} -> {len(out):>7,}")
    return out


def take(rows, n):
    random.shuffle(rows)
    return rows[:n]


# ------------------------------------------------------------------ build

print("loading sources...")

# --- medical (split three ways; ChatDoctor also yields the private set) ---
# 75k pulled: 60k private + 15k pool. No overlap between the two.
chat = rows_from("lavita/ChatDoctor-HealthCareMagic-100k", None, "medical",
                 "input", "output", limit=75_000)
random.shuffle(chat)
private_med = chat[:60_000]              # held out entirely from the pool
chat_pool = chat[60_000:]                # 15k into the public pool

drug_pool = rows_from("lewtun/drug-reviews", None, "medical",
                      "review", limit=15_000)
medq = rows_from("lavita/MedQuAD", None, "medical",
                 "question", "answer", limit=10_000)

# --- distractor slices ---------------------------------------------------
amazon = rows_from("SetFit/amazon_reviews_multi_en", None, "reviews",
                   "text", limit=100_000)
yelp = rows_from("Yelp/yelp_review_full", None, "reviews",
                 "text", limit=80_000)
news = rows_from("fancyzhx/ag_news", None, "news",
                 "text", limit=100_000)
squad = rows_from("rajpurkar/squad", None, "qa",
                  "context", limit=50_000)
legal = rows_from("coastalcph/lex_glue", "ledgar", "legal",
                  "text", limit=40_000)
fiqa = rows_from("LLukas22/fiqa", None, "finance",
                 "question", "answer")
phrase = rows_from("warwickai/financial_phrasebank_mirror", None, "finance",
                   "sentence")
# NOTE: DialogSum's medical rows are kept, not filtered -- natural overlap is
# realistic. They are caught by the content label instead.
dialog = rows_from("knkarthick/dialogsum", None, "dialogue",
                   "dialogue", "summary")

pool = (chat_pool + drug_pool + medq + amazon + yelp + news
        + squad + legal + fiqa + phrase + dialog)
random.shuffle(pool)

# --- content labelling over EVERY row, pool and private alike ------------
for r in pool:
    r["is_medical"] = is_medical(r["text"])
for r in private_med:
    r["is_medical"] = is_medical(r["text"])

# --- extra private domains for the generalisation table ------------------
private_legal = rows_from("coastalcph/lex_glue", "ledgar", "legal",
                          "text", split="test", limit=5_000)
private_fin = rows_from("LLukas22/fiqa", None, "finance",
                        "question", "answer", split="test", limit=2_000)
for r in private_legal + private_fin:
    r["is_medical"] = is_medical(r["text"])

# --- drug RAG index (the rows NOT used for training) ---------------------
drug_ds = load_dataset("lewtun/drug-reviews", cache_dir=CACHE)["train"]
index = []
for cond, drug, rev, rate, useful in zip(
        drug_ds["condition"], drug_ds["drugName"], drug_ds["review"],
        drug_ds["rating"], drug_ds["usefulCount"]):
    if not cond or "</span>" in str(cond) or not drug:
        continue                          # 1.1% junk found during inspection
    index.append({
        "drug": clean(drug),
        "condition": clean(cond),
        "review": clean(rev),
        "rating": rate,
        "useful": useful,
    })

# ------------------------------------------------------------------ write

def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


n_pool = dump(pool, f"{OUT}/pool_public.jsonl")
n_priv = dump(private_med, f"{OUT}/private_medical.jsonl")
dump(private_legal, f"{OUT}/private_legal.jsonl")
dump(private_fin, f"{OUT}/private_finance.jsonl")
n_idx = dump(index, f"{OUT}/drug_index.jsonl")

# ----------------------------------------------------------------- report

print(f"\npool: {n_pool:,} rows -> {OUT}/pool_public.jsonl")
print(f"{'role':<10} {'rows':>8} {'share':>7}   {'is_medical':>10}")
by_role = {}
for r in pool:
    a, b = by_role.setdefault(r["role"], [0, 0])
    by_role[r["role"]] = [a + 1, b + int(r["is_medical"])]
for role, (n, m) in sorted(by_role.items(), key=lambda x: -x[1][0]):
    print(f"{role:<10} {n:>8,} {100*n/n_pool:>6.1f}%   {m:>10,}")

src_med = sum(1 for r in pool if r["role"] == "medical")
con_med = sum(1 for r in pool if r["is_medical"])
print(f"\nmedical by SOURCE : {src_med:,} ({100*src_med/n_pool:.1f}%)")
print(f"medical by CONTENT: {con_med:,} ({100*con_med/n_pool:.1f}%)")
print("  ^ selection precision should be scored against the CONTENT figure")

print(f"\nprivate medical : {n_priv:,} -> {OUT}/private_medical.jsonl")
print(f"private legal   : {len(private_legal):,}")
print(f"private finance : {len(private_fin):,}")
print(f"drug RAG index  : {n_idx:,} -> {OUT}/drug_index.jsonl")
print(f"  distinct conditions: {len({r['condition'] for r in index}):,}")
print(f"  distinct drugs     : {len({r['drug'] for r in index}):,}")
