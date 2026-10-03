"""Download the public-pool candidate datasets and report what landed.

Safe to re-run. HuggingFace caches by (dataset, config, revision) under
cache_dir, so anything already fetched is reused rather than re-downloaded --
no collisions, no duplicate files. Only the new entries actually hit the network.

Two originals were dropped because datasets>=4 removed loading-script support:
    UCSD26/medical_dialog        -> replaced by lavita/ChatDoctor-HealthCareMagic-100k
    takala/financial_phrasebank  -> replaced by warwickai/financial_phrasebank_mirror
"""
import os
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "hf_cache")

# (hf_id, config, pool_role)
SETS = [
    # --- medical -------------------------------------------------------
    ("lewtun/drug-reviews",                    None,      "medical"),
    ("lavita/MedQuAD",                         None,      "medical"),
    ("lavita/ChatDoctor-HealthCareMagic-100k", None,      "medical"),   # NEW
    # --- news ----------------------------------------------------------
    ("fancyzhx/ag_news",                       None,      "news"),
    # --- general dialogue ----------------------------------------------
    ("knkarthick/dialogsum",                   None,      "dialogue"),
    # --- reviews -------------------------------------------------------
    ("SetFit/amazon_reviews_multi_en",         None,      "reviews"),
    ("Yelp/yelp_review_full",                  None,      "reviews"),
    # --- general QA ----------------------------------------------------
    ("rajpurkar/squad",                        None,      "qa"),
    # --- finance -------------------------------------------------------
    ("warwickai/financial_phrasebank_mirror",  None,      "finance"),   # NEW
    ("LLukas22/fiqa",                          None,      "finance"),
    # --- legal ---------------------------------------------------------
    ("coastalcph/lex_glue",                    "ledgar",  "legal"),
]


def cached_already(hf_id):
    """Best-effort check so the log shows what was reused vs fetched."""
    stem = hf_id.replace("/", "___")
    if not os.path.isdir(CACHE):
        return False
    return any(d.startswith(stem) or d.startswith(hf_id.split("/")[-1])
               for d in os.listdir(CACHE))


ok, failed = [], []

for name, config, role in SETS:
    pre = cached_already(name)
    try:
        ds = load_dataset(name, config, cache_dir=CACHE)
        splits = {k: len(v) for k, v in ds.items()}
        cols = ds[next(iter(ds))].column_names
        tag = "cached" if pre else "FETCHED"
        print(f"OK   {name:42s} [{role:8s}] {tag:7s} {splits}")
        print(f"     columns: {cols}")
        ok.append((name, role, splits, cols))
    except Exception as e:
        msg = str(e).replace("\n", " ")[:150]
        print(f"FAIL {name:42s} [{role:8s}] {type(e).__name__}: {msg}")
        failed.append((name, role, type(e).__name__))

print("\n" + "=" * 74)
print(f"{len(ok)} ok, {len(failed)} failed")
if failed:
    print("\nfailed:")
    for name, role, err in failed:
        print(f"  {name:42s} [{role}] {err}")

print("\nrows by pool role:")
roles = {}
for name, role, splits, _ in ok:
    roles[role] = roles.get(role, 0) + sum(splits.values())
for role, n in sorted(roles.items(), key=lambda x: -x[1]):
    print(f"  {role:10s} {n:>9,}")
print(f"\n  {'TOTAL':10s} {sum(roles.values()):>9,} rows available")
print("\nNote: these are raw availability counts, not pool proportions --")
print("the pool script subsamples these down to the target slice ratios.")
