"""Walk one private message through Step 2 by hand.

Type a patient message; it is treated as one more private record joining the
hospital's 60,000. The script prints every stage of Step 2 for it:

  1. votes      the 5 public topics it matches (route_private.py, centroid router)
  2. counts     the hospital's votes on those topics + this patient's
  3. noise      true count -> released count (noisy_select.py's mechanism)
  4. cutoff     which of its topics survive (released count >= T * sigma)
  5. retrieval  which public rows get picked from them

Each query is a fresh release with fresh noise, so asking the same thing twice
shows the noise moving. Nothing here changes Step 2: it imports its functions.

    python HardLLM/Step_2/query_step2.py                        # interactive
    python HardLLM/Step_2/query_step2.py -q "my knee hurts"     # one shot
    python HardLLM/Step_2/query_step2.py --eps inf              # no noise, to compare
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Step_1"))
from eval_unseen import EMB_CACHE, POOL, load_centroids  # noqa: E402
from noisy_select import paper_weights, sigma_analytic  # noqa: E402

ROUTES = os.path.join(HERE, "out", "routes_centroid.json")
TOP_K = 5
SOURCE_NAMES = {
    "lavita/ChatDoctor-HealthCareMagic-100k": "ChatDoctor",
    "lewtun/drug-reviews": "Drug reviews",
    "lavita/MedQuAD": "MedQuAD",
    "SetFit/amazon_reviews_multi_en": "Amazon",
    "fancyzhx/ag_news": "AG News",
    "Yelp/yelp_review_full": "Yelp",
    "rajpurkar/squad": "SQuAD",
    "coastalcph/lex_glue": "LexGLUE",
    "LLukas22/fiqa": "FiQA",
    "knkarthick/dialogsum": "DialogSum",
    "warwickai/financial_phrasebank_mirror": "PhraseBank",
}


def clip(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", help="single message; omit for interactive")
    ap.add_argument("--eps", type=float, default=0.5,
                    help="privacy level (default 0.5); 'inf' = no noise")
    ap.add_argument("--threshold", type=float, default=3,
                    help="cutoff in units of sigma (default 3)")
    ap.add_argument("--rows", type=int, default=2,
                    help="retrieved public rows to print per topic")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the noise; omit for fresh noise every query")
    args = ap.parse_args()

    print("loading clusters, public pool and the hospital's votes (~15s)...")
    ipca, centroids, cluster_ids, _ = load_centroids()
    pos = {int(c): i for i, c in enumerate(cluster_ids)}

    text, source, medical = [], [], []
    members = [[] for _ in cluster_ids]
    for row, line in enumerate(open(POOL)):
        o = json.loads(line)
        text.append(o["text"])
        source.append(SOURCE_NAMES.get(o["source"], o["source"]))
        medical.append(o.get("role") == "medical")
        members[pos[int(o["text_id"])]].append(row)
    members = [np.array(m) for m in members]
    medical = np.array(medical)
    med_frac = np.array([medical[m].mean() for m in members])

    R = json.load(open(ROUTES))
    hospital = np.zeros(len(cluster_ids), dtype=np.int64)
    for r in R["routes"]:
        for c in r:
            hospital[pos[c]] += 1

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = np.load(EMB_CACHE, mmap_mode="r")

    examples, weights = {}, {}

    def example(k):
        """The public row nearest the topic's centre: its most typical text."""
        if k not in examples:
            m = members[k]
            red = ipca.transform(np.asarray(emb[m], dtype=np.float32))
            order = m[np.argsort(((red - centroids[k]) ** 2).sum(axis=1))]
            short = [r for r in order[:15] if 30 <= len(text[r]) <= 200]
            examples[k] = short[0] if short else order[0]
        return text[examples[k]]

    def weight(k):
        if k not in weights:
            weights[k] = paper_weights([members[k]])[0][0]
        return weights[k]

    n = R["records"] + 1
    delta = 1.0 / n
    no_noise = math.isinf(args.eps)
    sigma = 0.0 if no_noise else sigma_analytic(args.eps, delta, math.sqrt(TOP_K))
    cutoff = args.threshold * sigma
    rng = np.random.default_rng(args.seed)

    def run(msg):
        # 1. votes
        red = ipca.transform(st.encode([msg], convert_to_numpy=True))
        top = np.argsort(((red[0] - centroids) ** 2).sum(axis=1))[:TOP_K]
        you = np.zeros(len(cluster_ids), dtype=np.int64)
        you[top] = 1
        true = hospital + you

        # 3. one full release, over all 2,208 topics
        noise = rng.normal(0, sigma, len(true))
        released = np.maximum(0, np.rint(true + noise)).astype(np.int64)
        kept = released >= max(cutoff, 1)

        bar = "-" * 78
        print(f"\n{bar}\nPRIVATE MESSAGE (stays inside the hospital)\n  {msg!r}\n{bar}")

        print("\n1. VOTES  the message votes for the 5 public topics it looks most like\n")
        for rank, k in enumerate(top, 1):
            print(f"  {rank}. topic {int(cluster_ids[k]):>4}   {len(members[k]):>3} public rows, "
                  f"{med_frac[k]:.0%} medical")
            print(f"     typical text: {clip(example(k), 100)!r}")

        print(f"\n2. COUNTS  votes per topic from all {n:,} patients "
              f"(inside the hospital, never released)\n")
        print(f"  {'topic':>7}  {'other patients':>15}  {'this patient':>13}  {'true count':>11}")
        for k in top:
            print(f"  {int(cluster_ids[k]):>7}  {hospital[k]:>15,}  {'+1':>13}  {true[k]:>11,}")

        if no_noise:
            print("\n3. NOISE  off (--eps inf): the true counts are released as they are\n")
        else:
            print(f"\n3. NOISE  privacy level eps = {args.eps:g}, delta = 1/{n:,}, "
                  f"noise size sigma = {sigma:.1f}\n")
        print(f"  {'topic':>7}  {'true count':>11}  {'noise':>8}  {'released':>9}")
        for k in top:
            print(f"  {int(cluster_ids[k]):>7}  {true[k]:>11,}  {noise[k]:>+8.1f}  {released[k]:>9,}")
        print(f"\n  Only the released numbers leave the hospital: {len(true):,} of them, "
              f"one per topic.")
        print(f"  First 10 as they'd be published: {', '.join(str(x) for x in released[:10])}, ...")
        k0 = top[0]
        if no_noise:
            print(f"\n  Can anyone tell this patient is in? YES. Topic {int(cluster_ids[k0])} reads "
                  f"{hospital[k0]:,} without them and {true[k0]:,} with them.")
        else:
            lo, hi = 2 * sigma, 2 * sigma
            print(f"\n  Can anyone tell this patient is in? Topic {int(cluster_ids[k0])} would be "
                  f"released as roughly")
            print(f"    without them: {hospital[k0] - lo:,.0f} to {hospital[k0] + hi:,.0f}")
            print(f"    with them:    {true[k0] - lo:,.0f} to {true[k0] + hi:,.0f}")
            print(f"  Their 1 vote is buried in noise of about ±{sigma:.0f}, so the two "
                  f"cases look the same.")

        print(f"\n4. CUTOFF  keep a topic only if its released count is at least "
              f"{args.threshold:g} x sigma = {cutoff:.1f}\n")
        for k in top:
            verdict = "kept" if kept[k] else "DROPPED (too small to trust)"
            sign = ">=" if kept[k] else "<"
            print(f"  topic {int(cluster_ids[k]):>4}: {released[k]:>6,} {sign} {cutoff:.1f}  ->  {verdict}")

        print("\n5. RETRIEVAL  public rows picked from each kept topic\n")
        for k in top:
            v, m = int(released[k]), members[k]
            if not kept[k]:
                print(f"  topic {int(cluster_ids[k]):>4}: dropped, nothing taken\n")
                continue
            if v >= len(m):
                picked = m
                print(f"  topic {int(cluster_ids[k]):>4}: asked for {v:,} rows, topic has "
                      f"{len(m)}  ->  took all {len(m)}")
            else:
                picked = rng.choice(m, size=v, replace=False, p=weight(k))
                print(f"  topic {int(cluster_ids[k]):>4}: asked for {v:,} rows, topic has "
                      f"{len(m)}  ->  randomly picked {v}")
            for r in rng.permutation(picked)[:args.rows]:
                print(f"     [{source[r]}] {clip(text[r], 110)!r}")
            print()

        rows = sum(min(int(released[k]), len(members[k])) for k in np.nonzero(kept)[0])
        med = sum(min(int(released[k]), len(members[k])) * med_frac[k]
                  for k in np.nonzero(kept)[0])
        print(f"WHOLE RELEASE  ({R['records']:,} hospital patients + this message, all 2,208 topics)")
        print(f"  topics kept {kept.sum():,} of {len(true):,}   ·   "
              f"public rows picked {rows:,} of {len(text):,}   ·   "
              f"medical ≈{100 * med / max(rows, 1):.0f}%")
        print("  The message itself appears nowhere in the output: only public rows do.\n")

    if args.query:
        run(args.query)
        return
    print("\nType a patient message and press enter. Ctrl-D or 'quit' to exit.\n")
    while True:
        try:
            msg = input("> ").strip()
        except EOFError:
            break
        if msg in ("quit", "exit"):
            break
        if msg:
            run(msg)


if __name__ == "__main__":
    main()
