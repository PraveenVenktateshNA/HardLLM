"""Step 2b: noisy vote counts -> public rows to synthesise from (the noise sweep).

Input is route_private.py's output: 5 cluster ids per private record. From
here on nothing private is read -- only vote counts, which get Gaussian noise,
and the public pool, which is what gets sampled. Everything after the noise is
post-processing, so it cannot weaken the privacy guarantee.

What changed from the paper's query_retrival_model.py, and why
--------------------------------------------------------------
1. The noise is actually applied. The original computes noisy counts, then
   overwrites them with the contents of noisy_output.jsonl, a file nothing
   writes -- so it crashes, and if it didn't, the noise would be discarded.
2. Sensitivity is sqrt(5), not 1. One record casts 5 votes (top-5), each +1 to
   a different cluster, so removing it moves the count vector by sqrt(5) in L2.
3. delta defaults to 1/60000 (one over the number of records), not 0.01. At
   0.01 the guarantee may fail for 1 record in 100 -- 600 patients here.
4. Noise goes on all 2,208 clusters, not only the ones that received a vote.
   Noising only non-zero clusters leaks which clusters had any patient at all.
   The cost is visible in the sweep: empty clusters drawing positive noise.
5. sigma is calibrated with the analytic Gaussian mechanism (Balle & Wang
   2018). The paper's formula sqrt(2 ln(1.25/delta)) * sens / eps is only
   proven for eps < 1; the sweep goes to 3. Both are printed.
6. The paper's sampling weight p is computed exactly but vectorised:
       p_i = 1 - r_i / (S - r_i)
   with r_i = sum of distances from i to the rest of its cluster and S = sum
   of all pairwise distances. Same numbers, seconds instead of ~5 hours.
   For tiny clusters p is <= 0 (n=3 always, by the triangle inequality), where
   the original crashes in np.random.choice; those fall back to uniform.

The sweep also reruns each eps with a threshold: clusters whose noisy count is
below `threshold_sigmas * sigma` are dropped. That is post-processing, so it is
free in privacy terms, and it is the standard fix for empty clusters drawing
positive noise.

    python HardLLM/Step_2/noisy_select.py --router centroid
    python HardLLM/Step_2/noisy_select.py --router centroid --eps 1 --threshold_sigmas 3
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Step_1"))
from eval_unseen import EMB_CACHE, POOL, SEED, load_centroids  # noqa: E402

OUT = os.path.join(HERE, "out")


# --------------------------------------------------------------------------
# noise calibration
# --------------------------------------------------------------------------
def sigma_classic(eps, delta, sens):
    """The paper's formula. Only a valid (eps, delta) guarantee for eps < 1."""
    return math.sqrt(2 * math.log(1.25 / delta)) * sens / eps


def sigma_analytic(eps, delta, sens):
    """Smallest sigma giving (eps, delta)-DP for the Gaussian mechanism.

    Balle & Wang 2018, Thm 8: (eps, delta)-DP iff
        Phi(s/2sig - eps*sig/s) - e^eps * Phi(-s/2sig - eps*sig/s) <= delta
    The left side falls as sigma grows, so bisect.
    """
    def delta_at(sig):
        a = sens / (2 * sig)
        b = eps * sig / sens
        return norm.cdf(a - b) - math.exp(eps + norm.logcdf(-a - b))

    lo, hi = 1e-6, 1.0
    while delta_at(hi) > delta:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if delta_at(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# public pool + the paper's sampling weights
# --------------------------------------------------------------------------
def load_pool(cluster_ids):
    lines = open(POOL).read().splitlines()
    pos = {int(c): i for i, c in enumerate(cluster_ids)}
    members = [[] for _ in cluster_ids]
    is_med = np.zeros(len(lines), dtype=bool)
    for row, line in enumerate(lines):
        o = json.loads(line)
        members[pos[int(o["text_id"])]].append(row)
        is_med[row] = bool(o.get("is_medical"))
    return lines, [np.array(m) for m in members], is_med


def paper_weights(members):
    """Per-cluster sampling distribution from the paper's compute_p.

    Returns (weights, fell_back) where fell_back marks clusters on which the
    original code would have crashed and we used uniform instead.
    """
    emb = np.load(EMB_CACHE, mmap_mode="r")
    weights, fell_back, spread = [], np.zeros(len(members), dtype=bool), []
    for k, m in enumerate(members):
        n = len(m)
        if n <= 2:
            weights.append(np.full(n, 1.0 / n))
            fell_back[k] = True
            continue
        E = np.asarray(emb[m], dtype=np.float64)
        d = pdist(E)
        S = d.sum()
        r = squareform(d).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = 1 - r / (S - r)
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            weights.append(np.full(n, 1.0 / n))
            fell_back[k] = True
            continue
        w = p / p.sum()
        weights.append(w)
        if n >= 10:
            spread.append(w.max() / w.min())
    return weights, fell_back, np.array(spread)


# --------------------------------------------------------------------------
# one configuration of the sweep
# --------------------------------------------------------------------------
def run_one(votes, eps, delta, sens, threshold_sigmas, members, weights,
            fell_back, is_med, lines, write_path):
    rng = np.random.default_rng(
        [SEED, int(0 if math.isinf(eps) else eps * 1000), int(threshold_sigmas * 100)])

    if math.isinf(eps):
        sigma = 0.0
        noisy = votes.copy()
    else:
        sigma = sigma_analytic(eps, delta, sens)
        noisy = np.maximum(0, np.rint(votes + rng.normal(0, sigma, len(votes)))).astype(int)

    if threshold_sigmas > 0 and sigma > 0:
        noisy[noisy < threshold_sigmas * sigma] = 0

    selected, whole, fallback_used = [], 0, 0
    for k in np.nonzero(noisy)[0]:
        m, v = members[k], int(noisy[k])
        if v >= len(m):
            # The paper takes the whole cluster here and does no sampling.
            selected.append(m)
            whole += 1
        else:
            selected.append(rng.choice(m, size=v, replace=False, p=weights[k]))
            fallback_used += bool(fell_back[k])
    rows = np.concatenate(selected) if selected else np.array([], dtype=int)

    true_p = votes / votes.sum()
    noisy_p = noisy / noisy.sum() if noisy.sum() else np.zeros_like(true_p, dtype=float)
    voted = votes > 0
    res = {
        "eps": "inf" if math.isinf(eps) else eps,
        "threshold_sigmas": threshold_sigmas,
        "sigma": round(sigma, 2),
        "clusters_selected": int((noisy > 0).sum()),
        "junk_clusters": int(((noisy > 0) & ~voted).sum()),
        "lost_clusters": int((voted & (noisy == 0)).sum()),
        "lost_vote_share": float(votes[voted & (noisy == 0)].sum() / votes.sum()),
        "whole_cluster_takes": whole,
        "uniform_fallbacks_used": fallback_used,
        "rows_selected": int(len(rows)),
        "rows_from_junk_clusters": int(sum(len(s) for s, k in zip(selected, np.nonzero(noisy)[0]) if not voted[k])),
        "medical_frac_selected": float(is_med[rows].mean()) if len(rows) else 0.0,
        "tv_distance": float(0.5 * np.abs(true_p - noisy_p).sum()),
    }
    if write_path:
        with open(write_path, "w") as f:
            for r in rows:
                f.write(lines[r] + "\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="centroid",
                    help="which routes_<router>.json to read")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 1, 2, 3])
    ap.add_argument("--delta", type=float, default=None,
                    help="default: 1 / number of private records")
    ap.add_argument("--threshold_sigmas", type=float, nargs="+", default=[0, 3],
                    help="drop clusters whose noisy count < T * sigma (0 = off)")
    ap.add_argument("--no_write", action="store_true",
                    help="skip writing the selected_*.jsonl files")
    args = ap.parse_args()

    routes_path = os.path.join(OUT, f"routes_{args.router}.json")
    R = json.load(open(routes_path))
    n_records, k = R["records"], R["votes_per_record"]
    delta = args.delta or 1.0 / n_records
    sens = math.sqrt(k)

    _, _, cluster_ids, _ = load_centroids()
    pos = {int(c): i for i, c in enumerate(cluster_ids)}
    votes = np.zeros(len(cluster_ids), dtype=np.int64)
    for r in R["routes"]:
        for c in r:
            if c in pos:
                votes[pos[c]] += 1

    print(f"routes: {routes_path}")
    print(f"{n_records:,} records x {k} votes = {votes.sum():,} votes "
          f"over {(votes > 0).sum():,}/{len(votes):,} clusters")
    print(f"delta = {delta:.3g}   L2 sensitivity = sqrt({k}) = {sens:.3f}")
    print(f"paper's code used sigma = {sigma_classic(1, 0.01, 1):.2f} "
          f"(eps=1, delta=0.01, sensitivity=1)")

    print("loading public pool and computing the paper's sampling weights...")
    lines, members, is_med = load_pool(cluster_ids)
    weights, fell_back, spread = paper_weights(members)
    print(f"  clusters where the paper's formula breaks (n<=2 or p<=0): "
          f"{fell_back.sum()} -> uniform")
    print(f"  max/min sampling weight within a cluster (n>=10): "
          f"median {np.median(spread):.4f}, worst {spread.max():.4f}")
    print(f"  (1.0000 would be plain uniform random sampling)")
    print(f"  medical share of the whole public pool: {is_med.mean():.3f}")

    configs = [(math.inf, 0)] + [(e, t) for e in args.eps for t in args.threshold_sigmas]
    results = []
    for eps, t in configs:
        eps_tag = "inf" if math.isinf(eps) else f"{eps:g}"
        name = f"selected_{args.router}_eps{eps_tag}_t{t:g}.jsonl"
        res = run_one(votes, eps, delta, sens, t, members, weights, fell_back,
                      is_med, lines, None if args.no_write else os.path.join(OUT, name))
        if not math.isinf(eps):
            res["sigma_paper_formula"] = round(sigma_classic(eps, delta, sens), 2)
        results.append(res)

    hdr = (f"{'eps':>5} {'T':>3} {'sigma':>7} {'clusters':>8} {'junk':>5} "
           f"{'lost':>5} {'lost%':>6} {'whole':>6} {'rows':>7} {'junkrows':>8} "
           f"{'med%':>6} {'TV':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{str(r['eps']):>5} {r['threshold_sigmas']:>3g} {r['sigma']:>7.2f} "
              f"{r['clusters_selected']:>8} {r['junk_clusters']:>5} "
              f"{r['lost_clusters']:>5} {100*r['lost_vote_share']:>5.1f}% "
              f"{r['whole_cluster_takes']:>6} {r['rows_selected']:>7} "
              f"{r['rows_from_junk_clusters']:>8} "
              f"{100*r['medical_frac_selected']:>5.1f}% {r['tv_distance']:>6.3f}")
    print("""
eps       privacy budget (smaller = stronger privacy, more noise; inf = no noise)
T         threshold: clusters with noisy count < T*sigma dropped (0 = paper)
clusters  clusters that contribute rows        junk  of those, ones no record voted for
lost      voted clusters the noise zeroed out  lost% share of all votes they held
whole     clusters taken in full (noisy count >= cluster size, no sampling done)
med%      share of selected public rows that are medical
TV        how far the noisy histogram is from the true one (0 = identical, 1 = disjoint)""")

    summary = {
        "router": args.router, "records": n_records, "votes_per_record": k,
        "delta": delta, "sensitivity": sens,
        "paper_code_sigma": sigma_classic(1, 0.01, 1),
        "weights_fell_back": int(fell_back.sum()),
        "weight_spread_median": float(np.median(spread)),
        "weight_spread_worst": float(spread.max()),
        "pool_medical_frac": float(is_med.mean()),
        "results": results,
    }
    path = os.path.join(OUT, f"noise_sweep_{args.router}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
