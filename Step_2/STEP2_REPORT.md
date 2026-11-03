# Step 2: Progress Report

**Scope:** Step 2 only (*Query Retrieval Model with Privacy Enhancement*): Section
V-B-2 and the DP-histogram / resampling lines of Algorithm 1 in *Hardening LLM
Fine-Tuning: From Differentially Private Data Selection to Trustworthy Model
Quantization* (IEEE TIFS, Vol. 20, 2025).

**Status:** Step 2 has been **run end to end on the real data** using the centroid
router. Output selected for Step 3: **36,201 public rows, 87.7% medical, at ε = 0.5.**
The DSI-router run (~7 h) has not been done yet. See [§8](#8-whats-left).

**Hardware:** Apple M1 (8 cores), 16 GB unified memory, MPS GPU backend. No cloud.

---

## 1. What Step 2 does

Step 1 produced a model that maps any text to one of 2,208 public topic clusters.
Step 2 uses it to choose public training data that looks like the private data,
without releasing the private data:

1. **Route.** Each of the 60,000 private medical records names its 5 closest
   clusters (5 votes per record, 300,000 in total).
2. **Add noise.** Count the votes per cluster and add Gaussian noise, sized so that no
   single record can be detected in the counts (differential privacy).
3. **Select.** For each cluster, take that many rows from the **public** pool.

Only noisy counts cross the privacy boundary. Everything after the noise reads public
data only, so it is post-processing and cannot weaken the guarantee.

**Nothing is trained in Step 2.** It is inference, counting and sampling.

| Input | Size |
|---|---|
| Private records (`pool/private_medical.jsonl`, ChatDoctor) | 60,000 |
| Public pool (`pool/pool_with_clusters.jsonl`) | 441,699 rows, 2,208 clusters |
| Medical share of the public pool | 9.3% (40,975 rows) |
| Cluster sizes | min 1, median 185, max 702 |

---

## 2. What was wrong with the original

The paper's `Step_2/query_retrival_model.py` could not run. More importantly, **its
privacy step did not work.**

| # | Problem | Severity |
|---|---|---|
| 1 | **The noise is thrown away.** Lines 87–97 compute the noisy counts, then reset the dict to `{}` and read it back from `noisy_output.jsonl`, a file nothing creates. | Fatal. `FileNotFoundError`, and if it ran, no DP at all |
| 2 | **Sensitivity hardcoded to 1.** With `num_return_sequences=5`, each record adds +1 to 5 bins, so the L2 sensitivity is √5. | Privacy claim wrong |
| 3 | **δ = 0.01.** On 60,000 records that permits failure for ~1 in 100. Standard practice is δ < 1/n. | Privacy claim wrong |
| 4 | **Noise only on clusters that received a vote.** Whether a cluster appears at all reveals that someone voted for it. | Privacy leak |
| 5 | **The noise formula `σ = √(2 ln(1.25/δ))·Δ/ε` is only proven for ε < 1.** | Invalid above ε = 1 |
| 6 | **`compute_p` is O(n³) per cluster.** Measured 6.0 × 10⁻⁷·n³ s; over our clusters (Σn³ = 3.06 × 10¹⁰) that is **5.1 hours**. It also crashes on clusters of size ≤ 3: zero denominator at n ≤ 2, and p ≤ 0 at n = 3 by the triangle inequality, which `np.random.choice` rejects. | Fatal on small clusters; impractical runtime |
| 7 | **Full dataset rescan per cluster** (line 112): 1.19 s × number of clusters = 0.7 h. | Wasteful |
| 8 | `MT5Tokenizer` no longer exists in transformers 5; SST-2 paths; field `sentence` instead of `text`; `checkpoint-140000` path. | Fatal: `ImportError` |

Under the paper's settings (ε = 1, δ = 0.01, Δ = 1) the code would use **σ = 3.11**. The
correct σ at the same ε = 1 is **8.08, 2.6× larger.**

---

## 3. What was built

Two scripts replace the original. They reuse Step 1's `load_centroids()` and
`build_int_token_ids()` from `Step_1/eval_unseen.py`, so the clustering stays in one
place.

### 3.1 `route_private.py`: private records → top-5 clusters

Two routers with the same output format:

| Router | Method | Measured time, 60k records |
|---|---|---|
| `centroid` | MiniLM embedding → PCA(48) → 5 nearest cluster centres. This is the rule k-means used to build the clusters. | **3.6 min** |
| `dsi` | The Step 1 mT5 retrieval model with the paper's decoding (20 beams, top 5) | **~7.1 h** (428 ms/record) |

The DSI mode writes each batch as it goes and resumes from where it stopped if
interrupted.

### 3.2 `noisy_select.py`: noise + selection, as a sweep

Fixes for every item in §2:

- **Noise is applied and used.** (#1)
- **Δ = √5, δ = 1/60,000.** (#2, #3)
- **Noise on all 2,208 clusters**, including those with zero votes. (#4)
- **σ from the analytic Gaussian mechanism** (Balle & Wang, 2018, Thm. 8), found by
  bisection. Valid for any ε. The paper's σ is printed alongside. (#5)
- **The paper's sampling weight, computed exactly but vectorised:**
  `p_i = 1 − r_i / (S − r_i)`, where r_i is the sum of distances from row i to the rest
  of its cluster and S is the sum of all pairwise distances. **Verified against the
  original `compute_p` on real embeddings: max difference 3.3 × 10⁻¹⁶** (machine
  precision). Runs in seconds instead of 5.1 h. Clusters where the formula breaks fall
  back to uniform; only 1 cluster needed this. (#6)
- **One pass over the pool**, grouped by cluster. (#7)
- **A noise threshold (new):** clusters whose noisy count is below T·σ are dropped.
  It only reads already-noisy counts, so it is post-processing and free in privacy
  terms. (§5 explains why it is needed.)

One command runs ε ∈ {0.5, 1, 2, 3} × T ∈ {0, 3}, plus a no-noise baseline.

---

## 4. Feasibility measurements (M1, before running)

| Stage | Config | Measured | 60,000 records |
|---|---|---|---|
| DSI inference | MPS, batch 8 | 428 ms/row | **7.1 h** |
| DSI inference | MPS, batch 16 | 471 ms/row | 7.8 h |
| DSI inference | CPU, batch 8 | 457 ms/row | 7.6 h |
| DSI inference | CPU, batch 16 | 672 ms/row | 11.2 h |
| MiniLM encoding | MPS, batch 256 | 2.76 ms/row | 2.8 min |
| Original `compute_p` | all clusters | 6.0e-7 × n³ s | 5.1 h |

Batch 8 is the fastest for DSI. Larger batches are slower because 20-beam search at
batch 16 holds 320 live sequences, which pushes 16 GB into memory pressure. MPS gives
only about a 6% gain over CPU here, since beam search runs step by step and keeps the
GPU underfed.

---

## 5. Results

### 5.1 Routing (centroid router, all 60,000 records)

| Metric | Value |
|---|---|
| Time | 3.6 min |
| Top-1 cluster is majority-medical | **98.3%** (Step 1's DSI on 500 unseen records: 97.6%) |
| Invalid cluster ids | 0 |
| Clusters receiving any vote | 812 / 2,208 |
| Busiest cluster | 7,159 votes (the largest cluster in the pool has 702 rows) |

### 5.2 Noise sweep

```
  eps   T   sigma clusters  junk  lost  lost%  whole    rows junkrows   med%     TV
  inf   0    0.00      812     0     0   0.0%    113   39710        0  80.6%  0.000
  0.5   0   15.18     1227   671   256   0.3%    113   50149     7846  64.1%  0.039
  0.5   3   15.18      188     1   625   1.2%    115   36201       46  87.7%  0.014
    1   0    8.08     1292   666   186   0.1%    113   45420     4540  70.5%  0.022
    1   3    8.08      231     1   582   0.8%    112   37552       32  85.1%  0.009
    2   0    4.33     1221   593   184   0.1%    113   42334     2263  75.7%  0.011
    2   3    4.33      286     4   530   0.5%    114   38441       58  83.4%  0.006
    3   0    3.03     1292   623   143   0.1%    113   41661     1750  76.8%  0.008
    3   3    3.03      315     1   498   0.4%    113   38631       10  82.9%  0.005
```

`junk` = clusters selected that no record voted for. `lost` = clusters with real votes
zeroed by noise or the threshold. `whole` = clusters taken in full. `TV` = total
variation distance between the true and noisy vote histograms.

**Reading it:**

- **Without a threshold (T = 0), correct noise floods the output with junk.** Once
  noise goes on all 2,208 clusters, roughly 600–670 empty clusters draw a positive
  count by chance. That pulls in up to 7,846 junk rows and drops medical content to
  64–77%. Stronger privacy makes it worse.
- **With T = 3, junk almost disappears** (10–58 rows) and medical content rises to
  83–88%. The clusters the threshold drops hold only 0.4–1.2% of the votes.
- **Privacy costs very little here.** Even at ε = 0.5 the noisy histogram is nearly
  identical to the true one (TV 0.014). 300,000 votes on 812 clusters gives the busy
  clusters counts in the hundreds or thousands, which ±15 noise barely moves.

### 5.3 Selected setting: ε = 0.5, T = 3

| | Clusters | Rows |
|---|---|---|
| Taken whole (noisy count ≥ cluster size) | 115 | 27,769 |
| Sampled (cluster larger than noisy count) | 73 | 8,432 |
| **Total** | **188** | **36,201** |

**Why 300,000 votes become 36,201 rows:** the busiest clusters asked for far more rows
than exist. For example, cluster 40 received 7,159 votes and has 292 rows, so all 292
are taken. The public pool's supply is the cap, not the votes.

**Medical coverage, by source:**

| Kind of public medical text | In pool | Selected |
|---|---|---|
| Patient questions (ChatDoctor) | 14,509 | **99.2%** |
| Drug reviews | 14,464 | 71.9% |
| Medical encyclopedia (MedQuAD) | 9,582 | 61.5% |
| **All medical rows** | **40,975** | **77.7%** |

The private data is entirely patient questions. Selection captures almost every public
row of that kind, and is more selective with medical text written in other styles.
This is the intended behaviour: the private data steers selection toward what it looks
like. Taking every medical row would make the private data and the DP pointless.

**The 12.3% non-medical rows (4,462), inspected:**

| Group | Rows | Examples |
|---|---|---|
| Medical text the classifier misses | ≥ 730 | From ChatDoctor / MedQuAD / drug reviews, labelled non-medical |
| Health-related, not clinical | ~1,000+ | Yelp reviews of doctors and dentists, WHO news, TB passages |
| Unrelated | ~2,050 (≈ 5.7% of output) | Amazon product reviews, general Wikipedia |

---

## 6. Findings

1. **The released code never applied differential privacy.** The noise step is dead
   code (§2 #1). Any privacy claim resting on this script is unsupported.
2. **The paper's privacy parameters understate the noise by 2.6×** (σ 3.11 vs 8.08 at
   ε = 1), due to Δ = 1 and δ = 0.01.
3. **The "informativeness" sampling is effectively uniform random sampling.** Within a
   cluster, the ratio of the largest to smallest sampling weight is **1.0024 at the
   median** (worst 1.2162; 1.0 is perfectly uniform). The formula divides a sum of n
   distances by a sum of ~n²/2 distances, so every p ≈ 1 − 2/n. Five hours of the
   original computation produce a distribution indistinguishable from
   `np.random.choice` without weights.
4. **About two-thirds of the output involves no sampling at all.** 115 clusters
   (27,769 of 36,201 rows) are copied whole because their noisy count exceeds their
   size. The median such cluster asked for 11.7× more rows than it holds. For these
   clusters, the choice of ε does not change which rows are selected.
5. **Noise on all clusters, done correctly, needs a threshold.** Without one, empty
   clusters contribute 1,750–7,846 junk rows. A 3σ threshold removes ~99% of them for
   no privacy cost.
6. **The threshold is not perfect.** Three sampled clusters passed with 0, 1 and 9 real
   votes (~140 rows, 0.4% of output). Raising T to 4 would remove them at the cost of
   more lost rare topics.

---

## 7. Verification

### Verified

- **End-to-end run on the real data:** routing and the full sweep both completed on
  all 60,000 records.
- **Vectorised sampling weight = original `compute_p`:** max difference 3.3 × 10⁻¹⁶
  across clusters of size 5, 12, 40 and 90, using real embeddings.
- **Small-cluster crash:** confirmed that n = 3 gives p ≤ 0 (−0.96, −1.04, −1.00).
- **Selected output inspected by hand:** source breakdown, classifier misses, and
  samples of the non-medical rows (§5.3).
- **Timing:** every figure in §4 was measured on this machine.

### Not verified

- **The DSI router has not been run through Step 2.** All results above use the
  centroid router. Step 1 measured DSI at 97.6% medical routing on 500 records, close
  to the centroid router's 98.3%, but the full-run comparison is pending.

---

## 8. What's left

1. **Run the DSI router overnight** (`route_private.py --router dsi`, ~7.1 h), then
   `noisy_select.py --router dsi`, and compare against the centroid run. This is the
   evidence for keeping or dropping the 1.2B-parameter model.
2. **Test with private data from a different source** than anything in the public pool
   (see §9.1). This is the number that matters for a real deployment.
3. Optional: try T = 4 to remove the residual junk clusters (§6 #6).

---

## 9. Risks and limitations

1. **The private and public patient questions come from the same dataset.** Step 1
   split ChatDoctor into a private part and a public part. Public look-alikes of the
   private data are therefore guaranteed to exist, which is why ChatDoctor coverage is
   99.2%. Real hospital records (clinical notes, abbreviations, lab values) have no such
   twin in any public corpus, and results would be weaker. **The 99.2% is an
   optimistic upper bound, not a deployment estimate.**
2. **Rare conditions are dropped.** 625 voted clusters (3,710 votes, 1.2%) fall below
   the threshold, the largest with 51 votes. This is the inherent privacy/utility
   trade-off: a rare condition is exactly where one patient stands out, so DP must
   blur it.
3. **The sweep spent more privacy budget than one release would.** Each ε in the sweep
   is a separate release from the same private data. Under basic composition, publishing
   all four would cost ε = 0.5 + 1 + 2 + 3 = 6.5 in total. The sweep is a research
   experiment to choose a setting. A deployment releases **once**, at ε = 0.5.
4. **"Medical" is our classifier's label, not ground truth.** It misses at least 730
   genuinely medical rows, so the 87.7% figure is conservative.
5. **The guarantee assumes each record's 5 votes go to 5 distinct clusters.** This holds
   for the centroid router by construction. For DSI, the 5 returned beams are distinct
   strings; invalid ids are discarded, which only lowers a record's contribution.

---

## 10. Files

```
Step_2/route_private.py           NEW   routing, centroid + DSI, resumable
Step_2/noisy_select.py            NEW   noise + selection sweep
Step_2/README.md                  NEW   how to run, how to read the output
Step_2/step2_architecture.mmd     NEW   compact diagram
Step_2/STEP2_REPORT.md            NEW   this document
Step_2/query_retrival_model.py    unchanged   original, kept for reference

Step_2/out/routes_centroid.json             5 clusters per private record
Step_2/out/noise_sweep_centroid.json        the full sweep table
Step_2/out/selected_centroid_*.jsonl        selected rows per setting (not in git)
Step_2/out/selected_centroid_eps0.5_t3.jsonl   → input to Step 3
```

**To reproduce** (from the outer `HardLLM/` folder):

```bash
python HardLLM/Step_2/route_private.py --router centroid   # ~3.6 min
python HardLLM/Step_2/noisy_select.py --router centroid    # ~1-2 min
```
