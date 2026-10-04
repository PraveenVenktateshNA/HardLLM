# Step 2: Turning private records into a safe public dataset

## What Step 2 does

Step 1 gave us a model that reads a record and says which of the 2,208 public
clusters it looks like. Step 2 uses that to build a training set for Step 3
without ever releasing a patient's text:

1. **Route.** Run all 60,000 private medical records through the router. Each
   record names its 5 closest clusters, which counts as 5 votes.
2. **Add noise.** Count the votes per cluster and add random noise to every
   count. The noise is sized so that no single patient's votes can be detected
   in the result. This is differential privacy.
3. **Select.** For each cluster, take that many rows from the **public** pool.

What reaches Step 3 is public text only, chosen in proportions that follow the
private data. The private text never leaves step 1 of this list.

Nothing gets trained in Step 2. It is inference, counting and sampling.

## Files

| File | What it does |
|---|---|
| `route_private.py` | Step 1: routes private records to clusters. Has two routers (below). |
| `noisy_select.py` | Steps 2 and 3, run as a sweep over privacy budgets. |
| `query_retrival_model.py` | The paper's original script, kept for reference. It does not run. |
| `out/` | Results. The `.json` summaries are committed; the `.jsonl` data files are not. |

## The two routers

| Router | How it picks clusters | Time for 60k on an M1 |
|---|---|---|
| `centroid` | Embeds the record with MiniLM and takes the 5 nearest cluster centres. This is the same rule that built the clusters. | ~3 min |
| `dsi` | The Step 1 retrieval model, with the paper's settings (20 beams, top 5). | ~7 h, measured at 428 ms per record |

The centroid router is the **dry run**. It finds every bug in minutes and gives
us a reference answer. The DSI run comes later, overnight, and the two get
compared.

## What we fixed in the paper's script, and why

The original `query_retrival_model.py` cannot run as written. More importantly,
its privacy step does not work.

| # | Problem in the original | What we do instead |
|---|---|---|
| 1 | It adds noise, then overwrites the noisy counts by reading `noisy_output.jsonl`, a file nothing creates. It crashes, and if it didn't, the noise would be thrown away. | The noisy counts are what gets used. |
| 2 | Sensitivity is set to 1. But each record casts 5 votes, so removing one record moves the counts by √5. | Sensitivity √5. |
| 3 | δ = 0.01, meaning the guarantee is allowed to fail for 1 record in 100. That is 600 patients here. | δ = 1/60,000. |
| 4 | Noise only goes on clusters that got at least one vote, so you can tell which clusters had any patient at all. | Noise goes on all 2,208 clusters. |
| 5 | The noise formula it uses is only proven for ε < 1. | Exact calibration for any ε (Balle & Wang, 2018). The paper's number is printed alongside. |
| 6 | The sampling formula loops in O(n³) per cluster, about 5 hours in total, and crashes on clusters with 1 to 3 rows. | Same formula, computed exactly in seconds. Uniform sampling where the formula breaks. |
| 7 | SST-2 paths, a tokenizer import that no longer exists, wrong field names. | Our paths, our data, working imports. |

**A finding, not a fix:** the paper's "pick the most informative rows" weights
come out almost identical for every row. In our clusters the largest weight is
typically only 0.2% bigger than the smallest. The formula is effectively plain
random sampling. Each run measures and prints this.

## The noise sweep

`noisy_select.py` tries ε = 0.5, 1, 2 and 3 (smaller ε means stronger privacy
and more noise), plus a no-noise baseline for comparison. Each ε is run twice:

- **T = 0**: the paper's way. Any cluster whose noisy count comes out above zero
  contributes rows.
- **T = 3**: any cluster whose noisy count falls below 3σ is dropped.

Why T = 3 exists: once noise goes on all 2,208 clusters, about half of the
clusters nobody voted for draw positive noise, and the paper's way samples
non-medical "junk" rows from them. A threshold removes that. It only looks at
counts that are already noisy, so it costs no extra privacy.

How to read the table it prints:

| Column | Meaning |
|---|---|
| `sigma` | Size of the noise added to each count |
| `clusters` | Clusters that contribute rows |
| `junk` | Of those, clusters that no private record voted for |
| `lost` / `lost%` | Clusters that had real votes but got zeroed by noise, and the share of all votes they held |
| `whole` | Clusters taken in full, because the noisy count was at least the cluster size. No sampling happens for these. |
| `rows` / `junkrows` | Public rows selected, and how many came from junk clusters |
| `med%` | Share of selected rows that are medical. Higher is better: the private data is medical. |
| `TV` | How different the noisy vote histogram is from the true one. 0 = identical, 1 = nothing in common. |

What a good setting looks like: high `med%`, low `TV`, few `junkrows`, small
`lost%`, at the smallest ε that achieves it.

## How to run

From the outer `HardLLM/` folder (the one containing `.venv` and `pool/`):

```bash
source .venv/bin/activate

# 1. Dry run: route all 60,000 records with the centroid router (~3 min)
python HardLLM/Step_2/route_private.py --router centroid

# 2. Noise sweep: eps 0.5/1/2/3, thresholds 0 and 3, plus the no-noise baseline
python HardLLM/Step_2/noisy_select.py --router centroid
```

Outputs in `Step_2/out/`:

- `routes_centroid.json`: the 5 clusters picked for each private record
- `noise_sweep_centroid.json`: the full sweep table and settings
- `selected_centroid_eps<ε>_t<T>.jsonl`: the selected public rows for each
  setting. These are what Step 3 trains on. Add `--no_write` to skip them.

## After the sweep (not done yet)

1. Pick the ε and T to carry forward, based on the sweep.
2. Run the DSI router overnight: `python HardLLM/Step_2/route_private.py --router dsi`.
   It saves progress as it goes. If it stops, run the same command again and it
   picks up where it left off.
3. Run `noisy_select.py --router dsi` and compare it with the centroid run. If
   both pick the same clusters, the 1.2-billion-parameter model is doing a job a
   3-minute calculation already does.
