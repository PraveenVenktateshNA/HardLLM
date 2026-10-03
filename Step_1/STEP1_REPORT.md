# Step 1 — Reconstruction & Repair Report

**Scope:** Step 1 only (*Training Retrieval Model via Document Index*), corresponding to
Section V-B-1 and Lines 1–9 of Algorithm 1 in *Hardening LLM Fine-Tuning: From
Differentially Private Data Selection to Trustworthy Model Quantization* (IEEE TIFS,
Vol. 20, 2025).

**Status:** Step 1 is now structurally complete and syntactically valid. It has **not**
been executed — see [§5 Verification](#5-verification) for exactly what was and was not
checked, and why.

---

## 1. What was wrong

Step 1 could not run at all. Five distinct blockers:

| # | File | Problem | Severity |
|---|------|---------|----------|
| 1 | `trainer.py` | **File absent.** `train_retrival_model.py:14` imports `DSITrainer` and `DocTqueryTrainer` from it | Fatal — `ImportError` at startup |
| 2 | `cluster_sst2.py:53` | `new_data.append(entry)` sat **outside** the `for` loop | Fatal (silent) — writes 1 row instead of ~233,000 |
| 3 | `download_data.py:12` | `map_amazon_label` returned `None` for labels 0/1/2 | Fatal — `datasets.map()` rejects `None` |
| 4 | `data.py:20` | `ignore_verifications=` was removed from `load_dataset` in modern `datasets` | Fatal — `TypeError` |
| 5 | `train_retrival_model.py:17,77` | Hard `import wandb` + unconditional `wandb.login()` | Blocks on an interactive prompt |

Bug #2 deserves emphasis: it fails *silently*. The script prints its success message and
exits 0, having written a single-line `sst2_with_clusters.jsonl`. Every downstream step
then trains on one example. On a remote GPU box this would have looked like a mysterious
convergence failure hours into a paid run.

---

## 2. The reconstruction — `trainer.py`

This is the substantive piece of work, so the derivation is worth recording.

### 2.1 How the contract was determined

`trainer.py` was **not** guessed. Its required behaviour is almost fully pinned by two
call sites that *do* ship in the repo, which act as an executable specification:

**Constraint A — the constructor**, from `train_retrival_model.py:127-140`:

```python
trainer = DSITrainer(
    model=model, tokenizer=tokenizer, args=training_args,
    train_dataset=train_dataset, eval_dataset=valid_dataset,
    data_collator=IndexingCollator(tokenizer, padding='longest'),
    compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
    restrict_decode_vocab=restrict_decode_vocab,   # <-- non-standard
    id_max_length=run_args.id_max_length,          # <-- non-standard
)
```

Everything except the last two arguments is stock `transformers.Trainer`. So `DSITrainer`
is a `Trainer` subclass taking exactly two extra keyword arguments.

**Constraint B — the output shapes**, from `make_compute_metrics` at
`train_retrival_model.py:47-68`:

```python
for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
    rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
    label_id  = tokenizer.decode(label, skip_special_tokens=True)
    hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
```

`batch_decode(beams)` means each element of `predictions` is itself a *batch of
sequences* — a ranked beam list. `decode(label)` means each element of `label_ids` is a
single sequence. And the `[:10]` slice with `Hits@1`/`Hits@10` tells us the beam list is
ranked best-first and at least 10 deep. That fixes the shapes exactly:

```
predictions : (num_examples, num_return_sequences, id_max_length)
label_ids   : (num_examples, id_max_length)
```

**Constraint C — the decoding must be constrained.** `restrict_decode_vocab` is built at
`train_retrival_model.py:105-118` as the set of integer-only token ids, and is passed in.
It can only be consumed by `generate(prefix_allowed_tokens_fn=...)`. This is the standard
Differentiable Search Index mechanism: the model must emit a valid cluster id, never
free-form text.

### 2.2 What was implemented

`DSITrainer` overrides exactly two methods:

- **`compute_loss`** — plain seq2seq cross-entropy from the model's own forward pass.
  Training the index is ordinary teacher-forced seq2seq; nothing exotic here.
- **`prediction_step`** — this is where DSI differs from a normal trainer. Rather than
  scoring logits, it runs **constrained beam search** (20 beams, 20 returned sequences),
  reshapes `generate()`'s flat `(batch × beams, len)` output into
  `(batch, beams, len)`, right-pads to `id_max_length`, and returns
  `(None, beams, labels)` so `compute_metrics` can compute hit rates.

Plus one helper, `_pad_to_id_max_length`. The padding is not cosmetic: `Trainer`
concatenates per-batch predictions into one array at the end of evaluation, so every
batch must agree on the final axis or the concatenation throws.

### 2.3 Three deliberate divergences from a naive reconstruction

These are places where I did *not* reproduce the obvious/original implementation, and
why. Flagged so they can be reviewed rather than trusted blindly.

1. **`-100` is stripped from labels before they are returned.**
   `IndexingCollator` (`data.py:59`) masks label padding with `-100` for the loss
   function. But `compute_metrics` calls `tokenizer.decode(label)` on those same labels,
   and decoding a negative token id raises `OverflowError` on current tokenizers. So
   `prediction_step` restores the real `pad_token_id` before returning. Without this,
   evaluation crashes the first time it runs — which, with `--eval_steps 10000`, would be
   hours into training.

2. **`compute_loss` accepts `**kwargs`.**
   transformers ≥ 4.46 passes `num_items_in_batch` for gradient-accumulation loss
   scaling. A fixed four-argument signature raises `TypeError` on those versions.

3. **`tokenizer=` is translated to `processing_class=` when needed.**
   transformers 4.46 renamed this `Trainer` argument and 5.x drops the old name. Rather
   than edit the call site, `_normalise_trainer_kwargs` inspects the installed
   `Trainer.__init__` signature and remaps if required. This keeps `train_retrival_model.py`
   untouched and makes the module version-portable.

`DocTqueryTrainer` is imported by `train_retrival_model.py` but **never instantiated** —
HardLLM uses the clustered-index arm, not the query-generation arm. It is implemented
faithfully (document → query, sampled decoding) so the import resolves and the variant
stays available, but it is not on the execution path.

---

## 3. The four repairs

All are minimal and surgical. No refactoring, no reformatting, no behavioural changes
beyond the defect itself.

**`cluster_sst2.py`** — indented the append into the loop:

```diff
 for idx, label in enumerate(labels):
     entry = train_data[idx].copy()
-    entry['text_id'] = int(label) 
-new_data.append(entry) 
+    entry['text_id'] = int(label)
+    new_data.append(entry)
```

**`download_data.py`** — moved `return` to function scope. The 0,1,2→negative /
3,4→positive mapping is the authors' choice and is preserved verbatim; only the control
flow is fixed:

```diff
     elif example['label'] in [3, 4]:
         example['label'] = 1
-        return example
+    return example
```

**`data.py`** — dropped the removed kwarg. Default verification behaviour is equivalent
for local JSONL:

```diff
-            ignore_verifications=False,
```

**`train_retrival_model.py`** — soft `wandb` import, and the explicit init now fires only
when logging was actually requested. Also corrects the main-process test: `local_rank` is
`-1` (not `0`) for single-process runs, so the original guard was wrong for the
single-GPU case this will actually run on.

```diff
-    if training_args.local_rank == 0:
+    if (training_args.local_rank in (-1, 0)
+            and wandb is not None
+            and "wandb" in (training_args.report_to or [])):
```

---

## 4. Files changed

```
Step_1/trainer.py               NEW  (~215 lines)  reconstructed
Step_1/cluster_sst2.py          repaired  (1 line)
Step_1/download_data.py         repaired  (1 line)
Step_1/data.py                  repaired  (1 line)
Step_1/train_retrival_model.py  repaired  (2 hunks)
Step_1/STEP1_REPORT.md          NEW  this document
```

---

## 5. Verification

### Verified

- **Syntax** — all five Python files parse cleanly under `ast.parse`.
- **`map_amazon_label`** — executed the patched function directly over inputs 0–4.
  Returns `[0, 0, 0, 1, 1]`, and never `None`.
- **Cluster loop** — replicated the patched loop over a 3-element fixture. Writes 3 rows,
  each carrying its own `text_id`. Pre-fix this produced 1 row.
- **Beam-shape contract** — simulated `generate()`'s flat output for
  `B=4, R=20, gen_len=6, id_max_length=20`, applied the exact reshape-and-pad from
  `prediction_step`, then ran the real consumer loop from `make_compute_metrics` against
  it. Result: `predictions (4, 20, 20)`, `labels (4, 20)`, all 4 examples consumed,
  beam ordering preserved (`beams[0][0]` is `generate()` row 0), and zero negative ids
  surviving in the labels.

### Not verified

**Nothing here has been run end-to-end.** Two hard reasons:

1. `torch`, `transformers`, and `sentence-transformers` are not installed on this
   machine, and the system Python is 3.14 — too new for that stack. A 3.11 environment is
   required.
2. mT5-Large needs roughly 20 GB with Adam optimizer states. This machine has 16 GB of
   unified memory. The training step in this repo **cannot** execute here at the paper's
   model size regardless of environment.

So: the reshape/padding logic is verified against its consumer by simulation, but the
`generate()` call itself, the constrained-decode integration, and convergence are all
**unproven until a GPU run happens**. Treat the first smoke test on GCP (§6, Step 6) as
the real verification gate.

---

## 6. GCP implementation runbook

### Step 0 — Request GPU quota *first*

**Do this before anything else; it is the longest pole.** New GCP projects default to
**zero** GPU quota, and approval takes anywhere from minutes to two business days.

IAM & Admin → Quotas → filter for:
- `NVIDIA_A100_GPUS` in your target region (request **1**)
- `GPUS_ALL_REGIONS` (request **1**)

### Step 1 — Pick the machine

**Recommendation: `a2-highgpu-1g` — 1× A100 40 GB.** mT5-Large needs ~20 GB, so this
fits with real headroom for a larger batch.

A note on your earlier H100 question: on GCP, H100s are sold almost exclusively as
`a3-highgpu-8g` — **eight** GPUs at roughly $88/hr. For a 1.2B model on 32-token
sequences that is badly wasteful; you would leave 7 GPUs idle. The A100 is the right
call here.

| Machine | GPU | ~On-demand | Verdict |
|---|---|---|---|
| `g2-standard-8` | 1× L4 24 GB | ~$0.85/hr | Tight; needs gradient checkpointing |
| **`a2-highgpu-1g`** | **1× A100 40 GB** | **~$3.67/hr** | **Recommended** |
| `a2-ultragpu-1g` | 1× A100 80 GB | ~$5.07/hr | Unnecessary headroom |
| `a3-highgpu-8g` | 8× H100 80 GB | ~$88/hr | Wasteful for this job |

```bash
gcloud compute instances create hardllm-step1 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

**Spot instances** cut cost ~60–70% (add `--provisioning-model=SPOT
--instance-termination-action=STOP`) but can be preempted. Only sane in combination with
frequent checkpointing to GCS (Step 7). For a 140k-step run, worth it.

### Step 2 — Environment

The Deep Learning VM image ships CUDA, drivers, and PyTorch. Add the rest:

```bash
gcloud compute ssh hardllm-step1 --zone=us-central1-a

nvidia-smi   # confirm the A100 is visible before going further

pip install "transformers>=4.40,<5" datasets sentence-transformers \
            sentencepiece protobuf scikit-learn accelerate
```

`sentencepiece` and `protobuf` are **not optional** — `MT5Tokenizer` fails to load
without them, and the failure message does not make the cause obvious.

### Step 3 — Get the code up

```bash
gcloud compute scp --recurse ./HardLLM/Step_1 hardllm-step1:~/Step_1 --zone=us-central1-a
```

### Step 4 — Create the output directory

```bash
cd ~/Step_1 && mkdir -p sst2 cache models
```

Not optional. `download_data.py` and `cluster_sst2.py` both call
`open('./sst2/...', 'w')` without creating the directory first, and will raise
`FileNotFoundError`. This is a known rough edge left unpatched, since one `mkdir` is
cleaner than editing three files.

### Step 5 — Data prep and clustering

```bash
python download_data.py            # ~minutes
wc -l sst2/sst_public.jsonl        # expect ~233,675
wc -l sst2/sst_private.jsonl       # expect ~33,675

python cluster_sst2.py             # ~10-25 min
wc -l sst2/sst2_with_clusters.jsonl   # MUST match sst_public.jsonl
```

**That last check is the regression test for bug #2.** If it prints `1`, the append fix
did not take effect. Do not proceed past a mismatch.

One external dependency to confirm early: `download_data.py` pulls
`SetFit/amazon_reviews_multi_en`. Amazon withdrew the original `amazon_reviews_multi`
dataset from the Hub at one point; if this mirror is also gone, substitute another
review corpus with a 0–4 star label field and note the substitution.

Expected geometry: `num_clusters = len(sentences) // 100` ≈ **2,336 clusters** of ~100
points each.

### Step 6 — Smoke test before the real run

**Do not skip this.** It is the first true execution of the reconstructed `trainer.py`,
and it costs about two minutes instead of discovering a defect eight hours in.

```bash
python train_retrival_model.py \
  --model_name google/mt5-small \
  --train_file ./sst2/sst2_with_clusters.jsonl \
  --output_dir ./models/smoke \
  --max_length 32 --id_max_length 20 --task DSI \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --max_steps 50 --eval_strategy steps --eval_steps 25 \
  --logging_steps 10 --report_to none --bf16
```

Success = it reaches the eval at step 25 and prints `Hits@1` / `Hits@10`. The numbers
will be near zero (50 steps trains nothing); **that is fine.** What matters is that
constrained beam search runs and `compute_metrics` consumes the output without a shape or
decode error. That single line clears every unverified item in §5.

### Step 7 — The real run

```bash
python train_retrival_model.py \
  --model_name google/mt5-large \
  --train_file ./sst2/sst2_with_clusters.jsonl \
  --output_dir ./models/sst2-cluster-mt5-large-DSI \
  --max_length 32 --id_max_length 20 --task DSI \
  --learning_rate 5e-4 --warmup_steps 10000 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
  --max_steps 140000 \
  --eval_strategy steps --eval_steps 10000 \
  --save_strategy steps --save_steps 10000 --save_total_limit 3 \
  --bf16 --dataloader_num_workers 8 --logging_steps 100 \
  --report_to none
```

`--max_steps 140000` mirrors the authors' shipped checkpoint path
(`sst2-cluster-mt5-large-DSI/checkpoint-140000`, referenced at
`Step_2/query_retrival_model.py:12`). Their batch size is not recorded anywhere in the
repo, so at batch 32 this is ~19 epochs over the 233k-row public set. Index memorisation
genuinely needs many passes — do not expect convergence in three.

On `--eval_strategy`: renamed from `--evaluation_strategy` in transformers 4.46. If you
pinned an older version, use the old flag.

**Estimated: 6–12 hours on one A100 40 GB** (~2–2.5× an H100). Roughly **$25–45**
on-demand, **$8–18** on spot.

Checkpoint off-box so a preemption or a stopped VM does not cost you the run:

```bash
gsutil -m rsync -r ./models gs://<your-bucket>/hardllm/step1/
```

### Step 8 — Shut down

```bash
gcloud compute instances stop hardllm-step1 --zone=us-central1-a
```

A stopped A100 instance still bills for its 200 GB SSD (~$34/month). Delete it once
checkpoints are safely in GCS.

---

## 7. What Step 1 produces

Artifacts:

| File | Contents |
|---|---|
| `sst2/sst_public.jsonl` | ~233k public rows (SST-2 half + Amazon) |
| `sst2/sst_private.jsonl` | ~33.7k held-out private rows — input to Step 2 |
| `sst2/sst2_with_clusters.jsonl` | Public rows, each tagged with its `text_id` (document index) |
| `models/sst2-cluster-mt5-large-DSI/checkpoint-140000/` | The trained retrieval model — the artifact Step 2 loads |

Presentable results:

- **Hits@1 / Hits@10** on the held-out subset, straight from `make_compute_metrics` — the
  headline quantitative result for Step 1.
- **Cluster size distribution** over the ~2,336 clusters.
- **Qualitative cluster inspection** — print member sentences from a few clusters to show
  semantic coherence. This is the most persuasive single artifact, because coherent
  clusters are the entire premise of the document-index design: if similar sentences
  don't share an index, nothing downstream works.
- **2D t-SNE / UMAP** of the PCA-reduced embeddings, coloured by cluster.

---

## 8. Remaining risks

1. **`trainer.py` is a reconstruction, not a recovery.** It satisfies every constraint the
   surviving call sites impose, and the shape contract is simulation-verified. But the
   authors' original may differ in details that change *numbers* rather than *behaviour* —
   beam count, length penalty, whether they evaluated on a held-out split at all
   (`valid_dataset` is `Subset(train_dataset, range(1000))`, i.e. **carved from the
   training set**, so reported Hits@k are optimistic by construction and are a
   memorisation measure, not generalisation).
2. **Upstream may have the real file.** The local clone has two commits (`fa9101e`,
   `dd6c759 "add codes"`) and looks like a partial dump. Check `hardllm/HardLLM` upstream,
   and the ancestor DSI-QG repo, before trusting this reconstruction for publication.
   A recovered original beats my reconstruction.
3. **mT5-Large hyperparameters are inferred.** LR 5e-4 and 10k warmup are reasonable
   T5-family defaults, not values recovered from the paper — the paper does not report
   Step 1 hyperparameters.
4. **Steps 2–4 remain broken** and are out of scope here. Step 2 in particular has a
   privacy-accounting defect (L2 sensitivity is √5, not the hardcoded 1, because
   `num_return_sequences=5` lets each private record vote in five bins) that must be fixed
   before any DP claim from this codebase is meaningful.
