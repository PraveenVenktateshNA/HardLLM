# Run Step 3 on someone else's NVIDIA PC, from an external drive

Nothing gets installed on the PC and it needs no internet. Python, PyTorch (CUDA)
and GPT-2 all live in this folder, and every output is written back onto the drive.

## 1. On the Mac, once (~3.5 GB download)

```bash
cd ~/HardLLM
bash HardLLM/Step_3/portable/build_windows_bundle.sh
```

This creates `win_python/` (Windows Python + libraries, ~5 GB) and `gpt2/` (base
weights, 0.5 GB) here. Both are git-ignored.

## 2. Copy to the drive

Format the drive as **exFAT** (NTFS also works; FAT32 doesn't, since a file can't
be over 4 GB there). The whole project works, but only these are needed:

```
HardLLM/HardLLM/Step_2/out/selected_centroid_eps0.5_t3.jsonl   (Step 3's input)
HardLLM/HardLLM/Step_3/                                        (scripts + this bundle)
```

Keep the same folder layout: Step 3 finds its input at `../Step_2/out/`.

## 3. On the PC

Open `Step_3\portable\` on the drive and double-click **`RUN_STEP3_ON_NVIDIA.bat`**.

1. It checks that PyTorch can see the GPU.
2. **Code tests** (~1 min, `tests/test_step3_qa.py`): data counts, Q&A format,
   parsing, the stop-if-broken check, resuming, the bundle and this .bat.
3. **Pre-flight** (~3 min, `preflight_qa.py`): one worst-case generation batch
   (every sample forced to 400 tokens) in a fresh process at batch 256, then 128,
   then 64, keeping the biggest that stays under 90% of GPU memory; generation
   then runs at that batch. Then 10 real training steps. Stops if nothing fits,
   and prints the real time estimate.
4. **Training**: GPT-2 on the 16,788 question -> answer pairs Step 2 picked
   (~10-15 min, estimated).
5. **Generation**: 100,000 synthetic question -> answer pairs (~35-60 min,
   estimated). After the first batch it prints 2 samples and stops at once if
   under half came out complete.
6. **Results check** (`tests/check_results_qa.py`): fails on broken output
   (wrong row count, bad format); reports duplicates, copies of the training
   data, lengths and medical-word share next to the real data, and 20 samples.

Every stage stops the whole run if it fails. Training and generation resume if
the run is stopped and started again.

First run (question-only mode, done 2026-10-23): training 7.5 min, generation
10.5 min. Those older outputs are not touched by --qa.

Bring back `Step_3\models\gpt2-qa\` and the whole `Step_3\out\` folder:

| File | What it is |
|---|---|
| `Step_3/models/gpt2-qa/` | the fine-tuned model |
| `Step_3/out/preflight_qa.json` | GPU memory and speed measured before the run |
| `Step_3/out/gen_batch_qa.txt` | the generation batch size the pre-flight picked |
| `Step_3/out/train_summary_qa.json` | loss, perplexity, time taken |
| `Step_3/out/synthetic_qa.jsonl` | 100,000 synthetic `{"question", "answer"}` pairs |
| `Step_3/out/synthetic_qa_stats.json` | how many were dropped and why, time taken |
| `Step_3/out/results_qa.json` | the results check |
| `Step_3/out/samples_qa.txt` | 20 random pairs to read |

## The only thing the PC must have

An NVIDIA driver recent enough for the 5070 Ti, which any 5070 Ti already has.
If the GPU check fails, update the driver from nvidia.com.

Training uses batch 16 for 3 epochs, as on the Mac. In --qa mode each batch is fed
as 4 rows x 4 accumulation steps: identical training, a quarter of the memory
(batch 16 at 512 tokens ran out of memory in testing). Generation batch size
changes speed only, not output quality.
