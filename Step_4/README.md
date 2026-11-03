# Step 4: fine-tune Llama-3.2-3B and measure what HardLLM's data is worth

## The experiment

Llama-3.2-3B, fine-tuned with LoRA 4 times. Same settings, same 16,000 training
rows each (1,000 steps at batch 16), only the data differs:

| Model | Trained on | Question it answers |
|---|---|---|
| base | nothing | starting point |
| **hardllm** | Step 3's synthetic Q&A pairs, cleaned | the full HardLLM pipeline |
| selected | the public pairs Step 2 selected (ε = 0.5), no synthesis | what does Step 3 add? |
| random | public pairs picked at random, same count | what is the private selection worth? |
| private | the real private records | the ceiling. HardLLM exists to never do this; it is only a measuring stick |

Scored the same way:

1. **Answer perplexity** on 1,000 held-out private patient questions with real
   doctor answers (no model trains on them). Lower is better. The main number.
2. **MMLU medical** (1,089 multiple-choice questions, 6 subjects): did training on
   synthetic answers, which are often factually wrong, damage medical knowledge?
   Compare each model with base.
3. **20 demo questions** answered by every model side by side, to read.

Every training and test set goes through the same Step 3 cleaner, and the cleaning
rules were fixed before any Step 4 result existed.

## Files

| File | What it does | Runs on |
|---|---|---|
| `prepare_data.py` | builds `data/` from Step 2, Step 3, the pool and the private set | Mac, plain Python, ~20 s |
| `train_lora.py` | trains one model (`--arm`) | PC |
| `evaluate_models.py` | scores base + every trained model | PC |
| `tests/test_step4.py` | code tests on a tiny model, run first by the .bat | PC |
| `tests/check_results_step4.py` | checks the outputs, run last by the .bat | PC |
| `portable/preflight_step4.py` | real memory and speed test, picks the fastest setting | PC |
| `portable/RUN_STEP4_ON_NVIDIA.bat` | runs everything in order | PC |

## On the PC

Copy the folder to the PC and double-click `Step_4\portable\RUN_STEP4_ON_NVIDIA.bat`.
It stops at the first failure. Training and evaluation resume if stopped.

Bring back `Step_4\out\` and `Step_4\models\`.

## Not done here

TQU, the paper's compression step, comes after this. It needs `ca_module.py`
rebuilt (missing from the paper's code), the `quantize.tqu` import fixed and a
patch for Llama 3's grouped-query attention.
