# Step 4 results

| Model | Trained on | Answer perplexity (lower is better) | MMLU medical (higher is better) |
|---|---|---|---|
| base | nothing (Llama-3.2-3B as released) | 14.70 | 63.1% |
| hardllm | HardLLM synthetic pairs (full pipeline) | 19.04 | 21.9% (-41.1% vs base) |
| selected | public pairs chosen by private-data selection | 10.26 | 30.9% (-32.2% vs base) |
| random | public pairs chosen at random | 10.86 | 29.3% (-33.8% vs base) |
| private | the real private data (ceiling, not private) | 10.07 | 24.3% (-38.8% vs base) |
| hardllm-filtered | HardLLM synthetic pairs, worse-matching half removed | 19.02 | 22.0% (-41.0% vs base) |
| hardllm-fullsample | HardLLM synthetic pairs, GPT-2 sampled without top-k/top-p cut | 17.36 | 21.9% (-41.1% vs base) |

MMLU, zero-shot: not valid for the fine-tuned models. They were trained to write a sentence after "Answer:", the same cue MMLU ends with, so they pick the letter that also starts sentences ("A"). Base is valid.

- hardllm: perplexity 19.04 vs random 10.86 and base 14.70
- hardllm-filtered: perplexity 19.02 vs random 10.86 and base 14.70
- hardllm-fullsample: perplexity 17.36 vs random 10.86 and base 14.70
- selected: perplexity 10.26 vs random 10.86 and base 14.70
- private: perplexity 10.07 vs random 10.86 and base 14.70
