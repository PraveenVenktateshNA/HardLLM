# Step 4 results

| Model | Trained on | Answer perplexity (lower is better) | MMLU medical (higher is better) |
|---|---|---|---|
| base | nothing (Llama-3.2-3B as released) | 14.70 | 63.1% |
| hardllm | HardLLM synthetic pairs (full pipeline) | 19.04 | 21.9% (-41.1% vs base) |
| selected | public pairs chosen by private-data selection | 10.26 | 30.9% (-32.2% vs base) |
| random | public pairs chosen at random | 10.86 | 29.3% (-33.8% vs base) |
| private | the real private data (ceiling, not private) | 10.07 | 24.3% (-38.8% vs base) |

HardLLM closes -743% of the gap between random selection and training on the private data (measured in answer loss).
