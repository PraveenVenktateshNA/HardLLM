# Step 5 results: the final model, compressed for the phone

Final model: Step 4 "selected" (Step 2 data). Step 4 scores for reference: base 14.70, selected 10.26.

| Version | File size | Answer perplexity (Step 4 test, lower is better) | llama.cpp perplexity (whole text) | Weights changed by |
|---|---|---|---|---|
| merged | 5.98 GB | 10.262 | - | 0.00% |
| f16 | 5.99 GB | 10.262 | 15.164 ± 0.253 | 0.00% |
| q8_0 | 3.19 GB | 10.265 | 15.167 ± 0.253 | 0.57% |
| q4_k_m | 1.88 GB | 10.435 | 15.312 ± 0.255 | 6.31% |

- 4-bit costs +0.173 answer perplexity (+1.7%) for a 3.2x smaller file; it is 4.27 better than base.
- Answer perplexity of a GGUF: the file's weights read back into PyTorch. The f16 row matching the merged row shows the reading back is exact.
- llama.cpp perplexity: the real runtime on the first 51,200 tokens of the test text, questions included.
