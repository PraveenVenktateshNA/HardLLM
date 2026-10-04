"""Step 3b: write synthetic text with the fine-tuned GPT-2.

Replaces the paper's generate-text.py (forces CUDA, loads weights through a
private transformers API that no longer exists, prompts on label columns our
data doesn't have). Sampling settings are the paper's: temperature 1.0,
top_k 50, top_p 0.9, no_repeat_ngram_size 2, 128 tokens, 100,000 samples.
The paper also declares num_beams=5 but never passes it to generate(), so it
has no effect there either; it is left out.

Writes as it goes and resumes if interrupted.

    python HardLLM/Step_3/generate_synthetic.py
    python HardLLM/Step_3/generate_synthetic.py --total 500 --output /tmp/smoke.jsonl
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "gpt2-generator")
DEFAULT_OUT = os.path.join(HERE, "out", "synthetic.jsonl")
SEED = 313


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--total", type=int, default=100_000, help="paper: 100,000")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--length", type=int, default=128, help="new tokens per sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_repeat_ngram", type=int, default=2,
                    help="paper: 2 (no word pair may repeat); 0 = off")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = sum(1 for _ in open(args.output)) if os.path.exists(args.output) else 0
    if done:
        print(f"resuming: {done:,} samples already written")
    print(f"device: {device}   model: {args.model}   target: {args.total:,}")

    # Different seed per resume point, so a restart doesn't repeat samples.
    torch.manual_seed(SEED + done)
    # Every sample starts from the end-of-text token: unconditional generation.
    prompt = torch.full((args.batch_size, 1), tokenizer.eos_token_id, device=device)
    mask = torch.ones_like(prompt)

    t0, start_done, empty = time.time(), done, 0
    with open(args.output, "a") as f:
        while done < args.total:
            with torch.no_grad():
                out = model.generate(
                    prompt, attention_mask=mask,
                    do_sample=True,
                    max_new_tokens=args.length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    no_repeat_ngram_size=args.no_repeat_ngram,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for seq in tokenizer.batch_decode(out[:, 1:], skip_special_tokens=True):
                text = " ".join(seq.split())
                if not text:
                    empty += 1
                    continue
                if done >= args.total:
                    break
                f.write(json.dumps({"text": text}) + "\n")
                done += 1
            f.flush()
            rate = (time.time() - t0) / max(done - start_done, 1)
            print(f"\r  {done:,}/{args.total:,}  {rate*1000:.0f} ms/sample  "
                  f"ETA {rate*(args.total-done)/3600:.1f} h", end="", flush=True)
    print(f"\nwrote {args.output}   (empty generations dropped: {empty})")


if __name__ == "__main__":
    main()
