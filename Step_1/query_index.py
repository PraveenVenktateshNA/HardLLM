"""Query the trained DSI index by hand.

Give it a sentence, it returns the cluster ids the retrieval model thinks that
sentence belongs to, ranked, together with what actually lives in those
clusters. This is the qualitative check -- Hits@1 tells you *how often* the
model is right, this tells you *what it does* when it is wrong.

    python HardLLM/Step_1/query_index.py                       # interactive
    python HardLLM/Step_1/query_index.py -q "my knee hurts"    # one shot

Decoding replicates train_retrival_model.py exactly: 20 beams constrained to
digit tokens via prefix_allowed_tokens_fn, top-k returned in rank order. The
tokenizer must come from google/mt5-base, NOT from the checkpoint directory --
loading it from the checkpoint silently produces different token ids.
"""
import argparse
import collections
import json
import os

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOL = os.path.join(ROOT, "pool", "pool_with_clusters.jsonl")
DEFAULT_CKPT = os.path.join(HERE, "models", "checkpoint-4000-cooldown")
CLUSTER_CACHE = os.path.join(HERE, "models", "cluster_summary.json")

SPIECE_UNDERLINE = "▁"


def build_int_token_ids(tokenizer):
    """The digits-only decode vocabulary, byte-identical to training."""
    ids = []
    for token, tid in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                ids.append(tid)
        if token == SPIECE_UNDERLINE:
            ids.append(tid)
        elif token.isdigit():
            ids.append(tid)
    ids.append(tokenizer.eos_token_id)
    return ids


def load_cluster_summary(n_samples=3):
    """cluster id -> {size, medical_frac, roles, samples}.

    One streaming pass over the 441k-row pool, cached because it takes ~20s
    and nothing about it changes between queries.
    """
    if os.path.exists(CLUSTER_CACHE):
        with open(CLUSTER_CACHE) as f:
            return {int(k): v for k, v in json.load(f).items()}

    print("building cluster summary (one-off, ~20s)...")
    size = collections.Counter()
    med = collections.Counter()
    roles = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)
    with open(POOL) as f:
        for line in f:
            r = json.loads(line)
            c = r["text_id"]
            size[c] += 1
            roles[c][r["role"]] += 1
            if r.get("is_medical"):
                med[c] += 1
            if len(samples[c]) < n_samples:
                samples[c].append(r["text"][:160])

    summary = {
        c: {
            "size": size[c],
            "medical_frac": round(med[c] / size[c], 3),
            "top_role": roles[c].most_common(1)[0][0],
            "samples": samples[c],
        }
        for c in size
    }
    os.makedirs(os.path.dirname(CLUSTER_CACHE), exist_ok=True)
    with open(CLUSTER_CACHE, "w") as f:
        json.dump(summary, f)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", help="single query; omit for interactive")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--top_k", type=int, default=5,
                    help="clusters to return (Step 2 uses 5)")
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # google/mt5-base, never the checkpoint dir -- see module docstring.
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(device).eval()

    int_token_ids = build_int_token_ids(tokenizer)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return int_token_ids

    clusters = load_cluster_summary()

    def run(text):
        enc = tokenizer(text, return_tensors="pt", truncation="only_first",
                        max_length=args.max_length)
        with torch.no_grad():
            beams = model.generate(
                enc.input_ids.to(device),
                attention_mask=enc.attention_mask.to(device),
                max_length=20,
                num_beams=args.num_beams,
                prefix_allowed_tokens_fn=restrict_decode_vocab,
                num_return_sequences=args.top_k,
                early_stopping=True,
            )
        print(f"\nquery: {text!r}\n")
        for rank, seq in enumerate(beams, 1):
            cid_str = tokenizer.decode(seq, skip_special_tokens=True).strip()
            info = clusters.get(int(cid_str)) if cid_str.isdigit() else None
            if info is None:
                # The digit constraint permits ids that no cluster uses; a model
                # emitting these is a real failure mode worth seeing.
                print(f"  {rank}. id={cid_str!r}  <- NOT A VALID CLUSTER")
                continue
            print(f"  {rank}. cluster {cid_str:>4}  "
                  f"n={info['size']:<5} medical={info['medical_frac']:.0%}  "
                  f"role={info['top_role']}")
            for s in info["samples"]:
                print(f"        {s!r}")
        print()

    if args.query:
        run(args.query)
        return
    print("\nType a sentence and press enter. Ctrl-D or 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if text in ("quit", "exit"):
            break
        if text:
            run(text)


if __name__ == "__main__":
    main()
