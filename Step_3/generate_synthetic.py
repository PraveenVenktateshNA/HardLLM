"""Step 3b: write synthetic text with the fine-tuned GPT-2.

Replaces the paper's generate-text.py (forces CUDA, loads weights through a
private transformers API that no longer exists, prompts on label columns our
data doesn't have). Sampling settings are the paper's: temperature 1.0,
top_k 50, top_p 0.9, no_repeat_ngram_size 2, 128 tokens, 100,000 samples.
The paper also declares num_beams=5 but never passes it to generate(), so it
has no effect there either; it is left out.

Writes as it goes and resumes if interrupted.

--qa uses the model from train_generator.py --qa. Every sample is prompted with
"Question:" and the model writes a question and its answer. Only complete pairs
are kept: the answer must end (end-of-text token) before the length cap, and the
text must split cleanly into one question and one answer. Rows are written as
{"question": ..., "answer": ...}; drop counts are printed at the end.

    python HardLLM/Step_3/generate_synthetic.py
    python HardLLM/Step_3/generate_synthetic.py --qa
    python HardLLM/Step_3/generate_synthetic.py --total 500 --output /tmp/smoke.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "gpt2-generator")
DEFAULT_OUT = os.path.join(HERE, "out", "synthetic.jsonl")
QA_MODEL = os.path.join(HERE, "models", "gpt2-qa")
QA_OUT = os.path.join(HERE, "out", "synthetic_qa.jsonl")
SEP = "\nAnswer:"   # must match train_generator.qa_text
MIN_ANSWER_WORDS = 3  # must match train_generator.MIN_ANSWER_WORDS
SEED = 313
NOREPEAT_QA = 0      # off for --qa: "no word pair twice" wrecks 400-token answers
QA_LENGTH = 400      # new tokens per --qa sample; 99% of real pairs fit in 512
QA_BATCH_CUDA = 256  # 256 x 400 tokens keeps the KV cache ~4 GB on a 16 GB card


def parse_qa(tokenizer, ids):
    """Turn one --qa generation (token ids after the "Question:" prompt) into
    a row, or say why it was dropped. Returns (row, None) or (None, reason)."""
    eos = tokenizer.eos_token_id
    if eos not in ids:                  # hit the length cap mid-answer
        return None, "unfinished"
    text = tokenizer.decode(ids[:ids.index(eos)])
    parts = ("Question:" + text).split(SEP)
    q = " ".join(parts[0][len("Question:"):].split())
    a = " ".join(parts[-1].split()) if len(parts) == 2 else ""
    if not q or len(a.split()) < MIN_ANSWER_WORDS:
        return None, "malformed"
    return {"question": q, "answer": a}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", action="store_true",
                    help="write question -> answer pairs (see top of file)")
    ap.add_argument("--model", default=None,
                    help="default: models/gpt2-generator, or models/gpt2-qa with --qa")
    ap.add_argument("--output", default=None,
                    help="default: out/synthetic.jsonl, or out/synthetic_qa.jsonl with --qa")
    ap.add_argument("--total", type=int, default=100_000, help="paper: 100,000")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="default: 512 on an NVIDIA GPU (256 with --qa), 128 otherwise (64 with --qa)")
    ap.add_argument("--length", type=int, default=None,
                    help="new tokens per sample; default 128 (paper), 400 with --qa")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_repeat_ngram", type=int, default=None,
                    help="paper: 2 (no word pair may repeat); 0 = off. "
                         "Default 2, or NOREPEAT_QA with --qa")
    args = ap.parse_args()
    if args.model is None:
        args.model = QA_MODEL if args.qa else DEFAULT_MODEL
    if args.output is None:
        args.output = QA_OUT if args.qa else DEFAULT_OUT
    if args.length is None:
        args.length = QA_LENGTH if args.qa else 128
    if args.no_repeat_ngram is None:
        args.no_repeat_ngram = NOREPEAT_QA if args.qa else 2

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16   # half precision: faster sampling
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32
    if args.batch_size is None:
        if device == "cuda":
            args.batch_size = QA_BATCH_CUDA if args.qa else 512
        else:
            args.batch_size = 64 if args.qa else 128
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = sum(1 for _ in open(args.output)) if os.path.exists(args.output) else 0
    if done:
        print(f"resuming: {done:,} samples already written")
    print(f"device: {device}   model: {args.model}   target: {args.total:,}")

    # Different seed per resume point, so a restart doesn't repeat samples.
    torch.manual_seed(SEED + done)
    if args.qa:
        # Training rows start with "Question:" (no leading end-of-text token),
        # so the prompt is exactly that.
        start = tokenizer("Question:")["input_ids"]
    else:
        # Every sample starts from the end-of-text token: unconditional generation.
        start = [tokenizer.eos_token_id]
    prompt = torch.tensor([start] * args.batch_size, device=device)
    mask = torch.ones_like(prompt)

    t0, start_done, empty = time.time(), done, 0
    unfinished = malformed = 0
    first_batch = True

    def write_stats(status):
        if not args.qa:
            return
        stats = {"status": status, "written_total": done,
                 "written_this_session": done - start_done,
                 "dropped_unfinished": unfinished, "dropped_malformed": malformed,
                 "minutes_this_session": round((time.time() - t0) / 60, 1),
                 "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
                 "args": vars(args)}
        with open(os.path.splitext(args.output)[0] + "_stats.json", "w") as sf:
            json.dump(stats, sf, indent=2)

    with open(args.output, "a", encoding="utf-8") as f:
        while done < args.total:
            before = done
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
            for ids in out[:, prompt.shape[1]:].tolist():
                if done >= args.total:
                    break
                if args.qa:
                    row, why = parse_qa(tokenizer, ids)
                    if why == "unfinished":
                        unfinished += 1
                        continue
                    if why == "malformed":
                        malformed += 1
                        continue
                else:
                    text = " ".join(tokenizer.decode(ids, skip_special_tokens=True).split())
                    if not text:
                        empty += 1
                        continue
                    row = {"text": text}
                f.write(json.dumps(row) + "\n")
                done += 1
            f.flush()
            if args.qa and first_batch:
                # Fail fast: a broken format should cost one batch, not an hour.
                ok = done - before
                print(f"first batch: {ok} of {args.batch_size} usable pairs "
                      f"({unfinished} unfinished, {malformed} malformed)")
                with open(args.output, encoding="utf-8") as rf:
                    last = rf.readlines()[-2:]
                for line in last:
                    r = json.loads(line)
                    print(f"  Q: {r['question'][:150]}\n  A: {r['answer'][:200]}\n")
                if ok < args.batch_size // 2:
                    write_stats("stopped: under half the first batch was usable")
                    sys.exit("[stopped] fewer than half the samples came out as complete "
                             "question-answer pairs. Nothing more was generated. "
                             "Bring back Step_3/out/ so it can be diagnosed.")
                first_batch = False
            rate = (time.time() - t0) / max(done - start_done, 1)
            print(f"\r  {done:,}/{args.total:,}  {rate*1000:.0f} ms/sample  "
                  f"ETA {rate*(args.total-done)/3600:.1f} h", end="", flush=True)
    print(f"\nwrote {args.output}")
    write_stats("complete")
    if args.qa:
        print(f"dropped this session: {unfinished:,} unfinished (hit {args.length} tokens), "
              f"{malformed:,} malformed")
    else:
        print(f"empty generations dropped: {empty}")


if __name__ == "__main__":
    main()
