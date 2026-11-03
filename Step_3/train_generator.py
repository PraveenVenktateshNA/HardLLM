"""Step 3a: fine-tune GPT-2 on the public rows Step 2 selected.

Replaces the paper's fine-tune-generator.py, which can't run here: it forces
CUDA, imports dp_transformers (not installed, and never used for privacy --
it builds the DP machinery then trains with a plain Trainer), and reads files
that don't exist.

Privacy: this script only ever sees PUBLIC text -- the rows Step 2 picked using
noisy counts. No private record is read, so training costs no privacy budget.

What it learns: plain language modelling on the selected text, so it can write
more text in the same style. The paper's version prompts on label columns; our
rows carry no labels, so generation is unconditional.

--qa trains on question -> answer pairs instead, so the synthetic data can teach
Step 4's model to ANSWER, not just to write patient questions. It keeps only
the selected rows whose `target` is a real answer to `text` (QA_SOURCES), and
formats each as "Question: ...\nAnswer: ...". Outputs go to separate paths, so
the question-only run is left untouched.

Settings: the paper and repo give none for this step (no epochs, lr or batch
size anywhere). These are standard GPT-2 fine-tuning defaults, all overridable.

    python HardLLM/Step_3/train_generator.py
    python HardLLM/Step_3/train_generator.py --qa
    python HardLLM/Step_3/train_generator.py --max_steps 30 --output_dir /tmp/smoke
"""
import argparse
import json
import math
import os
import random
import time

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

HERE = os.path.dirname(os.path.abspath(__file__))
STEP2_OUT = os.path.join(os.path.dirname(HERE), "Step_2", "out")
DEFAULT_DATA = os.path.join(STEP2_OUT, "selected_centroid_eps0.5_t3.jsonl")
DEFAULT_OUT = os.path.join(HERE, "models", "gpt2-generator")
QA_OUT = os.path.join(HERE, "models", "gpt2-qa")
SEED = 313

# Sources whose `target` answers the `text` question. DialogSum is left out on
# purpose: its target is a summary of the dialogue, not an answer to it.
QA_SOURCES = {
    "lavita/ChatDoctor-HealthCareMagic-100k",   # patient question -> doctor reply
    "lavita/MedQuAD",                           # NIH question -> answer
    "LLukas22/fiqa",                            # finance question -> answer
}


# Pairs whose answer is shorter than this are dropped: 27 ChatDoctor "answers"
# are just the scraped heading "Brief Answer". generate_synthetic.parse_qa
# rejects generations by the same rule (the tests check the two match).
MIN_ANSWER_WORDS = 3


def is_qa_pair(row):
    return (row["source"] in QA_SOURCES and bool(row.get("target"))
            and len(str(row["target"]).split()) >= MIN_ANSWER_WORDS)


def qa_text(question, answer):
    """One training example. Whitespace is flattened so the only newline is the
    separator, which is what generate_synthetic.py --qa splits on."""
    return f"Question: {' '.join(question.split())}\nAnswer: {' '.join(answer.split())}"


def make_collator(tokenizer):
    """Pads each batch to its longest row and masks only the padding.

    GPT-2 pads with its end-of-text token, and DataCollatorForLanguageModeling
    masks by token id, so it also masked every row's real end-of-text: the model
    never learned to stop (the first --qa run: 0 of 128 answers finished). Here
    padding is masked by position (attention_mask), so the end-of-text is learned.
    """
    def collate(features):
        batch = tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="selected_*.jsonl from Step 2")
    ap.add_argument("--model", default="gpt2", help="gpt2 = 124M parameters")
    ap.add_argument("--qa", action="store_true",
                    help="train on question -> answer pairs (see top of file)")
    ap.add_argument("--output_dir", default=None,
                    help="default: models/gpt2-generator, or models/gpt2-qa with --qa")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="rows per optimizer step. Default 16 (fed as 4 x 4 with --qa)")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="split each batch into this many chunks to save memory. "
                         "Default 1, or 4 with --qa: 512-token rows at batch 16 "
                         "need ~2x the 5070 Ti's 16 GB")
    ap.add_argument("--seq_len", type=int, default=None,
                    help="default 128 (paper); 512 with --qa, which fits 99%% of pairs")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="stop after N steps (for timing runs); -1 = full run")
    args = ap.parse_args()
    if args.output_dir is None:
        args.output_dir = QA_OUT if args.qa else DEFAULT_OUT
    if args.seq_len is None:
        args.seq_len = 512 if args.qa else 128
    if args.batch_size is None:
        args.batch_size = 16
    if args.grad_accum is None:
        args.grad_accum = 4 if args.qa else 1
    if args.batch_size % args.grad_accum:
        ap.error("--batch_size must be divisible by --grad_accum")
    micro = args.batch_size // args.grad_accum

    if torch.cuda.is_available():
        device = "cuda"
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"device: {device}")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.qa:
        texts = [qa_text(r["text"], r["target"]) for r in rows if is_qa_pair(r)]
        print(f"question -> answer pairs: {len(texts):,} of {len(rows):,} selected rows")
    else:
        texts = [r["text"] for r in rows]
    random.Random(SEED).shuffle(texts)
    n_val = int(len(texts) * args.val_frac)
    val, train = texts[:n_val], texts[n_val:]
    print(f"data: {args.data}")
    print(f"train rows: {len(train):,}   validation rows: {len(val):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token

    def tokenize(batch):
        return tokenizer([t + tokenizer.eos_token for t in batch["text"]],
                         truncation=True, max_length=args.seq_len)

    ds_train = Dataset.from_dict({"text": train}).map(
        tokenize, batched=True, remove_columns=["text"])
    ds_val = Dataset.from_dict({"text": val}).map(
        tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model)
    print(f"parameters: {model.num_parameters()/1e6:.0f}M")

    steps_per_epoch = math.ceil(len(ds_train) / args.batch_size)
    total = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    print(f"steps: {total:,} ({steps_per_epoch:,} per epoch, batch {args.batch_size}"
          f" = {micro} x {args.grad_accum})")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=max(int(0.03 * total), 1),   # 3% warmup
        weight_decay=0.01,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=micro * 2,
        eval_strategy="steps" if args.max_steps < 0 else "no",
        eval_steps=max(steps_per_epoch // 2, 1),
        save_strategy="steps" if args.max_steps < 0 else "no",
        save_steps=max(steps_per_epoch // 2, 1),
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
        seed=SEED,
        bf16=device == "cuda",                  # ~2x faster on NVIDIA; MPS stays fp32
        dataloader_pin_memory=device == "cuda",  # not supported on MPS
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=make_collator(tokenizer),
    )

    # Resume if a checkpoint exists: a multi-hour run shouldn't restart from 0.
    resume = None
    if args.max_steps < 0 and os.path.isdir(args.output_dir):
        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            resume = True
            print(f"resuming from latest checkpoint in {args.output_dir}")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0

    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"])
    print("=" * 60)
    print(f"trained {total:,} steps in {elapsed/60:.1f} min "
          f"({elapsed/total:.3f} s/step)")
    print(f"validation loss {metrics['eval_loss']:.3f}   perplexity {ppl:.1f}")
    result = {"steps": total, "seconds": elapsed, "val_loss": metrics["eval_loss"]}

    if args.max_steps < 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        summary_path = os.path.join(
            HERE, "out", "train_summary_qa.json" if args.qa else "train_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"data": args.data, "train_rows": len(train),
                       "val_rows": len(val), "steps": total,
                       "minutes": round(elapsed / 60, 1),
                       "val_loss": metrics["eval_loss"],
                       "val_perplexity": ppl, "eos_learned": True,
                       "args": vars(args)}, f, indent=2)
        print(f"saved model to {args.output_dir}")
        print(f"wrote {summary_path}")
    return result


if __name__ == "__main__":
    main()
