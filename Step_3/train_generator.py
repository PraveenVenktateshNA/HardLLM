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

Settings: the paper and repo give none for this step (no epochs, lr or batch
size anywhere). These are standard GPT-2 fine-tuning defaults, all overridable.

    python HardLLM/Step_3/train_generator.py
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
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

HERE = os.path.dirname(os.path.abspath(__file__))
STEP2_OUT = os.path.join(os.path.dirname(HERE), "Step_2", "out")
DEFAULT_DATA = os.path.join(STEP2_OUT, "selected_centroid_eps0.5_t3.jsonl")
DEFAULT_OUT = os.path.join(HERE, "models", "gpt2-generator")
SEED = 313


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="selected_*.jsonl from Step 2")
    ap.add_argument("--model", default="gpt2", help="gpt2 = 124M parameters")
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128, help="paper: 128")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="stop after N steps (for timing runs); -1 = full run")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    texts = [json.loads(l)["text"] for l in open(args.data)]
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
    print(f"steps: {total:,} ({steps_per_epoch:,} per epoch, batch {args.batch_size})")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=max(int(0.03 * total), 1),   # 3% warmup
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="steps" if args.max_steps < 0 else "no",
        eval_steps=max(steps_per_epoch // 2, 1),
        save_strategy="steps" if args.max_steps < 0 else "no",
        save_steps=max(steps_per_epoch // 2, 1),
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
        seed=SEED,
        dataloader_pin_memory=False,   # not supported on MPS
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        # mlm=False -> next-token prediction; pads each batch to its longest row
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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

    if args.max_steps < 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        summary_path = os.path.join(HERE, "out", "train_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"data": args.data, "train_rows": len(train),
                       "val_rows": len(val), "steps": total,
                       "minutes": round(elapsed / 60, 1),
                       "val_loss": metrics["eval_loss"],
                       "val_perplexity": ppl, "args": vars(args)}, f, indent=2)
        print(f"saved model to {args.output_dir}")
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
