"""Step 4, part 2: fine-tune Llama-3.2-3B with LoRA on one training set.

    python Step_4/train_lora.py --arm hardllm --model <path to Llama-3.2-3B>

--arm is one of hardllm, selected, random, private (see prepare_data.py). Every
arm uses identical settings, so the data is the only difference.

Format, the same as Step 3:  "Question: {q}\\nAnswer: {a}<end of text>"
The loss is on the answer and the end-of-text token only: the model learns to
answer and to stop, not to write patient questions.

Writes models/lora-{arm}/ (the adapter, ~100 MB) and out/train_{arm}.json.
Resumes from the last checkpoint if stopped; skips an arm that is already done.
"""
import argparse
import json
import math
import os
import shutil
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
ARMS = ["hardllm", "selected", "random", "private"]
EXTRA_ARMS = ["hardllm-filtered", "hardllm-fullsample"]   # added after the first results:
                                                         # filter_hardllm.py, build_fullsample_arm.py
SEED = 313

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PAD_TOKEN = "<|finetune_right_pad_id|>"   # Llama 3.2's reserved padding token


def prompt_text(question):
    return f"Question: {' '.join(question.split())}\nAnswer:"


def setup_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = PAD_TOKEN if PAD_TOKEN in tok.get_vocab() else tok.eos_token
    return tok


def encode(tok, question, answer, max_len):
    """input_ids and labels; the question part of labels is -100. The prompt and
    answer are tokenized separately so the boundary is exact."""
    p = tok(prompt_text(question))["input_ids"]                       # with <|begin_of_text|>
    a = tok(" " + " ".join(answer.split()), add_special_tokens=False)["input_ids"]
    ids = (p + a + [tok.eos_token_id])[:max_len]
    labels = ([-100] * len(p) + a + [tok.eos_token_id])[:max_len]
    return {"input_ids": ids, "labels": labels}


def make_collator(tok):
    """Pads to the longest row. Padding is masked by position (attention_mask),
    never by token id -- the Step 3 bug that hid every end-of-text token."""
    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        ids = torch.full((len(features), n), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), n), -100, dtype=torch.long)
        mask = torch.zeros((len(features), n), dtype=torch.long)
        for i, f in enumerate(features):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            labels[i, :k] = torch.tensor(f["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return collate


def add_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS + EXTRA_ARMS)
    ap.add_argument("--model", required=True, help="base Llama-3.2-3B folder")
    ap.add_argument("--output_dir", default=None, help="default: models/lora-{arm}")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--micro_batch", type=int, default=4,
                    help="rows per forward pass; batch_size / micro_batch = accumulation")
    ap.add_argument("--checkpointing", type=int, default=1,
                    help="1 = gradient checkpointing (less memory, ~30%% slower)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_steps", type=int, default=-1, help="for the pre-flight only")
    ap.add_argument("--rows", type=int, default=None, help="use only the first N rows (tests)")
    ap.add_argument("--worst_case", action="store_true",
                    help="pre-flight: stretch every row to --max_len tokens")
    ap.add_argument("--mem_cap", type=float, default=0.95,
                    help="share of GPU memory PyTorch may use. Above it: a clean out-of-memory "
                         "error instead of Windows silently spilling into (very slow) system RAM")
    args = ap.parse_args()
    if args.batch_size % args.micro_batch:
        ap.error("--batch_size must be divisible by --micro_batch")
    accum = args.batch_size // args.micro_batch
    out_dir = args.output_dir or os.path.join(HERE, "models", f"lora-{args.arm}")
    summary_path = os.path.join(OUT, f"train_{args.arm}.json")
    real_run = args.max_steps < 0

    if real_run and os.path.exists(summary_path) and \
            os.path.exists(os.path.join(out_dir, "adapter_model.safetensors")):
        print(f"[skip] {args.arm} is already trained ({summary_path})")
        return json.load(open(summary_path))

    cuda = torch.cuda.is_available()
    if cuda and args.mem_cap:
        torch.cuda.set_per_process_memory_fraction(args.mem_cap)
    print(f"arm: {args.arm}   device: {torch.cuda.get_device_name(0) if cuda else 'cpu'}")
    tok = setup_tokenizer(args.model)
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"train_{args.arm}.jsonl"), encoding="utf-8")]
    if args.rows:
        rows = rows[:args.rows]
    if args.worst_case:
        rows = [{"question": r["question"], "answer": (r["answer"] + " ") * 40} for r in rows]
    ds = Dataset.from_list([encode(tok, r["question"], r["answer"], args.max_len) for r in rows])
    n_tokens = sum(len(x) for x in ds["input_ids"])
    print(f"rows: {len(ds):,}   tokens: {n_tokens:,} (mean {n_tokens / len(ds):.0f})   "
          f"truncated at {args.max_len}: {sum(len(x) == args.max_len for x in ds['input_ids']):,}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if cuda else torch.float32)
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = add_lora(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA r={LORA_R}: {trainable / 1e6:.1f}M trainable parameters")

    steps = args.max_steps if not real_run else math.ceil(len(ds) / args.batch_size)
    print(f"steps: {steps:,}   batch {args.batch_size} = {args.micro_batch} x {accum}   "
          f"checkpointing: {'on' if args.checkpointing else 'off'}")

    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(int(0.03 * steps), 1),
        weight_decay=0.0,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=accum,
        train_sampling_strategy="group_by_length",   # longest batch first: OOM early
        logging_steps=25,
        save_strategy="steps" if real_run else "no",
        save_steps=250,
        save_total_limit=1,
        report_to=[],
        seed=SEED,
        bf16=cuda,
        dataloader_pin_memory=cuda,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=make_collator(tok), processing_class=tok)

    resume = None
    if real_run and os.path.isdir(out_dir) and any(
            d.startswith("checkpoint-") for d in os.listdir(out_dir)):
        resume = True
        print(f"resuming from the latest checkpoint in {out_dir}")

    t0 = time.time()
    res = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    # tokens actually trained on; for a short pre-flight run this is an estimate
    # (and pessimistic: length grouping feeds the longest rows first)
    done_tokens = min(steps * args.batch_size, len(ds)) * n_tokens / len(ds)
    result = {"arm": args.arm, "steps": steps, "seconds": elapsed,
              "tokens": n_tokens, "tokens_per_second": done_tokens / elapsed,
              "train_loss": res.training_loss,
              "first_logged_loss": losses[0] if losses else None,
              "last_logged_loss": losses[-1] if losses else None}
    if cuda:
        result["peak_reserved_frac"] = round(
            torch.cuda.max_memory_reserved() / torch.cuda.get_device_properties(0).total_memory, 3)
    print(f"trained {steps:,} steps in {elapsed / 60:.1f} min, loss {res.training_loss:.3f}")
    if not real_run:
        print("PROBE " + json.dumps(result))

    if real_run:
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        os.makedirs(OUT, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({**result, "minutes": round(elapsed / 60, 1), "rows": len(ds),
                       "answer_only_loss": True, "eos_learned": True,
                       "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                                "targets": LORA_TARGETS},
                       "args": vars(args)}, f, indent=2)
        for d in os.listdir(out_dir):   # adapter saved: checkpoints are no longer needed
            if d.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)
        print(f"saved {out_dir}\nwrote {summary_path}")
    return result


if __name__ == "__main__":
    main()
