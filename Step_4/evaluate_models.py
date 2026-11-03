"""Step 4, part 3: score the base model and every fine-tuned model the same way.

    python Step_4/evaluate_models.py --model <path to Llama-3.2-3B>

For each of base, hardllm, selected, random, private (those that are trained),
and with --with_extra also the extra arms (hardllm-filtered, hardllm-fullsample):

  1. answer perplexity on 1,000 held-out private pairs (data/test_private.jsonl):
     how well the model predicts a real doctor's answer to a real patient.
     Lower is better. This is the main number.
  2. MMLU medical, 1,089 multiple-choice questions, zero-shot: did fine-tuning
     on (possibly wrong) synthetic answers damage medical knowledge?
     Higher is better; compare each model with base.
  3. answers to the 20 demo questions, greedy, side by side, to read.

Writes out/eval_{name}.json per model (skipped if already there, so a stopped
run resumes), then out/results_step4.json, out/results_step4.md and
out/demo_answers.md. With --with_extra the report goes to
results_step4{suffix}.{md,json} and demo_answers{suffix}.md instead (--suffix,
default _filtered), so an earlier report is never overwritten.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM

# the bundled Windows Python does not put the script's folder on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lora as tl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = tl.DATA, tl.OUT
MODELS = ["base"] + tl.ARMS
LETTERS = ["A", "B", "C", "D"]
GEN_TOKENS = 256


def read(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name), encoding="utf-8")]


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


@torch.no_grad()
def answer_nll(model, tok, rows, max_len=512, batch=4):
    """Mean negative log-likelihood per answer token (answer + end-of-text),
    given the question. Same encoding as training."""
    collate = tl.make_collator(tok)
    enc = sorted((tl.encode(tok, r["question"], r["answer"], max_len) for r in rows),
                 key=lambda e: len(e["input_ids"]))
    total, count = 0.0, 0
    dev = next(model.parameters()).device
    for chunk in batches(enc, batch):
        b = {k: v.to(dev) for k, v in collate(chunk).items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        logits, labels = logits[:, :-1].float(), b["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="sum")
        total += loss.item()
        count += (labels != -100).sum().item()
    return total / count, count


def mmlu_prompt(r):
    subject = r["subject"].replace("_", " ")
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, r["choices"]))
    return (f"The following is a multiple choice question about {subject}.\n\n"
            f"{r['question']}\n{opts}\nAnswer:")


@torch.no_grad()
def mmlu(model, tok, rows, batch=8):
    """Zero-shot: the letter (" A".." D") with the highest probability after
    "Answer:" is the model's choice."""
    letter_ids = [tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in LETTERS]
    dev = next(model.parameters()).device
    right = {}
    for chunk in batches(rows, batch):
        enc = [tok(mmlu_prompt(r))["input_ids"] for r in chunk]
        n = max(map(len, enc))
        ids = torch.full((len(enc), n), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e)
            mask[i, :len(e)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
        last = mask.sum(1).to(dev) - 1
        scores = logits[torch.arange(len(enc), device=dev), last][:, letter_ids]
        for r, pick in zip(chunk, scores.argmax(-1).tolist()):
            right.setdefault(r["subject"], []).append(pick == r["answer"])
    per = {s: round(sum(v) / len(v), 4) for s, v in sorted(right.items())}
    allv = [x for v in right.values() for x in v]
    return round(sum(allv) / len(allv), 4), per


@torch.no_grad()
def demo_answers(model, tok, questions, max_new=GEN_TOKENS):
    """Greedy answers. Left padding so every prompt ends where generation starts."""
    tok.padding_side = "left"
    dev = next(model.parameters()).device
    prompts = [tl.prompt_text(q["question"]) for q in questions]
    out = []
    for chunk in batches(prompts, 10):
        b = tok(chunk, return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**b, do_sample=False, max_new_tokens=max_new,
                             repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        for row in gen[:, b["input_ids"].shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True)
            # a model that never learned to stop runs on into a new "Question:"
            out.append(" ".join(text.split("\nQuestion:")[0].split()))
    tok.padding_side = "right"
    return out


def load(base_path, name):
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16 if cuda else torch.float32)
    if name != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, os.path.join(HERE, "models", f"lora-{name}")).merge_and_unload()
    return model.to("cuda" if cuda else "cpu").eval()


def evaluate_one(base_path, name, tok, test, mm, demo):
    t0 = time.time()
    model = load(base_path, name)
    nll, n_tok = answer_nll(model, tok, test)
    acc, per = mmlu(model, tok, mm)
    answers = demo_answers(model, tok, demo)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"model": name, "answer_nll": round(nll, 4), "answer_perplexity": round(math.exp(nll), 3),
            "answer_tokens_scored": n_tok, "mmlu_medical": acc, "mmlu_by_subject": per,
            "demo_answers": answers, "minutes": round((time.time() - t0) / 60, 1)}


def report(results, demo, suffix=""):
    names = [m for m in MODELS + (tl.EXTRA_ARMS if suffix else []) if m in results]
    base = results.get("base")
    lines = ["# Step 4 results", "",
             "| Model | Trained on | Answer perplexity (lower is better) | MMLU medical (higher is better) |",
             "|---|---|---|---|"]
    what = {"base": "nothing (Llama-3.2-3B as released)",
            "hardllm": "HardLLM synthetic pairs (full pipeline)",
            "selected": "public pairs chosen by private-data selection",
            "random": "public pairs chosen at random",
            "private": "the real private data (ceiling, not private)",
            "hardllm-filtered": "HardLLM synthetic pairs, worse-matching half removed",
            "hardllm-fullsample": "HardLLM synthetic pairs, GPT-2 sampled without top-k/top-p cut"}
    for m in names:
        r = results[m]
        d = f" ({r['mmlu_medical'] - base['mmlu_medical']:+.1%} vs base)" if base and m != "base" else ""
        lines.append(f"| {m} | {what[m]} | {r['answer_perplexity']:.2f} | {r['mmlu_medical']:.1%}{d} |")
    if suffix:
        lines += ["", "MMLU, zero-shot: not valid for the fine-tuned models. They were trained to "
                      "write a sentence after \"Answer:\", the same cue MMLU ends with, so they "
                      "pick the letter that also starts sentences (\"A\"). Base is valid.", ""]
        if "random" in results:
            for k in ("hardllm", "hardllm-filtered", "hardllm-fullsample", "selected", "private"):
                if k in results:
                    lines.append(f"- {k}: perplexity {results[k]['answer_perplexity']:.2f} vs random "
                                 f"{results['random']['answer_perplexity']:.2f} and base "
                                 f"{results['base']['answer_perplexity']:.2f}")
    elif all(k in results for k in ("random", "hardllm", "private")):
        ra, ha, pa = (results[k]["answer_nll"] for k in ("random", "hardllm", "private"))
        if ra != pa:
            lines += ["", f"HardLLM closes {(ra - ha) / (ra - pa):.0%} of the gap between random "
                          f"selection and training on the private data (measured in answer loss)."]
    with open(os.path.join(OUT, f"results_step4{suffix}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, f"demo_answers{suffix}.md"), "w", encoding="utf-8") as f:
        for i, q in enumerate(demo):
            f.write(f"## {i + 1}. {q['question']}\n\n")
            for m in names:
                f.write(f"**{m}:** {results[m]['demo_answers'][i]}\n\n")
    with open(os.path.join(OUT, f"results_step4{suffix}.json"), "w") as f:
        json.dump({m: {k: v for k, v in results[m].items() if k != "demo_answers"} for m in names},
                  f, indent=2)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base Llama-3.2-3B folder")
    ap.add_argument("--with_extra", action="store_true",
                    help="also evaluate the extra arms; report to *{suffix} files")
    ap.add_argument("--suffix", default="_filtered", help="report file suffix with --with_extra")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tok = tl.setup_tokenizer(args.model)
    test, mm, demo = read("test_private.jsonl"), read("mmlu_medical.jsonl"), read("demo_questions.jsonl")
    results = {}
    for name in MODELS + (tl.EXTRA_ARMS if args.with_extra else []):
        path = os.path.join(OUT, f"eval_{name}.json")
        if os.path.exists(path):
            results[name] = json.load(open(path))
            print(f"[skip] {name}: already evaluated")
            continue
        if name != "base" and not os.path.exists(
                os.path.join(HERE, "models", f"lora-{name}", "adapter_model.safetensors")):
            print(f"[skip] {name}: not trained")
            continue
        print(f"== evaluating {name}")
        r = evaluate_one(args.model, name, tok, test, mm, demo)
        print(f"   answer perplexity {r['answer_perplexity']:.2f}   MMLU medical "
              f"{r['mmlu_medical']:.1%}   ({r['minutes']} min)")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        results[name] = r
    report(results, demo, args.suffix if args.with_extra else "")


if __name__ == "__main__":
    main()
