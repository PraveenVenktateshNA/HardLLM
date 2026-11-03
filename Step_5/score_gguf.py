"""Step 5, part 3: how much did compressing the model cost? Scored the same way
as Step 4, on the same 1,000 held-out real doctor answers.

    python Step_5/score_gguf.py [--arm selected]

Two measures per version (merged, f16, q8_0, q4_k_m):

  1. answer perplexity, exactly Step 4's number: the GGUF file's weights are
     read back (gguf-py, the reader that ships with llama.cpp), loaded into
     the PyTorch model and scored with Step_4/evaluate_models.answer_nll.
     The f16 file must give the merged model's score: that proves the
     reading back is right, so the q8_0 and q4_k_m scores are the files'.
  2. llama.cpp's own perplexity tool on the plain test text, running the
     file the way the phone will. Not the same number as 1 (it scores the
     questions too), so compare versions with each other, not with Step 4.

Also 20 demo answers per compressed version, and how much compression moved
the weights. Writes out/eval_{variant}.json and out/llamacpp_{variant}.json
(each skipped if there, so a stopped run resumes), out/results_step5.{md,json}
and out/demo_answers_step5.md.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.join(os.path.dirname(HERE), "Step_4")
OUT = os.path.join(HERE, "out")
# the bundled Windows Python does not put the script's folder on the path
sys.path.insert(0, STEP4)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "portable", "llama.cpp", "convert", "gguf-py"))
import evaluate_models as em  # noqa: E402
import export_gguf as ex  # noqa: E402
import gguf  # noqa: E402
import train_lora as tl  # noqa: E402

VARIANTS = ["merged", "f16", "q8_0", "q4_k_m"]
PPL_CTX, PPL_CHUNKS = 512, 100


def unpermute(w, n_head):
    """Undo the converter's reordering of the Q and K rows (llama.cpp's rope
    layout, conversion/llama.py LlamaModel.permute)."""
    return (w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:])
            .swapaxes(1, 2).reshape(w.shape))


def gguf_weights(path, model):
    """Yield (parameter, weight as float32 numpy) for every model parameter,
    read from a GGUF file. Fails if a parameter has no tensor or a tensor is
    left over, so nothing is silently kept from the model that was there."""
    cfg = model.config
    reader = gguf.GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, cfg.num_hidden_layers)
    used = set()
    for name, p in model.named_parameters():   # a tied lm_head is listed once, as embed_tokens
        g = names.get_name(name, try_suffixes=(".weight", ".bias"))
        if g not in tensors:
            raise SystemExit(f"{name}: no tensor {g} in {path}")
        t = tensors[g]
        w = gguf.quants.dequantize(t.data, t.tensor_type).reshape([int(x) for x in reversed(t.shape)])
        if name.endswith("q_proj.weight"):
            w = unpermute(w, cfg.num_attention_heads)
        elif name.endswith("k_proj.weight"):
            w = unpermute(w, cfg.num_key_value_heads)
        if tuple(w.shape) != tuple(p.shape):
            raise SystemExit(f"{name}: shape {w.shape} in the file, {tuple(p.shape)} in the model")
        used.add(g)
        yield p, w, t.tensor_type.name
    extra = set(tensors) - used - {"rope_freqs.weight"}   # llama 3 rope factors, rebuilt from config
    if extra:
        raise SystemExit(f"tensors in {path} the model has no place for: {sorted(extra)[:5]}")


@torch.no_grad()
def load_gguf(model, path, original):
    """Overwrite every weight with the file's. Returns the tensor types used
    and how far the weights moved from the merged model (relative, overall)."""
    types, diff2, norm2 = {}, 0.0, 0.0
    for p, w, kind in gguf_weights(path, model):
        new = torch.from_numpy(np.ascontiguousarray(w)).to(p.device, p.dtype)
        ref = original[p].to(p.device).float()
        diff2 += (new.float() - ref).pow(2).sum().item()
        norm2 += ref.pow(2).sum().item()
        p.copy_(new)
        types[kind] = types.get(kind, 0) + 1
    return types, math.sqrt(diff2 / norm2)


def llama_perplexity(path, text_file, log_dir=OUT):
    t0 = time.time()
    r = subprocess.run([ex.exe("llama-perplexity"), "-m", path, "-f", text_file,
                        "-c", str(PPL_CTX), "--chunks", str(PPL_CHUNKS), "-ngl", "99"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = r.stdout + r.stderr
    with open(os.path.join(log_dir, f"llamacpp_{os.path.basename(path)}.log"), "w", encoding="utf-8") as f:
        f.write(log)
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", log)
    if r.returncode or not m:
        raise SystemExit(f"llama-perplexity failed on {path}, see out/llamacpp_*.log")
    device = [l.strip() for l in log.splitlines() if re.search(r"Vulkan\d*:|ggml_vulkan: Found|offloaded", l)]
    return {"ppl": float(m.group(1)), "plus_minus": float(m.group(2)), "device_lines": device[:4],
            "tokens": PPL_CTX * PPL_CHUNKS, "minutes": round((time.time() - t0) / 60, 1)}


def write_test_text(rows, out_dir=OUT):
    path = os.path.join(out_dir, "test_text_for_llamacpp.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{tl.prompt_text(r['question'])} {' '.join(r['answer'].split())}\n\n")
    return path


def report(arm, results, demo):
    s4 = {m: json.load(open(os.path.join(STEP4, "out", f"eval_{m}.json"))) for m in ("base", arm)}
    lines = ["# Step 5 results: the final model, compressed for the phone", "",
             f"Final model: Step 4 \"{arm}\" (Step 2 data). Step 4 scores for reference: "
             f"base {s4['base']['answer_perplexity']:.2f}, {arm} {s4[arm]['answer_perplexity']:.2f}.", "",
             "| Version | File size | Answer perplexity (Step 4 test, lower is better) | "
             "llama.cpp perplexity (whole text) | Weights changed by |",
             "|---|---|---|---|---|"]
    for v in VARIANTS:
        r = results[v]
        lc = f"{r['llamacpp']['ppl']:.3f} ± {r['llamacpp']['plus_minus']:.3f}" if r.get("llamacpp") else "-"
        lines.append(f"| {v} | {r['file_gb']:.2f} GB | {r['answer_perplexity']:.3f} | {lc} | "
                     f"{r['weight_change']:.2%} |")
    q4, mg = results["q4_k_m"], results["merged"]
    lines += ["", f"- 4-bit costs {q4['answer_perplexity'] - mg['answer_perplexity']:+.3f} answer perplexity "
                  f"({q4['answer_perplexity'] / mg['answer_perplexity'] - 1:+.1%}) for a "
                  f"{mg['file_gb'] / q4['file_gb']:.1f}x smaller file; it is "
                  f"{s4['base']['answer_perplexity'] - q4['answer_perplexity']:.2f} better than base.",
              "- Answer perplexity of a GGUF: the file's weights read back into PyTorch. The f16 row "
              "matching the merged row shows the reading back is exact.",
              "- llama.cpp perplexity: the real runtime on the first "
              f"{PPL_CTX * PPL_CHUNKS:,} tokens of the test text, questions included."]
    with open(os.path.join(OUT, "results_step5.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "results_step5.json"), "w") as f:
        json.dump({v: {k: x for k, x in results[v].items() if k != "demo_answers"} for v in VARIANTS},
                  f, indent=2)
    with open(os.path.join(OUT, "demo_answers_step5.md"), "w", encoding="utf-8") as f:
        for i, q in enumerate(demo):
            f.write(f"## {i + 1}. {q['question']}\n\n**{arm} (Step 4):** {s4[arm]['demo_answers'][i]}\n\n")
            for v in ("q8_0", "q4_k_m"):
                f.write(f"**{v}:** {results[v]['demo_answers'][i]}\n\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="selected")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    merged = os.path.join(HERE, "models", f"{args.arm}-merged")
    files = {"merged": None, **{v: ex.gguf_path(args.arm, v) for v in VARIANTS[1:]}}
    for v, p in files.items():
        if p and not os.path.exists(p):
            raise SystemExit(f"[stopped] {p} is missing, run export_gguf.py first")

    tok = tl.setup_tokenizer(merged)
    test, demo = em.read("test_private.jsonl"), em.read("demo_questions.jsonl")
    text_file = write_test_text(test)
    model = None
    results = {}
    for v in VARIANTS:
        path = os.path.join(OUT, f"eval_{v}.json")
        if os.path.exists(path):
            results[v] = json.load(open(path))
            print(f"[skip] {v}: already scored")
            continue
        if model is None:
            cuda = torch.cuda.is_available()
            model = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16)
            model = model.to("cuda" if cuda else "cpu").eval()
            original = {p: p.detach().to("cpu", copy=True) for p in model.parameters()}
        print(f"== {v}")
        t0 = time.time()
        r = {"variant": v}
        if files[v]:
            r["tensor_types"], r["weight_change"] = load_gguf(model, files[v], original)
            r["file_gb"] = round(os.path.getsize(files[v]) / 2**30, 3)
        else:
            r["tensor_types"], r["weight_change"] = {"BF16": len(original)}, 0.0
            r["file_gb"] = round(sum(os.path.getsize(os.path.join(merged, f)) for f in os.listdir(merged)
                                     if f.endswith(".safetensors")) / 2**30, 3)
        nll, n_tok = em.answer_nll(model, tok, test)
        r.update(answer_nll=round(nll, 4), answer_perplexity=round(math.exp(nll), 3),
                 answer_tokens_scored=n_tok)
        if v in ("q8_0", "q4_k_m"):
            r["demo_answers"] = em.demo_answers(model, tok, demo)
        r["minutes"] = round((time.time() - t0) / 60, 1)
        print(f"   answer perplexity {r['answer_perplexity']:.3f}   weights changed {r['weight_change']:.2%}"
              f"   ({r['minutes']} min)")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        results[v] = r

    # llama.cpp needs the GPU memory PyTorch was holding
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for v in VARIANTS[1:]:
        path = os.path.join(OUT, f"llamacpp_{v}.json")
        if not os.path.exists(path):
            print(f"== llama.cpp perplexity, {v}")
            with open(path, "w") as f:
                json.dump(llama_perplexity(files[v], text_file), f, indent=2)
        results[v]["llamacpp"] = json.load(open(path))
        print(f"   {results[v]['llamacpp']['ppl']:.3f}   ({results[v]['llamacpp']['minutes']} min)")
    report(args.arm, results, demo)


if __name__ == "__main__":
    main()
