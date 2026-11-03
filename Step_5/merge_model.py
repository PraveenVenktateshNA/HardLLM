"""Step 5, part 1: fold the Step 4 LoRA adapter into Llama-3.2-3B, one normal model.

    python Step_5/merge_model.py --model <path to Llama-3.2-3B> [--arm selected]

The final model is the Step 2 arm ("selected"): Step 3's synthetic data made
the model worse than base (Step_4/out/results_step4_fullsample.md), so the
pipeline rolls back to the Step 2 output. The merge is the same one
Step_4/evaluate_models.py does in memory before scoring, saved to disk.

Writes models/{arm}-merged/ (~6.4 GB, bf16 safetensors, with the base
tokenizer files copied unchanged). Does nothing if it already exists.
"""
import argparse
import os
import shutil
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.join(os.path.dirname(HERE), "Step_4")
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def merged_dir(arm):
    return os.path.join(HERE, "models", f"{arm}-merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base Llama-3.2-3B folder")
    ap.add_argument("--arm", default="selected")
    args = ap.parse_args()
    out = merged_dir(args.arm)
    if os.path.exists(os.path.join(out, "config.json")):
        print(f"[skip] {out} already exists")
        return
    adapter = os.path.join(STEP4, "models", f"lora-{args.arm}")
    if not os.path.exists(os.path.join(adapter, "adapter_model.safetensors")):
        raise SystemExit(f"[stopped] {adapter} is missing")

    t0 = time.time()
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model.to("cuda" if cuda else "cpu"), adapter).merge_and_unload()
    tmp = out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    for f in TOKENIZER_FILES:
        shutil.copy(os.path.join(args.model, f), tmp)
    os.replace(tmp, out)
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
