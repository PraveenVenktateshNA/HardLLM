"""Step 5 code tests. RUN_STEP5_EXPORT.bat runs these first. About 1 min.

The real code end to end on a tiny random Llama (2 layers, the real
tokenizer): convert to GGUF, quantize with llama-quantize, read the files back
into PyTorch, run llama.cpp's perplexity tool. Works in a temporary folder;
models/ and out/ are not touched. Exit 1 on any failure.

    python Step_5/tests/test_step5.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import torch
from transformers import AutoConfig, LlamaForCausalLM

STEP5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STEP5)
import export_gguf as ex  # noqa: E402
import merge_model as mm  # noqa: E402
import score_gguf as sg  # noqa: E402

LLAMA = os.path.join(sg.STEP4, "portable", "llama-3.2-3b")
BAT = os.path.join(STEP5, "portable", "RUN_STEP5_EXPORT.bat")
results = []


def test(fn):
    try:
        fn()
        results.append((fn.__name__, None))
        print(f"  [ok]   {fn.__name__}")
    except Exception:
        results.append((fn.__name__, traceback.format_exc()))
        print(f"  [FAIL] {fn.__name__}\n{traceback.format_exc()}")
    return fn


def scripts_start_as_the_bat_runs_them():
    for s in ["merge_model.py", "export_gguf.py", "score_gguf.py", "run_convert.py"]:
        r = subprocess.run([sys.executable, os.path.join(STEP5, s), "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{s}: {r.stderr[-800:]}"


def unpermute_undoes_the_converter():
    sys.path.insert(0, os.path.join(STEP5, "portable", "llama.cpp", "convert"))
    from conversion.llama import LlamaModel
    cfg = AutoConfig.from_pretrained(LLAMA)
    for heads, kv in [(cfg.num_attention_heads, cfg.num_attention_heads),
                      (cfg.num_attention_heads, cfg.num_key_value_heads)]:
        w = torch.randn(kv * cfg.head_dim, 16)
        back = sg.unpermute(LlamaModel.permute(w, heads, kv).numpy(), kv)
        assert (torch.from_numpy(back) == w).all(), f"heads {heads}, kv {kv}"


def tiny_model_round_trip():
    tmp = tempfile.mkdtemp(prefix="step5_test_")
    try:
        cfg = AutoConfig.from_pretrained(LLAMA)
        cfg.update(dict(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=64))
        torch.manual_seed(0)
        model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
        hf = os.path.join(tmp, "tiny")
        model.save_pretrained(hf)
        for f in mm.TOKENIZER_FILES:
            shutil.copy(os.path.join(LLAMA, f), hf)

        f16 = os.path.join(tmp, "tiny-f16.gguf")
        ex.run([sys.executable, os.path.join(STEP5, "run_convert.py"), hf,
                "--outfile", f16, "--outtype", "f16"], f16)
        names = {t.name for t in sg.gguf.GGUFReader(f16).tensors}
        assert "rope_freqs.weight" in names, "the file lost Llama 3's rope scaling"
        files = {"f16": f16}
        for v, q in ex.QUANTS.items():
            files[v] = os.path.join(tmp, f"tiny-{v}.gguf")
            ex.run([ex.exe("llama-quantize"), f16, files[v], q], files[v])

        original = {p: p.detach().clone() for p in model.parameters()}
        ids = torch.randint(0, 128_000, (2, 24))
        with torch.no_grad():
            ref = model(input_ids=ids).logits.float()
        limits = {"f16": 1e-3, "q8_0": 0.02, "q4_k_m": 0.15}
        for v, path in files.items():
            types, change = sg.load_gguf(model, path, original)
            with torch.no_grad():
                out = model(input_ids=ids).logits.float()
            print(f"         {v}: weights changed {change:.3%}, types {types}, "
                  f"logits max diff {(out - ref).abs().max().item():.4f}")
            assert change < limits[v], f"{v}: weights changed by {change:.2%}"
            if v == "f16":
                assert torch.allclose(out, ref, atol=1e-2), "f16 file does not give the same model"
        assert "Q4_K" in sg.load_gguf(model, files["q4_k_m"], original)[0], "q4_k_m holds no 4-bit tensors"

        text = sg.write_test_text(sg.em.read("test_private.jsonl")[:300], tmp)
        r = sg.llama_perplexity(files["q4_k_m"], text, tmp)
        print(f"         llama.cpp on tiny q4_k_m: ppl {r['ppl']:.0f}, device {r['device_lines'][:2]}")
        assert r["ppl"] > 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bat_file():
    raw = open(BAT, "rb").read()
    assert raw.count(b"\r\n") == raw.count(b"\n"), "the .bat must have Windows line endings"
    s = raw.decode()
    for x in ["tests\\test_step5.py", "merge_model.py", "export_gguf.py", "score_gguf.py",
              "tests\\check_results_step5.py", "--model \"%LLAMA%\"", "..\\..\\Step_4"]:
        assert x in s, f"the .bat does not mention {x}"


def main():
    print("Step 5 code tests")
    for fn in [scripts_start_as_the_bat_runs_them, unpermute_undoes_the_converter,
               tiny_model_round_trip, bat_file]:
        test(fn)
    failed = [n for n, e in results if e]
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
