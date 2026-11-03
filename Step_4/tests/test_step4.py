"""Code tests for Step 4. RUN_STEP4_ON_NVIDIA.bat runs these first, on the PC,
and stops if any fail. About a minute.

The real training and evaluation code runs end to end on a tiny random Llama
(same tokenizer and architecture as Llama-3.2-3B, 2 layers of width 64), so a
code bug costs a minute instead of hours. What is NOT tested here: real memory
and speed (preflight_step4.py measures those on the real model) and real
quality (that is the experiment).

    python Step_4/tests/test_step4.py
"""
import collections
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
STEP4 = os.path.dirname(HERE)
PORT = os.path.join(STEP4, "portable")
LLAMA = os.path.join(PORT, "llama-3.2-3b")
MINILM = os.path.join(PORT, "minilm")
STEP3 = os.path.join(os.path.dirname(STEP4), "Step_3")
sys.path.insert(0, STEP4)
sys.path.insert(0, PORT)

import torch  # noqa: E402
import transformers  # noqa: E402

import evaluate_models as ev  # noqa: E402
import build_fullsample_arm as bf  # noqa: E402
import filter_hardllm as fh  # noqa: E402
import preflight_step4 as pf  # noqa: E402
import train_lora as tl  # noqa: E402

TOK = tl.setup_tokenizer(LLAMA)
RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def read(name):
    return [json.loads(l) for l in open(os.path.join(tl.DATA, name), encoding="utf-8")]


def norm(s):
    return " ".join(s.lower().split())


# ----------------------------------------------------------------------- data

@test
def data_files():
    """Every file is the one prepare_data.py wrote, with the expected sizes."""
    m = json.load(open(os.path.join(tl.DATA, "manifest.json")))
    for name, s in m["sets"].items():
        p = os.path.join(tl.DATA, s["file"])
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == s["sha256"], s["file"]
    for arm in tl.ARMS:
        assert m["sets"][arm]["rows"] == 16_000, arm
    assert m["sets"]["test_private"]["rows"] == 1_000
    mm = read("mmlu_medical.jsonl")
    assert len(mm) == 1_089 and all(len(r["choices"]) == 4 and r["answer"] in range(4) for r in mm)
    test_q = {norm(r["question"]) for r in read("test_private.jsonl")}
    assert all(norm(r["question"]) in test_q for r in read("demo_questions.jsonl"))


@test
def data_clean_and_disjoint():
    """Cleaned the same way everywhere; no test question in any training set."""
    test_q = {norm(r["question"]) for r in read("test_private.jsonl")}
    for name in [f"train_{a}.jsonl" for a in tl.ARMS] + ["test_private.jsonl"]:
        rows = read(name)
        assert all(set(r) == {"question", "answer"} for r in rows), name
        bad = [r for r in rows if "chat doctor" in norm(r["answer"]) or "chatdoctor" in norm(r["answer"])
               or "http:" in r["answer"] or "https:" in r["answer"]]   # a link, not the word
        assert not bad, (name, bad[0]["answer"][:120])
        assert all(len(r["answer"].split()) >= 15 for r in rows), name
        if name.startswith("train_"):
            assert not {norm(r["question"]) for r in rows} & test_q, f"{name} contains test questions"
    cand = read("hardllm_candidates.jsonl")
    assert len(cand) == 90_155 and not {norm(r["question"]) for r in cand} & test_q


# ------------------------------------------------------------ format and loss

@test
def tokenizer_and_format():
    assert TOK.pad_token == tl.PAD_TOKEN and TOK.pad_token_id != TOK.eos_token_id
    e = tl.encode(TOK, "I have  a headache.", "Drink water and rest.", 512)
    text = TOK.decode(e["input_ids"])
    assert text == f"{TOK.bos_token}Question: I have a headache.\nAnswer: Drink water and rest.{TOK.eos_token}", text
    n_prompt = len(TOK(tl.prompt_text("I have a headache."))["input_ids"])
    assert e["labels"][:n_prompt] == [-100] * n_prompt, "question must not be trained on"
    assert e["labels"][n_prompt:] == e["input_ids"][n_prompt:]
    assert e["labels"][-1] == TOK.eos_token_id, "end-of-text must be a label"
    long = tl.encode(TOK, "q " * 50, "answer " * 2000, 512)
    assert len(long["input_ids"]) == len(long["labels"]) == 512


@test
def collator_masks_padding_only():
    feats = [tl.encode(TOK, "Short?", "A short answer here.", 512),
             tl.encode(TOK, "A much longer question " * 5, "and a longer answer " * 8, 512)]
    b = tl.make_collator(TOK)(feats)
    for i, f in enumerate(feats):
        n = len(f["input_ids"])
        assert b["labels"][i, n - 1].item() == TOK.eos_token_id
        assert (b["labels"][i, n:] == -100).all() and (b["attention_mask"][i, n:] == 0).all()
        assert (b["attention_mask"][i, :n] == 1).all()


# ------------------------------------------------- tiny model, real code paths

TINY = None


def tiny_model_dir():
    """A 2-layer random Llama with the real tokenizer, saved like a real model."""
    global TINY
    if TINY:
        return TINY
    TINY = tempfile.mkdtemp()
    real = json.load(open(os.path.join(LLAMA, "config.json")))
    cfg = transformers.LlamaConfig(
        vocab_size=real["vocab_size"], hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, tie_word_embeddings=True,
        bos_token_id=real["bos_token_id"], eos_token_id=real["eos_token_id"])
    transformers.LlamaForCausalLM(cfg).save_pretrained(TINY)
    TOK.save_pretrained(TINY)
    return TINY


def run_train(*extra):
    out = tempfile.mkdtemp()
    argv = sys.argv
    try:
        sys.argv = ["train_lora.py", "--arm", "hardllm", "--model", tiny_model_dir(),
                    "--rows", "64", "--max_steps", "2", "--output_dir", out, *extra]
        with contextlib.redirect_stdout(io.StringIO()) as buf, \
             contextlib.redirect_stderr(io.StringIO()):
            res = tl.main()
    finally:
        sys.argv = argv
        shutil.rmtree(out, ignore_errors=True)
    return res, buf.getvalue()


@test
def training_runs_both_settings():
    """LoRA + Trainer + length grouping + collator, with and without checkpointing."""
    before = set(os.listdir(tl.OUT)) if os.path.isdir(tl.OUT) else set()
    for ckpt, micro in [(1, 4), (0, 2)]:
        res, log = run_train("--checkpointing", str(ckpt), "--micro_batch", str(micro))
        assert res["steps"] == 2 and math.isfinite(res["train_loss"]), res
        assert f"batch 16 = {micro} x {16 // micro}" in log, log[-400:]
        assert "PROBE " in log
    res, log = run_train("--worst_case")
    assert "truncated at 512: 64" in log, "worst case must stretch every row to 512"
    after = set(os.listdir(tl.OUT)) if os.path.isdir(tl.OUT) else set()
    assert before == after, "a test run must not write into out/"


@test
def save_load_and_evaluate():
    """Adapter saves, loads and merges; all three evaluations run on it."""
    from peft import PeftModel
    base = transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir())
    adir = tempfile.mkdtemp()
    tl.add_lora(base).save_pretrained(adir)
    assert os.path.exists(os.path.join(adir, "adapter_model.safetensors"))
    model = PeftModel.from_pretrained(
        transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir()), adir).merge_and_unload()
    model.eval()
    nll, n = ev.answer_nll(model, TOK, read("test_private.jsonl")[:3])
    assert math.isfinite(nll) and n > 0
    acc, per = ev.mmlu(model, TOK, read("mmlu_medical.jsonl")[:6])
    assert 0 <= acc <= 1 and per
    ans = ev.demo_answers(model, TOK, read("demo_questions.jsonl")[:2], max_new=8)
    assert len(ans) == 2 and all(isinstance(a, str) for a in ans)
    assert TOK.padding_side == "right", "generation must restore right padding"
    assert ev.load(tiny_model_dir(), "base") is not None
    shutil.rmtree(adir)


@test
def preflight_choice():
    ok = lambda t, p: {"ok": p <= pf.MEM_LIMIT, "tokens_per_second": t, "peak_reserved_frac": p}
    assert pf.choose({"a": ok(100, .5), "b": ok(300, .95), "c": ok(200, .8)}) == "c"
    assert pf.choose({"a": {"ok": False}, "b": ok(300, .99)}) is None


@test
def scripts_start_as_the_bat_runs_them():
    """Each script started by path, as the .bat does. The bundled Python does not
    put a script's folder on the import path, so a script that imports its
    neighbour must add the folder itself (evaluate_models.py once did not)."""
    import subprocess
    for script in [os.path.join(STEP4, "train_lora.py"), os.path.join(STEP4, "evaluate_models.py"),
                   os.path.join(STEP4, "filter_hardllm.py"), os.path.join(PORT, "preflight_step4.py"),
                   os.path.join(STEP3, "generate_synthetic.py"), os.path.join(STEP3, "clean_synthetic_qa.py")]:
        r = subprocess.run([sys.executable, script, "--help"], cwd=PORT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, (os.path.basename(script), r.stderr[-400:])


# ------------------------------------------------------- hardllm-filtered arm

@test
def filter_selection_rule():
    """Worse half dropped, 16,000 drawn at random from the better half, same every run."""
    rows = list(range(40_000))
    scores = [((i * 7919) % 40_000) / 40_000 for i in rows]
    picked, cutoff = fh.select(rows, scores)
    assert len(picked) == len(set(picked)) == 16_000
    assert min(scores[i] for i in picked) >= cutoff >= 0.49
    assert picked == fh.select(rows, scores)[0], "must be deterministic"
    try:
        fh.select(rows[:20_000], scores[:20_000])
        raise AssertionError("must refuse when the better half is too small")
    except SystemExit:
        pass


@test
def filter_embedder_scores_fit():
    """The real MiniLM: an answer about the question beats one about something else."""
    tok = transformers.AutoTokenizer.from_pretrained(MINILM)
    model = transformers.AutoModel.from_pretrained(MINILM).eval()
    rows = [{"question": "I get a bad headache and blurred vision every afternoon.",
             "answer": "Headaches with blurred vision can come from migraine or high blood pressure; check your eyes and blood pressure."},
            {"question": "I get a bad headache and blurred vision every afternoon.",
             "answer": "For heartburn after spicy food, take an antacid and avoid lying down after meals."}]
    scores, q, a = fh.pair_scores(model, tok, rows)
    assert scores[0] > scores[1] + 0.1, scores
    assert torch.allclose(q.norm(dim=-1), torch.ones(2), atol=1e-4)


@test
def extra_arm_reports_separately():
    """--with_extra writes *_filtered files with 6 models; the first report keeps 5."""
    fake = lambda m: {"model": m, "answer_nll": 2.3, "answer_perplexity": 10.0, "mmlu_medical": 0.5,
                      "demo_answers": ["x"] * 2}
    demo = [{"question": "q1"}, {"question": "q2"}]
    tmp, real_out = tempfile.mkdtemp(), ev.OUT
    try:
        ev.OUT = tmp
        results = {m: fake(m) for m in ev.MODELS + tl.EXTRA_ARMS}
        with contextlib.redirect_stdout(io.StringIO()):
            ev.report(results, demo)
            ev.report(results, demo, "_filtered")
        assert sorted(json.load(open(os.path.join(tmp, "results_step4.json")))) == sorted(ev.MODELS)
        assert sorted(json.load(open(os.path.join(tmp, "results_step4_filtered.json")))) == \
            sorted(ev.MODELS + tl.EXTRA_ARMS)
        assert "hardllm-filtered" in open(os.path.join(tmp, "demo_answers_filtered.md")).read()
    finally:
        ev.OUT = real_out
        shutil.rmtree(tmp, ignore_errors=True)


@test
def fullsample_vocabulary_measure():
    test_counts = collections.Counter("the rare drug cures the cough".split())
    v = bf.vocab([{"answer": "the drug cures the cough"}, {"answer": "The cough"}], test_counts)
    assert v["answer_words"] == 7 and v["distinct_per_100k"] == 4
    assert v["test_words_never_seen"] == round(1 / 6, 4)      # "rare"
    assert bf.SOURCE.endswith(os.path.join("Step_3", "out", "synthetic_qa_fullsample_clean.jsonl"))
    assert os.path.exists(os.path.join(STEP3, "models", "gpt2-qa", "model.safetensors"))


# --------------------------------------------------------------- bat, bundle

@test
def bat_file():
    raw = open(os.path.join(PORT, "RUN_STEP4_ON_NVIDIA.bat"), "rb").read()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), "needs CRLF only"
    s = raw.decode("ascii")
    for line in s.splitlines():
        if line.upper().startswith("REM"):
            assert not set(line) & set("<>|&"), f"special char in comment: {line}"
    order = ["tests\\test_step4.py", "preflight_step4.py", "set /p MICRO", "train_lora.py",
             "evaluate_models.py", "check_results_step4.py"]
    assert [s.index(o) for o in order] == sorted(s.index(o) for o in order), "stages out of order"
    assert "--micro_batch %MICRO% --checkpointing %CKPT%" in s
    assert "for %%A in (hardllm selected random private)" in s
    assert s.count("if errorlevel 1") >= 6


@test
def bat_file_evaluate_only():
    raw = open(os.path.join(PORT, "RUN_STEP4_EVALUATE_ONLY.bat"), "rb").read()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), "needs CRLF only"
    s = raw.decode("ascii")
    for line in s.splitlines():
        if line.upper().startswith("REM"):
            assert not set(line) & set("<>|&"), f"special char in comment: {line}"
    assert "train_lora.py" not in s and "preflight_step4.py" not in s, "must never train"
    order = ["tests\\test_step4.py", "adapter_model.safetensors", "evaluate_models.py",
             "check_results_step4.py"]
    assert [s.index(o) for o in order] == sorted(s.index(o) for o in order), "stages out of order"
    assert s.count("if errorlevel 1") >= 4


@test
def bat_file_filtered():
    raw = open(os.path.join(PORT, "RUN_STEP4_FILTERED.bat"), "rb").read()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), "needs CRLF only"
    s = raw.decode("ascii")
    for line in s.splitlines():
        if line.upper().startswith("REM"):
            assert not set(line) & set("<>|&"), f"special char in comment: {line}"
    assert "preflight_step4.py" not in s
    assert s.count("train_lora.py") == 1 and "--arm hardllm-filtered" in s, "trains only the new model"
    order = ["tests\\test_step4.py", "eval_%%A.json", "filter_hardllm.py", "train_lora.py",
             "evaluate_models.py\" --model \"%LLAMA%\" --with_extra", "check_results_filtered.py"]
    assert [s.index(o) for o in order] == sorted(s.index(o) for o in order), "stages out of order"
    assert s.count("if errorlevel 1") >= 6


@test
def bat_file_fullsample():
    raw = open(os.path.join(PORT, "RUN_STEP4_FULLSAMPLE.bat"), "rb").read()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), "needs CRLF only"
    s = raw.decode("ascii")
    for line in s.splitlines():
        if line.upper().startswith("REM"):
            assert not set(line) & set("<>|&"), f"special char in comment: {line}"
    assert "preflight_step4.py" not in s
    assert s.count("train_lora.py") == 1 and "--arm hardllm-fullsample" in s, "trains only the new model"
    assert "--top_k 0 --top_p 1.0 --temperature 1.0" in s
    order = ["tests\\test_step4.py", "eval_%%A.json", "generate_synthetic.py", "clean_synthetic_qa.py",
             "build_fullsample_arm.py", "train_lora.py", "--suffix _fullsample", "check_results_fullsample.py"]
    assert [s.index(o) for o in order] == sorted(s.index(o) for o in order), "stages out of order"
    assert s.count("if errorlevel 1") >= 8


@test
def bundle():
    sp = os.path.join(PORT, "win_python", "Lib", "site-packages")
    have = {d.split("-")[0].lower(): d for d in os.listdir(sp) if d.endswith(".dist-info")}
    for pkg, ver in [("torch", "2.11.0+cu128"), ("transformers", "5.16.1"), ("peft", "0.21.2"),
                     ("accelerate", "1.14.0"), ("datasets", "5.0.1")]:
        assert pkg in have and ver in have[pkg], (pkg, have.get(pkg))
    cfg = json.load(open(os.path.join(LLAMA, "config.json")))
    assert cfg["model_type"] == "llama" and cfg["num_hidden_layers"] == 28 and cfg["hidden_size"] == 3072
    idx = json.load(open(os.path.join(LLAMA, "model.safetensors.index.json")))
    for shard in set(idx["weight_map"].values()):
        assert os.path.getsize(os.path.join(LLAMA, shard)) > 1e9, shard


def main():
    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"  [ok]   {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {fn.__name__}\n" + "".join(
                "         " + l for l in traceback.format_exc().splitlines(True)))
    if TINY:
        shutil.rmtree(TINY, ignore_errors=True)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    if failed:
        print("TESTS FAILED: nothing was trained. Bring back the window text (a photo is fine).")
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
