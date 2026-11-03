"""Code tests for Step 3 --qa. RUN_STEP3_ON_NVIDIA.bat runs these first, on
the NVIDIA PC, and stops if any fail. About a minute: a fake model where a real
one isn't needed, and a 2-step training run.

    python Step_3/tests/test_step3_qa.py
    python Step_3/tests/test_step3_qa.py --ssd /Volumes/PortableSSD/Project

--ssd also checks the copy on the drive: same files as here, nothing missing,
no macOS junk files.

What is NOT tested here, and why: real output quality and real GPU speed and
memory. Those need the trained model and the 5070 Ti; preflight_qa.py checks
memory and speed on the PC before the real run, and generate_synthetic.py
stops after one batch if the output is broken.
"""
import argparse
import collections
import contextlib
import filecmp
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
STEP3 = os.path.dirname(HERE)
PORT = os.path.join(STEP3, "portable")
sys.path.insert(0, STEP3)

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import generate_synthetic as gen  # noqa: E402
import train_generator as tg  # noqa: E402

TOK = AutoTokenizer.from_pretrained(os.path.join(PORT, "gpt2"))
RESULTS = []


def test(fn):
    RESULTS.append(fn)
    return fn


def selected_pairs():
    rows = [json.loads(l) for l in open(tg.DEFAULT_DATA, encoding="utf-8")]
    return rows, [r for r in rows if tg.is_qa_pair(r)]


# ----------------------------------------------------------------------- data

@test
def data_pairs():
    """The input has the 16,788 pairs the preflight expects, from the right sources,
    and the junk "Brief Answer" rows are out."""
    rows, pairs = selected_pairs()
    assert len(rows) == 36_201, len(rows)
    by = collections.Counter(r["source"] for r in pairs)
    assert by == {"lavita/ChatDoctor-HealthCareMagic-100k": 14_789,
                  "lavita/MedQuAD": 1_957, "LLukas22/fiqa": 42}, by
    assert gen.MIN_ANSWER_WORDS == tg.MIN_ANSWER_WORDS
    import portable_preflight as pf
    assert len(pairs) == pf.EXPECTED_PAIRS
    assert not any(r["source"] == "knkarthick/dialogsum" for r in pairs)
    assert all(r["text"].strip() and str(r["target"]).strip() for r in pairs)


@test
def qa_format():
    """Exactly one separator per training example, whatever the raw text holds."""
    _, pairs = selected_pairs()
    for r in pairs:
        t = tg.qa_text(r["text"], r["target"])
        assert t.startswith("Question: ")
        assert t.count("\n") == 1 and t.count(gen.SEP) == 1, t[:200]
    t = tg.qa_text("line one\nline two\r\n  Answer: tricky", "a\n\nb")
    assert t == "Question: line one line two Answer: tricky\nAnswer: a b", t


@test
def prompt_is_prefix():
    """The generation prompt tokenises exactly like the start of every training row.
    If not, the model would be prompted with something it never saw."""
    _, pairs = selected_pairs()
    start = TOK("Question:")["input_ids"]
    for r in pairs:
        ids = TOK(tg.qa_text(r["text"], r["target"]))["input_ids"]
        assert ids[:len(start)] == start, (r["text"][:80], ids[:5])


@test
def parse_round_trip():
    """A generation shaped exactly like a training row parses back to that row."""
    _, pairs = selected_pairs()
    start = len(TOK("Question:")["input_ids"])
    eos = TOK.eos_token_id
    for r in pairs[::7]:
        ids = TOK(tg.qa_text(r["text"], r["target"]))["input_ids"][start:] + [eos] * 3
        row, why = gen.parse_qa(TOK, ids)
        assert why is None, (why, r["text"][:80])
        assert row["question"] == " ".join(r["text"].split())
        assert row["answer"] == " ".join(str(r["target"]).split())


@test
def parse_rejects():
    eos = TOK.eos_token_id
    ids = TOK(" what is this?\nAnswer: it is a test of the parser")["input_ids"]
    assert gen.parse_qa(TOK, ids)[1] == "unfinished"                 # no end token
    assert gen.parse_qa(TOK, TOK(" no answer here")["input_ids"] + [eos])[1] == "malformed"
    two = TOK(" q?\nAnswer: one two three\nAnswer: again")["input_ids"] + [eos]
    assert gen.parse_qa(TOK, two)[1] == "malformed"                  # two separators
    short = TOK(" q?\nAnswer: ok")["input_ids"] + [eos]
    assert gen.parse_qa(TOK, short)[1] == "malformed"                # < 3 words
    assert gen.parse_qa(TOK, ids + [eos])[1] is None


# ------------------------------------------------------ generation, end to end

def run_fake_generation(mode, out, total=150):
    """generate_synthetic.py --qa on CPU with a fake model of known behaviour.
    Real generate() pads finished rows with EOS; unfinished rows run to the cap
    without one, which the fake reproduces."""
    eos = TOK.eos_token_id
    prefix = len(TOK("Question:")["input_ids"])
    good = TOK(tg.qa_text("My baby has had a fever since last night. What should I do?",
                          "Hi, give paracetamol by weight and keep her hydrated."))["input_ids"]
    good = good[prefix:] + [eos]
    unfinished = good[:-1]
    noanswer = TOK(" just a question")["input_ids"] + [eos]
    filler = TOK(" and")["input_ids"][0]

    class Fake(torch.nn.Module):
        def generate(self, prompt, **kw):
            L, rows = kw["max_new_tokens"], []
            for i in range(prompt.shape[0]):
                if mode == "good":
                    g = good if i % 10 else unfinished if i % 20 else noanswer
                else:
                    g = unfinished if i % 2 else noanswer
                fill = eos if eos in g else filler
                rows.append(prompt[i].tolist() + (g + [fill] * L)[:L])
            return torch.tensor(rows)

    patches = [(transformers.AutoModelForCausalLM, "from_pretrained",
                staticmethod(lambda *a, **k: Fake())),
               (gen.AutoModelForCausalLM, "from_pretrained",
                staticmethod(lambda *a, **k: Fake())),
               (gen.torch.backends.mps, "is_available", lambda: False)]
    saved = [(o, n, inspect.getattr_static(o, n)) for o, n, _ in patches]
    argv = sys.argv
    try:
        for o, n, v in patches:
            setattr(o, n, v)
        sys.argv = ["g", "--qa", "--model", os.path.join(PORT, "gpt2"),
                    "--total", str(total), "--batch_size", "64", "--length", "60",
                    "--output", out]
        code = 0
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                gen.main()
            except SystemExit as e:
                code = e.code
        return code
    finally:
        sys.argv = argv
        for o, n, v in saved:
            setattr(o, n, v)


@test
def generation_good_and_resume():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "qa.jsonl")
    assert run_fake_generation("good", out) in (0, None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 150 and all(set(r) == {"question", "answer"} for r in rows)
    st = json.load(open(os.path.join(d, "qa_stats.json")))
    assert st["status"] == "complete" and st["written_total"] == 150
    assert st["dropped_unfinished"] > 0 and st["dropped_malformed"] > 0
    # resume: already complete -> nothing added, nothing duplicated
    assert run_fake_generation("good", out) in (0, None)
    assert sum(1 for _ in open(out)) == 150
    # resume from a partial file -> tops up to the target exactly
    with open(out) as f:
        keep = f.readlines()[:40]
    with open(out, "w") as f:
        f.writelines(keep)
    assert run_fake_generation("good", out) in (0, None)
    assert sum(1 for _ in open(out)) == 150
    shutil.rmtree(d)


@test
def generation_broken_stops_after_one_batch():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "qa.jsonl")
    code = run_fake_generation("bad", out)
    assert code not in (0, None), "should have exited non-zero"
    assert sum(1 for _ in open(out)) == 0, "should have written nothing"
    st = json.load(open(os.path.join(d, "qa_stats.json")))
    assert st["status"].startswith("stopped"), st["status"]
    shutil.rmtree(d)


# ------------------------------------------------------------------- training

@test
def training_two_steps_cpu():
    """The real training code, real data, --qa settings (4 x 4), 2 steps."""
    d = tempfile.mkdtemp()
    summary = os.path.join(STEP3, "out", "train_summary_qa.json")
    before = os.path.getmtime(summary) if os.path.exists(summary) else None
    argv = sys.argv
    real_mps = tg.torch.backends.mps.is_available
    try:
        tg.torch.backends.mps.is_available = lambda: False
        sys.argv = ["t", "--qa", "--model", os.path.join(PORT, "gpt2"),
                    "--max_steps", "2", "--output_dir", d]
        with contextlib.redirect_stdout(io.StringIO()) as buf, \
             contextlib.redirect_stderr(io.StringIO()):
            res = tg.main()
        log = buf.getvalue()
    finally:
        sys.argv = argv
        tg.torch.backends.mps.is_available = real_mps
    assert "batch 16 = 4 x 4" in log, log[-500:]
    assert "997 per epoch" in log, log[-500:]
    assert res["steps"] == 2 and res["val_loss"] < 6, res
    after = os.path.getmtime(summary) if os.path.exists(summary) else None
    assert before == after, "a test run must not touch the real training summary"
    shutil.rmtree(d)


@test
def eos_is_learned():
    """Every row's end-of-text token is a training label; only padding is masked.
    (The standard collator masked the end-of-text too, since GPT-2 pads with it,
    and the model never learned to stop.)"""
    TOK.pad_token = TOK.eos_token
    texts = [tg.qa_text("short q", "a short answer here"),
             tg.qa_text("a much longer question " * 5, "and a longer answer " * 8)]
    feats = [TOK(t + TOK.eos_token) for t in texts]
    b = tg.make_collator(TOK)(feats)
    eos = TOK.eos_token_id
    for i, f in enumerate(feats):
        n = len(f["input_ids"])
        assert b["labels"][i, n - 1].item() == eos, "end-of-text must be a label"
        assert (b["labels"][i, :n] == b["input_ids"][i, :n]).all()
        assert (b["labels"][i, n:] == -100).all(), "padding must be masked"
    assert (b["labels"] == -100).sum().item() == len(feats[1]["input_ids"]) - len(feats[0]["input_ids"])


@test
def no_stale_model():
    """A model trained before the end-of-text fix must not be resumed or used."""
    summary = os.path.join(STEP3, "out", "train_summary_qa.json")
    if os.path.isdir(tg.QA_OUT):
        ok = os.path.exists(summary) and json.load(open(summary)).get("eos_learned")
        assert ok, (f"{tg.QA_OUT} was trained before the end-of-text fix and never "
                    "learned to stop. Delete Step_3\\models\\gpt2-qa and "
                    "Step_3\\out\\synthetic_qa.jsonl, then run again.")


@test
def defaults_resolve():
    """--qa never writes over the first run's outputs."""
    assert tg.QA_OUT != tg.DEFAULT_OUT and gen.QA_OUT != gen.DEFAULT_OUT
    assert gen.QA_MODEL == tg.QA_OUT
    assert gen.QA_LENGTH + len(TOK("Question:")["input_ids"]) <= 1024   # GPT-2 context


@test
def preflight_batch_choice():
    """Biggest batch that fits wins; crashes and over-limit tries fall through."""
    import portable_preflight as pf
    assert pf.GEN_BATCHES[0] == gen.QA_BATCH_CUDA and list(pf.GEN_BATCHES) == sorted(pf.GEN_BATCHES, reverse=True)

    def fake(peaks):
        def probe(bs):
            p = peaks[bs]
            return None if p is None else {"batch": bs, "seconds": 1.0,
                                           "peak_reserved_frac": p, "peak_allocated_frac": p}
        return probe
    big, mid, small = pf.GEN_BATCHES
    chosen, tried = pf.choose_batch(fake({big: 0.5, mid: 0.3, small: 0.2}))
    assert chosen["batch"] == big and len(tried) == 1
    chosen, tried = pf.choose_batch(fake({big: 1.12, mid: 0.6, small: 0.3}))
    assert chosen["batch"] == mid and len(tried) == 2
    chosen, tried = pf.choose_batch(fake({big: None, mid: 0.95, small: 0.4}))
    assert chosen["batch"] == small and tried[0].get("crashed")
    chosen, tried = pf.choose_batch(fake({big: 1.2, mid: 1.0, small: 0.91}))
    assert chosen is None and len(tried) == 3


# ------------------------------------------------------------ bundle and .bat

@test
def bat_file():
    p = os.path.join(PORT, "RUN_STEP3_ON_NVIDIA.bat")
    raw = open(p, "rb").read()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), "needs CRLF only"
    s = raw.decode("ascii")
    for line in s.splitlines():
        if line.upper().startswith("REM"):
            assert not set(line) & set("<>|&"), f"special char in comment: {line}"
    assert '"%PORT%preflight_qa.py"' in s
    assert s.index("preflight_qa.py") < s.index("train_generator.py") < s.index("generate_synthetic.py")
    assert "train_generator.py\" --qa" in s and "generate_synthetic.py\" --qa" in s
    assert s.count("if errorlevel 1") >= 4, "every stage must stop the .bat on failure"
    # the batch size the preflight measured is the one generation uses
    assert 'set /p GENBS=<"%STEP3%\\out\\gen_batch_qa.txt"' in s
    assert s.index("preflight_qa.py") < s.index("set /p GENBS") < s.index("generate_synthetic.py")
    assert "generate_synthetic.py\" --qa --batch_size %GENBS%" in s
    for f in ["preflight_qa.py", "gpt2", os.path.join("win_python", "python.exe")]:
        assert os.path.exists(os.path.join(PORT, f)), f


@test
def windows_bundle():
    w = os.path.join(PORT, "win_python")
    pth = [f for f in os.listdir(w) if f.endswith("._pth")]
    assert len(pth) == 1 and "Lib\\site-packages" in open(os.path.join(w, pth[0])).read()
    sp = os.path.join(w, "Lib", "site-packages")
    have = {d.split("-")[0].lower(): d for d in os.listdir(sp) if d.endswith(".dist-info")}
    for pkg, ver in [("torch", "2.11.0+cu128"), ("transformers", "5.16.1"),
                     ("datasets", "5.0.1"), ("accelerate", "1.14.0")]:
        assert pkg in have and ver in have[pkg], (pkg, have.get(pkg))
    libs = os.listdir(os.path.join(sp, "torch", "lib"))
    assert any(l.startswith("cudart64") for l in libs), "torch bundle has no CUDA runtime"
    for f in ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]:
        assert os.path.exists(os.path.join(w, f)), f
    g = set(os.listdir(os.path.join(PORT, "gpt2")))
    assert {"config.json", "model.safetensors", "tokenizer.json", "vocab.json",
            "merges.txt"} <= g, g


# ------------------------------------------------------------------ the drive

def check_ssd(root):
    """The drive copy matches this Mac: same layout, same script contents, no junk."""
    problems = []
    junk = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs
            if f.startswith("._") or f == ".DS_Store"]
    if junk:
        problems.append(f"{len(junk)} macOS junk files, e.g. {junk[0]}")
    want = os.path.join(root, "Step_2", "out", "selected_centroid_eps0.5_t3.jsonl")
    if not os.path.exists(want) or not filecmp.cmp(want, tg.DEFAULT_DATA, shallow=False):
        problems.append("Step_2/out/selected_centroid_eps0.5_t3.jsonl missing or different")
    for rel in ["train_generator.py", "generate_synthetic.py",
                "portable/preflight_qa.py", "portable/RUN_STEP3_ON_NVIDIA.bat",
                "portable/README.md"]:
        a, b = os.path.join(STEP3, rel), os.path.join(root, "Step_3", rel)
        if not os.path.exists(b) or not filecmp.cmp(a, b, shallow=False):
            problems.append(f"Step_3/{rel} missing or out of date")
    for sub in ["portable/win_python", "portable/gpt2"]:
        a, b = os.path.join(STEP3, sub), os.path.join(root, "Step_3", sub)
        na = sum(len(fs) for _, _, fs in os.walk(a))
        nb = sum(len([f for f in fs if not f.startswith("._")]) for _, _, fs in os.walk(b))
        if na != nb:
            problems.append(f"Step_3/{sub}: {nb:,} files on the drive, {na:,} here")
    return problems


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssd", help="also check the project copy on the drive")
    args = ap.parse_args()

    # the preflight lives in portable/; import it by path for data_pairs()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "portable_preflight", os.path.join(PORT, "preflight_qa.py"))
    sys.modules["portable_preflight"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules["portable_preflight"])

    failed = 0
    for fn in RESULTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc(limit=3)
    if args.ssd:
        probs = check_ssd(args.ssd)
        for p in probs:
            print(f"  FAIL  drive: {p}")
        if not probs:
            print(f"  PASS  drive copy at {args.ssd}")
        failed += bool(probs)
    print(f"\n{'ALL PASSED' if not failed else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
