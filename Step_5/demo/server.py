"""The HardLLM demo: a web page on this laptop with a live chat to the final
model, patient intake with rule-based triage, a hospital queue, and the
results, every number read from the steps' own outputs.

    python3 HardLLM/Step_5/demo/server.py                  # opens http://127.0.0.1:8080
    python3 HardLLM/Step_5/demo/server.py --version f16    # the full 16-bit model
    python3 HardLLM/Step_5/demo/server.py --mock           # no model: recorded answers
    python3 HardLLM/Step_5/demo/server.py --lan            # phones on the same Wi-Fi can open it

The model runs through llama.cpp on the Apple GPU, offline (see ask_model.py,
same prompt and decoding). --mock serves the answers recorded in Step 5
instead, for checking the page or as a fallback; the page says so.

Plain Python, no torch. Endpoints:
  GET  /api/story   all numbers and recorded demo answers, with their source files
  POST /api/ask     {"question": "...", "filter": true} -> server-sent events [{"alert": ..}] {"t": piece} ... {"done": ...}
  POST /api/intake  {"text": "..."} -> {"assessment": triage result}
  GET  /api/cases   the hospital queue, most urgent first
  POST /api/cases   {"text": "...", "patient": "..."} -> the new case (triaged again here)
  POST /api/cases/<id>  {"status": "waiting"|"seen"} or {"remove": true}
"""
import argparse
import functools
import http.server
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import webbrowser

DEMO = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.dirname(DEMO)
ROOT = os.path.dirname(STEP5)
sys.path.insert(0, STEP5)
import ask_model  # noqa: E402
sys.path.insert(0, DEMO)
import triage  # noqa: E402
import answer_filter  # noqa: E402

MAX_QUESTION = 2000
MAX_BODY = 20000
MAX_PATIENT = 60
QUEUE = os.path.join(DEMO, "queue.json")
# shown above the answer when triage.red_flags finds an emergency sign, whatever the model says
# greetings get a prompt for symptoms instead of the model, which answers "hello" by inventing a patient's question
GREETING = re.compile(r"(?:hi+|hello|hey|yo|good (?:morning|afternoon|evening)|thanks?|thank you|ok(?:ay)?|test(?:ing)?)[\s!.?]*", re.I)
ASK_FOR_SYMPTOMS = "Hello. Describe what you feel, for example where it hurts, since when, and anything that makes it better or worse."
EMERGENCY = ("These symptoms can be a medical emergency. Call your local emergency number "
             "(112 / 108 in India, 911 in the US) or go to the nearest emergency department now. "
             "Do not wait for this answer.")
# Step 4/5 models shown on the page, in the order of the chart legend
MODELS = {
    "base": ("Base Llama-3.2-3B", "nothing: the model as released"),
    "hardllm": ("HardLLM, Step 3 data", "GPT-2 synthetic Q&A (paper's sampling)"),
    "hardllm-filtered": ("Step 3, filtered", "synthetic Q&A, worse-matching half removed"),
    "hardllm-fullsample": ("Step 3, full sampling", "synthetic Q&A, no top-k/top-p cut"),
    "random": ("Random public data", "public Q&A picked at random"),
    "selected": ("Final model", "public Q&A picked by private-data votes"),
    "q4_k_m": ("Final model, 4-bit (this demo)", "the final model compressed to 1.9 GB"),
    "private": ("Private data (ceiling)", "the real patient records, not private"),
}


def rel(path):
    return os.path.relpath(path, ROOT)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def p(*parts):
    return os.path.join(ROOT, *parts)


def story(version):
    """Everything the page shows, from the files that hold it."""
    src = {}

    clusters = load(p("Step_1", "models", "cluster_summary.json"))
    log = open(p("Step_1", "models", "cooldown.log"), encoding="utf-8", errors="replace").read()
    last = re.findall(r"'eval_Hits@1': '([0-9.]+)', 'eval_Hits@10': '([0-9.]+)'", log)[-1]
    unseen = load(p("Step_1", "models", "unseen_eval.json"))
    step1 = {"pool_rows": sum(c["size"] for c in clusters.values()), "clusters": len(clusters),
             "hits1": float(last[0]), "hits10": float(last[1]),
             "unseen_hits1": unseen["hits1"], "unseen_hits10": unseen["hits10"]}
    src["step1"] = [rel(p("Step_1", "models", f)) for f in
                    ("cluster_summary.json", "cooldown.log", "unseen_eval.json")]

    sweep = load(p("Step_2", "out", "noise_sweep_centroid.json"))
    run = next(r for r in sweep["results"] if r["eps"] == 0.5 and r["threshold_sigmas"] == 3)
    step2 = {"records": sweep["records"], "votes_per_record": sweep["votes_per_record"],
             "delta": sweep["delta"], "eps": run["eps"], "sigma": run["sigma"],
             "threshold_sigmas": run["threshold_sigmas"], "clusters_selected": run["clusters_selected"],
             "rows_selected": run["rows_selected"], "medical_frac_selected": run["medical_frac_selected"],
             "pool_medical_frac": sweep["pool_medical_frac"]}
    src["step2"] = [rel(p("Step_2", "out", "noise_sweep_centroid.json"))]

    gen = load(p("Step_3", "out", "synthetic_qa_stats.json"))
    full = load(p("Step_4", "out", "fullsample_arm.json"))["vocabulary"]
    step3 = {"train": load(p("Step_3", "out", "train_summary_qa.json"))["train_rows"],
             "generated": gen["written_total"], "top_k": gen["args"]["top_k"], "top_p": gen["args"]["top_p"],
             "kept": load(p("Step_3", "out", "clean_qa_report.json"))["kept_rows"],
             "distinct_words": {k: full[k]["distinct_per_100k"] for k in ("hardllm", "hardllm-fullsample", "private")}}
    src["step3"] = [rel(p("Step_3", "out", f)) for f in
                    ("train_summary_qa.json", "synthetic_qa_stats.json", "clean_qa_report.json")] + \
        [rel(p("Step_4", "out", "fullsample_arm.json"))]

    evals = {}
    for m in MODELS:
        f = p("Step_5", "out", f"eval_{m}.json") if m == "q4_k_m" else p("Step_4", "out", f"eval_{m}.json")
        evals[m] = load(f)
        src.setdefault("step4", []).append(rel(f))
    models = [{"id": m, "name": n, "trained_on": t, "ppl": evals[m]["answer_perplexity"]}
              for m, (n, t) in MODELS.items()]

    s5 = load(p("Step_5", "out", "results_step5.json"))
    step5 = [{"version": v, "file_gb": s5[v]["file_gb"], "ppl": s5[v]["answer_perplexity"],
              "llamacpp_ppl": (s5[v].get("llamacpp") or {}).get("ppl"), "weight_change": s5[v]["weight_change"]}
             for v in ("merged", "f16", "q8_0", "q4_k_m")]
    src["step5"] = [rel(p("Step_5", "out", "results_step5.json"))]

    questions = read_jsonl(p("Step_4", "data", "demo_questions.jsonl"))
    demo = [{"question": q["question"],
             "answers": {m: evals[m]["demo_answers"][i] for m in ("base", "hardllm", "selected", "q4_k_m", "private")}}
            for i, q in enumerate(questions)]
    src["demo"] = [rel(p("Step_4", "data", "demo_questions.jsonl"))]

    return {"version": version, "file_gb": next(s["file_gb"] for s in step5 if s["version"] == version),
            "step1": step1, "step2": step2, "step3": step3, "models": models, "step5": step5,
            "demo": demo, "sources": src}


class Cases:
    """The hospital queue, kept in queue.json so it survives a restart."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.cases = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.cases = json.load(f)["cases"]
            except (ValueError, KeyError, OSError) as e:
                bad = path + ".unreadable"
                os.replace(path, bad)
                print(f"queue file could not be read ({e}); moved to {bad}, starting empty", flush=True)

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"cases": self.cases}, f, indent=1)
        os.replace(tmp, self.path)

    def listing(self):
        with self.lock:
            return sorted(self.cases, key=lambda c: (c["status"] != "waiting", triage.RANK[c["assessment"]["level"]], c["created"]))

    def add(self, text, patient):
        with self.lock:
            # the same patient and words assessed again within 10 minutes is the same case, not a new one
            same = next((c for c in reversed(self.cases) if c["text"] == text and c["patient"] == patient
                         and c["status"] == "waiting" and time.time() - c["created"] < 600), None)
            if same:
                return same
            case = {"id": max((c["id"] for c in self.cases), default=0) + 1, "patient": patient, "text": text,
                    "created": time.time(), "status": "waiting", "assessment": triage.assess(text)}
            self.cases.append(case)
            self.save()
            return case

    def update(self, case_id, body):
        with self.lock:
            case = next((c for c in self.cases if c["id"] == case_id), None)
            if case is None:
                return None
            if body.get("remove") is True:
                self.cases.remove(case)
            elif body.get("status") in ("waiting", "seen"):
                case["status"] = body["status"]
                case["seen_at"] = time.time() if body["status"] == "seen" else None
            else:
                return False
            self.save()
            return case


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves web/ and the two API routes."""
    llama_port = None       # None = mock mode
    story_json = b"{}"
    recorded = []           # mock answers
    lock = threading.Lock()  # llama-server answers one question at a time
    cases = None            # Cases

    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            sys.stderr.write("  %s\n" % (fmt % args))

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")  # the page's files: always ask, so an update shows on a plain refresh
        super().end_headers()

    def send_json(self, obj, code=200):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_body(self):
        """The JSON object sent, or None (after answering 400) if it is not one."""
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= MAX_BODY:
                raise ValueError("size")
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("not an object")
            return body
        except (ValueError, UnicodeDecodeError):
            self.send_json({"error": "send a JSON object under 20 kB"}, 400)
            return None

    def do_GET(self):
        if self.path == "/api/story":
            self.send_json(self.story_json)
            return
        if self.path == "/api/cases":
            self.send_json({"cases": self.cases.listing(), "now": time.time()})
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/intake":
            body = self.read_body()
            if body is not None:
                self.send_json({"assessment": triage.assess(clean_text(body.get("text"), MAX_QUESTION))})
            return
        if self.path == "/api/cases":
            body = self.read_body()
            if body is None:
                return
            text = clean_text(body.get("text"), MAX_QUESTION)
            if not text:
                self.send_json({"error": "empty description"}, 400)
                return
            self.send_json(self.cases.add(text, clean_text(body.get("patient"), MAX_PATIENT)))
            return
        m = re.fullmatch(r"/api/cases/(\d+)", self.path)
        if m:
            body = self.read_body()
            if body is None:
                return
            case = self.cases.update(int(m.group(1)), body)
            if case is None:
                self.send_json({"error": "no such case"}, 404)
            elif case is False:
                self.send_json({"error": "send {status: waiting|seen} or {remove: true}"}, 400)
            else:
                self.send_json(case)
            return
        if self.path != "/api/ask":
            self.send_error(404)
            return
        body = self.read_body()
        if body is None:
            return
        q = clean_text(body.get("question"), MAX_QUESTION)
        if not q:
            self.send_error(400, "empty question")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # the Intake note asks for information only (instructions dropped); the chat shows the whole answer
        self.sentences = answer_filter.Sentences(keep_all=body.get("filter") is False)
        try:
            if triage.red_flags(q):
                self.event({"alert": EMERGENCY})
            if GREETING.fullmatch(q):
                self.event({"t": ASK_FOR_SYMPTOMS})
                self.event({"done": True, "canned": True})
            elif self.llama_port is None:
                self.mock_answer(q)
            else:
                with self.lock:
                    _, t = ask_model.ask(self.llama_port, q, out=self)
                self.finish_answer()
                self.event({"done": True, "tokens": t.get("predicted_n", 0),
                            "tps": round(t.get("predicted_per_second", 0), 1), "dropped": self.sentences.dropped})
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page was closed mid-answer

    def write(self, piece):
        """ask_model.ask streams into this; finished sentences that are information go out as events."""
        out = self.sentences.feed(piece)
        if out:
            self.event({"t": out})

    def finish_answer(self):
        out = self.sentences.finish()
        if out:
            self.event({"t": out})

    def flush(self):
        pass

    def event(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def mock_answer(self, q):
        # the recorded 4-bit answer of the demo question with most words in common
        words = set(q.lower().split())
        best = max(self.recorded, key=lambda r: len(words & set(r["question"].lower().split())))
        for w in best["answers"]["q4_k_m"].split(" "):
            self.write(w + " ")
            time.sleep(0.03)
        self.finish_answer()
        self.event({"done": True, "mock": True, "matched": best["question"][:120], "dropped": self.sentences.dropped})


class Server(http.server.ThreadingHTTPServer):
    request_queue_size = 128  # the default 5 refuses connections when several screens poll the queue at once
    daemon_threads = True


def clean_text(value, limit):
    return " ".join(str(value or "").split())[:limit]


def lan_address():
    """This laptop's address on the local network, for --lan."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=list(ask_model.VERSIONS), default="q4")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mock", action="store_true", help="no model, recorded answers")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--lan", action="store_true", help="also serve other devices on this Wi-Fi (no login: trusted networks only)")
    ap.add_argument("--queue", default=QUEUE, help="where the hospital queue is kept")
    args = ap.parse_args()

    data = story(ask_model.VERSIONS[args.version])
    data["mock"] = args.mock
    Handler.story_json = json.dumps(data).encode()
    Handler.recorded = data["demo"]
    Handler.cases = Cases(args.queue)
    server = None
    if not args.mock:
        model = os.path.join(STEP5, "models", f"selected-{ask_model.VERSIONS[args.version]}.gguf")
        if not os.path.exists(model):
            raise SystemExit(f"{model} is missing; --mock runs the page without it")
        os.makedirs(os.path.join(STEP5, "portable", "tmp"), exist_ok=True)
        port = ask_model.free_port()
        print(f"loading {os.path.basename(model)} ...", flush=True)
        server = ask_model.start_server(model, port, os.path.join(STEP5, "portable", "tmp", "llama_server_demo.log"))
        Handler.llama_port = port

    httpd = Server(
        ("0.0.0.0" if args.lan else "127.0.0.1", args.port), functools.partial(Handler, directory=os.path.join(DEMO, "web")))
    url = f"http://127.0.0.1:{args.port}"
    print(f"demo running at {url}  ({'recorded answers' if args.mock else 'live model'}); Ctrl-C to stop")
    if args.lan:
        print(f"on a phone on the same Wi-Fi: http://{lan_address()}:{args.port}/#intake")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    def stop(*_):
        raise KeyboardInterrupt  # so the finally below also stops llama-server
    signal.signal(signal.SIGTERM, stop)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
        if server:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
