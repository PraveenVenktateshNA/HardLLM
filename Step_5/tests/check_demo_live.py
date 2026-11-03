"""Check the RUNNING demo with the live model, before showing it. About 2 min.

Start the demo first (python3 HardLLM/Step_5/demo/server.py), then:

    python3 HardLLM/Step_5/tests/check_demo_live.py [--url http://127.0.0.1:8080]

Sends the 20 demo questions plus the Intake examples, a greeting and a plain
question to /api/ask as the Intake note asks (information only) and checks what
the page would show: every answer finishes, no sentence telling the reader what
to do gets through, no
"by your history" opener, emergencies get the alert and others do not, a
greeting gets the fixed reply, and each answer comes back in under 60 s. Then
two questions as the Ask tab asks (the whole answer): nothing dropped.
Writes out/demo_live_check.json with every answer shown. Exit 1 on any failure.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

STEP5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(STEP5, "demo"))
import answer_filter  # noqa: E402
import triage  # noqa: E402

EXTRA = [
    "i am having my left chest pain sharpy and im unable to breath",
    "I am having sharp pain in my left chest and I am unable to breathe",
    "My 2 year old has had a fever of 104 for three days",
    "Burning sensation when I urinate since yesterday",
    "Itchy red rash on my arm for a week, no fever",
    "what is diabetes?",
    "hello",
]
MAX_SECONDS = 60


def ask(url, q):
    req = urllib.request.Request(url + "/api/ask", json.dumps({"question": q}).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        ev = [json.loads(l[6:]) for l in r.read().decode().split("\n\n") if l.startswith("data: ")]
    return ev, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    story = json.load(urllib.request.urlopen(args.url + "/api/story", timeout=10))
    if story.get("mock"):
        raise SystemExit("the demo is running with --mock; start it without --mock to check the live model")
    with open(os.path.join(os.path.dirname(STEP5), "Step_4", "data", "demo_questions.jsonl"), encoding="utf-8") as f:
        questions = [json.loads(l)["question"] for l in f] + EXTRA

    fails, rows = [], []
    for i, q in enumerate(questions):
        ev, dt = ask(args.url, q)
        shown = "".join(e.get("t", "") for e in ev)
        done = ev[-1] if ev else {}
        alert = any("alert" in e for e in ev)
        emergency = bool(triage.red_flags(q))
        bad = []
        if not done.get("done"):
            bad.append("answer did not finish")
        if not shown.strip():
            bad.append("nothing shown")
        if dt > MAX_SECONDS:
            bad.append(f"took {dt:.0f} s")
        if alert != emergency:
            bad.append(f"alert {alert} but emergency {emergency}")
        if "your history" in shown.lower():
            bad.append("'your history' opener shown")
        if q == "hello" and not done.get("canned"):
            bad.append("greeting went to the model")
        model_text = shown.replace(answer_filter.NOTHING_LEFT, "") if not done.get("canned") else ""  # the page's own messages
        told = [s for s in answer_filter.split(model_text) if answer_filter.is_instruction(s)]
        if told:
            bad.append(f"instruction shown: {told}")
        rows.append({"question": q, "shown": shown, "seconds": round(dt, 1), "tokens": done.get("tokens"),
                     "dropped": done.get("dropped"), "alert": alert, "problems": bad})
        print(f"{'ok  ' if not bad else 'FAIL'} {i:2d} {dt:4.1f}s dropped={done.get('dropped')} alert={alert}  {shown[:110]}")
        fails += [f"{i}: {b}" for b in bad]

    # the chat (Ask tab) shows the whole answer: advice reaches the page, the opener is still reworded
    for q in ["i am having my left chest pain sharpy and im unable to breath", "I have burning pain in my upper stomach after meals for 2 weeks"]:
        req = urllib.request.Request(args.url + "/api/ask", json.dumps({"question": q, "filter": False}).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            ev = [json.loads(l[6:]) for l in r.read().decode().split("\n\n") if l.startswith("data: ")]
        dt = time.time() - t0
        shown = "".join(e.get("t", "") for e in ev)
        bad = [b for b, cond in [("did not finish", not ev[-1].get("done")), ("sentences dropped in chat mode", ev[-1].get("dropped")),
                                 ("nothing shown", not shown.strip()), ("'your history' opener shown", "your history" in shown.lower())] if cond]
        print(f"{'ok  ' if not bad else 'FAIL'} chat {shown[:110]}")
        rows.append({"question": q, "mode": "chat", "shown": shown, "seconds": round(dt, 1), "problems": bad})
        fails += [f"chat {q[:30]}: {b}" for b in bad]

    out = os.path.join(STEP5, "out", "demo_live_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"url": args.url, "version": story["version"], "checked": time.strftime("%Y-%m-%d %H:%M"),
                   "failures": fails, "answers": rows}, f, indent=1)
    times = [r["seconds"] for r in rows]
    print(f"\n{len(rows)} questions, {min(times):.1f}-{max(times):.1f} s each; answers in {os.path.relpath(out)}")
    print("all passed" if not fails else f"{len(fails)} FAILED:\n  " + "\n  ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
