"""Demo server API tests: intake, hospital queue, chat alert. Plain Python, ~10 s.

Starts server.py --mock (no model) on a free port with a temporary queue
file, drives every endpoint including bad input, restarts it to check the
queue survives, and stops it. The real queue.json is not touched. Exit 1 on
any failure.

    python3 Step_5/tests/test_demo_api.py
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")
FAILS = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILS.append(what)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(port, queue):
    proc = subprocess.Popen([sys.executable, os.path.join(DEMO, "server.py"), "--mock", "--no-browser",
                             "--port", str(port), "--queue", queue],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/cases", timeout=1)
            return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit("server did not start:\n" + proc.stdout.read())


def call(port, path, body=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode()
            return r.status, (json.loads(text) if r.headers.get_content_type() == "application/json" else text)
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        try:
            return e.code, json.loads(text)
        except ValueError:
            return e.code, text


def events(text):
    return [json.loads(l[6:]) for l in text.split("\n\n") if l.startswith("data: ")]


def main():
    tmp = tempfile.mkdtemp()
    queue = os.path.join(tmp, "queue.json")
    port = free_port()
    proc = start(port, queue)
    try:
        # intake
        s, r = call(port, "/api/intake", {"text": "i am having my left chest pain sharpy and im unable to breath"})
        check(s == 200 and r["assessment"]["level"] == "emergency", "intake: chest pain + can't breathe is an emergency")
        s, r = call(port, "/api/intake", {"text": "itchy rash on my arm"})
        check(s == 200 and r["assessment"]["level"] == "routine" and r["assessment"]["department"] == "Dermatology",
              "intake: rash is routine, dermatology")
        s, r = call(port, "/api/intake", {"text": ""})
        check(s == 200 and r["assessment"]["level"] == "review", "intake: empty text needs review, no crash")
        s, r = call(port, "/api/intake", {"text": 12345})
        check(s == 200 and r["assessment"]["level"] == "review", "intake: a number instead of text does not crash")
        s, r = call(port, "/api/intake", {"text": "cough " * 5000})
        check(s == 400, "intake: body over 20 kB refused")
        s, _ = call(port, "/api/intake", raw=b"not json")
        check(s == 400, "intake: invalid JSON refused")
        s, _ = call(port, "/api/intake", raw=b"[1, 2]")
        check(s == 400, "intake: JSON that is not an object refused")

        # queue: add in an order that the sort must fix
        s, a = call(port, "/api/cases", {"text": "itchy rash on my arm", "patient": "Routine R"})
        check(s == 200 and a["id"] == 1 and a["status"] == "waiting", "queue: add routine case")
        s, b = call(port, "/api/cases", {"text": "my 2 year old has had a fever of 104 for three days", "patient": "Urgent U"})
        s, c = call(port, "/api/cases", {"text": "sudden chest pain and sweating", "patient": "<script>alert(1)</script>"})
        check(s == 200 and c["assessment"]["level"] == "emergency", "queue: add emergency case")
        s, dup = call(port, "/api/cases", {"text": "sudden chest pain and sweating", "patient": "<script>alert(1)</script>"})
        check(s == 200 and dup["id"] == c["id"], "queue: the same patient and words again is the same case, not a duplicate")
        s, _ = call(port, "/api/cases", {"text": "   ", "patient": "x"})
        check(s == 400, "queue: empty description refused")
        s, _ = call(port, "/api/cases", {"text": "cough", "patient": "p" * 500})
        s, r = call(port, "/api/cases")
        ids = [x["id"] for x in r["cases"]]
        check(ids[:3] == [3, 2, 1], f"queue: sorted emergency, urgent, routine (got {ids})")
        check(len(next(x for x in r["cases"] if x["id"] == 4)["patient"]) == 60, "queue: patient name cut to 60 characters")
        check(r["cases"][0]["patient"] == "<script>alert(1)</script>", "queue: names stored as plain text (the page shows them with textContent)")

        # status changes
        s, r = call(port, "/api/cases/3", {"status": "seen"})
        check(s == 200 and r["status"] == "seen" and r["seen_at"], "queue: mark seen")
        s, r = call(port, "/api/cases")
        check([x["id"] for x in r["cases"]][-1] == 3, "queue: seen cases sort after waiting ones")
        s, r = call(port, "/api/cases/3", {"status": "waiting"})
        check(s == 200 and r["status"] == "waiting", "queue: back to waiting")
        s, _ = call(port, "/api/cases/3", {"status": "deleted"})
        check(s == 400, "queue: unknown status refused")
        s, _ = call(port, "/api/cases/999", {"status": "seen"})
        check(s == 404, "queue: unknown case is 404")
        s, _ = call(port, "/api/cases/4", {"remove": True})
        s, r = call(port, "/api/cases")
        check(s == 200 and 4 not in [x["id"] for x in r["cases"]], "queue: remove")

        # many at once: no lost or duplicate ids
        def add(i):
            call(port, "/api/cases", {"text": f"cough for {i % 9 + 1} days", "patient": f"P{i}"})
        threads = [threading.Thread(target=add, args=(i,)) for i in range(40)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        s, r = call(port, "/api/cases")
        ids = [x["id"] for x in r["cases"]]
        check(len(ids) == 43 and len(set(ids)) == 43, f"queue: 40 parallel adds, {len(ids)} cases, all ids distinct")

        # chat: the emergency alert comes before the answer, the opener is reworded
        s, text = call(port, "/api/ask", {"question": "i am having my left chest pain sharpy and im unable to breath"})
        ev = events(text)
        check(s == 200 and "alert" in ev[0] and ev[-1].get("done"), "ask: emergency alert first, then the answer")
        answer = "".join(e.get("t", "") for e in ev)
        check("your history" not in answer.lower(), "ask: no 'by your history' in the answer")
        s, text = call(port, "/api/ask", {"question": "i am having my left chest pain sharpy and im unable to breath", "filter": False})
        ev = events(text)
        whole = "".join(e.get("t", "") for e in ev)
        check(s == 200 and "alert" in ev[0] and ev[-1].get("dropped") == 0 and "ECG" in whole and "your history" not in whole.lower(),
              "ask, chat mode: the whole answer, advice included, opener reworded, alert still first")
        s, text = call(port, "/api/ask", {"question": "I have a mild rash on my arm"})
        check(s == 200 and not any("alert" in e for e in events(text)), "ask: no alert for a mild rash")
        s, text = call(port, "/api/ask", {"question": "hello"})
        ev = events(text)
        check(s == 200 and ev[-1].get("canned") and "Describe what you feel" in ev[0]["t"], "ask: 'hello' gets the fixed reply, not the model")
        s, _ = call(port, "/api/ask", {"question": ""})
        check(s == 400, "ask: empty question refused")

        # the page and its files
        s, html = call(port, "/")
        check(s == 200 and 'id="tab-intake"' in html and 'id="tab-queue"' in html, "page: has the Intake and Hospital queue tabs")
        check('data-tab="compare"' not in html and 'data-tab="pipeline"' not in html, "page: Compare and How it works are gone")
        s, story = call(port, "/api/story")
        check(s == 200 and story["mock"] is True, "story: loads")
    finally:
        proc.terminate()
        proc.wait()

    # restart: the queue is still there
    proc = start(port, queue)
    try:
        s, r = call(port, "/api/cases")
        check(s == 200 and len(r["cases"]) == 43, "restart: queue kept (43 cases)")
    finally:
        proc.terminate()
        proc.wait()

    # a broken queue file does not stop the server
    with open(queue, "w") as f:
        f.write("{broken")
    proc = start(port, queue)
    try:
        s, r = call(port, "/api/cases")
        check(s == 200 and r["cases"] == [] and os.path.exists(queue + ".unreadable"),
              "broken queue file: set aside, server starts empty")
    finally:
        proc.terminate()
        proc.wait()

    print(f"\n{'all passed' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
