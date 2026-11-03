#!/data/data/com.termux/files/usr/bin/bash
# Start the demo on the phone, fully offline, and open it in the browser.
#
#     bash ~/HardLLM_phone/HardLLM/Step_5/phone/run_demo.sh          # live model
#     bash ~/HardLLM_phone/HardLLM/Step_5/phone/run_demo.sh --mock   # no model, recorded answers
#
# Loading the model takes 20-60 s on a phone. Ctrl-C (or closing Termux) stops it.
set -uo pipefail

STEP5="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
PORT="${PORT:-8080}"
LOG="$STEP5/phone/demo.log"
URL="http://127.0.0.1:$PORT/#intake"

termux-wake-lock 2>/dev/null || true  # keep Android from pausing Termux while the screen is on the browser
"$PY" -u "$STEP5/demo/server.py" --no-browser --port "$PORT" "$@" >"$LOG" 2>&1 &
PID=$!
trap 'kill -TERM $PID 2>/dev/null; wait $PID 2>/dev/null; termux-wake-unlock 2>/dev/null || true' EXIT INT TERM HUP

echo "loading the model (20-60 s on a phone)..."
for _ in $(seq 1 600); do
  if grep -q "demo running" "$LOG"; then
    echo "demo running at $URL"
    termux-open-url "$URL" 2>/dev/null || echo "open $URL in the browser"
    echo "Ctrl-C here stops it."
    wait $PID
    exit $?
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "the demo stopped while starting:"
    tail -20 "$LOG"
    exit 1
  fi
  sleep 0.5
done
echo "the demo did not start in 5 min:"
tail -20 "$LOG"
exit 1
