#!/bin/bash
# Build the phone bundle on the Mac: only what the demo needs, in the repo's own layout.
#
#     bash HardLLM/Step_5/phone/build_phone_bundle.sh
#
# Writes Step_5/phone/dist/HardLLM_phone.zip (about 1.9 GB, the model is most of it).
# Copy that zip to the phone's Download folder, then follow Step_5/phone/README.md.
set -euo pipefail

STEP5="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$STEP5")"
DIST="$STEP5/phone/dist"
OUT="$DIST/HardLLM_phone"
MODEL="Step_5/models/selected-q4_k_m.gguf"

[ -f "$ROOT/$MODEL" ] || { echo "missing $MODEL (bring it back from the PC run)"; exit 1; }
rm -rf "$OUT" "$DIST/HardLLM_phone.zip"
mkdir -p "$OUT/HardLLM"

files=(
  # the numbers on the Results tab, read at start-up (server.py story())
  Step_1/models/cluster_summary.json Step_1/models/cooldown.log Step_1/models/unseen_eval.json
  Step_2/out/noise_sweep_centroid.json
  Step_3/out/train_summary_qa.json Step_3/out/synthetic_qa_stats.json Step_3/out/clean_qa_report.json
  Step_4/out/fullsample_arm.json Step_4/data/demo_questions.jsonl
  Step_4/out/eval_base.json Step_4/out/eval_hardllm.json Step_4/out/eval_hardllm-filtered.json
  Step_4/out/eval_hardllm-fullsample.json Step_4/out/eval_random.json Step_4/out/eval_selected.json
  Step_4/out/eval_private.json
  Step_5/out/eval_q4_k_m.json Step_5/out/results_step5.json
  # the demo itself
  Step_5/ask_model.py Step_5/demo/server.py Step_5/demo/triage.py Step_5/demo/answer_filter.py
  Step_5/demo/web/index.html Step_5/demo/web/app.js Step_5/demo/web/app.css
  Step_5/tests/check_demo_live.py Step_5/tests/test_triage.py Step_5/tests/test_answer_filter.py
  Step_5/phone/setup_termux.sh Step_5/phone/run_demo.sh Step_5/phone/README.md
)
for f in "${files[@]}"; do
  mkdir -p "$OUT/HardLLM/$(dirname "$f")"
  cp "$ROOT/$f" "$OUT/HardLLM/$f"
done
mkdir -p "$OUT/HardLLM/Step_5/models"
cp -c "$ROOT/$MODEL" "$OUT/HardLLM/$MODEL" 2>/dev/null || cp "$ROOT/$MODEL" "$OUT/HardLLM/$MODEL"  # APFS clone: no extra space

# the model is already compressed: store it, do not deflate it again
(cd "$DIST" && zip -qr -0 HardLLM_phone.zip HardLLM_phone)
echo "built $(du -h "$DIST/HardLLM_phone.zip" | cut -f1) $DIST/HardLLM_phone.zip"
