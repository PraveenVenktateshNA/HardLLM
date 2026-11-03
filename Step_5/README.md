# Step 5: the final model, made phone-ready

The final model is Step 4's `selected` arm (trained on the Step 2 selection).
Step 3's synthetic data was assessed in Step 4 and made the model worse than
base (`Step_4/out/results_step4_fullsample.md`), so the pipeline rolls back to
the Step 2 output.

| Script | Does |
|---|---|
| `merge_model.py` | folds `Step_4/models/lora-selected` into Llama-3.2-3B -> `models/selected-merged/` |
| `export_gguf.py` | GGUF 16-bit, then 8-bit (`q8_0`) and 4-bit (`q4_k_m`) with llama.cpp -> `models/*.gguf` |
| `score_gguf.py` | each version on the Step 4 test (answer perplexity), plus llama.cpp's own perplexity -> `out/` |
| `run_convert.py` | starts llama.cpp's converter with the bundled Python |
| `ask_model.py` | ask the model questions in the terminal, on the Mac |
| `demo/server.py` | the demo web page: live chat, patient intake, hospital queue, results |
| `demo/triage.py` | the intake's rule-based triage (urgency, symptoms, duration, age, department) |
| `phone/` | the same demo on an Android phone, offline (Termux): see `phone/README.md` |
| `demo/answer_filter.py` | keeps the model's answers to information: drops sentences that tell the reader what to do |

Run on the PC: put this folder next to the bundle's `Step_4` folder and run
`portable/RUN_STEP5_EXPORT.bat` (about 20-25 min). llama.cpp is in
`portable/llama.cpp/` (release b11552, see `VERSION.txt`).

## On the Mac, after the PC run

Needs `models/selected-q4_k_m.gguf` (and `selected-f16.gguf` for `--version f16`).
llama.cpp's Mac build is in `portable/llama.cpp-mac/` (same release), runs on
the Apple GPU, offline. Plain Python, no torch.

    python3 HardLLM/Step_5/ask_model.py                      # terminal chat
    python3 HardLLM/Step_5/ask_model.py -q "my knee hurts"   # one question
    python3 HardLLM/Step_5/demo/server.py                    # demo page, http://127.0.0.1:8080
    python3 HardLLM/Step_5/demo/server.py --mock             # page with recorded answers, no model
    python3 HardLLM/Step_5/demo/server.py --lan              # phones on the same Wi-Fi can open it (no login)

Tabs open directly at `#ask`, `#intake`, `#queue`, `#results`.

Intake: fixed rules (`demo/triage.py`), not the model, set the urgency
(emergency / urgent / needs review / routine) and the summary, and show the
words that caused them; negated symptoms ("no chest pain") do not count. The
model's answer is shown below as an "AI note", skipped for emergencies and
flagged if it sounds reassuring under an urgent result. In the chat and the
note alike, the model only explains: `answer_filter.py` drops every sentence
that tells the reader what to do (take, avoid, consult, doses, "you should",
sign-offs) and the page says how many were hidden. A greeting gets a fixed
reply instead of the model. The hospital queue is
kept in `demo/queue.json` (delete it to start empty).

Tests, plain Python, seconds, safe on the Mac (no model):

    python3 HardLLM/Step_5/tests/test_triage.py      # 96 descriptions (incl. the 20 real demo questions) -> expected urgency
    python3 HardLLM/Step_5/tests/test_demo_api.py    # every endpoint, bad input, restart
    python3 HardLLM/Step_5/tests/test_answer_filter.py   # advice dropped, facts kept, on the real answers

With the demo running live (not --mock), before showing it (about 2 min):

    python3 HardLLM/Step_5/tests/check_demo_live.py  # 27 questions through the live model -> out/demo_live_check.json
Every number on the page is read from the steps' output files at start-up
(listed under "Where every number comes from" on the Results tab).
