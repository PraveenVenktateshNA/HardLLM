"""Answer filter tests: the model's answers keep information, lose instructions. Plain Python, instant.

Checks sentence by sentence on hand-written cases and on the real 4-bit model's
recorded answers to the 20 demo questions (Step_5/out/eval_q4_k_m.json), and
that streaming in random pieces gives the same text as filtering the whole
answer. Exit 1 on any failure.

    python3 Step_5/tests/test_answer_filter.py
"""
import json
import os
import random
import sys

STEP5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(STEP5, "demo"))
import answer_filter as f  # noqa: E402

INSTRUCTIONS = [
    "Take omeprazole 20 mg twice daily.", "Avoid spicy food.", "You should consult a cardiologist.",
    "So get done ECG and 2d echo to rule out cardiac cause.", "I suggest that you visit a dermatologist.",
    "Better to consult a doctor.", "It is advisable to get a blood test.", "Need not worry.",
    "Don't worry, it will settle.", "Hope this helps.", "Thanks for your query.", "Please drink plenty of fluids.",
    "You can take paracetamol for fever.", "Apply ice packs on the area.", "Go for an ultrasound abdomen.",
    "If you take it regularly then it will help.", "You may need antibiotics.", "Rest for a few days.",
    "Kindly revert with reports.", "Also do warm saline gargles.", "Your child should be seen by a paediatrician.",
    "Get your thyroid checked.", "Let me know if you have more questions.", "Take care.",
]
INFORMATION = [
    "This can be due to gastritis or a peptic ulcer.", "Musculoskeletal pain is common with bad posture.",
    "Pernicious anaemia is caused by a lack of intrinsic factor.", "It is present at birth.",
    "Doctor may prescribe muscle relaxants.", "These symptoms may be due to many reasons like low blood pressure.",
    "Infection is a common reason to return to hospital after surgery.", "You are having symptoms of TMJ disorder.",
    "It is not possible to diagnose this without an examination.", "Smoking is a major cause of lung cancer.",
]
# what must survive in the real answers (one fact per answer that should be shown)
REAL_KEEP = {3: "musculoskeletal pain", 4: "worm", 13: "gastritis", 14: "high cholesterol", 17: "gastritis", 19: "congenital"}
ACTION_WORDS = ["you should", "take ", "avoid ", "consult", "i suggest", "better to", "get done", "go for", "need not worry", " mg"]


def main():
    fails = []
    for s in INSTRUCTIONS:
        if not f.is_instruction(s):
            fails.append(f"kept an instruction: {s!r}")
    for s in INFORMATION:
        if f.is_instruction(s):
            fails.append(f"dropped information: {s!r}")

    ev = json.load(open(os.path.join(STEP5, "out", "eval_q4_k_m.json"), encoding="utf-8"))
    rng = random.Random(0)
    for i, a in enumerate(ev["demo_answers"]):
        whole, dropped = f.keep_information(a)
        low = whole.lower()
        for w in ACTION_WORDS:
            if w in low and not low.startswith("doctor may"):
                # "take" is fine inside a description ("Doctor may ... take impression of teeth")
                if not (w == "take " and "doctor may order x-ray of jaw and take" in low and low.count("take ") == 1):
                    fails.append(f"answer {i}: shown text still has {w!r}: {whole[:160]!r}")
        if "your history" in low:
            fails.append(f"answer {i}: 'your history' opener shown")
        if i in REAL_KEEP and REAL_KEEP[i] not in low:
            fails.append(f"answer {i}: lost the information {REAL_KEEP[i]!r}")
        # streaming in random pieces gives the same kept sentences as the whole answer (bar a cut-off tail)
        st = f.Sentences()
        out, pos = "", 0
        while pos < len(a):
            n = rng.randint(1, 12)
            out += st.feed(a[pos:pos + n])
            pos += n
        out += st.finish()
        sents = f.split(a)
        no_tail, _ = f.keep_information(" ".join(sents[:-1])) if sents and not sents[-1].rstrip().endswith((".", "!", "?")) else (whole, 0)
        if out not in (whole, no_tail, f.NOTHING_LEFT):
            fails.append(f"answer {i}: streamed {out[:100]!r} differs from whole {whole[:100]!r}")
    # a pronoun pointing at a dropped sentence goes with it
    whole, _ = f.keep_information("You may have some cyst. You should go for a biopsy. It will give you exact diagnosis.")
    if whole != "You may have some cyst.":
        fails.append(f"dangling pronoun kept: {whole!r}")
    whole, _ = f.keep_information("It can be due to gastritis. It can also be due to an ulcer.")
    if whole != "It can be due to gastritis. It can also be due to an ulcer.":
        fails.append(f"pronoun after a kept sentence dropped: {whole!r}")
    # the model writing as a patient is not an answer
    whole, _ = f.keep_information("I have a question regarding antibiotics. I am taking an antibiotic called Augmentin for 5 days.")
    if whole:
        fails.append(f"patient voice kept: {whole!r}")
    # the chat keeps every sentence, only rewording the opener
    st = f.Sentences(keep_all=True)
    out = st.feed("By your history and description, it is gastritis. Take omeprazole 20 mg. ") + st.finish()
    if out != "From your description, it is gastritis. Take omeprazole 20 mg." or st.dropped:
        fails.append(f"keep_all changed the answer: {out!r}")
    # all advice: a clear note instead of an empty box
    st = f.Sentences()
    out = st.feed("Take rest. Avoid stress. ") + st.finish()
    if out != f.NOTHING_LEFT:
        fails.append(f"all-advice answer gave {out!r}")
    # a cut-off last sentence is dropped
    st = f.Sentences()
    out = st.feed("It can be due to gastritis. Doctor may prescribe anti-inflamma") + st.finish()
    if out != "It can be due to gastritis.":
        fails.append(f"cut-off tail kept: {out!r}")

    for x in fails:
        print("FAIL ", x)
    print(f"{'all passed' if not fails else str(len(fails)) + ' FAILED'} "
          f"({len(INSTRUCTIONS)} instructions, {len(INFORMATION)} facts, {len(ev['demo_answers'])} real answers)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
