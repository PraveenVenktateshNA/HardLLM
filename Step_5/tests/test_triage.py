"""Triage rule tests for the demo's Intake tab. Plain Python, under a second.

Each case is a description a patient or health worker might type and the
urgency, department or details the rules must give. Exit 1 on any failure.

    python3 Step_5/tests/test_triage.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo"))
import triage  # noqa: E402

# (description, expected level, extra checks on the result)
CASES = [
    # --- emergencies
    ("i am having my left chest pain sharpy and im unable to breath", "emergency", {"department": "Emergency department"}),
    ("Sudden pressure in my chest spreading to my left arm, sweating", "emergency", {}),
    ("I think I'm having a heart attack", "emergency", {}),
    ("my dad suddenly has slurred speech and his face is drooping", "emergency", {}),
    ("weakness on one side of the body since an hour", "emergency", {}),
    ("my wife fainted twice today", "emergency", {"sex": "female"}),
    ("she is not responding when we call her", "emergency", {}),
    ("my son had a seizure for 2 minutes", "emergency", {"sex": "male"}),
    ("worst headache of my life, came on suddenly", "emergency", {}),
    ("I have been coughing up blood since yesterday", "emergency", {}),
    ("vomiting blood this morning", "emergency", {}),
    ("I'm 8 weeks pregnant and bleeding", "emergency", {}),
    ("my throat is swelling after eating peanuts", "emergency", {}),
    ("I feel like I want to die and I have thought about suicide", "emergency", {"specialty": "Psychiatry"}),
    ("my child swallowed bleach", "emergency", {}),
    ("snake bite on the leg an hour ago", "emergency", {}),
    ("Can't breathe properly, gasping for air", "emergency", {}),
    ("shortness of breath when climbing stairs for a week", "emergency", {}),
    ("my 2 month old baby has a fever", "emergency", {"age": "2 months"}),
    ("high fever and a stiff neck since this morning", "emergency", {}),
    ("temperature of 106 F and confused", "emergency", {}),
    ("cut on my hand that won't stop bleeding", "emergency", {}),
    ("Chest tightness and I can’t catch my breath", "emergency", {}),
    ("I suddenly lost vision in my right eye", "emergency", {}),

    # --- urgent
    ("my 2 year old has had a fever of 104 for three days", "urgent", {"age": "2 years"}),
    ("fever for 5 days with body aches", "urgent", {}),
    ("I've had fever since monday and a cough", "urgent", {}),
    ("blood in my stool for 2 days", "urgent", {"department": "Gastroenterology"}),
    ("severe stomach pain on the right side", "urgent", {}),
    ("vomiting everything, can't keep water down since morning", "urgent", {}),
    ("my heart is racing and pounding at rest", "urgent", {"department": "Cardiology"}),
    ("dog bite on the hand this morning", "urgent", {}),
    ("I am 30 weeks pregnant with mild back pain", "urgent", {"department": "Obstetrics and gynaecology"}),
    ("temperature 39.8 C", "urgent", {"temperature_c": 39.8}),
    ("my 70 year old mother has diarrhoea for 3 days", "urgent", {"age": "70 years"}),
    ("hit my head on the door, mild headache", "urgent", {"department": "Neurology"}),

    # --- routine
    ("I have a cough and runny nose for 2 days", "routine", {"duration": "2 days"}),
    ("itchy red rash on my arm", "routine", {"department": "Dermatology"}),
    ("pimples on my face for months", "routine", {"department": "Dermatology"}),
    ("lower back pain after lifting boxes", "routine", {"department": "Orthopaedics"}),
    ("knee pain when climbing stairs, I am 55", "routine", {"age": "55 years"}),
    ("burning sensation when I urinate", "routine", {"department": "Urology"}),
    ("I can't sleep at night and feel stressed", "routine", {"department": "Psychiatry"}),
    ("mild headache since yesterday", "routine", {"duration": "since yesterday"}),
    ("migraine again, light hurts my eyes", "routine", {"department": "Neurology"}),
    ("I have fever of 100 since yesterday", "routine", {}),
    ("toothache on the left side", "routine", {"department": "Dental"}),
    ("acidity and bloating after meals", "routine", {"department": "Gastroenterology"}),
    ("my 6 year old son has a cold and cough", "routine", {"department": "Paediatrics", "age": "6 years"}),
    ("irregular periods for 3 months", "routine", {"department": "Obstetrics and gynaecology"}),
    ("cough for 2 weeks and fever since yesterday", "routine", {}),

    # --- negations must not raise the level
    ("no chest pain, just a mild cough", "routine", {}),
    ("I don't have chest pain or shortness of breath, only a sore throat", "routine", {"department": "ENT"}),
    ("no fever, but a rash on my legs", "routine", {}),
    ("denies fainting, has knee pain", "routine", {}),
    ("I don't feel faint, just tired", "routine", {}),
    ("not pregnant, period pain", "routine", {}),
    ("I had no appetite and now chest pain", "emergency", {}),  # "no" belongs to appetite
    ("not sure if it is chest pain or acidity", "emergency", {}),  # unsure is not a negation
    ("no cough or fever but chest pain since an hour", "emergency", {}),  # "but" ends the negation
    ("never had this before: chest pain and sweating", "emergency", {}),

    # --- nothing recognisable: a nurse reviews it
    ("hello", "review", {}),
    ("I need a medical certificate", "review", {}),
    ("", "review", {}),

    # --- capital letters and word edges
    ("No chest pain. Mild cough since Monday.", "routine", {}),
    ("NO FEVER, just a runny nose", "routine", {}),
    ("Witch doctor said it was nothing, my knee hurts", "routine", {"department": "Orthopaedics"}),
    ("I want to get pregnant, irregular periods", "routine", {}),
    ("my 82.5 year old father has a cough", "routine", {"age": "82 years"}),
    ("pain in the middle of my chest since yesterday", "emergency", {}),
    ("my chest really hurts when I breathe in", "emergency", {}),
    ("chest X-ray was clear, mild cough", "routine", {}),
    ("stung by a bee on my lip, lips swelling", "emergency", {}),
    ("bee sting on the arm, a bit red", "routine", {}),

    # --- words that look like symptoms but are not
    ("I have cold hands in winter and itchy skin", "routine", {}),
    ("my 5 year old daughter has an itchy rash", "routine", {"age": "5 years", "duration": None}),
]


# The 20 Step 4 demo questions (real ChatDoctor patients, Step_4/data/demo_questions.jsonl):
# index -> (level a triage nurse would give, extra checks)
DEMO_EXPECT = {
    0: ("review", {}),                     # B12 injections, asking about a blood test
    1: ("urgent", {"age": "82 years"}),    # fluid on the lung, E. coli in the blood, after surgery
    2: ("review", {}),                     # skin hurts when the shirt touches it, no detail
    3: ("emergency", {}),                  # pressure in the middle of the chest, worse lying flat
    4: ("urgent", {}),                     # stomach and testicles aching
    5: ("routine", {"age": "26 years", "sex": "female"}),  # dizzy spells for three years, tingling
    6: ("routine", {"department": "General surgery"}),     # painful lump
    7: ("routine", {"department": "General surgery"}),     # lump on the jaw, swollen ("Witch" is not itching)
    8: ("routine", {}),                    # fertility treatment question
    9: ("routine", {"department": "Orthopaedics"}),        # lower back pain, wants to get pregnant
    10: ("routine", {}),                   # past wound infection, asking about the surgeon's care
    11: ("routine", {}),                   # jaw pain (TMJ)
    12: ("routine", {"department": "Urology"}),            # stinging when urine touches a sore area
    13: ("urgent", {}),                    # vomiting, nothing stays down
    14: ("review", {}),                    # blood test results, no symptoms given
    15: ("routine", {}),                   # unusual bleeding while trying to conceive
    16: ("routine", {}),                   # hurts to swallow after thyroid surgery
    17: ("urgent", {}),                    # abdominal pain 5 days, gets severe; "No fever."
    18: ("emergency", {}),                 # bee sting under the tongue, dizzy, nausea
    19: ("urgent", {}),                    # pregnant, heart defect
}


def main():
    fails = 0
    demo = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Step_4", "data", "demo_questions.jsonl")
    import json
    with open(demo, encoding="utf-8") as f:
        questions = [json.loads(l)["question"] for l in f]
    CASES.extend((questions[i], lvl, chk) for i, (lvl, chk) in DEMO_EXPECT.items())
    r17 = triage.assess(questions[17])
    if "fever" in r17["symptoms"]:
        fails += 1
        print("FAIL  demo question 17 says 'No fever.' but fever was counted")
    for text, level, checks in CASES:
        r = triage.assess(text)
        bad = [] if r["level"] == level else [f"level {r['level']} != {level}"]
        bad += [f"{k} {r[k]!r} != {v!r}" for k, v in checks.items() if r[k] != v]
        if text and r["level"] != "review" and not r["reasons"]:
            bad.append("no reasons given")
        if bad:
            fails += 1
            print(f"FAIL  {text!r}\n      " + "; ".join(bad) + f"\n      symptoms={r['symptoms']} reasons={r['reasons']}")
    # determinism: same input, same output
    for text, _, _ in CASES:
        assert triage.assess(text) == triage.assess(text)
    # every emergency reason reaches the chat tab's alert
    assert triage.red_flags("i am having my left chest pain sharpy and im unable to breath")
    assert not triage.red_flags("I have a mild rash on my arm")
    print(f"{len(CASES) - fails}/{len(CASES)} triage cases pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
