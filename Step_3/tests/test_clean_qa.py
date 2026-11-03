"""Tests for clean_synthetic_qa.py: real answers from synthetic_qa.jsonl (some
shortened) and what the cleaner must turn them into. Plain Python, no model.

    python Step_3/tests/test_clean_qa.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clean_synthetic_qa as c  # noqa: E402

CASES = [
    # greeting and sign-off go; "As per your query" is content, not a greeting
    ("Hi, Thanks for your query. As per your query your back pain is mostly due to back "
     "pain arising from trauma. You need to visit your doctor. Hope I have answered your "
     "query. Take care",
     "As per your query your back pain is mostly due to back pain arising from trauma. "
     "You need to visit your doctor."),
    # glued "HelloThanks", scraping artefact "come have gone"
    ("HelloThanks for query.come have gone through your question. It appears to be a "
     "prostate cancer. Consult a gynecologist for clinical evaluation and management. Regards.",
     "It appears to be a prostate cancer. Consult a gynecologist for clinical evaluation "
     "and management."),
    # doctor signature junk after the sign-off
    ("Hello, Dear, Welcome to Chat Doctor . There are many causes for abscess like "
     "gallstones. You should go for culture. Wishing you a good health, Take Care Chat "
     "Doctor. N. Genl-CVTS, Genl-CVTS S. Genl-CVTS S. Genl-CVTS & L-CVTS. Thanks.",
     "There are many causes for abscess like gallstones. You should go for culture."),
    # list numbers glued to sentences; "Chat Doctor" standing in for a drug name
    ("Hello, You had a gum infection. I recommend following Chat Doctor.1. Avoid taking "
     "food, especially fatty foods.2. Avoid large meals, instead take small meals.3. Chat "
     "Doctor.4. Avoid stress and anxiety. I hope I answered your query. Thank you.",
     "You had a gum infection. Avoid taking food, especially fatty foods. Avoid large "
     "meals, instead take small meals. Avoid stress and anxiety."),
    # "best" and "I hope" as content at the end must stay
    ("Hi. You likely have reflux. It is best to avoid spicy food. I hope the scan comes "
     "back normal soon and you recover.",
     "You likely have reflux. It is best to avoid spicy food. I hope the scan comes back "
     "normal soon and you recover."),
    # content in the same sentence as the greeting must stay
    ("Hi, Welcome to Chat Doctor, Based on the information provided by you, it appears "
     "your child has a severe jaw pain. See a dentist.",
     "Based on the information provided by you, it appears your child has a severe jaw "
     "pain. See a dentist."),
    ("Hello, It is a rare occurrence for a cyst in the kidney to form in a person due to "
     "an illness. Get an ultrasound.",
     "It is a rare occurrence for a cyst in the kidney to form in a person due to an "
     "illness. Get an ultrasound."),
    ("Hi welcome to Chat Doctor, as you have mentioned that you are having a mild upper "
     "abdominal pain, it may be gastritis. Take antacids.",
     "As you have mentioned that you are having a mild upper abdominal pain, it may be "
     "gastritis. Take antacids."),
    # chained pleasantries; short real sentences ("Take iron.") stay
    ("HelloThanks for writing to us, and thank you for the information you provided. I "
     "understand your concern and I will try to help you as much as I can. Your report "
     "shows anemia. Take iron.",
     "Your report shows anemia. Take iron."),
    ("Hi, I think your son is having his second blood test done on the 2nd day of his "
     "cycle. Wait for results.",
     "I think your son is having his second blood test done on the 2nd day of his cycle. "
     "Wait for results."),
    # "There" starts content, it is not "Hi there"
    ("There are many causes of chest pain. Get an ECG.",
     "There are many causes of chest pain. Get an ECG."),
    # unfinished last sentence and a promised list that never came
    ("Hello, There are many causes of fever. Viral infection is the most common. You "
     "should get the following tests:",
     "There are many causes of fever. Viral infection is the most common."),
    ("Hi, Gastritis is likely. Avoid spicy food. Also the reasons could be",
     "Gastritis is likely. Avoid spicy food."),
    # URL sentence goes
    ("Hi, Rest your knee. If you require more of my help in this aspect, please use this "
     "URL. http", "Rest your knee."),
    # self-introductions in front of the pleasantries
    ("Hello, My name is Chat Doctor. I am Chat Doctor answering your query. I have gone "
     "through your query and can understand your concern. By your description, your baby "
     "is underweight.", "By your description, your baby is underweight."),
    ("HiT hank you for asking Chat Doctor . I understand your concern. As per your "
     "complaint it is indigestion.", "As per your complaint it is indigestion."),
    # nothing but pleasantries
    ("Hi thanks for asking question", ""),
]


def main():
    failed = 0
    for raw, want in CASES:
        got = c.clean_answer(raw)
        if got != want:
            failed += 1
            print(f"[FAIL]\n  in:   {raw}\n  want: {want}\n  got:  {got}")
    # whole-row drops
    shell = ("These resources address the diagnosis or management of psoriasis: - Genetic "
             "Testing Registry: Psoriasis - MedlinePlus Encyclopedia: Psoriasis")
    loop = "the pain can lead to liver failure and " * 4
    good = " ".join(["Gastritis is a common cause of this pain."] * 1) + \
        " Take an antacid after meals and avoid spicy food for two weeks."
    for q, a, want in [("What are the treatments for Psoriasis ?", shell,
                        "medquad_link_list_or_missing_table"),
                       ("I have pain in my side, what is it?", loop, "repetition_loop"),
                       ("pain?", good, "question_too_short"),
                       ("I have pain in my side, what is it?", "Rest. Take care.",
                        "answer_too_short_after_cleaning"),
                       ("I have pain in my side, what is it?", good, None)]:
        got = c.drop_reason(q, a, c.clean_answer(a))
        if got != want:
            failed += 1
            print(f"[FAIL] drop_reason: want {want}, got {got}\n  {a[:80]}")
    total = len(CASES) + 5
    print(f"{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
