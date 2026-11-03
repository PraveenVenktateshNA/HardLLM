"""Rule-based intake triage: urgency, symptoms, duration, age, department.

The model never decides urgency. Fixed rules do, so the same description
always gets the same result and every result says which words caused it.
When in doubt the rules over-triage (shortness of breath alone is an
emergency), because sending a well patient to a doctor costs less than the
opposite. Negated symptoms ("no chest pain") do not count.

    import triage
    triage.assess("my 2 year old has had a fever of 104 for three days")

Plain Python, no dependencies.
"""
import re

# "review": no symptom recognised, so a nurse talks to the patient before anything is booked
LEVELS = ("emergency", "urgent", "review", "routine")
RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
EMERGENCY_DEPT = "Emergency department"
DEFAULT_DEPT = "General medicine"

# (label, pattern, level, department). Order matters only for picking the
# department among symptoms of the same level: the first one listed wins.
SYMPTOMS = [
    # ---- emergency: any one of these sends the patient to emergency now
    ("chest pain", r"chest\b[^.!?\n]{0,25}?\b(?:pain\w*|tight\w*|pressure|discomfort|heav\w*|hurts?|hurting|ache|aching|burning)|(?:pain|tightness|pressure|heaviness|discomfort|ache|aching)\b[^.!?\n]{0,30}?\bchest\b|heart attack", "emergency", "Cardiology"),
    ("difficulty breathing", r"(?:can'?t|cannot|can not|could not|couldn'?t|unable to|not able to|hard to|difficult(?:y)?(?: in)?|trouble|struggling to|struggle to) (?:to )?breath\w*|short(?:ness)? of breath|breathless|gasping for (?:air|breath)|choking", "emergency", "Pulmonology"),
    ("fainting or unconscious", r"\bfaint(?:ed|ing)\b|feel(?:s|ing)? faint|passed out|pass(?:es|ing)? out|unconscious|unresponsive|not responding|(?:won'?t|will not|not) wak(?:e|ing) up|collapsed", "emergency", "Neurology"),
    ("stroke signs", r"\bstroke\b|slurred speech|slurring|face (?:is )?drooping|drooping face|droop(?:ing|y) (?:on one side|face|mouth)|(?:numb|numbness|weak|weakness|paralys\w*) (?:on|in|of) (?:one|the left|the right|left|right) side|sudden (?:confusion|weakness|numbness)|can'?t (?:move|lift|feel) (?:my |his |her )?(?:arm|leg|face)", "emergency", "Neurology"),
    ("seizure", r"seizure|convuls\w*", "emergency", "Neurology"),
    ("worst or sudden severe headache", r"worst headache|thunderclap|sudden(?:ly)? (?:a )?severe headache|sudden(?:ly)? (?:a )?(?:very )?bad headache", "emergency", "Neurology"),
    ("vomiting or coughing blood", r"(?:cough|vomit|throw)(?:ing|ed|s)? (?:up )?blood|blood in (?:my |the |his |her )?vomit|vomit(?:ing)? blood", "emergency", "Gastroenterology"),
    ("heavy bleeding", r"severe bleeding|heavy bleeding|bleeding (?:heavily|a lot|badly|that won'?t stop|won'?t stop|will not stop)|won'?t stop bleeding|can'?t stop (?:the )?bleeding", "emergency", "General surgery"),
    ("bleeding in pregnancy", r"pregnan\w*\b.{0,60}\bbleed\w*|bleed\w*\b.{0,60}\bpregnan\w*", "emergency", "Obstetrics and gynaecology"),
    ("throat or tongue swelling", r"throat (?:is )?(?:swelling|swollen|closing)|swollen (?:throat|tongue|lips)|(?:tongue|lips) (?:is |are )?swollen|anaphyla\w*", "emergency", "Emergency department"),
    ("thoughts of suicide or self-harm", r"suicid\w*|kill (?:my|him|her)sel(?:f|ves)|end (?:my|his|her) life|want(?:s)? to die|self[- ]harm|hurt(?:ing)? (?:my|him|her)self", "emergency", "Psychiatry"),
    ("overdose or poisoning", r"overdose|overdosed|poison\w*|swallowed (?:bleach|pills|chemicals?|a battery)", "emergency", "Emergency department"),
    ("blood infection or fluid on the lungs", r"sepsis|septic|(?:infection|bacteria|e\.? ?coli)\b[^.!?\n]{0,30}?\bin (?:his|her|my|the) blood|fluid (?:on|in) (?:\w+ ){0,2}lungs?", "urgent", "General medicine"),
    ("testicle pain", r"(?:testic\w*|testes|scrot\w*|balls?)\b[^.!?\n]{0,20}?\b(?:pain\w*|ache|aching|hurts?|hurting|swollen|swelling)", "urgent", "Urology"),
    ("snake bite", r"snake ?bite|bitten by a snake|snake bit", "emergency", "Emergency department"),
    ("sudden vision loss", r"sudden(?:ly)? (?:lost |losing |loss of )?(?:my )?vision|lost (?:my )?(?:vision|sight)|suddenly can'?t see|went blind", "emergency", "Ophthalmology"),

    # ---- urgent: see a doctor today
    ("high fever", r"high fever|very high temperature", "urgent", "General medicine"),
    ("blood in urine or stool", r"blood in (?:my |the |his |her )?(?:urine|stool|stools|poop|pee)|bloody (?:stool|stools|urine|diarrh\w*)|black (?:tarry )?stools?|passing blood", "urgent", "Gastroenterology"),
    ("severe pain", r"(?:severe|unbearable|excruciating|intense|terrible|extreme) (?:\w+ ){0,2}pain|pain (?:is )?(?:severe|unbearable|excruciating)|(?:gets|becomes|getting|became) (?:very )?(?:severe|unbearable)", "urgent", "General medicine"),
    ("can't keep fluids down", r"can'?t keep (?:anything|food|water|fluids|liquids) down|nothing (?:stays|staying|will stay) down|vomit\w* (?:all day|nonstop|non-stop|repeatedly|frequently|everything|every time)", "urgent", "Gastroenterology"),
    ("dehydration", r"dehydrat\w*|(?:hasn'?t|haven'?t|not) (?:peed|urinated|passed urine)|no urine", "urgent", "General medicine"),
    ("stiff neck", r"stiff neck|neck (?:is )?stiff", "urgent", "Neurology"),
    ("palpitations", r"palpitation\w*|heart (?:is )?(?:racing|pounding)|racing heart|irregular heartbeat", "urgent", "Cardiology"),
    ("burn injury", r"(?:got|was|been|is|am) (?:badly )?burn(?:ed|t)|burn (?:injury|wound)|scald\w*", "urgent", "General surgery"),
    ("animal bite", r"(?:dog|cat|monkey|animal) (?:bite|bit)|bitten by a (?:dog|cat|monkey)", "urgent", "General medicine"),
    # being pregnant, not wanting to be ("I want to get pregnant" is not a pregnancy)
    ("pregnancy", r"(?:am|is|are|was|i'm|im|she's|weeks?|months?|currently|now) (?:\w+ ){0,2}pregnant|(?:during|in) (?:my |her )?pregnancy", "urgent", "Obstetrics and gynaecology"),
    ("head injury", r"head injury|hit (?:my|his|her) head|fell on (?:my|his|her) head|banged (?:my|his|her) head", "urgent", "Neurology"),

    # ---- routine: book an appointment or home care
    ("migraine", r"migraine", "routine", "Neurology"),
    ("headache", r"headache|head (?:is )?(?:aching|hurting|pain)", "routine", "General medicine"),
    ("dizziness", r"dizz\w*|vertigo|light[- ]?headed", "routine", "General medicine"),
    ("fever", r"fever|feverish|high temperature|temperature of", "routine", "General medicine"),
    ("cough", r"cough\w*", "routine", "General medicine"),
    ("cold or runny nose", r"runny nose|blocked nose|stuffy nose|sneez\w*|nasal congestion|(?:a|the|common|bad|have) cold\b(?! (?:hands|feet|sweats?|water|drinks?))|\bflu\b", "routine", "General medicine"),
    ("sore throat", r"sore throat|throat (?:is )?(?:pain|sore|hurting|hurts)|painful swallowing|hurts? to swallow|pain (?:when|while) swallowing|difficulty swallowing|caught in (?:my |the )?throat", "routine", "ENT"),
    ("insect sting", r"stung|(?:bee|wasp|hornet|insect) stings?|sting (?:by|from) a", "routine", "General medicine"),
    ("ear pain", r"ear ?ache|ear (?:pain|infection)|pain in (?:my |the )?ear", "routine", "ENT"),
    ("toothache", r"tooth ?ache|tooth pain|gum pain", "routine", "Dental"),
    ("eye problem", r"(?:red|itchy|watery|swollen|painful) eyes?|eye (?:pain|infection|redness)|blurr(?:y|ed) vision|conjunctivitis", "routine", "Ophthalmology"),
    ("rash or itching", r"rash\w*|itch\w*|hives|eczema|psoriasis", "routine", "Dermatology"),
    ("acne", r"acne|pimples?", "routine", "Dermatology"),
    ("hair loss", r"hair (?:loss|fall|falling)|losing (?:my )?hair", "routine", "Dermatology"),
    ("stomach pain", r"(?:stomach|abdomen|abdominal|tummy|belly)\b[^.!?\n]{0,20}?\b(?:pain\w*|ache|aching|hurts?|hurting|cramps?)|stomach ?ache|(?:pain|ache|cramps?)\b[^.!?\n]{0,30}?\b(?:stomach|abdomen|belly|tummy)\b", "routine", "Gastroenterology"),
    ("vomiting or nausea", r"vomit\w*|nause\w*|throwing up|threw up", "routine", "Gastroenterology"),
    ("diarrhoea", r"diarrh\w*|loose (?:motions?|stools?)|watery stools?", "routine", "Gastroenterology"),
    ("constipation", r"constipat\w*", "routine", "Gastroenterology"),
    ("acidity or heartburn", r"heartburn|acidity|acid reflux|gerd\b|indigestion|bloat\w*|gas pain", "routine", "Gastroenterology"),
    ("back pain", r"back ?(?:pain|ache)|pain in (?:my |the )?(?:lower |upper )?back|sciatica", "routine", "Orthopaedics"),
    ("joint pain", r"(?:joint|knee|hip|shoulder|ankle|wrist|elbow)s?\b[^.!?\n]{0,15}?\b(?:pain\w*|ache|aching|hurts?|hurting|swelling|swollen)|pain in (?:my |the )?(?:joints?|knees?|hips?|shoulders?|ankles?|wrists?)|arthritis|gout", "routine", "Orthopaedics"),
    ("muscle pain or sprain", r"muscle (?:pain|ache|cramps?|spasms?)|sprain\w*|body ?aches?|neck pain", "routine", "Orthopaedics"),
    ("urinary symptoms", r"(?:burning|stinging) (?:sensation )?(?:when|while|during) (?:i )?(?:urinat\w*|pee\w*)|urin\w*\b[^.!?\n]{0,40}?\b(?:sting\w*|burn\w*)|painful urination|frequent urination|uti\b|urinary tract infection", "routine", "Urology"),
    ("lump or swelling", r"lumps?\b|swelling|swollen|cysts?\b|growth on|ball like", "routine", "General surgery"),
    ("wound or infection", r"wound|incision|pus\b|abscess|infected|infection", "routine", "General surgery"),
    ("tingling or numbness", r"pins and needles|tingl\w*|numb\w*", "routine", "Neurology"),
    ("period problems", r"\b(?:my|her|missed|late|irregular|heavy|painful|no) periods?\b|period (?:pain|cramps?)|menstrua\w*|vaginal (?:discharge|itching)|yeast infection", "routine", "Obstetrics and gynaecology"),
    ("low mood or anxiety", r"anxi\w*|depress\w*|panic attacks?|stress\w*|feel(?:ing)? (?:low|sad|hopeless)", "routine", "Psychiatry"),
    ("sleep problems", r"insomnia|can'?t sleep|trouble sleeping|not sleeping", "routine", "Psychiatry"),
    ("blood pressure", r"blood pressure|hypertension|\bbp\b", "routine", "Cardiology"),
    ("blood sugar", r"diabet\w*|blood sugar|sugar levels?", "routine", "Endocrinology"),
    ("thyroid", r"thyroid", "routine", "Endocrinology"),
    ("tiredness", r"tired\w*|fatigue\w*|weak(?:ness)?\b|exhaust\w*", "routine", "General medicine"),
]
SYMPTOMS = [(label, re.compile(r"\b(?:" + pat + ")", re.I), lvl, dept) for label, pat, lvl, dept in SYMPTOMS]

# words that cancel a symptom when they come just before it in the same clause
NEGATION = re.compile(
    r"\b(?:no|never|without|denies|denied|deny|free of|negative for)\b"
    r"|\bnot\b(?! (?:sure|only|just|certain))"
    r"|\b(?:don'?t|doesn'?t|didn'?t|do not|does not|did not) (?:have|get|feel)\b"
    r"|\b(?:haven'?t|hasn'?t|have not|has not) (?:had|got)\b"
    r"|\b(?:don'?t|doesn'?t|didn'?t|do not|does not|did not|isn'?t|is not|wasn'?t|was not)$", re.I)
CLAUSE_END = re.compile(r"[.;:!?\n,()\u2013\u2014]|\b(?:and|but|now|also|then|though|although|however|plus)\b", re.I)
NEGATION_WORDS = 4  # how far back a negation reaches, in words

WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "few": 3, "couple of": 2, "several": 4}
UNIT_DAYS = {"hour": 1 / 24, "hr": 1 / 24, "day": 1, "week": 7, "wk": 7, "month": 30, "year": 365, "yr": 365}
DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|couple of|several)"
    r"\s*(hour|hr|day|week|wk|month|year|yr)s?\b(?![- ]?old)", re.I)
SINCE = re.compile(r"\bsince (yesterday|last night|this morning|morning|today|last week|last month|(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b", re.I)
SINCE_DAYS = {"yesterday": 1, "last night": 1, "this morning": 0.3, "morning": 0.3, "today": 0.3,
              "last week": 7, "last month": 30}
AGE_YEARS = re.compile(r"(?<![\d.])(\d{1,3})(?:\.\d+)?[- ]?(?:years?|yrs?|y)[- ]?old\b|\baged? (\d{1,3})\b|\bi(?:'m| am) (\d{1,3})\b(?! ?(?:kg|kgs|cm|lbs?|%|days?|weeks?|months?|hours?|times?))", re.I)
AGE_MONTHS = re.compile(r"\b(\d{1,2})[- ]?(?:months?|mo)[- ]?old\b|\b(\d{1,2})[- ]?(?:weeks?|wks?)[- ]?old\b|\bnew ?born\b", re.I)
SEX = re.compile(r"\b(male|female|man|woman|boy|girl|son|daughter|husband|wife|father|mother|dad|mom|mum)\b", re.I)
SEX_OF = {"male": "male", "man": "male", "boy": "male", "son": "male", "husband": "male", "father": "male", "dad": "male",
          "female": "female", "woman": "female", "girl": "female", "daughter": "female", "wife": "female",
          "mother": "female", "mom": "female", "mum": "female"}
TEMP = re.compile(r"(?:fever|temperature|temp)\D{0,25}?(\d{2,3}(?:\.\d+)?)\s*(?:°|degrees?|deg)?\s*([fc])?\b"
                  r"|(\d{2,3}(?:\.\d+)?)\s*(?:°|degrees?|deg)?\s*([fc])?\s*(?:of )?(?:fever|temperature)", re.I)

ADVICE = {
    "emergency": "Go to the emergency department now, or call an ambulance (112 / 108 in India, 911 in the US).",
    "urgent": "See a doctor today. Go to emergency if it gets worse.",
    "review": "No symptoms recognised. A nurse should talk to the patient before anything is booked.",
    "routine": "Book a regular appointment. Go sooner if new warning signs appear.",
}


def normalise(text):
    return " ".join(text.replace("’", "'").replace("‘", "'").split())


def negated(text, start):
    """True if a negation word sits just before position start, in the same clause."""
    before = text[:start]
    cut = 0
    for m in CLAUSE_END.finditer(before):
        cut = m.end()
    words = before[cut:].split()
    if NEGATION.search(" ".join(words[-NEGATION_WORDS:])):
        return True
    # "no chest pain or shortness of breath": the negation covers the whole "or" list
    clause = " ".join(words)
    neg = NEGATION.search(clause)
    return bool(neg and len(words) <= 12 and re.search(r"\b(?:or|nor)\b", clause[neg.end():]))


def find_symptoms(text):
    """[(label, level, dept, matched words)] for every symptom mentioned and not negated."""
    found = []
    for label, pat, lvl, dept in SYMPTOMS:
        for m in pat.finditer(text):
            if not negated(text, m.start()):
                found.append((label, lvl, dept, m.group(0), m.start()))
                break
    # a symptom that is part of a bigger one already found (e.g. "fever" inside "high fever") is dropped
    spans = [(f[4], f[4] + len(f[3])) for f in found]
    keep = []
    for i, f in enumerate(found):
        s, e = spans[i]
        inside = any(j != i and spans[j][0] <= s and e <= spans[j][1] and (spans[j][1] - spans[j][0]) > (e - s)
                     for j in range(len(found)))
        if not inside:
            keep.append(f)
    return keep


def durations(text):
    """[(phrase as written, length in days, position)] for every duration mentioned."""
    out = []
    for m in DURATION.finditer(text):
        n = m.group(1).lower()
        out.append((m.group(0), (float(n) if n[0].isdigit() else WORD_NUM[n]) * UNIT_DAYS[m.group(2).lower()], m.start()))
    for m in SINCE.finditer(text):
        out.append((m.group(0), SINCE_DAYS.get(m.group(1).lower(), 3), m.start()))  # a weekday: a few days
    return sorted(out, key=lambda d: d[2])


def duration_near(text, pos):
    """The duration closest to position pos in the same sentence, (phrase, days) or (None, None)."""
    lo = max([m.end() for m in re.finditer(r"[.!?\n]", text[:pos])], default=0)
    hi = next((pos + m.start() for m in re.finditer(r"[.!?\n]", text[pos:])), len(text))
    near = [d for d in durations(text) if lo <= d[2] < hi]
    if not near:
        return None, None
    d = min(near, key=lambda d: abs(d[2] - pos))
    return d[0], d[1]


def age(text):
    """Age in years (fractions for babies), or None."""
    m = AGE_MONTHS.search(text)
    if m:
        if m.group(1):
            return int(m.group(1)) / 12
        if m.group(2):
            return int(m.group(2)) / 52
        return 0.0  # newborn
    m = AGE_YEARS.search(text)
    if m:
        years = int(next(g for g in m.groups() if g))
        if 0 < years <= 120:
            return float(years)
    if re.search(r"\b(?:baby|infant)\b", text, re.I):
        return 0.5
    return None


def temperature(text):
    """Highest plausible body temperature mentioned, in °C, or None."""
    best = None
    for m in TEMP.finditer(text):
        value = float(m.group(1) or m.group(3))
        unit = (m.group(2) or m.group(4) or "").lower()
        if unit == "f" or (not unit and value >= 90):
            c = (value - 32) * 5 / 9
        else:
            c = value
        if 35 <= c <= 44:
            best = c if best is None else max(best, c)
    return best


def assess(text):
    """The triage result for one description. Deterministic."""
    t = normalise(text)
    found = find_symptoms(t)
    labels = {f[0] for f in found}
    reasons = []   # (level, why)
    for label, lvl, dept, words, _ in found:
        if lvl != "routine":
            reasons.append((lvl, f"{label} (“{words}”)"))

    all_d = durations(t)
    years = age(t)
    temp_c = temperature(t)
    has_fever = bool(labels & {"fever", "high fever"}) or temp_c is not None and temp_c >= 37.8
    fever_at = next((f[4] for f in found if f[0] in ("high fever", "fever")), None)
    fever_phrase, fever_days = duration_near(t, fever_at) if fever_at is not None else (None, None)

    # combinations and numbers
    if temp_c is not None:
        shown = f"{temp_c:.1f} °C / {temp_c * 9 / 5 + 32:.1f} °F"
        if temp_c >= 40.6:
            reasons.append(("emergency", f"very high temperature ({shown})"))
        elif temp_c >= 39.4:
            reasons.append(("urgent", f"high temperature ({shown})"))
    if "insect sting" in labels and (labels & {"dizziness", "fainting or unconscious", "difficulty breathing", "throat or tongue swelling",
                                               "lump or swelling", "rash or itching", "vomiting or nausea"}
                                     or re.search(r"\b(?:mouth|tongue|throat|lips?)\b", t, re.I)):
        reasons.append(("emergency", "insect sting with signs of an allergic reaction, or in the mouth or throat"))
    if has_fever and "stiff neck" in labels:
        reasons.append(("emergency", "fever with a stiff neck (warning sign of meningitis)"))
    if has_fever and years is not None and years < 0.25:
        reasons.append(("emergency", "fever in a baby under 3 months"))
    elif has_fever and years is not None and years < 3 and temp_c is not None and temp_c >= 39:
        reasons.append(("urgent", "high temperature in a child under 3"))
    if fever_days is not None and fever_days >= 3:
        reasons.append(("urgent", f"fever for {fever_phrase}"))
    gut = next((f[4] for f in found if f[0] in ("vomiting or nausea", "diarrhoea")), None)
    if years is not None and years >= 65 and gut is not None:
        gut_phrase, gut_days = duration_near(t, gut)
        if gut_days is not None and gut_days >= 2:
            reasons.append(("urgent", f"vomiting or diarrhoea for {gut_phrase} at age {int(years)}"))

    recognised = bool(found) or temp_c is not None
    level = min((r[0] for r in reasons), key=RANK.get, default="routine" if recognised else "review")
    top = [f for f in found if f[1] == level]
    if level == "emergency":
        dept = EMERGENCY_DEPT
        specialty = next((f[2] for f in top if f[2] != EMERGENCY_DEPT), None)
    else:
        dept = top[0][2] if top else (found[0][2] if found else DEFAULT_DEPT)
        specialty = None
        if years is not None and years < 16 and dept == DEFAULT_DEPT:
            dept = "Paediatrics"

    sex_m = SEX.search(t)
    return {
        "level": level,
        "advice": ADVICE[level],
        "reasons": [why for lvl, why in sorted(reasons, key=lambda r: RANK[r[0]]) if lvl == level]
                   or (["no warning signs found in the description"] if found else []),
        "other_reasons": [why for lvl, why in sorted(reasons, key=lambda r: RANK[r[0]]) if lvl != level],
        "symptoms": [f[0] for f in found],
        "duration": all_d[0][0] if all_d else None,
        "age": None if years is None else (f"{round(years * 12)} months" if years < 2 else f"{int(years)} years"),
        "sex": SEX_OF[sex_m.group(1).lower()] if sex_m else None,
        "temperature_c": None if temp_c is None else round(temp_c, 1),
        "department": dept,
        "specialty": specialty,
        "recognised": recognised,
    }


def red_flags(text):
    """Emergency reasons for a question, [] if none. The chat tab's alert uses this."""
    r = assess(text)
    return r["reasons"] if r["level"] == "emergency" else []
