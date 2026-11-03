"""Keep the model to information: drop the sentences that tell the reader what to do.

The model learned ChatDoctor's doctor voice, which mixes explanation ("this can
be due to gastritis") with orders ("take omeprazole twice daily", "consult a
cardiologist", "avoid stress"). The demo shows only the explanation. Whole
sentences are kept or dropped, never edited, so what is shown is the model's
own words. Streaming works sentence by sentence (see Sentences).

    import answer_filter
    answer_filter.is_instruction("Take omeprazole 20 mg twice daily.")   # True
"""
import re

# ChatDoctor's opener reads as if the model remembered an earlier chat; each question is sent on its own
HISTORY_OPENER = re.compile(r"\b(?:by|as per|from) your (?:history and description|description and history|history)\b", re.I)
# a sentence ends at . ! ? followed by a space and a capital, digit or quote, or at a line break
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
# words a sentence may start with before its verb ("So get done an ECG")
LEAD = re.compile(r"^(?:(?:so|and|but|also|then|hence|therefore|thus|now|first(?:ly)?|second(?:ly)?|finally|meanwhile|otherwise|hi|hello|dear|ok(?:ay)?|well|yes|no)\b[\s,:]*)+", re.I)
# verbs that make a sentence an order when it starts with them
IMPERATIVE = re.compile(
    r"^(?:please |kindly |just |always |never |do not |don'?t )?(?:take|avoid|consult|see|visit|go|get|do|try|use|apply|drink|eat|"
    r"start|stop|keep|continue|rest|wait|call|check|follow|maintain|ask|consider|contact|seek|have|make|quit|reduce|increase|"
    r"limit|monitor|practi[cs]e|sleep|wear|wash|give|put|let|be|undergo|repeat|book|schedule|cut|add|include|prefer|change|"
    r"switch|discuss|inform|tell|show|report|remember|note|hold|lie|sit|stand|elevate|massage|gargle|inhale|chew|swallow|"
    r"exercise|walk|lose|gain|watch|observe|write|meet|refer|share|send|upload|mention|need not|no need)\b", re.I)
# phrases that make a sentence advice wherever they appear
ADVICE = re.compile(
    r"\b(?:you|he|she|they|one|your (?:child|son|daughter|wife|husband|father|mother|baby|kid))\s+"
    r"(?:should|must|need to|needs to|have to|has to|ought to|had better|may need to|might need to|can take|could take|may take|"
    r"can use|could use|may use|can try|could try|may try|can go|may go|can consult|may consult|can get|may get|can apply|"
    r"may apply|can start|may start|need not|needn'?t|will need to|would need to)\b"
    r"|\byou (?:may|might|will|would) need\b|\bif you (?:take|use|follow|avoid|apply|stop|continue|start|keep|do)\b"
    r"|\b(?:i|we)\s+(?:would |will |do |strongly )?(?:suggest|advise|advice|recommend|prescribe|want you|would like you|feel you should|think you should)\w*\b"
    r"|\bmy (?:advice|suggestion|recommendation)\b|\b(?:it is|it's|it would be|would be|is)\s+(?:better|best|advisable|wise|recommended|"
    r"necessary|important|essential|mandatory|preferable|good|safer)\s+(?:to|that|if|for you)\b|\bbetter (?:to|if|you)\b"
    r"|\b(?:advisable|recommended|necessary) (?:to|that)\b|\bgo for\b|\bget (?:done|it done|yourself|an?|the) \b|\bget \w+ (?:done|checked|tested|examined)\b"
    r"|\bshould be (?:taken|done|avoided|consulted|seen|checked|treated|started|stopped|given|applied|used|examined|evaluated|investigated|monitored)\b"
    r"|\b(?:please|kindly)\b|\bmake sure\b|\btry to\b|\bdon'?t worry\b|\bdo not worry\b|\bno need to\b|\bneed not\b"
    r"|\b(?:hope|wish) (?:this|i|you|that)\b|\blet me know\b|\bfeel free\b|\btake care\b|\bthanks?\b|\bthank you\b|\bregards\b"
    r"|\bget well\b|\bfollow up\b|\bfollow-up\b|\bwrite (?:to|back)\b|\bcontact (?:me|us|your)\b|\bconsult (?:a|an|your|the|with)\b"
    r"|\bsee (?:a|an|your) (?:doctor|physician|specialist|dentist|gynaecologist|gynecologist|surgeon|dermatologist|neurologist|cardiologist|urologist|ent)\b"
    r"|\bvisit (?:a|an|your|the) \b|\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|iu)\b|\b(?:once|twice|thrice|three times) (?:a |per )?(?:day|daily|week)\b"
    r"|\b(?:before|after) (?:meals|food|breakfast|dinner)\b|\bfor (?:\d+|one|two|three|five|seven|ten) (?:days|weeks) (?:course|only)\b", re.I)
# the model writing as a patient instead of answering ("I am taking an antibiotic called ...")
PATIENT_VOICE = re.compile(r"\b(?:my doctor (?:prescribed|said|told|gave|suggested)|i am taking|i'm taking|i was prescribed|i have been taking|"
                           r"i have a question|i was diagnosed|i have been diagnosed|my (?:husband|wife|son|daughter|mother|father) (?:has|had|is|was))\b", re.I)
# a sentence leaning on the one before ("It will give you exact diagnosis." after a dropped "Go for a biopsy.")
LEANS_BACK = re.compile(r"^(?:it|this|that|these|those|he|she|they|which|such|both|the same)\b", re.I)
NOTHING_LEFT = "No explanation to show for this question: the model only gave advice, which this demo hides. Try describing the symptoms in more detail."


def reword(sentence):
    return HISTORY_OPENER.sub(lambda m: ("F" if m.group(0)[0].isupper() else "f") + "rom your description", sentence)


def is_instruction(sentence):
    """True if the sentence tells someone what to do (an order, advice, a dose, or a sign-off),
    or is not an answer at all (the model writing as a patient)."""
    s = LEAD.sub("", sentence.strip().strip("-*• ").strip())
    return bool(IMPERATIVE.match(s) or ADVICE.search(s) or PATIENT_VOICE.search(s))


def drops(sentence, previous_dropped):
    """True if the sentence is not shown: an instruction, or a pronoun pointing at a dropped one."""
    return is_instruction(sentence) or (previous_dropped and bool(LEANS_BACK.match(LEAD.sub("", sentence.strip()))))


def split(text):
    return [s for s in SENTENCE_END.split(text) if s.strip()]


def keep_information(text):
    """(kept text, number of sentences dropped) for a finished answer."""
    kept, dropped, last = [], 0, False
    for s in split(text):
        last = drops(s, last)
        if last:
            dropped += 1
        else:
            kept.append(reword(s.strip()))
    return " ".join(kept), dropped


class Sentences:
    """Feed streamed pieces in; get back finished, kept sentences as they complete."""

    def __init__(self, keep_all=False):
        self.keep_all = keep_all  # True: show every sentence (the chat), only reworded; False: information only
        self.buf = ""
        self.kept = 0
        self.dropped = 0
        self.last_dropped = False

    def feed(self, piece):
        self.buf += piece
        out = []
        while True:
            m = SENTENCE_END.search(self.buf)
            if not m:
                break
            sentence, self.buf = self.buf[:m.start()], self.buf[m.end():]
            out += self._judge(sentence)
        return "".join(out)

    def finish(self):
        tail = self.buf.strip()
        # the answer hit the length limit mid-sentence: drop the fragment unless it is all there is
        if tail and not re.search(r"[.!?\"')\]]$", tail) and self.kept:
            tail = ""
        out = self._judge(tail)
        self.buf = ""
        if not self.kept and not self.keep_all:
            out.append(NOTHING_LEFT)
        return "".join(out)

    def _judge(self, sentence):
        if not sentence.strip():
            return []
        self.last_dropped = not self.keep_all and drops(sentence, self.last_dropped)
        if self.last_dropped:
            self.dropped += 1
            return []
        self.kept += 1
        return [("" if self.kept == 1 else " ") + reword(sentence.strip())]
