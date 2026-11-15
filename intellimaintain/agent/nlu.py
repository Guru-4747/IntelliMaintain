"""Understand questions written the way people really type.

Handles chat-style text ("yo my bering makin weird grindin noise wat shud i do??"):
  1. clean-up       lower-case, emojis/symbols removed, stretched letters ("sooo" -> "soo")
  2. short forms    u -> you, wat -> what, shud -> should, pls -> please, abt -> about, ...
  3. spelling       words close to the knowledge-base vocabulary are corrected ("bering" -> "bearing")
  4. domain terms   everyday words are mapped to maintenance terms ("grinding noise" -> vibration, "getting hot"
                    -> temperature, "oil/grease" -> lubrication, "outer ring" -> outer race, ...)
  5. intent         greeting / thanks / help / diagnosis / knowledge / out_of_scope
The cleaned search query feeds the retriever and the answer generator; the original text is kept for the LLM.
Everything is rule-based and offline (no internet, deterministic, testable).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

SHORT_FORMS = {
    "u": "you", "ur": "your", "r": "are", "y": "why", "wat": "what", "wht": "what", "wats": "what is",
    "whats": "what is", "wt": "what", "hw": "how", "abt": "about", "pls": "please", "plz": "please", "plss": "please",
    "shud": "should", "shld": "should", "cud": "could", "wud": "would", "cn": "can", "dnt": "don't", "dont": "don't",
    "idk": "i do not know", "bcoz": "because", "coz": "because", "cuz": "because", "bcz": "because", "b4": "before",
    "msg": "message", "thx": "thanks", "thnx": "thanks", "tnx": "thanks", "ty": "thanks", "k": "ok", "kk": "ok",
    "okk": "ok", "smth": "something", "sth": "something", "w/o": "without", "w/": "with", "info": "information",
    "temp": "temperature", "vib": "vibration", "vibs": "vibration", "brg": "bearing", "brgs": "bearings",
    "mtr": "motor", "maint": "maintenance", "pm": "preventive maintenance", "asap": "urgently", "rn": "right now",
    "gonna": "going to", "wanna": "want to", "gotta": "have to", "ya": "yes", "yep": "yes", "nope": "no",
    "makin": "making", "doin": "doing", "runnin": "running", "grindin": "grinding", "comin": "coming",
    "hv": "have", "hav": "have", "nd": "and", "n": "and", "im": "i am", "isnt": "is not", "doesnt": "does not",
    "cant": "can not", "wont": "will not", "its": "it is", "thats": "that is", "whts": "what is",
    "ths": "this", "dis": "this", "da": "the", "wid": "with", "gud": "good", "pblm": "problem", "prob": "problem",
    "prblm": "problem", "becos": "because", "y'all": "you", "abt.": "about", "q": "question", "ans": "answer",
}

# everyday wording -> maintenance vocabulary (added to the search query, the original words are kept)
DOMAIN_SYNONYMS = [
    (r"\b(noise|noisy|sound|sounds|rattl\w*|grind\w*|humm?\w*|whin\w*|click\w*|knock\w*|shak\w*|wobbl\w*|"
     r"rough|rumbl\w*|buzz\w*|squeal\w*)\b", "vibration impacting bearing defect"),
    (r"\b(hot|heat\w*|warm|overheat\w*|burning|smok\w*)\b", "temperature overheating lubrication"),
    (r"\b(oil|oiling|grease|greasing|greased|lube|lubricat\w*)\b", "lubrication relubrication grease interval"),
    (r"\bouter (ring|track|race)\b", "outer race BPFO"),
    (r"\binner (ring|track|race)\b", "inner race BPFI"),
    (r"\b(balls?|rollers?|rolling elements?)\b", "ball rolling element BSF"),
    (r"\b(change|swap|replac\w*|new one|fit a new)\b", "bearing replacement mounting procedure"),
    (r"\b(remove|take out|pull out|dismount\w*)\b", "dismounting puller removal"),
    (r"\b(safe|safety|danger\w*|lock ?out|isolat\w*|before (i|we) (open|work))\b",
     "safety lockout tagout isolate energy"),
    (r"\b(how bad|serious|urgent\w*|critical|keep running|run (it )?(till|until|for)|how long|stop the machine|"
     r"can i (still )?(run|use))\b", "severity urgency stage of degradation replacement alarm level"),
    (r"\b(broken|damag\w*|crack\w*|pitt\w*|spall\w*|worn|wear)\b", "damage spall fatigue failure mode"),
    (r"\b(peak|spike|impulse|bang)\w*\b", "impulsive kurtosis crest factor"),
    (r"\b(misalign\w*|alignment|coupling)\b", "misalignment 1x 2x axial"),
    (r"\b(unbalanc\w*|imbalanc\w*)\b", "unbalance 1x running speed"),
    (r"\b(loose|looseness|bolt)\w*\b", "mechanical looseness harmonics"),
    (r"\b(electric\w*|current|vfd|drive|fluting)\b", "electrical erosion fluting shaft current"),
    (r"\b(dirt|dust|sand|water|contamin\w*)\b", "contamination seals"),
    (r"\b(fine|ok|okay|healthy|normal|good)\b", "healthy normal condition baseline"),
]

GREETING_RE = re.compile(r"^(hi+|hey+|hello+|hlo|helo|hai|yo|hola|namaste|vanakkam|good (morning|afternoon|evening)|"
                         r"sup|wassup|what is up|whats up)\b( (there|all|team|everyone|guys|bro|sir|madam|buddy|"
                         r"friend|copilot|intellimaintain))?[\s!.?]*$")
THANKS_RE = re.compile(r"^(ok |okay |cool |great |nice )?(thanks?|thank you|thx|ty|tq|great|awesome|cool|nice|"
                       r"super|got it|ok|okay)\b[\s!.?]*( (so much|a lot|bro|man|sir))?[\s!.?]*$")
HELP_RE = re.compile(r"\b(what can (you|u) do|who are (you|u)|how (do|to) (i )?use (this|you|the app)|help me|"
                     r"what is this (app|tool)|features|capabilities)\b")
DIAGNOSIS_RE = re.compile(r"\b(this|my|our|the) (bearing|motor|pump|machine|signal|data|file|reading|fan|"
                          r"compressor|gearbox)\b|\b(what'?s wrong|whats wrong|is it (ok|fine|safe|bad|damaged)|"
                          r"what is wrong|wrong with|condition|diagnos\w*|status|health|problem|issue|failing|"
                          r"what happened|check (it|this)|is it|should i|can i|too high|too much|result|reading)\b")
GENERAL_RE = re.compile(r"^(what (is|are|does|do)|how (do|does|to|can)|why|explain|define|definition|difference|"
                        r"tell me about|which|when should)\b")
DOMAIN_RE = re.compile(r"\b(bearing|vibration|motor|pump|fan|compressor|gearbox|shaft|race|ball|lubrica|grease|oil|"
                       r"kurtosis|crest|rms|fft|spectrum|envelope|bpf|bsf|ftf|frequency|rpm|load|maintenance|"
                       r"inspection|replace|failure|fault|defect|spall|noise|temperature|alignment|cwru|iso|"
                       r"lockout|safety|machine|sensor|accelerometer|wear|crack|damage|seal|monitor)", re.I)

# What the question is really about. Checked in order; the first match wins. "diagnosis" = general condition.
FOCUS_PATTERNS = [
    ("diagnosis", r"\b(condition|what'?s wrong|what is wrong|wrong with|status|health\w*|diagnos\w*|"
                  r"is (it|my|this|the)( \w+)? (ok|okay|fine|good|bad|normal|damaged|faulty))\b"),
    ("severity", r"\b(keep (it )?running|run (it )?(till|until|for|to)|how long|how much time|remaining|last|"
                 r"urgent\w*|serious|critical|how bad|dangerous|stop (the|it|this)|shut ?down|can (i|we) (still )?"
                 r"(run|use|operate)|safe to (run|use|operate)|till|until|friday|tomorrow|next week|next month|"
                 r"when (should|must|do) (i|we) (replace|change|stop))\b"),
    ("safety", r"\b(safety|safe(ly)?|lock ?out|tag ?out|isolat\w*|ppe|danger|hazard|before (i|we) (open|work|"
               r"replace|remove|change))\b"),
    ("lubrication", r"\b(grease|greasing|oil|oiling|lubric\w*|lube|relubric\w*|viscosity)\b"),
    ("replacement", r"\b(replac\w*|change (the|it|this)|swap|mount\w*|dismount\w*|install\w*|remove|removal|"
                    r"puller|heater|fit a new)\b"),
    ("indicators", r"\b(kurtosis|crest factor|crest|rms|peak value|skewness|indicator\w*|statistic\w*|"
                   r"anomaly score|anomaly)\b"),
    ("frequencies", r"\b(bpfo|bpfi|bsf|ftf|defect frequenc\w*|fault frequenc\w*|characteristic frequenc\w*|"
                    r"envelope|spectrum|harmonic\w*|sideband\w*|rpm|shaft speed|physics)\b"),
    ("cwru", r"\b(cwru|case western|dataset|data set|similar (case|recording)\w*|compar\w* (with|to)|"
             r"training data)\b"),
    ("model", r"\b(why (did|does|is) (the |this |your )?(model|classifier|ai|system|it) (predict|say|think|"
              r"classif)\w*|confidence|probabilit\w*|how (sure|confident|accurate)|accuracy|trust|"
              r"random ?forest|cnn|classifier|model)\b"),
    ("location", r"\b(which (part|component|ring|race|element)|where (is|are) (the )?(damage|fault|defect|"
                 r"problem)|what (part|component) (is|are) (damaged|faulty|broken)|inner or outer|damaged part)\b"),
    ("mechanism", r"\b(occur\w*|happen\w*|develop\w*|caus\w*|mechanism|reason|origin|"
                  r"why (do|does|did) \w+( \w+)? (fail|break|wear))\b"),
    ("actions", r"\b(what (should|shall|do|can|must) (i|we) do|next steps?|recommend\w*|actions?|fix|repair|"
                r"what now|maintenance (action|plan|task))\b"),
]
FOCUS_RE = [(name, re.compile(pat, re.I)) for name, pat in FOCUS_PATTERNS]


def question_focus(text: str) -> str:
    """The specific thing the user asks about; 'diagnosis' when it is a general condition question."""
    t = (text or "").lower()
    for name, rx in FOCUS_RE:
        if rx.search(t):
            return name
    return "diagnosis"


STOP = set("""a an the and or of to in on for with by is are was were be been at as it its this that these those from
into than then so such can may should must not no do does did how what which who when where why i you we my our your
me us please hi hey ok yes""".split())


@dataclass
class Query:
    original: str
    cleaned: str
    search_query: str
    intent: str                       # greeting | thanks | help | diagnosis | knowledge | out_of_scope
    corrections: dict[str, str] = field(default_factory=dict)
    added_terms: list[str] = field(default_factory=list)
    focus: str = "diagnosis"          # what the question asks about (see FOCUS_PATTERNS)

    @property
    def needs_tools(self) -> bool:
        return self.intent in ("diagnosis", "knowledge", "out_of_scope")


def _strip_accents_and_emoji(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch if (ch.isalnum() or ch in " .,?!'/-%x") else " " for ch in text)


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    """Words of the knowledge base (used to correct spelling). Falls back to a built-in domain list."""
    words = set()
    try:
        from .. import config

        for p in config.CORPUS_DIR.rglob("*.md"):
            words |= set(re.findall(r"[a-z]{4,}", p.read_text(encoding="utf-8", errors="ignore").lower()))
        chunks = config.INDEX_DIR / "chunks.jsonl"
        if chunks.exists():
            words |= set(re.findall(r"[a-z]{4,}", chunks.read_text(encoding="utf-8", errors="ignore").lower()))
    except Exception:  # noqa: BLE001
        pass
    words |= {"bearing", "bearings", "vibration", "kurtosis", "lubrication", "grease", "temperature", "replacement",
              "inspection", "maintenance", "misalignment", "frequency", "spectrum", "envelope", "outer", "inner",
              "race", "ball", "motor", "pump", "noise", "grinding", "damage", "failure", "fault", "safety"}
    return frozenset(words)


def _correct(word: str, vocab: frozenset[str]) -> str:
    if len(word) < 4 or word in vocab or not word.isalpha():
        return word
    match = difflib.get_close_matches(word, vocab, n=1, cutoff=0.82)
    return match[0] if match else word


def understand(text: str, has_signal: bool = False) -> Query:
    original = (text or "").strip()
    t = _strip_accents_and_emoji(original.lower())
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)            # "sooooo" -> "soo", "hiiii" -> "hii"
    t = re.sub(r"\s+", " ", t).strip()
    tokens = re.findall(r"[a-z0-9/'.%-]+|[?!,]", t)
    expanded = []
    for tok in tokens:
        core = tok.strip(".'")
        expanded.append(SHORT_FORMS.get(core, tok))
    t = " ".join(expanded)
    vocab = vocabulary()
    corrections = {}
    words = []
    for w in t.split():
        c = _correct(w, vocab)
        if c != w:
            corrections[w] = c
        words.append(c)
    cleaned = re.sub(r"\s+([?!,.])", r"\1", " ".join(words)).strip()
    plain = cleaned.strip(" ?!.,")

    added = []
    for pattern, terms in DOMAIN_SYNONYMS:
        if re.search(pattern, cleaned):
            added.extend(x for x in terms.split() if x not in added and x not in cleaned)

    if not plain:
        intent = "diagnosis" if has_signal else "help"
    elif GREETING_RE.match(plain) or GREETING_RE.match(cleaned):
        intent = "greeting"
    elif THANKS_RE.match(plain):
        intent = "thanks"
    elif HELP_RE.search(cleaned):
        intent = "help"
    elif has_signal and (DIAGNOSIS_RE.search(cleaned) or not GENERAL_RE.match(plain)):
        intent = "diagnosis"           # with a signal, anything not clearly a general question is about that signal
    elif DOMAIN_RE.search(cleaned) or added:
        intent = "knowledge"
    else:
        intent = "out_of_scope"
    content = [w for w in re.findall(r"[a-z0-9]+", cleaned) if w not in STOP]
    search_query = " ".join(content + added) or cleaned
    # focus is read from the original words too: spelling correction can turn "till" into "still"
    focus = question_focus(original.lower() + " . " + cleaned)
    return Query(original, cleaned, search_query, intent, corrections, added, focus)


GREETING_REPLY = ("Hi! I'm IntelliMaintain, your bearing maintenance copilot. Upload a vibration file (or pick a demo "
                  "sample) and ask me what's wrong, or ask any question about bearing faults, vibration analysis, "
                  "lubrication or safe replacement.")
THANKS_REPLY = "You're welcome! Ask me anything else about the bearing, or upload another signal to diagnose."
HELP_REPLY = ("I can: (1) diagnose a bearing from a vibration signal (normal, inner race, outer race or ball fault, "
              "with confidence and a physics check), (2) find the matching recordings in the CWRU dataset, "
              "(3) answer questions about fault signs, lubrication, inspection, safety and replacement using the "
              "maintenance manuals, with citations, and (4) record your review of each answer. You can type "
              "normally - for example: \"my pump bearing is making a grinding noise, what should I do?\"")
