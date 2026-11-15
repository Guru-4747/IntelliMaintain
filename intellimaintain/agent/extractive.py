"""Offline, deterministic answer generator (no LLM).

Builds the answer only from tool outputs and sentences copied from the retrieved
evidence, each with its citation. It cannot hallucinate, so it is the reference
point for faithfulness and lets the whole system run without any API key.
"""
from __future__ import annotations

import re

from ..rag.retriever import tokenize
from ..schemas import ClassifyResult, EvidenceChunk, FaultFrequencies

FAULT_KEYWORDS = {
    "Outer race": ("outer race", "bpfo", "outer ring", "outer track"),
    "Inner race": ("inner race", "bpfi", "inner ring", "inner track"),
    "Ball": ("ball fault", "rolling element", "ball defect", "bsf", "ball spin"),
    "Normal": ("healthy", "normal condition", "no fault"),
}


def _semi(text: str) -> str:
    """Join multi-sentence text into one citable sentence ("A. B." -> "A; b")."""
    text = text.strip().rstrip(".")

    def repl(m):
        word = m.group(1)
        return "; " + (word if len(word) > 1 and word[:2].isupper() else word[0].lower() + word[1:])

    return re.sub(r"[.!?] ([A-Z]\w*)", repl, text)


def guess_fault_from_text(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(difference|diff|versus|vs|compare|comparison|between)\b", t) or \
            sum(w in t for w in ("inner", "outer", "ball")) > 1:
        return "Unknown"  # comparison questions are not a diagnosis
    hits = [ft for ft, kws in FAULT_KEYWORDS.items() if any(k in t for k in kws)]
    return hits[0] if len(hits) == 1 else "Unknown"


def _sentences(text: str) -> list[str]:
    """Prose sentences of a chunk; headings (no final punctuation) and list items are skipped."""
    out = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-* ").strip()
        if not line or re.match(r"^\d+\.\s", line) or not re.search(r"[.!?]$", line):
            continue
        for s in re.split(r"(?<=[^0-9][.!?])\s+", line):
            s = s.strip()
            if 25 < len(s) < 400:
                if s.startswith("WO-"):
                    s = "Similar past work order " + s.replace(" | ", ", ")
                out.append(s)
    return out


def _actions_from(chunk: EvidenceChunk) -> list[str]:
    """Numbered procedure steps ("1. Verify ...") contained in a chunk."""
    items = []
    for line in chunk.text.split("\n"):
        m = re.match(r"^\s*\d+\.\s+([A-Z].+)$", line.strip())
        if m and 15 < len(m.group(1)) < 400 and not re.match(r"^\d+\.\s+[A-Z][^.]*$", line.strip()):
            items.append(m.group(1).strip())
    return items


def generate(question: str, ml: ClassifyResult | None, ff: FaultFrequencies | None,
             evidence: list[EvidenceChunk], min_relevance_ok: bool = True, intent: str = "diagnosis",
             about_machine: bool = False) -> dict:
    """intent/about_machine come from nlu.understand(): a general "knowledge" question gets a document answer,
    a question about the user's own machine without a signal gets guidance plus a request for a recording."""
    if ml:
        fault = ml.fault_type
    elif intent == "diagnosis" or about_machine:
        fault = guess_fault_from_text(question)
    else:
        fault = "Unknown"
    expl: list[str] = []
    if ml:
        conf_txt = "low" if ml.low_confidence else "high" if ml.confidence >= 0.85 else "moderate"
        expl.append(f"The classifier predicts {ml.label} ({ml.fault_type}) with {conf_txt} calibrated confidence "
                    f"{ml.confidence:.2f}; signal kurtosis is {ml.key_features.get('kurtosis', 0):.1f} and crest factor "
                    f"{ml.key_features.get('crest_factor', 0):.1f} [ML].")
    if ml and ml.similar_cases:
        c = ml.similar_cases[0]
        same = sum(x["fault_type"] == ml.fault_type for x in ml.similar_cases)
        expl.append(f"The most similar recording in the CWRU dataset is {c['cwru_file']} ({c['label']}, {c['load_hp']} HP, "
                    f"similarity {c['similarity']}), and {same} of the {len(ml.similar_cases)} closest recordings share "
                    f"the predicted fault type [ML].")
    if ff and ff.envelope_match:
        expl.append(f"At {ff.rpm:.0f} rpm the defect frequencies are BPFO {ff.bpfo_hz:.1f} Hz, BPFI {ff.bpfi_hz:.1f} Hz "
                    f"and BSF {ff.bsf_hz:.1f} Hz; the envelope check is {ff.consistency} with the ML result "
                    f"({_semi(ff.consistency_note)}) [PHYS].")

    actions: list[str] = []
    if evidence and min_relevance_ok:
        q_terms = set(tokenize(question + " " + (fault if fault != "Unknown" else "")))
        fault_terms = set(tokenize(fault)) if fault not in ("Unknown",) else set()
        scored = []
        for n, ev in enumerate(evidence, 1):
            for s in _sentences(ev.text):
                st = set(tokenize(s))
                score = len(st & q_terms) + 2 * len(st & fault_terms)
                if not re.match(r"^\d\.", s) and not s.lower().startswith(("type:", "title:")):
                    scored.append((score, n, s))
        scored.sort(key=lambda t: (-t[0], t[1]))
        used = set()
        for score, n, s in scored:
            if score <= 1 or len(expl) >= (6 if ml else 5):
                break
            if s not in used:
                expl.append(f"{_semi(s)} [{n}].")
                used.add(s)
        # actions: numbered steps from the most relevant evidence chunk that has them
        # prefer the procedure written for this fault type (title match), then any text mention
        candidates = [(n, ev) for n, ev in enumerate(evidence, 1) if _actions_from(ev)]
        if fault != "Unknown":
            ranked = ([c for c in candidates if fault.lower() in c[1].title.lower()] +
                      [c for c in candidates if fault.lower() in c[1].text.lower()])
        else:  # general question: only a procedure whose title matches what was asked
            def fit(c):
                n_, ev_ = c
                title_hit = len(set(tokenize(ev_.title)) & q_terms)
                body_hit = len(set(tokenize(" ".join(_actions_from(ev_)))) & q_terms)
                return (title_hit, body_hit)
            asks_for_steps = re.search(r"\b(what (should|shall|do|can) (i|we) do|steps?|procedure|how (to|do i|do we|"
                                       r"should i)|recommend\w*|actions?|fix|repair|next)\b", question.lower())
            ranked = sorted([c for c in candidates if fit(c)[0] >= 1 and fit(c)[1] >= 2], key=fit, reverse=True) \
                if asks_for_steps else []
        if ranked:
            n, ev = ranked[0]
            actions = [f"{_semi(a)} [{n}]." for a in _actions_from(ev)[:6]]

    insufficient = (not evidence or not min_relevance_ok) and ml is None
    if insufficient:
        summary = "Insufficient evidence: no ML output and no relevant documents are available for this question."
        fault = "Unknown"
    elif ml is None and not expl:
        summary = ("Insufficient evidence: the maintenance documents do not answer this question directly; "
                   "please rephrase it or ask about bearing faults, vibration, lubrication, inspection or safety.")
        insufficient = True
    elif ml is None and fault == "Unknown" and about_machine:
        summary = ("I cannot identify the exact fault from a text description alone - upload a vibration recording "
                   "for a full diagnosis. General guidance from the maintenance documents is given below [1].")
    elif ml is None and fault == "Unknown":
        summary = "Here is what the maintenance documents say about this [1]."
    else:
        summary = (f"Most likely condition: {fault} [ML]." if ml else
                   f"From your description this sounds like {'an' if fault[0] in 'AEIOU' else 'a'} {fault.lower()} "
                   f"problem [1]; upload a vibration "
                   f"recording to confirm it.")
        if ff and ff.consistency == "conflict":
            summary += " The physics check disagrees with the classifier, so analyst verification is required [PHYS]."
        elif ml and ml.low_confidence:
            summary += " Confidence is low, so the result should be verified by an analyst [ML]."
    if not actions and fault == "Normal" and ml is not None:
        actions = ["No corrective action is indicated by the data; continue routine monitoring and trend RMS and "
                   "kurtosis against this measurement as a baseline [ML]."]
    return {"fault_type": fault, "summary": summary, "explanation": " ".join(expl),
            "recommended_actions": actions, "insufficient_evidence": insufficient}
