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


def _sentences(text: str, max_len: int = 400) -> list[str]:
    """Prose sentences of a chunk; headings (no final punctuation) and list items are skipped."""
    out = []
    for line in text.split("\n"):
        line = line.strip().lstrip("-* ").strip()
        if not line or re.match(r"^\d+\.\s", line) or not re.search(r"[.!?]$", line):
            continue
        for s in re.split(r"(?<=[^0-9][.!?])\s+", line):
            s = s.strip()
            if 25 < len(s) < max_len:
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
            title_hit = min(len(set(tokenize(ev.title)) & q_terms), 2)  # passages from a document about the topic
            for s in _sentences(ev.text):
                st = set(tokenize(s))
                score = len(st & q_terms) + 2 * len(st & fault_terms) + (title_hit if len(st & q_terms) else 0)
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


MECHANISM_TERMS = {
    "Outer race": ("outer", "stationary", "load zone", "bpfo", "equal", "constant", "same position", "spall"),
    "Inner race": ("inner", "rotates", "revolution", "modulat", "sideband", "bpfi", "load zone", "spall"),
    "Ball": ("ball", "rolling element", "spin", "bsf", "cage", "ftf", "irregular", "both races", "alternately"),
    "Normal": ("healthy", "gaussian", "no repeating", "stage 1", "stage 2", "first", "fatigue", "spall",
               "trend", "quiet"),
    "Unknown": ("fatigue", "spall", "crack", "impact", "resonance", "defect frequency", "envelope", "load zone"),
}
# words that mark a sentence as being about ANOTHER fault type (kept out of this fault's mechanism)
MECHANISM_EXCLUDE = {
    "Outer race": ("inner ring", "inner race", "bpfi", "ball spin", "bsf", "cage", "rotates with the shaft",
                   "ball defect", "rolling element defect"),
    "Inner race": ("outer ring", "outer race", "bpfo", "ball spin", "bsf", "cage", "ftf", "stationary",
                   "ball defect", "rolling element defect"),
    "Ball": ("bpfi", "bpfo", "inner ring rotates", "outer ring is", "inner race defect", "outer race defect",
             "inner race damage", "outer race damage"),
    "Normal": ("bpfi", "bpfo", "bsf", "cage", "inner ring", "outer ring", "sideband"),
    "Unknown": (),
}
_NOT_MECHANISM = ("classifier", "predict", "analyst", "should be", "recommend", "dataset", "model", "monitoring "
                  "interval", "replace", "work order", "schedule")
_GENERAL_MECH = ("spall", "fatigue", "crack", "strike", "impact", "resonance", "repeat", "load zone", "envelope",
                 "subsurface", "contact stress")


def mechanism(fault: str, evidence: list[EvidenceChunk], max_sentences: int = 5) -> str:
    """How the fault develops and why it vibrates the way it does: the best mechanism sentences of the evidence,
    each cited. Prefers the fault-mechanism guide and passages written about this fault type; sentences about
    other fault types, about the models or about actions are left out (those belong to other answer parts)."""
    terms = MECHANISM_TERMS.get(fault, MECHANISM_TERMS["Unknown"])
    exclude = MECHANISM_EXCLUDE.get(fault, ())
    scored = []
    for n, ev in enumerate(evidence, 1):
        is_mech_doc = "how bearing faults occur" in ev.title.lower() or "mechanism" in ev.title.lower()
        for pos, s in enumerate(_sentences(ev.text)):
            low = s.lower()
            if low.startswith(("similar past work order", "type:", "title:")) or "|" in s:
                continue
            if any(x in low for x in exclude) or any(x in low for x in _NOT_MECHANISM):
                continue
            hits = sum(t in low for t in terms)
            general = sum(t in low for t in _GENERAL_MECH)
            if hits == 0 and general < 2:
                continue
            score = 3 * hits + general + (3 if is_mech_doc else 0)
            scored.append((score, n, pos, s))
    if not scored:
        return ""
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    picked = sorted(scored[:max_sentences], key=lambda t: (t[1], t[2]))  # keep the reading order of the sources
    seen, out = set(), []
    for _, n, _, s in picked:
        if s not in seen:
            out.append(f"{_semi(s)} [{n}].")
            seen.add(s)
    return " ".join(out)


# ====================================================================== question-focused chat answers
FOCUS_TERMS = {
    "severity": ("severity", "remaining life", "stop", "early stage", "quickly", "progress", "trend", "replace",
                 "planned", "urgent", "short", "rises", "increasing"),
    "safety": ("lockout", "tagout", "isolate", "safety", "energy", "ppe", "permit", "before"),
    "lubrication": ("lubric", "grease", "oil", "relubric", "interval", "viscosity", "film"),
    "replacement": ("mount", "dismount", "puller", "heater", "press", "replace", "install", "fit"),
    "indicators": ("kurtosis", "crest", "rms", "gaussian", "impact", "peak", "healthy", "3"),
    "frequencies": ("bpfo", "bpfi", "bsf", "ftf", "envelope", "harmonic", "sideband", "frequency"),
    "cwru": ("cwru", "dataset", "recording", "seeded", "load", "class"),
    "model": ("classifier", "confidence", "calibrat", "randomforest", "cnn", "probability", "accuracy"),
    "mechanism": ("fatigue", "spall", "crack", "impact", "load zone", "strike", "cause", "lubrication",
                  "contamination", "misalignment", "overload", "mounting", "fit"),
    "actions": ("inspect", "check", "monitor", "replace", "schedule", "verify", "trend", "confirm"),
    "location": ("outer race", "inner race", "rolling element", "ball", "ring"),
    "diagnosis": (),
}
_PART = {"Outer race": "the outer race (the stationary outer ring)",
         "Inner race": "the inner race (the ring that turns with the shaft)",
         "Ball": "a rolling element (one of the balls)"}


DEFINITION_RE = re.compile(r"^(what is|what are|define|meaning of|what does) (?:a |an |the )?([a-z0-9 -]{3,40}?)"
                           r"(?: mean)?\s*\??(?:\s.*)?$")


def best_sentences(question: str, evidence: list[EvidenceChunk], focus: str = "diagnosis", k: int = 3,
                   fault: str = "Unknown", min_score: int = 2) -> list[str]:
    """The k evidence sentences that answer the question best, each with its citation."""
    q_terms = set(tokenize(question))
    f_terms = FOCUS_TERMS.get(focus, ())
    m = DEFINITION_RE.match(question.strip().lower())
    definition = m.group(2).strip() if m else ""
    exclude = MECHANISM_EXCLUDE.get(fault, ()) if fault in MECHANISM_EXCLUDE else ()
    scored = []
    for n, ev in enumerate(evidence, 1):
        title_terms = set(tokenize(ev.title))
        for pos, s in enumerate(_sentences(ev.text, max_len=700)):
            low = s.lower()
            if "|" in s or low.startswith(("type:", "title:", "similar past work order")):
                continue
            if fault != "Unknown" and any(x in low for x in exclude):
                continue
            st = set(tokenize(s))
            score = len(st & q_terms) + sum(t in low for t in f_terms) + min(len(title_terms & q_terms), 2)
            if fault not in ("Unknown", "Normal") and fault.lower() in low:
                score += 1
            if definition and (low.startswith((f"a {definition} is", f"an {definition} is", f"{definition} is",
                                               f"the {definition} is")) or f"{definition}, also called" in low):
                score += 6
            if score >= min_score:
                scored.append((score, n, pos, s))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    out, seen = [], set()
    for _, n, _, s in scored:
        if s not in seen:
            out.append(f"{_semi(s)} [{n}].")
            seen.add(s)
        if len(out) >= k:
            break
    return out


def _ml_sentence(ml: ClassifyResult) -> str:
    if ml.fault_type == "Normal":
        return (f"The bearing looks healthy: the classifier predicts Normal with calibrated confidence "
                f"{ml.confidence:.2f} [ML].")
    size = f", defect size about {ml.fault_size_in} in" if ml.fault_size_in else ""
    art = "an" if ml.fault_type[0] in "AEIOU" else "a"
    sent = (f"The classifier finds {art} {ml.fault_type.lower()} fault ({ml.label}{size}) with calibrated "
            f"confidence {ml.confidence:.2f} [ML].")
    if ml.low_confidence:
        sent += " That confidence is low, so an analyst should verify it [ML]."
    return sent


def _phys_sentence(ff: FaultFrequencies | None) -> str:
    if ff is None or not ff.envelope_match:
        return ""
    if ff.consistency == "consistent":
        return f"The physics check agrees: the envelope spectrum points to {ff.physics_suggests} [PHYS]."
    if ff.consistency == "conflict":
        return (f"The physics check disagrees: the envelope spectrum points to {ff.physics_suggests or 'no clear '
                f'defect frequency'}, so an analyst should verify the result [PHYS].")
    return "The envelope spectrum shows no clear defect frequency, so the physics check is inconclusive [PHYS]."


def focused_answer(question: str, focus: str, ml: ClassifyResult | None, ff: FaultFrequencies | None,
                   evidence: list[EvidenceChunk], rel_ok: bool = True) -> str:
    """A direct, conversational reply to the question that was asked (no LLM), built only from the tool
    outputs and evidence sentences, each cited."""
    ev = evidence if rel_ok else []
    fault = ml.fault_type if ml else guess_fault_from_text(question)
    # keep sentences about a different fault type out of the reply (not for general-topic questions)
    ff_fault = fault if focus not in ("cwru", "lubrication", "safety", "replacement") else "Unknown"
    parts: list[str] = []
    if focus == "indicators" and ml:
        kf = ml.key_features
        parts.append(f"For this signal the RMS is {kf.get('rms_g', 0)} g, the kurtosis is {kf.get('kurtosis', 0)} "
                     f"and the crest factor is {kf.get('crest_factor', 0)} [ML].")
        ref = ml.cwru_reference or {}
        if "kurtosis" in ref:
            lo, hi = ref["kurtosis"]
            k = kf.get("kurtosis", 0)
            where = "inside" if lo <= k <= hi else "above" if k > hi else "below"
            parts.append(f"In the CWRU data, class {ref['cwru_class']} has a kurtosis of {lo} to {hi}, so this "
                         f"signal is {where} that range [ML].")
        if "anomal" in question.lower():
            parts.append(f"The anomaly score is {ml.anomaly_score:.2f}, on a scale from 0 (looks like the training "
                         f"data) to 1 (unusual) [ML].")
        parts += best_sentences(question + " kurtosis healthy impacts", ev, focus, 2, ff_fault)
    elif focus == "frequencies" and ff:
        parts.append(f"At {ff.rpm:.0f} rpm the shaft turns at {ff.shaft_hz:.2f} Hz, which gives BPFO "
                     f"{ff.bpfo_hz:.1f} Hz, BPFI {ff.bpfi_hz:.1f} Hz, BSF {ff.bsf_hz:.1f} Hz and FTF "
                     f"{ff.ftf_hz:.1f} Hz [PHYS].")
        if ff.envelope_match:
            top = max(ff.envelope_match.items(), key=lambda kv: kv[1])
            parts.append(f"In the envelope spectrum the strongest line is {top[0]} with a signal-to-noise ratio of "
                         f"{top[1]:.1f}, where above 5 means the frequency is present [PHYS].")
            if p := _phys_sentence(ff):
                parts.append(p)
        parts += best_sentences(question, ev, focus, 1, ff_fault)
    elif focus == "cwru" and ml:
        if ml.similar_cases:
            c = ml.similar_cases[0]
            same = sum(x["fault_type"] == ml.fault_type for x in ml.similar_cases)
            parts.append(f"The closest CWRU recording is {c['cwru_file']} ({c['label']}, {c['load_hp']} HP) with "
                         f"similarity {c['similarity']}, and {same} of the {len(ml.similar_cases)} closest recordings "
                         f"have the same fault type as the prediction [ML].")
        ref = ml.cwru_reference or {}
        if "rms_g" in ref and "kurtosis" in ref:
            parts.append(f"Typical CWRU {ref['cwru_class']} recordings have RMS {ref['rms_g'][0]} to "
                         f"{ref['rms_g'][1]} g and kurtosis {ref['kurtosis'][0]} to {ref['kurtosis'][1]}, while this "
                         f"signal has RMS {ml.key_features.get('rms_g')} g and kurtosis "
                         f"{ml.key_features.get('kurtosis')} [ML].")
        parts += best_sentences(question + " cwru", ev, focus, 1, "Unknown")
    elif focus == "model" and ml:
        top = sorted(ml.class_probabilities.items(), key=lambda kv: -kv[1])[:3]
        parts.append(f"The {ml.model_name} classified {ml.n_windows} windows of the signal and averaged their "
                     f"probabilities: {', '.join(f'{k} {v:.2f}' for k, v in top)} [ML].")
        parts.append(_ml_sentence(ml))
        if p := _phys_sentence(ff):
            parts.append(p)
        parts += best_sentences(question, ev, focus, 1, ff_fault)
    elif focus == "location" and ml:
        if ml.fault_type == "Normal":
            parts.append(_ml_sentence(ml) + " No damaged part is indicated.")
        else:
            parts.append(f"The damage is most likely on {_PART.get(ml.fault_type, ml.fault_type.lower())}: the "
                         f"classifier predicts {ml.label} with calibrated confidence {ml.confidence:.2f} [ML].")
            if p := _phys_sentence(ff):
                parts.append(p)
            parts += best_sentences(f"{ml.fault_type} defect", ev, "location", 1, fault)
    elif focus == "severity":
        if ml:
            parts.append(_ml_sentence(ml))
        parts += best_sentences(question + " severity remaining life stop", ev, focus, 3, fault)
        parts += [s for s in best_sentences("keep running decision engineer trend stopped", ev, focus, 1)
                  if s not in parts]
    elif len(mentioned := [ft for ft, word in (("Inner race", "inner"), ("Outer race", "outer"),
                                               ("Ball", "ball")) if re.search(rf"\b{word}\b", question.lower())]) > 1:
        # comparison ("difference between inner and outer race faults"): the best sentences for each fault type
        for ft in mentioned:
            parts += [x for x in best_sentences(f"{ft} fault impacts", ev, "mechanism", 2, ft) if x not in parts]
    else:  # diagnosis, actions, mechanism, safety, lubrication, replacement, anything else
        if ml and focus in ("diagnosis", "actions"):
            parts.append(_ml_sentence(ml))
            if p := _phys_sentence(ff):
                parts.append(p)
        k = 2 if (ml and focus in ("diagnosis", "actions")) else 4
        q = question
        if fault not in ("Unknown", "Normal") and focus in ("diagnosis", "actions", "mechanism", "location"):
            q += f" {fault}"
        if focus == "mechanism" and fault not in ("Unknown", "Normal"):
            q += " causes damage"
        parts += best_sentences(q, ev, focus, k, ff_fault)
    parts = [p for p in parts if p]
    if not parts and ml:
        parts.append(_ml_sentence(ml))
    return " ".join(parts)
