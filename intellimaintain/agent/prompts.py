"""Prompt templates (grounded, cite-everything, refuse-when-unsupported)."""
from __future__ import annotations

from ..schemas import ClassifyResult, EvidenceChunk, FaultFrequencies

SYSTEM_PROMPT = """You are IntelliMaintain, a maintenance decision-support assistant for rolling-element bearings.
You help a human maintenance engineer; your output is ADVISORY and will be reviewed before any action.

STRICT RULES
1. Use ONLY the information in the CONTEXT below. Do not use outside knowledge, do not guess numbers.
2. Cite EVERY factual sentence and every recommended action:
   - [1], [2], ... for numbered evidence passages,
   - [ML] for the machine-learning classifier output,
   - [PHYS] for the physics / fault-frequency analysis.
   A sentence may carry several citations, e.g. "... [ML][2]". Never cite a number that is not in the context.
3. If the context does not contain enough information to answer, set "insufficient_evidence": true and say
   "Insufficient evidence" in the summary instead of inventing an answer.
4. If the ML prediction has low confidence or conflicts with the physics check, say so explicitly and
   recommend verification by an analyst.
5. The fault type is decided by the ML classifier and the physics check - explain it, never change it.
   When the ML output lists typical CWRU values for the predicted class, compare the signal's own RMS, kurtosis
   and crest factor with them and say whether they fall inside, above or below the CWRU range.
6. Copy numbers exactly as they appear in the context (no rounding, no new numbers). Reuse the wording of the
   evidence where possible: every sentence is automatically checked against the source it cites, and
   sentences that cannot be verified are removed.
7. Be concise and practical. Recommended actions must be concrete steps taken from the evidence.
8. You are in a CHAT. "answer" is your reply to the LATEST question, written like a helpful expert talking to an
   engineer: answer exactly what was asked in the first sentence, then add only what helps (2-6 sentences).
   Do not repeat the whole diagnosis when the user asks something specific (e.g. grease, safety, kurtosis, CWRU,
   how long it can run). Use the CONVERSATION SO FAR to understand follow-ups such as "why?" or "and then?".
   Whatever the topic - the bearing, the data, the CWRU dataset, the models, maintenance or how the system
   works - answer it from the evidence. "summary" stays a one-line statement of the bearing condition.
9. In "fault_mechanism" explain, step by step, HOW this fault physically develops and WHY it produces the vibration
   seen in the data: where the damage starts, what causes it, how the rolling elements hit the defect, how that
   creates repeating impacts at the defect frequency, and why this fault's signature differs from the others.
   For a Normal (healthy) bearing return an empty string - there is no fault to explain. Every sentence needs a
   citation. If the evidence does not describe the mechanism, return an empty string.
10. Evidence passages from "Wikipedia: ..." are online sources: use them only where the documents do not cover
   the question, and cite them with their number like any other passage.
11. Users type casually (typos, short forms, everyday words like "grinding noise" or "getting hot", other
   languages). Work out what they mean from the QUESTION and its interpretation, and answer that intent in
   simple, clear English (English keeps every sentence verifiable against the English sources).

Return ONLY a JSON object with exactly these keys:
{"fault_type": "Normal | Inner race | Outer race | Ball | Unknown",
 "answer": "direct chat reply to the latest question, 2-6 sentences, each with citations",
 "summary": "one or two sentences with citations",
 "explanation": "3-6 sentences, each ending with citations",
 "fault_mechanism": "3-6 sentences: how this fault develops and why it vibrates this way, each with citations",
 "recommended_actions": ["action with citation", "..."],
 "insufficient_evidence": false}"""

NO_RETRIEVAL_NOTE = "(No documents were retrieved for this configuration.)"


def format_ml(ml: ClassifyResult | None) -> str:
    if ml is None:
        return "Not available (ML output not used in this configuration)."
    top = sorted(ml.class_probabilities.items(), key=lambda kv: -kv[1])[:3]
    feats = ", ".join(f"{k}={v}" for k, v in ml.key_features.items())
    return (f"Predicted class: {ml.label} (fault type: {ml.fault_type}"
            f"{f', defect size {ml.fault_size_in} in' if ml.fault_size_in else ''}); "
            f"calibrated confidence {ml.confidence:.2f}{' (LOW CONFIDENCE)' if ml.low_confidence else ''}; "
            f"top-3 probabilities: {', '.join(f'{k} {v:.2f}' for k, v in top)}; "
            f"anomaly score {ml.anomaly_score:.2f} (0 healthy .. 1 anomalous); "
            f"windows analysed: {ml.n_windows}; signal indicators: {feats}."
            + (" Most similar recordings in the CWRU training dataset: "
               + "; ".join(f"{c['cwru_file']} ({c['label']}, {c['fault_type']}, {c['load_hp']} HP, similarity "
                           f"{c['similarity']})" for c in ml.similar_cases) + "." if ml.similar_cases else "")
            + _format_reference(ml))


def _format_reference(ml: ClassifyResult) -> str:
    ref = ml.cwru_reference or {}
    if not ref:
        return ""
    names = {"rms_g": ("RMS", " g"), "kurtosis": ("kurtosis", ""), "crest_factor": ("crest factor", "")}
    parts = []
    for key, (name, unit) in names.items():
        if key in ref:
            lo, hi = ref[key]
            own = ml.key_features.get(key)
            where = ""
            if own is not None:
                where = " (this signal: {}{}, {} the CWRU range)".format(
                    own, unit, "inside" if lo <= own <= hi else "above" if own > hi else "below")
            parts.append(f"{name} {lo} to {hi}{unit}{where}")
    return (f" Typical CWRU values for class {ref['cwru_class']} over {ref.get('n_recordings', '?')} recordings: "
            + "; ".join(parts) + ".")


def format_physics(ff: FaultFrequencies | None) -> str:
    if ff is None:
        return "Not available."
    snr = ", ".join(f"{k} {v:.1f}" for k, v in ff.envelope_match.items())
    return (f"Shaft speed {ff.rpm:.0f} rpm ({ff.shaft_hz:.2f} Hz). Bearing {ff.bearing}. Defect frequencies: "
            f"BPFO {ff.bpfo_hz:.1f} Hz, BPFI {ff.bpfi_hz:.1f} Hz, BSF {ff.bsf_hz:.1f} Hz, FTF {ff.ftf_hz:.1f} Hz. "
            f"Envelope-spectrum harmonic SNR (>5 means the frequency is present): {snr}. "
            f"Physics points to: {ff.physics_suggests or 'no clear defect frequency'}. "
            f"Consistency with ML: {ff.consistency.upper()} - {ff.consistency_note}")


def format_evidence(evidence: list[EvidenceChunk], max_chars: int = 1800) -> str:
    if not evidence:
        return NO_RETRIEVAL_NOTE
    blocks = []
    for i, e in enumerate(evidence, 1):
        text = e.text if len(e.text) <= max_chars else e.text[:max_chars] + " ..."
        where = f"online source {e.url}" if e.url else f"page {e.page}"
        blocks.append(f"[{i}] (source: {e.source}, {where})\n{text}")
    return "\n\n".join(blocks)


FOCUS_HINT = {
    "diagnosis": "the overall condition of the bearing",
    "severity": "how serious it is and whether / how long the machine can keep running",
    "safety": "safety precautions",
    "lubrication": "lubrication (grease / oil, interval)",
    "replacement": "how to replace or mount the bearing",
    "indicators": "the signal indicators (RMS, kurtosis, crest factor, anomaly score) and what they mean",
    "frequencies": "the bearing defect frequencies and the envelope / physics check",
    "cwru": "the CWRU dataset and how this signal compares with it",
    "model": "how the model reached its prediction and how much to trust it",
    "location": "which part of the bearing is damaged",
    "mechanism": "how the fault develops and why",
    "actions": "what to do next",
}


def format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-6:]:
        who = "User" if h.get("role") == "user" else "Copilot"
        text = " ".join(str(h.get("content", "")).split())
        lines.append(f"{who}: {text[:400]}")
    return "CONVERSATION SO FAR\n" + "\n".join(lines) + "\n\n"


def build_user_prompt(question: str, ml: ClassifyResult | None, ff: FaultFrequencies | None,
                      evidence: list[EvidenceChunk], nq=None, history: list[dict] | None = None) -> str:
    interp = ""
    if nq is not None and (nq.cleaned != question.strip().lower() or nq.added_terms):
        interp = (f"\nInterpreted as: \"{nq.cleaned}\" (intent: {nq.intent}; related terms: "
                  f"{', '.join(nq.added_terms) or 'none'})")
    if nq is not None:
        interp += f"\nThe user is asking about: {FOCUS_HINT.get(nq.focus, nq.focus)}."
    return f"""{format_history(history)}LATEST QUESTION (as typed by the user)
{question}{interp}

CONTEXT
[ML] Machine-learning classifier output:
{format_ml(ml)}

[PHYS] Physics / fault-frequency analysis:
{format_physics(ff)}

Evidence passages:
{format_evidence(evidence)}

Write the JSON answer now. Remember: cite every sentence; use "insufficient_evidence": true if the context is not enough."""


PLANNER_PROMPT = """You are the planner of a bounded maintenance agent. Decide the NEXT step.
Available tools (call each at most twice; a maximum of {max_steps} tool calls in total):
- classify_signal: run the ML fault classifier on the uploaded vibration signal. args: {{}}
- fault_frequency_calc: compute bearing defect frequencies and the envelope physics check. args: {{"rpm": number}}
- retrieve_evidence: search the maintenance knowledge base. args: {{"query": "search text"}}
- web_search: search online sources (Wikipedia) when the knowledge base is not enough{web_note}. args: {{"query": "few keywords"}}
- finish: stop calling tools and write the answer. args: {{}}

QUESTION: {question}
SIGNAL AVAILABLE: {has_signal}   RPM: {rpm}
STEPS USED: {used}/{max_steps}
OBSERVATIONS SO FAR:
{observations}

Good practice: classify the signal first, then compute fault frequencies, then retrieve evidence using the
predicted fault type in the query. Reply with JSON only: {{"tool": "<name>", "args": {{...}}, "reason": "<short>"}}"""
