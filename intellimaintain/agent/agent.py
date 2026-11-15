"""Bounded diagnostic agent: <= MAX_AGENT_STEPS calls to 3 approved tools, then a grounded LLM answer.

Two planners:
  fixed  (default) deterministic plan  classify -> fault frequencies -> retrieve (-> refine)  : reproducible
  llm    the LLM chooses the next tool each step (ReAct-style JSON), validated against the whitelist;
         any invalid step falls back to the fixed plan, so the agent can never loop or call unknown tools.
"""
from __future__ import annotations

import json

import numpy as np

from .. import config
from ..rag.retriever import load_retriever
from ..schemas import ClassifyResult, Diagnosis, EvidenceChunk, FaultFrequencies, ToolCall
from . import extractive, nlu, tools
from .llm import LLM, LLMError, parse_json
from .prompts import PLANNER_PROMPT, SYSTEM_PROMPT, build_user_prompt, format_ml, format_physics
from ..security import redact
from .verify import citation_coverage, clean_invalid_citations, split_sentences, verify_sentences

VALID_TYPES = {"Normal", "Inner race", "Outer race", "Ball", "Unknown"}
DEFAULT_RPM = 1772.0


def ml_informed_query(question: str, ml: ClassifyResult | None) -> str:
    """Augment the user's question with the ML findings so retrieval targets the right fault."""
    if ml is None:
        return question
    parts = [question]
    if ml.fault_type != "Normal":
        parts.append(f"{ml.fault_type.lower()} fault")
        k = ml.key_features.get("kurtosis", 0)
        if k > 4:
            parts.append("high kurtosis impacting")
        parts.append("recommended inspection actions")
    else:
        parts.append("healthy bearing baseline monitoring no fault")
    return " ".join(parts)


class DiagnosisAgent:
    def __init__(self, llm: LLM | None = None, use_ml: bool = True, retrieval: str = "improved",
                 planner: str = "fixed", max_steps: int = config.MAX_AGENT_STEPS, top_k: int = config.TOP_K,
                 conflict_policy: str = "abstain", classifier: str = "rf"):
        if retrieval not in ("none", "basic", "improved"):
            raise ValueError("retrieval must be none | basic | improved")
        if planner not in ("fixed", "llm"):
            raise ValueError("planner must be fixed | llm")
        if conflict_policy not in ("abstain", "trust_ml"):
            raise ValueError("conflict_policy must be abstain | trust_ml")
        self.conflict_policy = conflict_policy
        if classifier not in ("rf", "cnn"):
            raise ValueError("classifier must be rf | cnn")
        self.classifier = classifier
        self.llm = llm or LLM()
        self.use_ml = use_ml
        self.retrieval = retrieval
        self.planner = planner
        self.max_steps = max(1, min(int(max_steps), 8))
        self.top_k = top_k

    @property
    def config_name(self) -> str:
        return f"{'ML' if self.use_ml else 'noML'}+{self.retrieval}"

    # ================================================================== public
    def run(self, question: str, signal: np.ndarray | None = None, fs: int | None = config.FS,
            rpm: float | None = None, override_label: str | None = None) -> Diagnosis:
        state = _State(question=question.strip() or "Diagnose the bearing condition.")
        state.nq = nlu.understand(question, has_signal=signal is not None)
        if state.nq.intent in ("greeting", "thanks", "help"):
            reply = {"greeting": nlu.GREETING_REPLY, "thanks": nlu.THANKS_REPLY, "help": nlu.HELP_REPLY}
            return Diagnosis(question=state.question, fault_type="Unknown", summary=reply[state.nq.intent],
                             llm_provider=self.llm.provider, llm_model=self.llm.model, config_name=self.config_name,
                             answer_source="assistant", citation_coverage=1.0)
        if signal is not None:
            signal = np.asarray(signal, dtype=np.float64).ravel()
            if fs and fs != config.FS:
                from ..ml.classifier import to_target_fs
                signal = to_target_fs(signal, fs)
        state.signal = signal
        if rpm is None and signal is not None:
            state.warnings.append(f"Shaft speed not provided; assumed {DEFAULT_RPM:.0f} rpm for defect frequencies.")
        state.rpm = float(rpm) if rpm else DEFAULT_RPM

        if self.planner == "llm" and not self.llm.is_offline:
            self._run_llm_planner(state, override_label)
        self._run_fixed_plan(state, override_label)  # completes whatever the LLM planner did not do
        return self._answer(state)

    # ================================================================== tools
    def _call(self, state: _State, tool: str, args: dict, fn) -> object:
        if len(state.trace) >= self.max_steps:
            state.warnings.append(f"Step limit ({self.max_steps}) reached; skipped {tool}.")
            return None
        step = len(state.trace) + 1
        try:
            out = fn()
            state.trace.append(ToolCall(step=step, tool=tool, arguments=args, summary=_summarise(out), ok=True))
            return out
        except Exception as e:  # noqa: BLE001 - a failing tool must not crash the diagnosis
            state.trace.append(ToolCall(step=step, tool=tool, arguments=args, summary=f"ERROR: {e}", ok=False))
            state.warnings.append(f"Tool {tool} failed: {e}")
            return None

    def _do_classify(self, state, override_label):
        if state.ml is None and self.use_ml and state.signal is not None and "classify_signal" not in state.done:
            state.done.add("classify_signal")
            state.ml = self._call(state, "classify_signal", {"n_samples": len(state.signal), "model": self.classifier},
                                  lambda: tools.classify_signal(state.signal, fs=config.FS, override_label=override_label,
                                                                model=self.classifier))

    def _do_physics(self, state, rpm=None):
        if state.ff is None and self.use_ml and state.signal is not None and "fault_frequency_calc" not in state.done:
            state.done.add("fault_frequency_calc")
            r = float(rpm or state.rpm)
            state.ff = self._call(state, "fault_frequency_calc", {"rpm": r},
                                  lambda: tools.fault_frequency_calc(r, state.signal,
                                                                     state.ml.fault_type if state.ml else None))

    def _do_retrieve(self, state, query):
        if self.retrieval == "none":
            return
        res = self._call(state, "retrieve_evidence", {"query": query, "mode": self.retrieval},
                         lambda: tools.retrieve_evidence(query, top_k=self.top_k, mode=self.retrieval))
        if res:
            seen = {e.chunk_id for e in state.evidence}
            state.evidence.extend(e for e in res if e.chunk_id not in seen)
        state.queries.append(query)

    def _run_fixed_plan(self, state: _State, override_label):
        self._do_classify(state, override_label)
        self._do_physics(state)
        if self.retrieval != "none" and not state.queries:
            self._do_retrieve(state, ml_informed_query(state.nq.search_query, state.ml))
        if self.retrieval != "none" and state.queries and len(state.queries) < 2:
            # self-check: refine once if the evidence looks off-topic
            try:
                rel = load_retriever().relevance(state.nq.search_query, state.evidence)
                state.relevance = rel
                if rel < load_retriever().min_relevance and state.ml is not None and state.ml.fault_type != "Normal":
                    self._do_retrieve(state, f"{state.ml.fault_type} bearing fault recommended actions inspection")
            except Exception:  # noqa: BLE001
                pass
        state.evidence = state.evidence[: self.top_k + 2]

    def _run_llm_planner(self, state: _State, override_label):
        for _ in range(self.max_steps):
            obs = "\n".join(f"- step {t.step} {t.tool}({json.dumps(t.arguments)}) -> {t.summary}" for t in state.trace)
            prompt = PLANNER_PROMPT.format(max_steps=self.max_steps, question=state.question,
                                           has_signal=state.signal is not None, rpm=state.rpm,
                                           used=len(state.trace), observations=obs or "(none)")
            try:
                step = parse_json(self.llm.chat("You are a careful tool-using planner.", prompt, json_mode=True,
                                                max_tokens=200))
            except Exception as e:  # noqa: BLE001
                state.warnings.append(f"Planner fell back to the fixed plan ({type(e).__name__}).")
                return
            tool = str(step.get("tool", "")).strip()
            args = step.get("args") or {}
            if tool == "finish":
                return
            if tool == "classify_signal":
                if "classify_signal" in state.done or state.signal is None or not self.use_ml:
                    return
                self._do_classify(state, override_label)
            elif tool == "fault_frequency_calc":
                if "fault_frequency_calc" in state.done or state.signal is None or not self.use_ml:
                    return
                try:
                    rpm = float(args.get("rpm", state.rpm))
                except (TypeError, ValueError):
                    rpm = state.rpm
                self._do_physics(state, rpm if 100 < rpm < 20000 else state.rpm)
            elif tool == "retrieve_evidence":
                if self.retrieval == "none" or len(state.queries) >= 2:
                    return
                q = str(args.get("query") or ml_informed_query(state.nq.search_query, state.ml))[:300]
                self._do_retrieve(state, q)
            else:
                state.warnings.append(f"Planner requested unknown tool '{tool}' - ignored.")
                return

    # ================================================================== answer
    def _answer(self, state: _State) -> Diagnosis:
        ml, ff, ev = state.ml, state.ff, state.evidence
        rel_ok = True
        if ev:
            try:
                state.relevance = state.relevance or load_retriever().relevance(state.nq.search_query, ev)
                rel_ok = state.relevance >= load_retriever().min_relevance or ml is not None
            except Exception:  # noqa: BLE001
                pass
        if not rel_ok and ml is None:
            ev = state.evidence = []  # off-topic question: do not show unrelated documents
        if ff is not None and ff.consistency == "conflict":
            state.warnings.append("Physics check conflicts with the ML prediction: " + ff.consistency_note)
        if ml is not None and ml.low_confidence:
            state.warnings.append(f"Low ML confidence ({ml.confidence:.2f} < {config.CONFIDENCE_THRESHOLD}).")

        has_ml, has_phys = ml is not None, ff is not None and bool(ff.envelope_match)

        def clean(out: dict):
            summary, r1 = clean_invalid_citations(str(out.get("summary", "")), len(ev), has_ml, has_phys)
            expl, r2 = clean_invalid_citations(str(out.get("explanation", "")), len(ev), has_ml, has_phys)
            acts_raw = out.get("recommended_actions") or []
            if isinstance(acts_raw, str):
                acts_raw = [acts_raw]
            acts, r3 = [], []
            for a in acts_raw:
                ca, rr = clean_invalid_citations(str(a), len(ev), has_ml, has_phys)
                if ca.strip():
                    acts.append(ca.strip())
                r3 += rr
            return summary, expl, acts, r1 + r2 + r3

        def offline_answer() -> dict:
            return extractive.generate(state.nq.cleaned + " " + " ".join(state.nq.added_terms), ml, ff, ev, rel_ok,
                                       intent=state.nq.intent,
                                       about_machine=bool(nlu.DIAGNOSIS_RE.search(state.nq.cleaned)))

        grounding, answer_source = 1.0, "offline"
        if self.llm.is_offline:
            out = offline_answer()
        else:
            try:
                raw = self.llm.chat(SYSTEM_PROMPT, build_user_prompt(state.question, ml, ff, ev, state.nq), json_mode=True)
                out = parse_json(raw)
                answer_source = f"{self.llm.provider} (verified)"
            except (LLMError, ValueError) as e:
                state.warnings.append(f"LLM unavailable or returned invalid JSON ({str(e)[:150]}); "
                                      "used the offline extractive generator instead.")
                out = offline_answer()
                answer_source = f"{self.llm.provider} -> offline fallback"

        summary, expl, actions, removed = clean(out)
        fault = str(out.get("fault_type", "Unknown")).strip()
        fault = next((t for t in VALID_TYPES if t.lower() == fault.lower()), "Unknown")

        if answer_source.endswith("(verified)"):
            # 1) fault-type lock: the data (ML + physics) decides the fault, the LLM only explains it
            if ml is not None and fault != ml.fault_type:
                state.warnings.append(f"The LLM suggested '{fault}'; replaced by the data-based result "
                                      f"'{ml.fault_type}' from the classifier.")
                fault = ml.fault_type
            # 2) every sentence and number must be supported by the source it cites
            sources = {str(i): f"{e.title} {e.text}" for i, e in enumerate(ev, 1)}
            if ml is not None:
                sources["ML"] = format_ml(ml)
            if has_phys:
                sources["PHYS"] = format_physics(ff)
            ok_e, bad_e = verify_sentences(split_sentences(expl), sources)
            ok_a, bad_a = verify_sentences(actions, sources)
            ok_s, _ = verify_sentences(split_sentences(summary), sources)
            total = len(ok_e) + len(bad_e) + len(ok_a) + len(bad_a)
            grounding = round((len(ok_e) + len(ok_a)) / total, 3) if total else 0.0
            if grounding < 0.5 or not ok_e:
                state.warnings.append(f"Only {grounding:.0%} of the LLM answer could be verified against the data "
                                      "and documents; the verified offline answer is shown instead.")
                summary, expl, actions, removed = clean(offline_answer())
                answer_source = f"{self.llm.provider} -> offline fallback (failed verification)"
            else:
                if bad_e or bad_a:
                    reason = (bad_e + bad_a)[0][1]
                    state.warnings.append(f"Removed {len(bad_e) + len(bad_a)} unverified sentence(s) from the LLM "
                                          f"answer (e.g. {reason}).")
                expl, actions = " ".join(ok_e), ok_a
                summary = " ".join(ok_s) if ok_s else clean(offline_answer())[0]
        if removed:
            state.warnings.append(f"Removed {len(removed)} invalid citation(s): {', '.join(sorted(set(removed)))}.")
        coverage = citation_coverage([expl] + actions)
        if expl and coverage < 0.8:
            state.warnings.append(f"Only {coverage:.0%} of explanation sentences are cited - review carefully.")
        insufficient = bool(out.get("insufficient_evidence", False)) or "insufficient evidence" in summary.lower()
        if self.conflict_policy == "abstain" and ff is not None and ff.consistency == "conflict" and ml is not None:
            # governance rule: never hand a contradicted label to the reviewer as the final answer
            phys = ff.physics_suggests or "no defect frequency"
            summary = (f"Uncertain - analyst verification required: the classifier predicts {ml.fault_type} [ML] "
                       f"but the envelope spectrum points to {phys} [PHYS]. " + summary)
            fault = "Unknown"

        return Diagnosis(
            question=state.question, fault_type=fault,
            confidence=ml.confidence if ml else 0.0,
            summary=summary.strip(), explanation=expl.strip(), recommended_actions=[a for a in actions if a],
            evidence=ev, ml_result=ml, fault_frequencies=ff, insufficient_evidence=insufficient,
            warnings=[redact(w) for w in dict.fromkeys(state.warnings)], citation_coverage=coverage,
            grounding_score=grounding, answer_source=answer_source,
            tool_trace=state.trace, llm_provider=self.llm.provider, llm_model=self.llm.model,
            config_name=self.config_name,
        )


class _State:
    def __init__(self, question: str):
        self.question = question
        self.signal: np.ndarray | None = None
        self.rpm: float = DEFAULT_RPM
        self.ml: ClassifyResult | None = None
        self.ff: FaultFrequencies | None = None
        self.evidence: list[EvidenceChunk] = []
        self.trace: list[ToolCall] = []
        self.warnings: list[str] = []
        self.queries: list[str] = []
        self.done: set[str] = set()
        self.relevance: float = 0.0
        self.nq = nlu.understand(question)


def _summarise(out) -> str:
    if isinstance(out, ClassifyResult):
        return f"{out.label} ({out.fault_type}), confidence {out.confidence:.2f}, anomaly {out.anomaly_score:.2f}"
    if isinstance(out, FaultFrequencies):
        s = f"BPFO {out.bpfo_hz:.1f} Hz, BPFI {out.bpfi_hz:.1f} Hz, BSF {out.bsf_hz:.1f} Hz"
        if out.envelope_match:
            s += f"; physics -> {out.physics_suggests or 'none'} ({out.consistency})"
        return s
    if isinstance(out, list):
        return f"{len(out)} chunks: " + ", ".join(f"{e.source} p.{e.page}" for e in out[:5])
    return str(out)[:200]


def diagnose(question: str, signal: np.ndarray | None = None, rpm: float | None = None, **kwargs) -> Diagnosis:
    """Convenience one-liner used by scripts and notebooks."""
    return DiagnosisAgent(**kwargs).run(question, signal=signal, rpm=rpm)
