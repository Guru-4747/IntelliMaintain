import json

import pytest

from conftest import needs_index, needs_model, synthetic_bearing_signal
from intellimaintain import config
from intellimaintain.agent.agent import DiagnosisAgent, ml_informed_query
from intellimaintain.agent.llm import LLM, LLMError, parse_json
from intellimaintain.agent.verify import citation_coverage, clean_invalid_citations, split_sentences
from intellimaintain.schemas import Diagnosis

pytestmark = [needs_model, needs_index]

GOOD_REPLY = json.dumps({
    "fault_type": "Outer race",
    "summary": "Outer race fault is likely [ML][1].",
    "explanation": "The classifier predicts an outer race fault [ML]. Envelope peaks at BPFO confirm it [PHYS]. "
                   "Outer race defects show BPFO harmonics [1]. This sentence cites a missing source [9].",
    "recommended_actions": ["Verify the BPFO peak [1].", "Plan replacement at the next stop [2]."],
    "insufficient_evidence": False,
})


# ------------------------------------------------------------------ helpers / verification
def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": {"b": 2}} hope it helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_citation_cleaning_and_coverage():
    text, removed = clean_invalid_citations("A [1]. B [7]. C [ML]. D [PHYS].", n_evidence=2, has_ml=True,
                                            has_phys=False)
    assert removed == ["[7]", "[PHYS]"] and "[1]" in text and "[ML]" in text
    assert citation_coverage(["One claim [1]. Another claim without citation."]) == 0.5
    assert len(split_sentences("Kurtosis is high. [2] BPFO is visible [1].")) == 2


def test_ml_informed_query():
    from intellimaintain.schemas import ClassifyResult

    ml = ClassifyResult(label="OR_007", fault_type="Outer race", confidence=0.9, key_features={"kurtosis": 7.0})
    q = ml_informed_query("what now?", ml)
    assert "outer race fault" in q and "high kurtosis" in q
    assert ml_informed_query("q", None) == "q"


def test_unknown_provider_and_missing_key(monkeypatch):
    with pytest.raises(ValueError):
        LLM("not-a-provider")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError):
        LLM("gemini", api_key=None).chat("s", "u")


# ------------------------------------------------------------------ offline agent end-to-end
@pytest.mark.parametrize("use_ml,retrieval", [(True, "improved"), (True, "basic"), (True, "none"),
                                              (False, "improved"), (False, "none")])
def test_agent_all_configs_produce_valid_diagnosis(use_ml, retrieval, demo_signal):
    d = DiagnosisAgent(LLM("offline"), use_ml=use_ml, retrieval=retrieval).run(
        "What is wrong with this bearing?", signal=demo_signal, rpm=1797)
    assert isinstance(d, Diagnosis)
    assert d.fault_type in {"Normal", "Inner race", "Outer race", "Ball", "Unknown"}
    assert len(d.tool_trace) <= config.MAX_AGENT_STEPS
    assert (d.ml_result is not None) == use_ml
    assert (len(d.evidence) > 0) == (retrieval != "none")
    Diagnosis.model_validate_json(d.model_dump_json())  # serialisable


def test_agent_offline_answer_is_fully_cited(demo_signal):
    d = DiagnosisAgent(LLM("offline")).run("Diagnose and recommend actions", signal=demo_signal, rpm=1797)
    assert d.citation_coverage >= 0.99 and d.explanation and d.summary
    assert [t.tool for t in d.tool_trace][:3] == ["classify_signal", "fault_frequency_calc", "retrieve_evidence"]


def test_agent_out_of_scope_question_is_refused():
    d = DiagnosisAgent(LLM("offline"), use_ml=False).run("What is the best recipe for chicken biryani?")
    assert d.insufficient_evidence and d.fault_type == "Unknown"


def test_agent_step_limit(demo_signal):
    d = DiagnosisAgent(LLM("offline"), max_steps=1).run("q", signal=demo_signal, rpm=1797)
    assert len(d.tool_trace) == 1 and any("Step limit" in w for w in d.warnings)


def test_conflict_policy_abstains_on_wrong_label():
    from intellimaintain.ml.physics import defect_frequencies

    x = synthetic_bearing_signal(defect_frequencies(1797)["bpfo_hz"], seconds=3)
    wrong = DiagnosisAgent(LLM("offline")).run("q", signal=x, rpm=1797, override_label="IR_014")
    assert wrong.fault_frequencies.consistency == "conflict"
    assert wrong.fault_type == "Unknown" and "verification" in wrong.summary
    trusting = DiagnosisAgent(LLM("offline"), conflict_policy="trust_ml").run("q", signal=x, rpm=1797,
                                                                             override_label="IR_014")
    assert trusting.fault_type == "Inner race"


# ------------------------------------------------------------------ real LLM providers (mock HTTP server)
def test_gemini_request_format(mock_llm_server):
    url, handler = mock_llm_server
    handler.reply_text = '{"ok": true}'
    out = LLM("gemini", model="gemini-3.5-flash-lite", base_url=url, api_key="test-key").chat(
        "system text", "user text", json_mode=True)
    assert parse_json(out) == {"ok": True}  # the "thought" part is ignored
    req = handler.requests[-1]
    assert req["path"] == "/v1beta/models/gemini-3.5-flash-lite:generateContent"
    assert req["headers"]["x-goog-api-key"] == "test-key"
    assert req["body"]["systemInstruction"]["parts"][0]["text"] == "system text"
    assert req["body"]["contents"][0]["parts"][0]["text"] == "user text"
    assert req["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_retries_after_rate_limit(mock_llm_server):
    url, handler = mock_llm_server
    handler.reply_text = "OK"
    handler.errors = [(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0s"}]}})]
    assert LLM("gemini", base_url=url, api_key="k").chat("s", "u") == "OK"
    assert len(handler.requests) == 2


@pytest.mark.parametrize("code,payload,expected", [
    (404, {"error": {"code": 404}}, "not available"),
    (429, {"error": {"code": 429, "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}},
     "daily quota"),
    (400, {"error": {"code": 400, "message": "API key not valid"}}, "rejected"),
])
def test_gemini_errors_are_explained(mock_llm_server, code, payload, expected):
    url, handler = mock_llm_server
    handler.errors = [(code, payload)]
    with pytest.raises(LLMError, match=expected):
        LLM("gemini", base_url=url, api_key="k").chat("s", "u")


def test_quota_exhausted_falls_back_to_offline(mock_llm_server, demo_signal):
    url, handler = mock_llm_server
    handler.errors = [(429, {"error": {"code": 429, "details": [{"quotaId": "PerDay"}]}})]
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k")).run("q", signal=demo_signal, rpm=1797)
    assert d.explanation and any("offline extractive generator" in w for w in d.warnings)


def test_agent_with_llm_cleans_invalid_citations(mock_llm_server, demo_signal):
    url, handler = mock_llm_server
    handler.reply_text = GOOD_REPLY
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k")).run("What is wrong?", signal=demo_signal, rpm=1797)
    assert d.llm_provider == "gemini"
    assert "[9]" not in d.explanation and any("invalid citation" in w for w in d.warnings)
    assert d.recommended_actions and d.summary.startswith("Outer race") or d.fault_type == "Unknown"
    # the grounded prompt was sent with numbered evidence and the tool outputs
    user_msg = handler.requests[-1]["body"]["contents"][0]["parts"][0]["text"]
    assert "[ML] Machine-learning classifier output" in user_msg and "[1] (source:" in user_msg


def test_agent_falls_back_when_llm_returns_garbage(mock_llm_server, demo_signal):
    url, handler = mock_llm_server
    handler.reply_text = "I am not JSON at all"
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k")).run("q", signal=demo_signal, rpm=1797)
    assert any("offline extractive generator" in w for w in d.warnings) and d.explanation


def test_llm_planner_is_bounded(mock_llm_server, demo_signal):
    """A planner that keeps asking for the same tool must not loop or exceed the step limit."""
    url, handler = mock_llm_server
    handler.reply_text = json.dumps({"tool": "retrieve_evidence", "args": {"query": "outer race"},
                                     "fault_type": "Outer race", "summary": "s [1].", "explanation": "e [1].",
                                     "recommended_actions": [], "insufficient_evidence": False})
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k"), planner="llm", max_steps=4).run(
        "q", signal=demo_signal, rpm=1797)
    assert len(d.tool_trace) <= 4
    assert {"classify_signal", "fault_frequency_calc", "retrieve_evidence"} <= {t.tool for t in d.tool_trace}


def test_llm_unreachable_falls_back(demo_signal):
    d = DiagnosisAgent(LLM("gemini", base_url="http://127.0.0.1:9", api_key="k", timeout=2, max_retries=1)).run(
        "q", signal=demo_signal, rpm=1797)
    assert d.explanation and any("offline extractive generator" in w for w in d.warnings)


def test_agent_with_cnn_classifier(demo_signal):
    from intellimaintain import config

    if not (config.MODELS_DIR / "cnn_classifier.pt").exists():
        pytest.skip("run scripts/train_cnn.py")
    d = DiagnosisAgent(LLM("offline"), classifier="cnn").run("What is wrong?", signal=demo_signal, rpm=1797)
    assert d.ml_result.model_name.startswith("Spectrogram CNN")
    assert d.tool_trace[0].arguments["model"] == "cnn"
    with pytest.raises(ValueError):
        DiagnosisAgent(LLM("offline"), classifier="svm")


# ------------------------------------------------------------------ grounding guard (online answers)
def _grounded_reply(demo_signal, **override):
    """A realistic LLM reply: built from the verified offline answer for the same input."""
    d0 = DiagnosisAgent(LLM("offline")).run("What is wrong?", signal=demo_signal, rpm=1797)
    reply = {"fault_type": d0.fault_type if d0.fault_type != "Unknown" else d0.ml_result.fault_type,
             "summary": d0.summary, "explanation": d0.explanation,
             "recommended_actions": d0.recommended_actions, "insufficient_evidence": False}
    reply.update(override)
    return d0, reply


def test_verify_sentences_checks_words_and_numbers():
    from intellimaintain.agent.verify import verify_sentences

    src = {"1": "Inner race faults show BPFI harmonics with 1x rpm sidebands; kurtosis 7.94 indicates impacts."}
    ok, bad = verify_sentences([
        "Inner race faults show BPFI harmonics with sidebands [1].",
        "Kurtosis 7.9 indicates impacts [1].",                       # rounding tolerated
        "Kurtosis 12.5 indicates impacts [1].",                      # invented number
        "Replace the gearbox with a titanium housing [1].",          # not in the source
        "Inner race faults show BPFI harmonics.",                    # no citation
    ], src)
    assert len(ok) == 2 and len(bad) == 3


def test_grounded_llm_answer_is_kept_and_unsupported_sentence_removed(mock_llm_server, demo_signal):
    url, handler = mock_llm_server
    d0, reply = _grounded_reply(demo_signal)
    reply["explanation"] += " The bearing must be replaced with a titanium alloy rated for 9000 rpm [1]."
    handler.reply_text = json.dumps(reply)
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k")).run("What is wrong?", signal=demo_signal, rpm=1797)
    assert d.answer_source == "gemini (verified)"
    assert "titanium" not in d.explanation and "9000" not in d.explanation
    assert any("unverified sentence" in w for w in d.warnings)
    assert d.grounding_score >= 0.5


def test_llm_cannot_change_the_data_based_fault_type(mock_llm_server, demo_signal):
    url, handler = mock_llm_server
    d0, reply = _grounded_reply(demo_signal)
    wrong = "Inner race" if d0.ml_result.fault_type != "Inner race" else "Outer race"
    handler.reply_text = json.dumps({**reply, "fault_type": wrong})
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k"), conflict_policy="trust_ml").run(
        "What is wrong?", signal=demo_signal, rpm=1797)
    assert d.fault_type == d.ml_result.fault_type
    assert any("replaced by the data-based result" in w for w in d.warnings)


def test_hallucinated_llm_answer_falls_back_to_verified_offline_answer(mock_llm_server, demo_signal):
    url, handler = mock_llm_server
    handler.reply_text = json.dumps({
        "fault_type": "Outer race", "summary": "The pump is fine [1].",
        "explanation": "Our records show the motor was serviced by Siemens in 2019 [1]. "
                       "Vibration is caused by cosmic rays at 42 GHz [2].",
        "recommended_actions": ["Call the manufacturer hotline 1800-555 [1]."], "insufficient_evidence": False})
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k")).run("What is wrong?", signal=demo_signal, rpm=1797)
    assert "failed verification" in d.answer_source
    assert "Siemens" not in d.explanation and "cosmic" not in d.explanation and d.explanation
    assert d.citation_coverage >= 0.99


def test_offline_answer_uses_similar_dataset_cases(demo_signal):
    d = DiagnosisAgent(LLM("offline")).run("What is wrong?", signal=demo_signal, rpm=1797)
    assert d.ml_result.similar_cases and len(d.ml_result.similar_cases) == 3
    assert "CWRU dataset" in d.explanation


def test_full_evaluation_runs_end_to_end(monkeypatch, tmp_path):
    """Smoke test of scripts/run_eval.py on 3 cases (catches wiring errors the unit tests cannot)."""
    from intellimaintain import config
    from intellimaintain.evaluation import evaluate

    cases = evaluate.load_cases()
    small = cases[cases.case_id.isin(["TC02", "TC12", "TC20"])].reset_index(drop=True)
    monkeypatch.setattr(evaluate, "load_cases", lambda *a, **k: small)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    for clf in ("rf", "cnn"):
        if clf == "cnn" and not (config.MODELS_DIR / "cnn_classifier_crossload.pt").exists():
            continue
        out = evaluate.run_all("offline", classifier=clf, verbose=False)
        assert out["summary"] and (tmp_path / "eval_results.md").exists()
