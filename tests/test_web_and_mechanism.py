"""Online evidence (web_search tool), the 'how this fault occurs' explanation and the CWRU class reference."""
import json

import pytest

from conftest import needs_index, needs_model
from intellimaintain import config
from intellimaintain.agent import extractive, tools
from intellimaintain.agent.agent import DiagnosisAgent
from intellimaintain.agent.llm import LLM
from intellimaintain.agent.prompts import format_evidence, format_ml
from intellimaintain.rag import web
from intellimaintain.schemas import EvidenceChunk

pytestmark = [needs_model, needs_index]

WEB_CHUNK = EvidenceChunk(chunk_id="web-1", source="Wikipedia: Rolling-element bearing", page=1,
                          text="A rolling-element bearing carries a load by placing rolling elements between two "
                               "races. Spalling of the raceway is a common failure mode caused by fatigue.",
                          retriever="web", title="Rolling-element bearing",
                          url="https://en.wikipedia.org/wiki/Rolling-element_bearing")


# ------------------------------------------------------------------ web module (no internet needed)
class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_web_search_parses_wikipedia(monkeypatch):
    long_para = ("Spalling is a fatigue failure of the raceway surface in a rolling bearing. Repeated contact "
                 "stress starts a crack below the surface that grows until a flake of metal breaks out. ") * 2
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(params)
        assert "IntelliMaintain" in headers["User-Agent"]
        if params.get("list") == "search":
            return _FakeResp({"query": {"search": [{"title": "Spall"}, {"title": "Banana"}]}})
        return _FakeResp({"query": {"pages": [
            {"title": "Spall", "extract": "Intro line too short.\n\n== Causes ==\n" + long_para},
            {"title": "Banana", "extract": "A banana is an elongated, edible fruit produced by several kinds of large "
                                          "herbaceous flowering plants in the genus Musa, eaten fresh worldwide."}]}})

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    out = web.web_search("bearing spalling fatigue crack", top_k=2)
    assert out and out[0].source == "Wikipedia: Spall" and out[0].retriever == "web"
    assert out[0].url == "https://en.wikipedia.org/wiki/Spall" and "fatigue" in out[0].text
    assert all("banana" not in e.text.lower() for e in out)  # irrelevant pages are ranked out (no shared terms)
    assert len(calls) == 2


def test_web_search_failure_raises_clean_error(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(web.WebSearchError, match="unavailable"):
        web.web_search("bearing")


def test_web_query_is_short_and_on_topic():
    q = web.web_query("my pump bearing is making a grinding noise what should i do about it today please",
                      "Outer race")
    assert q.startswith("outer race bearing fault") and len(q.split()) <= 12


# ------------------------------------------------------------------ agent integration
def test_web_off_never_calls_the_tool(monkeypatch, demo_signal):
    monkeypatch.setattr(tools, "web_search", lambda q, **k: pytest.fail("web_search must not run"))
    d = DiagnosisAgent(LLM("offline"), web="off").run("What is wrong?", signal=demo_signal, rpm=1797)
    assert "web_search" not in [t.tool for t in d.tool_trace] and d.web_search == "off"


def test_web_always_adds_cited_online_evidence(monkeypatch, demo_signal):
    monkeypatch.setattr(tools, "web_search", lambda q, **k: [WEB_CHUNK])
    d = DiagnosisAgent(LLM("offline"), web="always").run("What is wrong?", signal=demo_signal, rpm=1797)
    assert "web_search" in [t.tool for t in d.tool_trace]
    assert d.evidence[-1].url and d.evidence[-1].retriever == "web"  # numbered after the documents
    assert d.config_name.endswith("+web")
    assert "online source https://en.wikipedia.org" in format_evidence(d.evidence)


def test_web_auto_answers_questions_the_documents_cannot(monkeypatch):
    monkeypatch.setattr(tools, "web_search", lambda q, **k: [WEB_CHUNK])
    d = DiagnosisAgent(LLM("offline"), web="auto").run("who was leonardo da vinci")
    assert any(e.retriever == "web" for e in d.evidence)


def test_web_unavailable_is_a_warning_not_a_crash(monkeypatch, demo_signal):
    def offline(q, **k):
        raise web.WebSearchError("online search unavailable (ConnectionError)")

    monkeypatch.setattr(tools, "web_search", offline)
    d = DiagnosisAgent(LLM("offline"), web="always").run("What is wrong?", signal=demo_signal, rpm=1797)
    assert d.explanation and any("unavailable" in w for w in d.warnings)


def test_web_mode_is_validated():
    with pytest.raises(ValueError):
        DiagnosisAgent(LLM("offline"), web="sometimes")


# ------------------------------------------------------------------ fault mechanism
def test_diagnosis_explains_how_the_fault_occurs(demo_signal):
    d = DiagnosisAgent(LLM("offline")).run("What is wrong and why?", signal=demo_signal, rpm=1797)
    assert d.fault_mechanism, "a fault diagnosis must explain how the fault develops"
    assert "[" in d.fault_mechanism and len(d.tool_trace) <= config.MAX_AGENT_STEPS
    if d.fault_type == "Outer race":
        assert "BPFI" not in d.fault_mechanism  # no sentences about another fault type


def test_normal_bearing_gets_no_fault_mechanism():
    from intellimaintain.ml.signal_io import load_signal

    s = load_signal(config.SAMPLES_DIR / "normal_3hp.csv")
    d = DiagnosisAgent(LLM("offline")).run("What is the condition of this bearing and why?", signal=s.x, fs=s.fs,
                                            rpm=s.rpm)
    assert d.fault_type == "Normal" and d.fault_mechanism == ""
    assert [t.tool for t in d.tool_trace].count("retrieve_evidence") == 1  # no mechanism search either


def test_mechanism_question_without_signal():
    d = DiagnosisAgent(LLM("offline")).run("How does an inner race fault occur?")
    assert d.fault_mechanism and "inner" in d.fault_mechanism.lower()
    assert "BPFO" not in d.fault_mechanism


def test_mechanism_sentences_stay_on_the_fault_type():
    ev = [EvidenceChunk(chunk_id="a", source="kb11.md", page=5, title="How Bearing Faults Occur",
                        text="The outer ring is normally stationary in the housing, so an outer race defect stays at "
                             "the same position. The inner ring rotates with the shaft, so an inner race defect "
                             "travels around with it.")]
    out = extractive.mechanism("Outer race", ev)
    assert "outer race defect stays" in out and "inner ring" not in out and out.endswith("[1].")


def test_llm_answer_gets_a_verified_mechanism(mock_llm_server, demo_signal):
    """If Gemini leaves fault_mechanism empty, the verified extractive mechanism is used instead."""
    url, handler = mock_llm_server
    handler.reply_text = json.dumps({
        "fault_type": "Outer race", "summary": "The classifier predicts an outer race fault [ML].",
        "explanation": "The classifier predicts an outer race fault [ML].", "fault_mechanism": "",
        "recommended_actions": [], "insufficient_evidence": False})
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k")).run("What is wrong?", signal=demo_signal, rpm=1797)
    sent = handler.requests[-1]["body"]["systemInstruction"]["parts"][0]["text"]
    assert '"fault_mechanism"' in sent
    if d.ml_result and d.ml_result.fault_type != "Normal":
        assert d.fault_mechanism


# ------------------------------------------------------------------ CWRU reference
def test_cwru_reference_compares_signal_with_its_class(demo_signal):
    from intellimaintain.ml.classifier import classify_signal

    r = classify_signal(demo_signal)
    ref = r.cwru_reference
    assert ref["cwru_class"] == r.label and ref["n_recordings"] == 4
    lo, hi = ref["kurtosis"]
    assert lo <= hi
    text = format_ml(r)
    assert f"Typical CWRU values for class {r.label}" in text and "the CWRU range" in text
