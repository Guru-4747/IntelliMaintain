"""Questions typed casually (typos, short forms, everyday words) must be understood."""
import pytest

from conftest import needs_index, needs_model
from intellimaintain.agent.nlu import understand


@pytest.mark.parametrize("text,has_signal,intent", [
    ("hiii", False, "greeting"), ("hello there!", False, "greeting"), ("good morning", True, "greeting"),
    ("thx a lot", False, "thanks"), ("what can u do", False, "help"), ("", False, "help"), ("", True, "diagnosis"),
    ("yo my bering is makin weird grindin noise wat shud i do??", False, "knowledge"),
    ("how do i grease it pls", False, "knowledge"), ("kurtosis 8 means wat", False, "knowledge"),
    ("best chicken biryani recipe", False, "out_of_scope"), ("who won the match yesterday", False, "out_of_scope"),
    ("is ths bearing ok?", True, "diagnosis"), ("???", True, "diagnosis"),
    ("can i keep running it till friday", True, "diagnosis"), ("what is BPFO", True, "knowledge"),
])
def test_intent(text, has_signal, intent):
    assert understand(text, has_signal).intent == intent


def test_short_forms_spelling_and_domain_words():
    q = understand("yo my bering is makin weird grindin noise wat shud i do?? 😟")
    assert q.corrections.get("bering") == "bearing"
    assert "what" in q.cleaned and "should" in q.cleaned and "making" in q.cleaned
    assert "vibration" in q.search_query          # "grinding noise" -> vibration
    assert "lubrication" in understand("how often shud i oil it").search_query
    assert "BPFO" in understand("outer ring damaged").search_query
    assert "lockout" in understand("is it safe to open the machine").search_query


@needs_model
@needs_index
@pytest.mark.parametrize("text", ["hiii", "what can u do", "thanks!!"])
def test_small_talk_gets_a_friendly_reply_without_tools(text):
    from intellimaintain.agent.agent import DiagnosisAgent

    d = DiagnosisAgent().run(text)
    assert d.answer_source == "assistant" and d.summary and not d.tool_trace


@needs_model
@needs_index
def test_casual_questions_get_document_answers():
    from intellimaintain.agent.agent import DiagnosisAgent

    ag = DiagnosisAgent()
    d = ag.run("how do i grease it pls")
    assert not d.insufficient_evidence and "grease" in d.explanation.lower() and d.citation_coverage >= 0.99
    d = ag.run("is it safe to change the bearing without stopping the motor")
    assert "lockout" in d.explanation.lower()
    d = ag.run("whats the diff between inner n outer race fault")
    assert d.fault_type == "Unknown" and "BPFI" in d.explanation
    d = ag.run("best chicken biryani recipe")
    assert d.insufficient_evidence and not d.evidence


@needs_model
@needs_index
def test_casual_question_with_signal_is_a_diagnosis(demo_signal):
    from intellimaintain.agent.agent import DiagnosisAgent

    d = DiagnosisAgent().run("is ths bearing ok?", signal=demo_signal, rpm=1797)
    assert d.ml_result is not None and d.fault_type in ("Normal", "Inner race", "Outer race", "Ball", "Unknown")
    assert "CWRU dataset" in d.explanation


@needs_index
def test_casual_benchmark_language_layer_helps():
    from intellimaintain.evaluation.evaluate import casual_benchmark

    _, s = casual_benchmark()
    assert s["intent_accuracy"] >= 0.95
    assert s["hit@5_normalised"] >= s["hit@5_raw"]
