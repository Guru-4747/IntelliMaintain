"""Chat behaviour: every question gets its own answer, follow-ups use the conversation, the signal is not
re-classified on every turn."""
import json

import pytest

from conftest import needs_index, needs_model
from intellimaintain.agent import nlu
from intellimaintain.agent.agent import DiagnosisAgent
from intellimaintain.agent.llm import LLM

pytestmark = [needs_model, needs_index]


@pytest.mark.parametrize("question,focus", [
    ("What is the condition of this bearing, why, and what maintenance action is recommended?", "diagnosis"),
    ("what does the kurtosis value of this signal mean?", "indicators"),
    ("can i run it till friday", "severity"),
    ("which part of the bearing is damaged?", "location"),
    ("what are the defect frequencies?", "frequencies"),
    ("how does this compare with the CWRU dataset?", "cwru"),
    ("how confident are you", "model"),
    ("what grease should I use?", "lubrication"),
    ("what safety steps before I open it?", "safety"),
    ("how did this fault happen?", "mechanism"),
])
def test_question_focus(question, focus):
    assert nlu.understand(question).focus == focus


def test_different_questions_get_different_answers(demo_signal):
    agent = DiagnosisAgent(LLM("offline"))
    qs = ["what is wrong with this bearing?", "what grease should I use?", "what are the defect frequencies?",
          "how does this compare with the CWRU dataset?", "what safety steps before replacing it?"]
    answers = [agent.run(q, signal=demo_signal, rpm=1797).answer for q in qs]
    assert all(answers) and len(set(answers)) == len(answers)
    assert "BPFO" in answers[2] and "Hz" in answers[2]
    assert "CWRU" in answers[3]
    assert "grease" in answers[1].lower() or "lubric" in answers[1].lower()


def test_follow_up_uses_the_conversation_and_cached_diagnosis(demo_signal):
    agent = DiagnosisAgent(LLM("offline"))
    first = agent.run("what is wrong with this bearing?", signal=demo_signal, rpm=1797)
    history = [{"role": "user", "content": "what grease should I use?"},
               {"role": "assistant", "content": "Use the specified grease type [1]."}]
    d = agent.run("how often?", signal=demo_signal, rpm=1797, history=history,
                  ml=first.ml_result, ff=first.fault_frequencies)
    tools = [t.tool for t in d.tool_trace]
    assert "classify_signal" not in tools and "fault_frequency_calc" not in tools  # reused, not recomputed
    assert d.question_focus == "lubrication" and "lubric" in d.answer.lower()
    assert d.ml_result is first.ml_result


def test_unanswerable_question_gets_a_polite_reply():
    d = DiagnosisAgent(LLM("offline"), web="off").run("who won the cricket world cup")
    assert "won't guess" in d.answer and "Online search" in d.answer


def test_gemini_gets_the_conversation_and_focus(mock_llm_server, demo_signal):
    url, handler = mock_llm_server
    handler.reply_text = json.dumps({"fault_type": "Outer race", "answer": "", "summary": "s [ML].",
                                     "explanation": "The classifier predicts an outer race fault [ML].",
                                     "recommended_actions": [], "insufficient_evidence": False})
    history = [{"role": "user", "content": "is my bearing ok?"}, {"role": "assistant", "content": "No [ML]."}]
    d = DiagnosisAgent(LLM("gemini", base_url=url, api_key="k")).run(
        "what grease should I use?", signal=demo_signal, rpm=1797, history=history)
    user_msg = handler.requests[-1]["body"]["contents"][0]["parts"][0]["text"]
    assert "CONVERSATION SO FAR" in user_msg and "User: is my bearing ok?" in user_msg
    assert "asking about: lubrication" in user_msg
    assert d.answer  # empty Gemini reply -> the verified, question-focused offline reply is used
