"""The Gemini API key must never be visible: not in the UI, errors, warnings, reports or GitHub."""
from pathlib import Path

import pytest

from conftest import needs_index, needs_model
from intellimaintain.agent.llm import LLM, LLMError
from intellimaintain.security import MASK, mask_key, redact, scan_for_keys

ROOT = Path(__file__).resolve().parent.parent
FAKE_KEY = "AIza" + "SyD3moFakeKeyForTests_0123456789abc"  # same shape as a real Google key (39 chars)


def test_redact_removes_google_keys_and_configured_key(monkeypatch):
    assert len(FAKE_KEY) == 39
    assert FAKE_KEY not in redact(f"error calling ...?key={FAKE_KEY} failed")
    assert MASK in redact(f"bad key {FAKE_KEY}")
    monkeypatch.setenv("GEMINI_API_KEY", "my-custom-secret-value-123")
    assert "my-custom-secret-value-123" not in redact("leak: my-custom-secret-value-123")
    assert mask_key(None) == "not set" and "SyD3" not in mask_key(FAKE_KEY)


def test_llm_errors_never_contain_the_key():
    err = LLMError(f"Gemini rejected request with key {FAKE_KEY}")
    assert FAKE_KEY not in str(err)


def test_key_is_sent_in_a_header_not_in_the_url(mock_llm_server):
    url, handler = mock_llm_server
    handler.reply_text = "OK"
    LLM("gemini", base_url=url, api_key=FAKE_KEY).chat("s", "u")
    req = handler.requests[-1]
    assert FAKE_KEY not in req["path"]                      # URLs end up in logs and error messages
    assert req["headers"]["x-goog-api-key"] == FAKE_KEY


@needs_model
@needs_index
def test_key_echoed_by_server_is_hidden_in_diagnosis(mock_llm_server, demo_signal, monkeypatch):
    from intellimaintain.agent.agent import DiagnosisAgent

    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    url, handler = mock_llm_server
    handler.errors = [(400, {"error": {"message": f"API key not valid: {FAKE_KEY}"}})]
    d = DiagnosisAgent(LLM("gemini", base_url=url)).run("q", signal=demo_signal, rpm=1797)
    assert FAKE_KEY not in d.model_dump_json()               # warnings, report download, review log
    assert any(MASK in w for w in d.warnings)


def test_no_api_key_is_written_in_any_publishable_file():
    assert scan_for_keys(ROOT) == []


def test_scanner_finds_a_key_in_a_file(tmp_path):
    (tmp_path / "notes.md").write_text(f"my key is {FAKE_KEY}\n")
    (tmp_path / "app.py").write_text("x = 1\n")
    assert scan_for_keys(tmp_path) == [("notes.md", 1)]


def test_env_and_secrets_files_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore and ".streamlit/secrets.toml" in ignore


@needs_model
@needs_index
def test_interface_has_no_api_key_field(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120).run()
    assert not at.exception
    labels = [w.label.lower() for w in list(at.text_input) + list(at.text_area)]
    assert not any("key" in lab for lab in labels)           # nowhere to paste or reveal a key
    rendered = " ".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.caption) + list(at.info)
                        + list(at.warning) + list(at.error) + list(at.success))
    assert FAKE_KEY not in rendered and "configured" in rendered
