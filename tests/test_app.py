"""Headless Streamlit test: the dashboard renders and a full diagnosis runs without exceptions."""
from pathlib import Path

import pytest

from conftest import needs_index, needs_model
from intellimaintain import config

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
pytestmark = [needs_model, needs_index,
              pytest.mark.skipif(not any(config.SAMPLES_DIR.glob("*.csv")), reason="run scripts/make_samples.py")]


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=120)


def test_app_renders():
    at = _app().run()
    assert not at.exception
    assert any("Bearing Fault Diagnosis Copilot" in m.value for m in at.markdown)


@pytest.mark.parametrize("sample_idx", [0, 1, 2, 3, 4, 5])
def test_app_full_diagnosis_on_every_demo_sample(sample_idx):
    at = _app().run()
    samples = at.selectbox[3] if len(at.selectbox) > 3 else None  # provider, retrieval, planner, sample
    sb = next(s for s in at.selectbox if s.label == "Sample")
    sb.select_index(sample_idx).run()
    next(b for b in at.button if "Run diagnosis" in b.label).click().run()
    assert not at.exception, at.exception
    headers = [h.value for h in at.subheader]
    assert "1 · Prediction" in headers and "3 · Evidence" in headers and "4 · Human review" in headers
    del samples


def test_app_question_only_and_review(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REVIEW_LOG", tmp_path / "reviews.csv")
    at = _app().run()
    at.radio[0].set_value("No signal (question only)").run()
    next(b for b in at.button if "Run diagnosis" in b.label).click().run()
    assert not at.exception
    next(b for b in at.button if "Save review" in b.label).click().run()
    assert not at.exception
    assert any("Review saved" in s.value for s in at.success)


def test_app_runs_with_cnn_classifier():
    if not (config.MODELS_DIR / "cnn_classifier.pt").exists():
        pytest.skip("run scripts/train_cnn.py")
    at = _app().run()
    clf = next(s for s in at.selectbox if s.label == "Classifier")
    clf.select_index(1).run()
    next(b for b in at.button if "Run diagnosis" in b.label).click().run()
    assert not at.exception
    assert any("Spectrogram CNN" in m.value for m in at.markdown)


def test_app_reads_gemini_key_from_online_secrets(monkeypatch):
    """On Streamlit Community Cloud the key comes from st.secrets; the app must pick Gemini automatically."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = _app()
    at.secrets["GEMINI_API_KEY"] = "test-key-from-secrets"
    at.run()
    assert not at.exception
    assert next(s for s in at.selectbox if s.label == "Provider").value == "gemini"
    assert any(s.label == "Gemini model" for s in at.selectbox)
    import os
    os.environ.pop("GEMINI_API_KEY", None)  # do not leak into other tests


def test_app_without_key_defaults_to_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = _app().run()
    assert not at.exception
    assert next(s for s in at.selectbox if s.label == "Provider").value == "offline"


def test_app_handles_casual_and_small_talk_questions():
    at = _app().run()
    at.radio[0].set_value("No signal (question only)").run()
    for q in ("hiii", "how do i grease it pls", "best chicken biryani recipe"):
        at.text_area[0].set_value(q).run()
        next(b for b in at.button if "Run diagnosis" in b.label).click().run()
        assert not at.exception, q


def test_no_secrets_banner_when_running_locally(monkeypatch):
    """Locally there is no secrets.toml (the key is in .env): the app must not show any error banner."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = _app().run()
    assert not at.exception
    assert not any("secrets" in e.value.lower() for e in at.error)


def test_results_dashboard_renders_with_and_without_results(tmp_path):
    """The results tab must work with full results, with old/partial results and with no results at all."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(APP).parent))
    import results_dashboard as rd
    from streamlit.testing.v1 import AppTest

    at = _app().run()  # full results (tabs render all content)
    assert not at.exception
    assert any("2 × 3 ablation" in m.value for m in at.markdown)

    # partial / old-format results: only a minimal ml_metrics.json, no CNN, no eval runs
    (tmp_path / "ml_metrics.json").write_text(json.dumps({"cross_load": {"RandomForest": {"accuracy": 1.0}}}))
    script = f"""
import sys
sys.path.insert(0, {str(Path(APP).parent)!r}); sys.path.insert(0, {str(Path(APP).parent.parent)!r})
from pathlib import Path
import results_dashboard as rd
rd.R = Path({str(tmp_path)!r}); rd.FIG = Path({str(tmp_path)!r})
rd.render()
"""
    at2 = AppTest.from_string(script, default_timeout=60).run()
    assert not at2.exception
    empty = tmp_path / "empty"
    empty.mkdir()
    at3 = AppTest.from_string(script.replace(str(tmp_path), str(empty)), default_timeout=60).run()
    assert not at3.exception
    assert rd.fmt(None) == "–" and rd.fmt(0.12345) == "0.123"
