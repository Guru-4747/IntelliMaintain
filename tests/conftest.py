import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intellimaintain import config  # noqa: E402

HAS_MODEL = (config.MODELS_DIR / "classifier.joblib").exists()
HAS_INDEX = (config.INDEX_DIR / "faiss.index").exists()
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="run scripts/train_ml.py first")
needs_index = pytest.mark.skipif(not HAS_INDEX, reason="run scripts/build_index.py first")


@pytest.fixture(autouse=True)
def _offline_by_default(monkeypatch):
    """Tests never use a real Gemini key: hide any key from .env so results are reproducible and free.
    Tests that need Gemini start the local mock server or set a fake key themselves."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "IM_LLM_PROVIDER", "IM_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "offline")
    monkeypatch.setattr(config, "LLM_MODEL", "")


def synthetic_bearing_signal(fault_hz: float = 107.4, seconds: float = 2.0, fs: int = 12000, seed: int = 0):
    """Impulses at fault_hz exciting a 3 kHz resonance + noise (a textbook outer-race signal)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = 0.05 * rng.standard_normal(n)
    t_imp = np.arange(0, seconds, 1 / fault_hz)
    ring = np.exp(-np.arange(200) / 25) * np.sin(2 * np.pi * 3000 * np.arange(200) / fs)
    for ti in t_imp:
        i = int(ti * fs)
        seg = ring[: max(0, min(200, n - i))]
        x[i:i + len(seg)] += seg
    return x


@pytest.fixture
def demo_signal():
    return synthetic_bearing_signal()


class _Handler(BaseHTTPRequestHandler):
    """Imitates the Gemini REST endpoint (v1beta/models/<model>:generateContent)."""
    reply_text = "{}"
    requests: list = []
    errors: list = []  # queued (status, body) responses returned before the normal reply

    def log_message(self, *a):  # silence
        pass

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
        if not self.path.endswith(":generateContent"):
            self._send(404, {"error": {"code": 404, "message": "model not found"}})
            return
        if _Handler.errors:
            code, payload = _Handler.errors.pop(0)
            self._send(code, payload)
            return
        self._send(200, {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "internal reasoning", "thought": True}, {"text": _Handler.reply_text}]},
            "finishReason": "STOP"}]})


@pytest.fixture
def mock_llm_server(monkeypatch):
    """Local HTTP server that imitates the Gemini REST API."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    _Handler.requests = []
    _Handler.errors = []
    yield f"http://127.0.0.1:{server.server_port}", _Handler
    server.shutdown()
