"""LLM backend: Google Gemini (free API tier) with an offline fallback.

  gemini   Google Gemini via the official REST API (free key from https://aistudio.google.com/apikey)
           default model gemini-3.5-flash-lite (fast, free); gemini-3.8-flash gives richer answers.
  offline  no LLM: deterministic, extractive, citation-safe generator. Used automatically when no key is set,
           when the free quota is used up, or when Gemini is unreachable - the app never breaks.

Free-tier friendly: requests are paced (IM_LLM_RPM, default 8 per minute) and HTTP 429 "quota" answers are
retried after the delay Gemini asks for. Only `requests` is needed - no vendor SDK.
"""
from __future__ import annotations

import os
import re
import json
import threading
import time

from .. import config

DEFAULT_MODELS = {"gemini": "gemini-3.5-flash-lite", "offline": "extractive-v1"}
SUGGESTED_GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
PROVIDERS = list(DEFAULT_MODELS)
GEMINI_URL = "https://generativelanguage.googleapis.com"
_PACE_LOCK = threading.Lock()
_LAST_CALL = [0.0]


class LLMError(RuntimeError):
    """Error whose message can never contain an API key (redacted on creation)."""

    def __init__(self, message: str = ""):
        from ..security import redact

        super().__init__(redact(message))


def gemini_key_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float = config.LLM_TEMPERATURE, timeout: int = config.LLM_TIMEOUT,
                 base_url: str | None = None, api_key: str | None = None, rpm: float | None = None,
                 max_retries: int = 5):
        prov = (provider or config.LLM_PROVIDER or "offline").lower()
        if prov not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{prov}'. Choose from {PROVIDERS}")
        self.provider = prov
        self.model = model or (config.LLM_MODEL if provider is None else "") or DEFAULT_MODELS[prov]
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = base_url
        self.api_key = api_key
        self.rpm = float(os.getenv("IM_LLM_RPM", "8")) if rpm is None else float(rpm)
        self.max_retries = max(1, int(max_retries))
        self.calls = 0
        self.total_seconds = 0.0

    @property
    def is_offline(self) -> bool:
        return self.provider == "offline"

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"

    def _key(self) -> str:
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey and "
                           "put it in .env (local) or in the app's Secrets (online).")
        return key

    def _pace(self) -> None:
        """Keep under the free-tier requests-per-minute limit (shared by all LLM objects in the process)."""
        if self.rpm <= 0 or self.base_url:  # no pacing for local test servers
            return
        gap = 60.0 / self.rpm
        with _PACE_LOCK:
            wait = _LAST_CALL[0] + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.time()

    # ------------------------------------------------------------------ main entry
    def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 2048) -> str:
        if self.is_offline:
            raise LLMError("offline provider has no chat(); the agent uses the extractive generator instead")
        import requests

        t0 = time.time()
        last: Exception | None = None
        for attempt in range(self.max_retries):
            self._pace()
            try:
                text = self._gemini(requests, system, user, json_mode, max_tokens)
                self.calls += 1
                self.total_seconds += time.time() - t0
                return text
            except LLMError:
                raise
            except requests.HTTPError as e:
                resp = e.response
                code = resp.status_code if resp is not None else 0
                body = resp.text[:400] if resp is not None else ""
                if code == 429:  # free-tier quota: wait as long as Gemini asks, then retry
                    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', body)
                    delay = min(float(m.group(1)) + 1 if m else 10.0 * (attempt + 1), 65.0)
                    if "PerDay" in body or "per day" in body.lower():
                        raise LLMError("Gemini free daily quota used up (resets at midnight Pacific time). "
                                       "The offline generator is used meanwhile.") from e
                    last = LLMError(f"Gemini rate limit (429); retried after {delay:.0f}s")
                    time.sleep(delay)
                    continue
                if code == 404:
                    raise LLMError(f"Gemini model '{self.model}' is not available for this key. Try one of: "
                                   f"{', '.join(SUGGESTED_GEMINI_MODELS)}") from e
                if code in (400, 401, 403):
                    raise LLMError(f"Gemini rejected the request (HTTP {code}): {body}") from e
                last = LLMError(f"Gemini HTTP {code}: {body}")
            except requests.RequestException as e:
                last = LLMError(f"Could not reach Gemini: {e}")
            time.sleep(1.5 * (attempt + 1))
        raise last or LLMError("Gemini request failed")

    def _gemini(self, requests, system: str, user: str, json_mode: bool, max_tokens: int) -> str:
        url = f"{(self.base_url or GEMINI_URL).rstrip('/')}/v1beta/models/{self.model}:generateContent"
        gen = {"temperature": self.temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen}
        r = requests.post(url, json=body, timeout=self.timeout,
                          headers={"x-goog-api-key": self._key(), "Content-Type": "application/json"})
        r.raise_for_status()
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no answer ({reason})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMError(f"Gemini returned an empty answer (finishReason={cands[0].get('finishReason')})")
        return text

    def health_check(self) -> tuple[bool, str]:
        """Connectivity test used by the app sidebar."""
        if self.is_offline:
            return True, "Offline extractive generator (no LLM needed)"
        try:
            out = self.chat("You are a connectivity test.", "Reply with the single word OK.", max_tokens=512)
            return True, f"Connected to {self.describe()} (reply: {out.strip()[:20]})"
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:300]


def parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM reply (handles ```json fences and chatter)."""
    if not text:
        raise ValueError("empty LLM reply")
    t = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise ValueError("no JSON object found in LLM reply")
