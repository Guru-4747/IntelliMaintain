"""Keep API keys out of everything a user, a log or GitHub can see.

* `redact()` removes Google API keys (pattern `AIza...`) and the currently configured key from any text
  before it is shown in the app, written to a warning, a report or an exception message.
* `scan_for_keys()` finds keys accidentally written into project files (used by scripts/check_secrets.py,
  scripts/prepare_deploy.py and a test) so a key can never be committed to GitHub.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
KEY_PARAM_RE = re.compile(r"(?i)((?:api[_-]?key|key|token)=)[^&\s\"']+")
KEY_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
MASK = "***hidden***"


def _configured_keys() -> list[str]:
    return [v for v in (os.getenv(n, "") for n in KEY_ENV_NAMES) if len(v) >= 8]


def redact(text, extra_secrets: list[str] | None = None) -> str:
    """Return `text` with every API key replaced by ***hidden***."""
    s = str(text)
    for secret in _configured_keys() + [x for x in (extra_secrets or []) if x and len(x) >= 8]:
        s = s.replace(secret, MASK)
    s = GOOGLE_KEY_RE.sub(MASK, s)
    return KEY_PARAM_RE.sub(lambda m: m.group(1) + MASK, s)


def mask_key(key: str | None) -> str:
    """Show only that a key exists, e.g. 'AIza...7Qx' -> '••••••••7Qx' (never the full value)."""
    if not key:
        return "not set"
    return "••••••••" + key[-3:] if len(key) > 6 else "••••"


SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "raw", "processed", "pdf"}
TEXT_EXT = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".env",
            ".example", ".sh", ".html", ".js"}


def _candidate_files(root: Path) -> list[Path]:
    try:  # prefer exactly what git would publish
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return [root / f for f in out.stdout.splitlines()]
    except Exception:  # noqa: BLE001 - git not installed / not a repo
        pass
    return [p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS)]


def scan_for_keys(root: Path) -> list[tuple[str, int]]:
    """List (file, line) of every Google API key found in files that would be pushed to GitHub."""
    hits = []
    for f in _candidate_files(root):
        if f.suffix.lower() not in TEXT_EXT and f.name not in (".env", "secrets.toml"):
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if GOOGLE_KEY_RE.search(line):
                    hits.append((str(f.relative_to(root)), i))
        except OSError:
            continue
    return hits
