"""Central configuration for IntelliMaintain.

Every path and hyper-parameter lives here so experiments are reproducible.
Values can be overridden with environment variables (see README) or a `.env` file.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# macOS / Anaconda: PyTorch and FAISS each ship their own OpenMP runtime; loading both can crash the
# process silently (segfault) or abort with "OMP: Error #15". These settings make them coexist safely.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __import__("sys").platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# ----------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORPUS_DIR = DATA_DIR / "corpus"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
INDEX_DIR = MODELS_DIR / "rag_index"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVAL_DIR = ROOT / "eval"
REVIEW_LOG = DATA_DIR / "reviews.csv"

for _d in (RAW_DIR, CORPUS_DIR, PROCESSED_DIR, SAMPLES_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- signal / ML
FS = 12_000                 # target sampling rate (Hz) for every signal
NORMAL_FS = 48_000          # CWRU "Normal baseline" files are recorded at 48 kHz
WINDOW = 2048               # samples per window
OVERLAP = 0.5               # 50 % overlap
N_FFT_BANDS = 16            # equal-width FFT band energies 0 .. FS/2
RANDOM_STATE = 42
TRAIN_LOADS = (0, 1, 2)     # cross-load split: train on 0-2 HP
TEST_LOADS = (3,)           #                   test on 3 HP
CONFIDENCE_THRESHOLD = 0.60  # below this the prediction is flagged "low confidence"

# CWRU drive-end bearing: SKF 6205-2RS JEM deep-groove ball bearing
BEARING_6205 = {
    "name": "SKF 6205-2RS JEM (CWRU drive end)",
    "n_balls": 9,
    "ball_diameter_in": 0.3126,
    "pitch_diameter_in": 1.537,
    "contact_angle_deg": 0.0,
}

# ----------------------------------------------------------------- RAG
EMBED_MODEL = os.getenv("IM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("IM_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # or BAAI/bge-reranker-base
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_TOKENS = 600          # target chunk size (within the 500-800 range)
CHUNK_OVERLAP = 100         # token overlap between consecutive chunks
TOP_K = 5
RRF_K = 60                  # reciprocal-rank-fusion constant

# ----------------------------------------------------------------- LLM
# provider: gemini (free Google API key) | offline (no LLM). Default: gemini when a key is configured.
LLM_PROVIDER = os.getenv("IM_LLM_PROVIDER") or (
    "gemini" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "offline")
LLM_MODEL = os.getenv("IM_LLM_MODEL", "")
LLM_TEMPERATURE = float(os.getenv("IM_LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = int(os.getenv("IM_LLM_TIMEOUT", "120"))
MAX_AGENT_STEPS = 5         # hard cap on tool calls per diagnosis
