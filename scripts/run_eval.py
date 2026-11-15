"""P4: run the full evaluation (retrieval benchmark, C0 vs C1, 2x3 ablation, wrong-ML stress test).

Usage:
  python scripts/run_eval.py                         # offline generator (no LLM, deterministic)
  python scripts/run_eval.py --provider gemini       # with Gemini (GEMINI_API_KEY in .env)
  python scripts/run_eval.py --provider gemini --model gemini-3.8-flash --judge
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellimaintain.agent.llm import PROVIDERS  # noqa: E402
from intellimaintain.evaluation.evaluate import run_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=PROVIDERS, default=None, help="LLM provider (default: from .env)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scores (needs a real LLM)")
    ap.add_argument("--classifier", choices=["rf", "cnn"], default="rf", help="ML model used by the agent")
    a = ap.parse_args()
    run_all(a.provider, a.model, judge=a.judge, classifier=a.classifier)
