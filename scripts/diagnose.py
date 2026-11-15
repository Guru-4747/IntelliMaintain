"""Command-line copilot: diagnose a vibration file and print the cited answer.

Usage:
  python scripts/diagnose.py data/samples/outer_race_007_3hp.csv
  python scripts/diagnose.py my_signal.csv --rpm 1750 --provider gemini --question "Is it safe to run until Friday?"
  python scripts/diagnose.py --question "What does high kurtosis mean?"        (no signal)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellimaintain.agent.agent import DiagnosisAgent  # noqa: E402
from intellimaintain.agent.llm import LLM, PROVIDERS  # noqa: E402
from intellimaintain.ml.signal_io import load_signal  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help=".csv/.txt/.mat/.npy vibration signal")
    ap.add_argument("--question", default="What is the condition of this bearing and what should we do?")
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--provider", choices=PROVIDERS, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--retrieval", choices=["none", "basic", "improved"], default="improved")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--planner", choices=["fixed", "llm"], default="fixed")
    ap.add_argument("--json", action="store_true", help="print the full Diagnosis JSON")
    a = ap.parse_args()

    x = fs = rpm = None
    if a.file:
        sig = load_signal(a.file)
        x, fs, rpm = sig.x, a.fs or sig.fs or 12000, a.rpm or sig.rpm
    agent = DiagnosisAgent(LLM(a.provider, a.model), use_ml=not a.no_ml, retrieval=a.retrieval, planner=a.planner)
    d = agent.run(a.question, signal=x, fs=fs, rpm=rpm)
    if a.json:
        print(d.model_dump_json(indent=2))
        sys.exit(0)
    print(f"\nFAULT TYPE : {d.fault_type}" + (f"   (ML {d.ml_result.label}, confidence {d.ml_result.confidence:.2f})"
                                               if d.ml_result else ""))
    if d.fault_frequencies and d.fault_frequencies.envelope_match:
        print(f"PHYSICS    : {d.fault_frequencies.consistency} - {d.fault_frequencies.consistency_note}")
    print(f"\nSUMMARY\n{d.summary}\n\nEXPLANATION\n{d.explanation}\n\nRECOMMENDED ACTIONS")
    for i, act in enumerate(d.recommended_actions, 1):
        print(f"  {i}. {act}")
    print("\nEVIDENCE")
    for i, e in enumerate(d.evidence, 1):
        print(f"  [{i}] {e.source}, page {e.page}")
    for w in d.warnings:
        print(f"WARNING: {w}")
    print(f"\n(citation coverage {d.citation_coverage:.0%}, LLM {d.llm_provider}:{d.llm_model}, "
          f"{len(d.tool_trace)} tool calls)")
