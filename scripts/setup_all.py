"""One command to build everything from scratch.

  1. download CWRU data                 (scripts/download_cwru.py)
  2. train + evaluate the ML layer      (scripts/train_ml.py, scripts/train_cnn.py)
  3. generate synthetic logs, download open PDFs, build the BGE/FAISS index
  4. create demo CSV samples for the app
  5. run the copilot evaluation (offline generator)

Usage: python scripts/setup_all.py [--skip-pdfs] [--skip-eval]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(title: str, *args: str, allow_fail: bool = False) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode != 0 and not allow_fail:
        hint = " (the process crashed - see the README 'Troubleshooting' section)" if r.returncode < 0 else ""
        sys.exit(f"Step failed: {title} [exit code {r.returncode}]{hint}\n"
                 f"Re-run just this step to see the full error:  python {args[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-pdfs", action="store_true", help="do not download manufacturer PDFs")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    step("1/6 Download CWRU bearing data", "scripts/download_cwru.py")
    step("2/6 Train and evaluate the ML classifier", "scripts/train_ml.py")
    step("2b/6 Train the spectrogram CNN and compare it with the RandomForest (~6 min on CPU)",
         "scripts/train_cnn.py", allow_fail=True)
    step("3/6 Generate synthetic maintenance logs", "scripts/generate_maintenance_logs.py")
    if not a.skip_pdfs:
        step("4/6 Download open bearing-maintenance PDFs", "scripts/download_corpus.py", allow_fail=True)
    step("5/6 Build the RAG index (BGE + FAISS)", "scripts/build_index.py")
    step("6/6 Create demo samples", "scripts/make_samples.py")
    if not a.skip_eval:
        step("Evaluation (offline)", "scripts/run_eval.py")
    print("\nAll done. Start the app with:  streamlit run app/app.py")
