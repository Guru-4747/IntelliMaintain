"""P2: parse the knowledge base (data/corpus), chunk, embed (BGE) and store in FAISS.

Usage: python scripts/build_index.py [--embedder bge|lsa] [--reuse-missing-pdfs]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellimaintain.rag.retriever import build_index, retrieve_evidence  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", default="bge", choices=["bge", "lsa"])
    ap.add_argument("--reuse-missing-pdfs", action="store_true",
                    help="keep the chunks of PDFs that are in the current index but not in data/corpus "
                         "(e.g. after a fresh git clone, where the downloaded PDFs are not included)")
    args = ap.parse_args()
    build_index(prefer=args.embedder, reuse_missing_pdfs=args.reuse_missing_pdfs)
    q = "outer race fault high kurtosis inspection action"
    print(f"\nTest query: '{q}'")
    for i, ev in enumerate(retrieve_evidence(q), 1):
        print(f"[{i}] {ev.source} p.{ev.page}  score={ev.score:.3f} ({ev.retriever})  {ev.text[:90]}...")
