"""P1: build features from CWRU, run all ML experiments, save the deployable model.

Usage:  python scripts/train_ml.py [--no-cache]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellimaintain.ml import dataset, train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true", help="re-extract features from the .mat files")
    args = ap.parse_args()
    if args.no_cache and dataset.CACHE.exists():
        dataset.CACHE.unlink()
    train.run(verbose=True)
