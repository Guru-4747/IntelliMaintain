"""Train and evaluate the spectrogram CNN on the same splits as the RandomForest; writes the comparison table.

Usage: python scripts/train_cnn.py [--epochs 14]
Run python scripts/train_ml.py first so the RF numbers exist for the comparison.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellimaintain.ml import train_cnn  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=14)
    a = ap.parse_args()
    train_cnn.run(epochs=a.epochs)
