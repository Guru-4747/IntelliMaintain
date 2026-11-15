"""P1: download the 40 CWRU recordings (12 kHz drive end, 10 classes x 4 loads) into data/raw/.

Tries the official CWRU website first, then a public GitHub mirror. About 130 MB.
Usage: python scripts/download_cwru.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intellimaintain.ml import cwru  # noqa: E402

if __name__ == "__main__":
    files = cwru.download()
    print(f"\n{len(files)}/{len(cwru.CATALOG)} CWRU files ready in data/raw/")
    if len(files) < len(cwru.CATALOG):
        sys.exit(1)
