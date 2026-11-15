"""P2: download the openly available bearing-maintenance PDFs listed in data/corpus_sources.csv.

PDFs are saved to data/corpus/pdf/ (git-ignored: manufacturer documents are copyrighted, keep them local).
Usage: python scripts/download_corpus.py      then: python scripts/build_index.py
"""
import csv
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intellimaintain import config  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (IntelliMaintain academic project; PDF download)"}


def main() -> None:
    out_dir = config.CORPUS_DIR / "pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(config.DATA_DIR / "corpus_sources.csv", encoding="utf-8")))
    ok = 0
    for r in rows:
        target = out_dir / r["file_name"]
        if target.exists() and target.stat().st_size > 10_000:
            print(f"[skip] {target.name} (already downloaded)")
            ok += 1
            continue
        try:
            resp = requests.get(r["url"], headers=HEADERS, timeout=90)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF (the link may have moved)")
            target.write_bytes(resp.content)
            print(f"[ok]   {target.name}  ({len(resp.content) / 1e6:.1f} MB)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {r['file_name']}: {e}\n       -> open {r['url']} in a browser and save it to {out_dir}")
    print(f"\n{ok}/{len(rows)} PDFs available in {out_dir}. Now run: python scripts/build_index.py")


if __name__ == "__main__":
    main()
