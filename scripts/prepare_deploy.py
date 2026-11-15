"""Check the project is ready to run ONLINE (Streamlit Community Cloud) and pin model-critical versions.

Run this on the laptop where you trained the models, BEFORE pushing to GitHub:
    python scripts/prepare_deploy.py

It (1) checks that the trained models, the RAG index and the demo samples exist (they must be committed,
the online server cannot train), (2) pins scikit-learn/joblib in app/requirements.txt to the versions that
created models/*.joblib (a different scikit-learn version may fail to load them), (3) warns about secrets
and copyrighted text, and (4) prints the Python version to choose in the deploy dialog.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"


def main() -> int:
    problems = 0
    print("IntelliMaintain - deployment check\n")

    print("1. Files the online app needs (must be committed to git)")
    needed = ["models/classifier.joblib", "models/rag_index/faiss.index", "models/rag_index/chunks.jsonl",
              "models/rag_index/meta.json", "app/app.py", "app/requirements.txt", ".streamlit/config.toml"]
    for f in needed:
        exists = (ROOT / f).exists()
        problems += not exists
        print(f"{OK if exists else FAIL} {f}")
    for f in ["models/cnn_classifier.pt"]:
        print(f"{OK if (ROOT / f).exists() else WARN} {f}" + ("" if (ROOT / f).exists() else
                                                              "  (CNN option hidden online; run scripts/train_cnn.py)"))
    samples = list((ROOT / "data/samples").glob("*.csv"))
    print(f"{OK if samples else FAIL} data/samples: {len(samples)} demo CSV files")
    problems += not samples

    meta_path = ROOT / "models/rag_index/meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        emb = meta.get("embed_model")
        print(f"{OK if meta.get('embedder') == 'bge' else WARN} index built with {emb} "
              f"({meta.get('n_chunks')} chunks, {len(meta.get('sources', []))} documents)")
        pdfs = [s for s in meta.get("sources", []) if s.lower().endswith(".pdf")]
        if pdfs:
            print(f"{WARN} the index contains text from {len(pdfs)} manufacturer PDFs/papers. Keep the GitHub repo "
                  "PRIVATE (Streamlit Cloud can deploy private repos) or rebuild the index without data/corpus/pdf.")

    print("\n2. Pin versions that created the saved models")
    req = ROOT / "app/requirements.txt"
    import joblib
    import sklearn

    pins = {"scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    text = req.read_text()
    for pkg, ver in pins.items():
        text, n = re.subn(rf"(?m)^{re.escape(pkg)}\s*[<>=!~].*$", f"{pkg}=={ver}", text)
        if n == 0:
            text += f"\n{pkg}=={ver}\n"
        print(f"{OK} {pkg}=={ver}")
    req.write_text(text)

    print("\n3. Secrets")
    from intellimaintain.security import scan_for_keys

    hits = scan_for_keys(ROOT)
    if hits:
        problems += 1
        for f, line in hits:
            print(f"{FAIL} API key written in {f}:{line} - move it to .env / Secrets and delete it there")
    else:
        print(f"{OK} no API key written in any file that would be pushed")
    tracked = subprocess.run(["git", "ls-files", ".env", ".streamlit/secrets.toml"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    if tracked:
        problems += 1
        print(f"{FAIL} {tracked} is tracked by git - remove it: git rm --cached {' '.join(tracked)}")
    else:
        print(f"{OK} no API keys tracked by git (.env and secrets.toml are ignored)")

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"\n4. In Streamlit Community Cloud choose: main file = app/app.py, Python version = {py},")
    print("   Advanced settings -> Secrets:  GEMINI_API_KEY = \"your-key\"")
    print("\nResult:", "READY - commit and push." if problems == 0 else f"{problems} problem(s) to fix first.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
