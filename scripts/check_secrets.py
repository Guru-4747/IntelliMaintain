"""Make sure no API key can reach GitHub.

    python scripts/check_secrets.py                 # scan the files git would publish
    python scripts/check_secrets.py --install-hook  # run this scan automatically before every git commit
"""
import argparse
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intellimaintain.security import scan_for_keys  # noqa: E402

HOOK = """#!/bin/sh
# IntelliMaintain: block commits that contain an API key
python scripts/check_secrets.py || { echo "Commit blocked: remove the API key shown above."; exit 1; }
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install-hook", action="store_true")
    a = ap.parse_args()
    if a.install_hook:
        hook = ROOT / ".git" / "hooks" / "pre-commit"
        if not hook.parent.exists():
            print("Not a git repository yet - run `git init` first.")
            return 1
        hook.write_text(HOOK)
        hook.chmod(hook.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Installed pre-commit hook: {hook}")
    hits = scan_for_keys(ROOT)
    if hits:
        print("API key found in files that would be published:")
        for f, line in hits:
            print(f"  {f}:{line}")
        print("Move the key into .env (laptop) or Streamlit Secrets (online) and delete it from these files.")
        return 1
    print("No API keys found in publishable files. (.env and .streamlit/secrets.toml are git-ignored.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
