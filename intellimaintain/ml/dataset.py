"""Build the windowed feature table from the raw CWRU files (cached as parquet/csv)."""
from __future__ import annotations

import pandas as pd

from .. import config
from . import cwru
from .features import FEATURE_NAMES, signal_features

CACHE = config.PROCESSED_DIR / "features.csv"


def build_feature_table(use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    if use_cache and CACHE.exists():
        df = pd.read_csv(CACHE)
        if set(FEATURE_NAMES).issubset(df.columns):
            return df
    files = cwru.available()
    if len(files) == 0:
        raise FileNotFoundError(
            f"No CWRU .mat files found in {config.RAW_DIR}. Run: python scripts/download_cwru.py"
        )
    missing = sorted(set(c.number for c in cwru.CATALOG) - set(c.number for c in files))
    if missing and verbose:
        print(f"[warn] {len(missing)} CWRU files missing: {missing}")
    parts = []
    for c in files:
        x = cwru.load_file(c)
        f = signal_features(x)
        f["label"] = c.label
        f["fault_type"] = cwru.fault_type_of(c.label)
        f["load_hp"] = c.load_hp
        f["rpm"] = c.rpm
        f["file_id"] = c.number
        parts.append(f)
        if verbose:
            print(f"  {c.number:>4}.mat  {c.label:<7} {c.load_hp} HP  -> {len(f):4d} windows")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(CACHE, index=False)
    return df
