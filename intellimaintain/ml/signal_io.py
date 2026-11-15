"""Read vibration signals uploaded as CSV (or .mat / .npy) in a forgiving way.

Accepted CSV layouts
  * a single numeric column (with or without header)
  * several columns; we pick one named like DE / drive_end / signal / acc / vibration / value,
    otherwise the first numeric column that is not called time/t
Optional metadata comment lines at the top, e.g.:
  # fs=12000
  # rpm=1797
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

PREFERRED = ("de", "drive_end", "de_time", "signal", "acc", "accel", "acceleration", "vibration", "value", "x")
TIME_NAMES = ("time", "t", "timestamp", "seconds", "sec", "index")
_META_RE = re.compile(r"#\s*(\w+)\s*[=:]\s*([-\w.@ ]+)")


@dataclass
class LoadedSignal:
    x: np.ndarray
    fs: int | None = None
    rpm: float | None = None
    meta: dict = field(default_factory=dict)
    column: str = ""


def _read_bytes(src: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(src, bytes):
        return src
    if isinstance(src, (str, Path)):
        return Path(src).read_bytes()
    data = src.read()
    return data if isinstance(data, bytes) else data.encode()


def load_signal_csv(src: str | Path | bytes | BinaryIO) -> LoadedSignal:
    raw = _read_bytes(src).decode("utf-8", errors="replace")
    meta: dict[str, str] = {}
    for line in raw.splitlines()[:20]:
        if line.lstrip().startswith("#"):
            for k, v in _META_RE.findall(line):
                meta[k.lower()] = v.strip()
    df = pd.read_csv(io.StringIO(raw), comment="#", header=None, skip_blank_lines=True)
    if df.empty:
        raise ValueError("The CSV file contains no data rows.")
    # detect a header row: first row non-numeric
    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().any():
        df.columns = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    num = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in num.columns if num[c].notna().mean() > 0.95]
    if not numeric_cols:
        raise ValueError("No numeric column found in the CSV.")
    col = None
    lower = {c.lower(): c for c in numeric_cols}
    for name in PREFERRED:
        for lc, orig in lower.items():
            if lc == name or lc.endswith("_" + name) or lc.startswith(name + "_"):
                col = orig
                break
        if col:
            break
    if col is None:
        candidates = [c for c in numeric_cols if c.lower() not in TIME_NAMES]
        col = candidates[0] if candidates else numeric_cols[0]
    x = num[col].dropna().to_numpy(dtype=np.float64)
    fs = int(float(meta["fs"])) if "fs" in meta else None
    # infer fs from a time column if present
    if fs is None:
        tcols = [c for c in numeric_cols if c.lower() in ("time", "t", "seconds", "sec")]
        if tcols:
            t = num[tcols[0]].dropna().to_numpy()
            if len(t) > 2 and np.all(np.diff(t[:100]) > 0):
                fs = int(round(1.0 / np.median(np.diff(t[:1000]))))
    rpm = float(meta["rpm"]) if "rpm" in meta else None
    return LoadedSignal(x=x, fs=fs, rpm=rpm, meta=meta, column=str(col))


def load_signal(src: str | Path, name: str | None = None) -> LoadedSignal:
    """Load .csv / .txt / .npy / .mat by extension."""
    name = (name or str(src)).lower()
    if name.endswith(".npy"):
        return LoadedSignal(x=np.load(src).astype(np.float64).ravel())
    if name.endswith(".mat"):
        import scipy.io as sio

        mat = sio.loadmat(src)
        keys = [k for k in mat if k.endswith("_DE_time")] or [k for k in mat if not k.startswith("__")]
        rpm_keys = [k for k in mat if k.endswith("RPM")]
        rpm = float(np.ravel(mat[rpm_keys[0]])[0]) if rpm_keys else None
        return LoadedSignal(x=np.asarray(mat[keys[0]], dtype=np.float64).ravel(), rpm=rpm, column=keys[0])
    return load_signal_csv(src)
