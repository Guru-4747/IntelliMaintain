"""CWRU bearing dataset: catalogue, download and loading.

We use the standard 10-class benchmark from the 12 kHz drive-end experiments:
Normal + {Ball, Inner race, Outer race (@6 o'clock)} x {0.007", 0.014", 0.021"},
each recorded at motor loads 0, 1, 2 and 3 HP (40 files in total).

Important data-quality note (often missed in student projects): the CWRU
"Normal baseline" files were recorded at 48 kHz, while the 12k drive-end
fault files are at 12 kHz (Smith & Randall, MSSP 2015). We therefore
anti-alias and decimate the Normal files by 4 so that every class shares the
same 12 kHz time base. Without this, the classifier can learn the sampling
rate instead of the fault.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from .. import config

LABELS = [
    "Normal",
    "B_007", "B_014", "B_021",
    "IR_007", "IR_014", "IR_021",
    "OR_007", "OR_014", "OR_021",
]

FAULT_TYPE = {"Normal": "Normal", "B": "Ball", "IR": "Inner race", "OR": "Outer race"}


@dataclass(frozen=True)
class CwruFile:
    number: int
    label: str
    load_hp: int
    rpm: int
    mirror_path: str  # path inside the GitHub mirror repository

    @property
    def fs(self) -> int:
        return config.NORMAL_FS if self.label == "Normal" else config.FS


def _f(num, label, load, rpm, path):
    return CwruFile(num, label, load, rpm, path)


_DE = "12k_Drive_End_Bearing_Fault_Data"
CATALOG: list[CwruFile] = [
    _f(97, "Normal", 0, 1797, "Normal/97_Normal_0.mat"),
    _f(98, "Normal", 1, 1772, "Normal/98_Normal_1.mat"),
    _f(99, "Normal", 2, 1750, "Normal/99_Normal_2.mat"),
    _f(100, "Normal", 3, 1730, "Normal/100_Normal_3.mat"),
]
_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
for _label, _nums, _dir, _suffix in [
    ("B_007", (118, 119, 120, 121), "B/007", ""),
    ("B_014", (185, 186, 187, 188), "B/014", ""),
    ("B_021", (222, 223, 224, 225), "B/021", ""),
    ("IR_007", (105, 106, 107, 108), "IR/007", ""),
    ("IR_014", (169, 170, 171, 172), "IR/014", ""),
    ("IR_021", (209, 210, 211, 212), "IR/021", ""),
    ("OR_007", (130, 131, 132, 133), "OR/007/@6", "@6"),
    ("OR_014", (197, 198, 199, 200), "OR/014", "@6"),
    ("OR_021", (234, 235, 236, 237), "OR/021/@6", ""),
]:
    for _load, _num in enumerate(_nums):
        CATALOG.append(_f(_num, _label, _load, _RPM[_load], f"{_DE}/{_dir}/{_num}{_suffix}_{_load}.mat"))

BY_NUMBER = {c.number: c for c in CATALOG}

OFFICIAL_URL = "https://engineering.case.edu/sites/default/files/{num}.mat"
MIRROR_URL = "https://raw.githubusercontent.com/s-whynot/CWRU-dataset/main/{path}"


def fault_type_of(label: str) -> str:
    return FAULT_TYPE[label.split("_")[0]]


def fault_size_of(label: str) -> float | None:
    if label == "Normal":
        return None
    return int(label.split("_")[1]) / 1000.0


# ------------------------------------------------------------------ download
def download(dest: Path = config.RAW_DIR, timeout: int = 60) -> list[Path]:
    """Download all 40 files. Tries the official CWRU site, then a GitHub mirror."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for c in CATALOG:
        existing = find_file(c.number, dest)
        if existing is not None:
            saved.append(existing)
            continue
        target = dest / f"{c.number}.mat"
        urls = [OFFICIAL_URL.format(num=c.number),
                MIRROR_URL.format(path=urllib.parse.quote(c.mirror_path))]
        last_err = None
        for url in urls:
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                if len(r.content) < 10_000:
                    raise ValueError("file too small, probably an error page")
                target.write_bytes(r.content)
                sio.loadmat(target)  # validate it is a real .mat file
                print(f"[ok] {c.number}.mat  ({c.label}, {c.load_hp} HP)  <- {url.split('/')[2]}")
                saved.append(target)
                break
            except Exception as e:  # noqa: BLE001 - try next mirror
                last_err = e
                target.unlink(missing_ok=True)
        else:
            print(f"[FAILED] {c.number}.mat: {last_err}")
    return saved


# ------------------------------------------------------------------ loading
_NUM_RE = re.compile(r"^(\d+)")


def find_file(number: int, root: Path = config.RAW_DIR) -> Path | None:
    """Find a .mat file for a CWRU file number anywhere under `root`.

    Accepts official names ('97.mat') and mirror names ('97_Normal_0.mat', '130@6_0.mat').
    """
    for p in root.rglob("*.mat"):
        m = _NUM_RE.match(p.name)
        if m and int(m.group(1)) == number:
            return p
    return None


def load_de_signal(path: Path, number: int) -> np.ndarray:
    """Return the drive-end accelerometer signal (1-D float64) of a .mat file."""
    mat = sio.loadmat(path)
    preferred = f"X{number:03d}_DE_time"
    if preferred in mat:
        key = preferred
    else:  # some files (e.g. 99.mat) contain several recordings; take the matching or first DE key
        de_keys = [k for k in mat if k.endswith("_DE_time")]
        if not de_keys:
            raise KeyError(f"No drive-end signal in {path.name}")
        key = sorted(de_keys)[-1]
    return np.asarray(mat[key], dtype=np.float64).ravel()


def load_file(c: CwruFile, root: Path = config.RAW_DIR) -> np.ndarray:
    """Load one catalogue entry and return it resampled to config.FS (12 kHz)."""
    path = find_file(c.number, root)
    if path is None:
        raise FileNotFoundError(
            f"CWRU file {c.number}.mat not found under {root}. Run: python scripts/download_cwru.py"
        )
    x = load_de_signal(path, c.number)
    if c.fs != config.FS:
        x = resample_poly(x, up=config.FS, down=c.fs)  # anti-aliased decimation 48k -> 12k
    return x


def available(root: Path = config.RAW_DIR) -> list[CwruFile]:
    return [c for c in CATALOG if find_file(c.number, root) is not None]
