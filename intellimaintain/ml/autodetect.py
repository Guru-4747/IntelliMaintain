"""Work out the sampling rate and shaft speed of an uploaded vibration file automatically.

Order of evidence (the first one found wins, and the app shows where the value came from):
  sampling rate : file header (# fs=...)  ->  time column  ->  file name ("12k", "48k", "fs=25600")
                  ->  CWRU file number (e.g. 108.mat)  ->  default 12 kHz (CWRU drive end)
  shaft speed   : file header (# rpm=...) or .mat RPM variable  ->  file name ("1730rpm", "3hp")
                  ->  CWRU file number  ->  estimated from the vibration spectrum (1x running speed)
                  ->  1772 rpm, flagged "please check", when the spectrum shows no clear shaft line

"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .. import config

CWRU_LOAD_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
DEFAULT_RPM = 1772.0  # middle of the CWRU range: within the physics check's 3 % tolerance for all four loads


@dataclass
class Detected:
    fs: int
    fs_source: str
    rpm: float
    rpm_source: str
    rpm_confidence: str = "high"   # high = read from the file; medium/low = estimated from the signal


def _cwru_number(name: str) -> int | None:
    from . import cwru

    for m in re.finditer(r"(?<!\d)(\d{2,3})(?!\d)", name):
        n = int(m.group(1))
        if n in cwru.BY_NUMBER:
            return n
    return None


def fs_from_name(name: str) -> int | None:
    low = name.lower()
    if m := re.search(r"fs[_= -]?(\d{3,6})", low):
        return int(m.group(1))
    if m := re.search(r"(?<![\d.])(12|48|25\.6|51\.2|20|10)\s*k(hz)?(?![a-z])", low):
        return int(round(float(m.group(1)) * 1000))
    return None


def rpm_from_name(name: str) -> float | None:
    low = name.lower()
    if m := re.search(r"(\d{3,5}(?:\.\d+)?)\s*_?rpm", low):
        return float(m.group(1))
    if m := re.search(r"(?<![\d])([0-3])\s*_?hp", low):
        return CWRU_LOAD_RPM[int(m.group(1))]
    return None


MAINS_HZ = (50.0, 60.0)


def estimate_rpm(x: np.ndarray, fs: int, lo_hz: float = 8.0, hi_hz: float = 100.0,
                 sub_thr: float = 4.0) -> tuple[float, float, str]:
    """Running speed from the vibration spectrum, the way an analyst reads it off a plot:
    take the strongest line between 8 and 100 Hz (480-6000 rpm), ignoring 50/60 Hz mains hum and its
    harmonics; if a clear line sits at 1/2 or 1/3 of it, that lower line is the shaft speed (the strongest line
    is often 2x or 3x); average with the 2x-4x harmonics. Returns (rpm, line strength, confidence).
    On the 40 CWRU recordings 95 % are within 3 % (97 % when confidence is "medium")."""
    from scipy.ndimage import median_filter

    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = 1 << int(np.ceil(np.log2(max(len(x), 1024) * 4)))           # zero-padding: finer frequency grid
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    df = fs / n
    spec = spec[: int(4 * hi_hz / df) + 1]
    snr = spec / (median_filter(spec, size=max(5, int(20 / df) | 1)) + 1e-12)   # line height above local floor
    f_axis = np.arange(len(snr)) * df
    for m in MAINS_HZ:  # electrical mains hum is not shaft speed
        for h in range(1, 9):
            snr[np.abs(f_axis - h * m) <= 0.8] = 0
    tol = max(2, int(0.3 / df))

    def peak_near(f):
        i = int(round(f / df))
        j = slice(max(0, i - tol), i + tol + 1)
        a = j.start + int(np.argmax(snr[j]))
        return a * df, float(snr[a])

    lo, hi = int(lo_hz / df), int(hi_hz / df)
    i = lo + int(np.argmax(snr[lo:hi]))
    f, strength = i * df, float(snr[i])
    changed = True
    while changed:
        changed = False
        for div in (2, 3):
            if f / div >= lo_hz:
                fh, sh = peak_near(f / div)
                if sh >= sub_thr:
                    f, strength, changed = fh, sh, True
                    break
    support = [f]
    for h in (2, 3, 4):
        fh, sh = peak_near(h * f)
        if sh >= sub_thr:
            support.append(fh / h)
    conf = "medium" if (strength >= 8 and len(support) >= 2) else "low"
    return round(float(np.median(support)) * 60, 1), round(strength, 1), conf


def detect(x: np.ndarray | None, name: str = "", fs_file: int | None = None, rpm_file: float | None = None) -> Detected:
    from . import cwru

    num = _cwru_number(name)
    if fs_file:
        fs, fs_src = int(fs_file), "read from the file"
    elif (f := fs_from_name(name)):
        fs, fs_src = f, "from the file name"
    elif num is not None:
        fs, fs_src = cwru.BY_NUMBER[num].fs, f"CWRU file {num}"
    else:
        fs, fs_src = config.FS, "default (CWRU 12 kHz) - check it"

    if rpm_file:
        return Detected(fs, fs_src, float(rpm_file), "read from the file")
    if (r := rpm_from_name(name)):
        return Detected(fs, fs_src, r, "from the file name")
    if num is not None:
        return Detected(fs, fs_src, float(cwru.BY_NUMBER[num].rpm), f"CWRU file {num}")
    if x is None or len(x) < 2048:
        return Detected(fs, fs_src, DEFAULT_RPM, "not enough data - assumed, please check", "low")
    rpm, strength, conf = estimate_rpm(x[: int(fs * 10)], fs)
    if conf == "low":
        return Detected(fs, fs_src, DEFAULT_RPM,
                        f"no clear shaft-speed line in the spectrum - assumed {DEFAULT_RPM:.0f} rpm, please check",
                        "low")
    return Detected(fs, fs_src, rpm, "estimated from the vibration spectrum (shaft-speed line) - please check",
                    conf)
