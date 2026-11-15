"""Windowing and hand-crafted vibration features (time domain + FFT bands)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .. import config

TIME_FEATURES = [
    "rms", "std", "peak", "peak_to_peak", "kurtosis", "skewness",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
]
SPECTRAL_FEATURES = ["spectral_centroid", "spectral_spread", "high_freq_ratio"]
BAND_FEATURES = [f"fft_band_{i:02d}" for i in range(config.N_FFT_BANDS)]
FEATURE_NAMES = TIME_FEATURES + SPECTRAL_FEATURES + BAND_FEATURES


def sliding_windows(x: np.ndarray, window: int = config.WINDOW, overlap: float = config.OVERLAP) -> np.ndarray:
    """Cut a 1-D signal into overlapping windows -> array (n_windows, window)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if len(x) < window:
        raise ValueError(f"Signal has {len(x)} samples; need at least {window} (one window).")
    step = max(1, int(window * (1 - overlap)))
    n = 1 + (len(x) - window) // step
    idx = np.arange(window)[None, :] + step * np.arange(n)[:, None]
    return x[idx]


def window_features(windows: np.ndarray, fs: int = config.FS) -> np.ndarray:
    """Vectorised feature extraction for a batch of windows -> (n_windows, n_features)."""
    w = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    w = w - w.mean(axis=1, keepdims=True)
    eps = 1e-12
    absw = np.abs(w)
    rms = np.sqrt((w**2).mean(axis=1))
    std = w.std(axis=1)
    peak = absw.max(axis=1)
    p2p = w.max(axis=1) - w.min(axis=1)
    kurt = kurtosis(w, axis=1, fisher=False)  # 3.0 for a Gaussian signal
    skw = skew(w, axis=1)
    mean_abs = absw.mean(axis=1)
    crest = peak / (rms + eps)
    shape = rms / (mean_abs + eps)
    impulse = peak / (mean_abs + eps)
    clearance = peak / (np.sqrt(absw).mean(axis=1) ** 2 + eps)

    spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(w.shape[1], 1.0 / fs)
    total = spec.sum(axis=1) + eps
    centroid = (spec * freqs).sum(axis=1) / total
    spread = np.sqrt((spec * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total)
    hf_ratio = spec[:, freqs >= 2000].sum(axis=1) / total
    # relative energy in equal-width bands (load-robust: amplitude normalised out)
    edges = np.linspace(0, spec.shape[1], config.N_FFT_BANDS + 1).astype(int)
    bands = np.stack([spec[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1) / total[:, None]

    feats = np.column_stack([
        rms, std, peak, p2p, kurt, skw, crest, shape, impulse, clearance,
        centroid / 1000.0, spread / 1000.0, hf_ratio, bands,
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def signal_features(x: np.ndarray, fs: int = config.FS) -> pd.DataFrame:
    """Window a signal and return a feature DataFrame (one row per window)."""
    return pd.DataFrame(window_features(sliding_windows(x), fs), columns=FEATURE_NAMES)


def summary_features(x: np.ndarray) -> dict[str, float]:
    """Whole-signal headline indicators shown in the app and passed to the LLM."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    return {
        "rms_g": round(rms, 4),
        "peak_g": round(peak, 4),
        "kurtosis": round(float(kurtosis(x, fisher=False)), 2),
        "skewness": round(float(skew(x)), 3),
        "crest_factor": round(peak / (rms + 1e-12), 2),
    }
