"""Turn vibration windows into time-frequency images (log-power STFT spectrograms) for the CNN."""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

from .. import config

NPERSEG = 128      # 128-sample frames -> 65 frequency bins (0..6 kHz, ~94 Hz resolution)
NOVERLAP = 96      # 75 % frame overlap -> 61 time frames for a 2048-sample window
IMG_SHAPE = (64, 64)  # (freq, time) after cropping/padding


def window_spectrograms(windows: np.ndarray, fs: int = config.FS) -> np.ndarray:
    """(n, 2048) windows -> (n, 1, 64, 64) float32 images, each standardised on its own.

    Per-image standardisation removes the absolute vibration level, so the CNN must learn the
    *shape* of the energy distribution (which bands, how impulsive in time) instead of the
    load-dependent amplitude.
    """
    w = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    w = w - w.mean(axis=1, keepdims=True)
    _, _, z = stft(w, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, boundary=None, padded=False, axis=-1)
    img = np.log1p(np.abs(z) ** 2 * 1e4)                 # (n, 65, T)
    img = img[:, 1:1 + IMG_SHAPE[0], :]                    # drop DC bin -> 64 freq bins
    t = img.shape[2]
    if t >= IMG_SHAPE[1]:
        img = img[:, :, :IMG_SHAPE[1]]
    else:
        img = np.pad(img, ((0, 0), (0, 0), (0, IMG_SHAPE[1] - t)), mode="edge")
    mu = img.mean(axis=(1, 2), keepdims=True)
    sd = img.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((img - mu) / sd).astype(np.float32)[:, None, :, :]


def signal_spectrogram(x: np.ndarray, fs: int = config.FS, nperseg: int = 256):
    """Spectrogram of a whole signal for display: returns (times_s, freqs_hz, power_db)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    f, t, z = stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return t, f, 10 * np.log10(np.abs(z) ** 2 + 1e-12)
