"""Physics layer: bearing defect frequencies and envelope-spectrum cross-check.

The envelope (demodulation) spectrum is the classic, explainable way that
vibration analysts confirm a bearing fault: a localized defect produces
impacts that repeat at a characteristic frequency (BPFO, BPFI, BSF).
We use it as an *independent* check on the ML prediction - if the two
disagree, the agent warns the human reviewer (wrong-ML stress test).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .. import config
from ..schemas import FaultFrequencies


def defect_frequencies(
    rpm: float,
    n_balls: int = config.BEARING_6205["n_balls"],
    ball_diameter: float = config.BEARING_6205["ball_diameter_in"],
    pitch_diameter: float = config.BEARING_6205["pitch_diameter_in"],
    contact_angle_deg: float = config.BEARING_6205["contact_angle_deg"],
) -> dict[str, float]:
    """Standard kinematic bearing frequencies in Hz (assumes rolling without slip)."""
    if rpm <= 0:
        raise ValueError("rpm must be positive")
    if not (0 < ball_diameter < pitch_diameter):
        raise ValueError("ball diameter must be positive and smaller than the pitch diameter")
    if n_balls < 1:
        raise ValueError("n_balls must be >= 1")
    fr = rpm / 60.0
    ratio = (ball_diameter / pitch_diameter) * math.cos(math.radians(contact_angle_deg))
    return {
        "shaft_hz": fr,
        "ftf_hz": fr / 2 * (1 - ratio),
        "bpfo_hz": n_balls * fr / 2 * (1 - ratio),
        "bpfi_hz": n_balls * fr / 2 * (1 + ratio),
        "bsf_hz": pitch_diameter / (2 * ball_diameter) * fr * (1 - ratio**2),
    }


def envelope_spectrum(x: np.ndarray, fs: int = config.FS, band: tuple[float, float] = (2000.0, 5000.0)):
    """Band-pass -> Hilbert envelope -> FFT. Returns (freqs, amplitude)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    hi = min(band[1], 0.45 * fs)
    lo = min(band[0], hi * 0.5)
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)
    env = np.abs(hilbert(xf))
    env = env - env.mean()
    n = len(env)
    amp = np.abs(np.fft.rfft(env * np.hanning(n))) / n
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return freqs, amp


def _harmonic_snr(freqs, amp, f0: float, n_harm: int = 3, tol: float = 0.03) -> float:
    """Mean ratio of the peak near k*f0 to the local median background (k=1..n_harm)."""
    ratios = []
    for k in range(1, n_harm + 1):
        fk = k * f0
        if fk >= freqs[-1]:
            break
        peak_band = (freqs >= fk * (1 - tol)) & (freqs <= fk * (1 + tol))
        bg_band = (freqs >= fk * 0.7) & (freqs <= fk * 1.3) & ~peak_band
        if not peak_band.any() or not bg_band.any():
            continue
        bg = np.median(amp[bg_band]) + 1e-12
        ratios.append(float(amp[peak_band].max() / bg))
    return float(np.mean(ratios)) if ratios else 0.0


PHYSICS_THRESHOLD = 5.0  # harmonic SNR above which a defect frequency is "present" (all healthy CWRU records < 4.3)
STRONG_SNR = 8.0         # SNR above which physics evidence is strong enough to contradict the ML model
BALL_OVERRIDE_SNR = 25.0  # race-fault SNR that contradicts even a 'Ball' prediction (CWRU ball records reach <= 18.5)


def physics_check(x: np.ndarray, rpm: float, fs: int = config.FS) -> FaultFrequencies:
    """Compute defect frequencies and how strongly each appears in the envelope spectrum."""
    f = defect_frequencies(rpm)
    seg = np.asarray(x, dtype=np.float64)[: fs * 10]  # up to 10 s is plenty
    if len(seg) < 4096:
        raise ValueError("Need at least 4096 samples for the envelope analysis")
    freqs, amp = envelope_spectrum(seg, fs)
    match = {
        "Outer race": _harmonic_snr(freqs, amp, f["bpfo_hz"]),
        "Inner race": _harmonic_snr(freqs, amp, f["bpfi_hz"]),
        "Ball": _harmonic_snr(freqs, amp, 2 * f["bsf_hz"]),  # ball defects strike both races -> 2xBSF
    }
    best = max(match, key=match.get)
    suggests = best if match[best] >= PHYSICS_THRESHOLD else None
    return FaultFrequencies(
        rpm=float(rpm),
        bearing=config.BEARING_6205["name"],
        envelope_match={k: round(v, 2) for k, v in match.items()},
        physics_suggests=suggests,
        **{k: round(v, 3) for k, v in f.items()},
    )


def cross_check(ml_fault_type: str, ff: FaultFrequencies) -> tuple[str, str]:
    """Compare the ML fault type with the envelope evidence.

    Returns (status, message) with status in {consistent, conflict, inconclusive}.
    Ball faults are deliberately treated as 'inconclusive': it is well documented that
    CWRU ball faults rarely show clean 2xBSF lines (Smith & Randall, MSSP 2015).
    """
    snr = ff.envelope_match
    phys = ff.physics_suggests
    if ml_fault_type == "Ball":
        race = max(("Outer race", "Inner race"), key=lambda k: snr.get(k, 0.0))
        if snr.get(race, 0.0) >= BALL_OVERRIDE_SNR:
            return "conflict", (f"ML predicts a ball fault, but the envelope spectrum shows very strong {race} "
                                f"defect-frequency harmonics (SNR {snr[race]:.1f}). Investigate the {race.lower()} "
                                "before trusting the ball label.")
        return "inconclusive", ("Ball-spin (2xBSF) lines are weak for ball defects, so the envelope "
                                "spectrum cannot confirm or reject this prediction.")
    if ml_fault_type == "Normal":
        if phys is not None and snr.get(phys, 0) >= STRONG_SNR:
            return "conflict", (f"ML says Normal, but the envelope spectrum shows strong {phys} defect "
                                f"frequency harmonics (SNR {snr[phys]:.1f}). Inspect before trusting 'Normal'.")
        return "consistent", "No strong bearing defect frequency in the envelope spectrum."
    ml_snr = snr.get(ml_fault_type, 0.0)
    if phys == ml_fault_type:
        return "consistent", f"Envelope spectrum shows {ml_fault_type} defect-frequency harmonics (SNR {ml_snr:.1f})."
    if phys is not None and snr[phys] >= STRONG_SNR and snr[phys] >= 1.5 * max(ml_snr, 1e-9):
        return "conflict", (f"ML predicts {ml_fault_type} (SNR {ml_snr:.1f}) but the envelope spectrum points to "
                            f"{phys} (SNR {snr[phys]:.1f}). Treat the ML label with caution.")
    if phys is None and ml_snr < PHYSICS_THRESHOLD and max(snr.values()) < 4.5:
        return "conflict", (f"ML predicts {ml_fault_type}, but no defect-frequency harmonics are visible "
                            "(the signal looks healthy to the envelope analysis).")
    return "inconclusive", f"Envelope evidence for {ml_fault_type} is weak (SNR {ml_snr:.1f}); manual check advised."
