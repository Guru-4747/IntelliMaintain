"""Inference: classify_signal() -> ClassifyResult (the P1 contract used by the agent)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
from scipy.signal import resample_poly

from .. import config
from ..schemas import ClassifyResult
from . import cwru
from .features import sliding_windows, summary_features, window_features
from .train import CROSSLOAD_MODEL_PATH, MODEL_PATH  # noqa: F401

# The evaluation scripts switch this to the cross-load model (trained on 0-2 HP only) so that
# test recordings (3 HP) were never seen by the classifier.
ACTIVE_MODEL_PATH = MODEL_PATH


@lru_cache(maxsize=2)
def load_bundle(path: str = str(MODEL_PATH)) -> dict:
    import warnings

    import sklearn

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # version warnings are reported below in plain words
            bundle = joblib.load(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Trained model not found at {path}. Run: python scripts/train_ml.py") from e
    except Exception as e:  # noqa: BLE001 - e.g. pickle from an incompatible scikit-learn version
        raise RuntimeError(f"Could not load {path} ({e}). Re-train with: python scripts/train_ml.py") from e
    trained = bundle.get("sklearn_version")
    if trained and trained.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        warnings.warn(f"Model trained with scikit-learn {trained}, running {sklearn.__version__}. "
                      "If predictions look wrong, re-train with: python scripts/train_ml.py", stacklevel=2)
    return bundle


def to_target_fs(x: np.ndarray, fs: int | None) -> np.ndarray:
    if fs is None or fs == config.FS:
        return x
    if fs <= 0:
        raise ValueError("Sampling rate must be positive")
    from math import gcd

    g = gcd(int(fs), config.FS)
    return resample_poly(x, config.FS // g, int(fs) // g)


MODELS = ("rf", "cnn")


def cnn_ready() -> bool:
    from . import cnn

    return cnn.torch_available() and cnn.CNN_PATH.exists()


def similar_cases(X: np.ndarray, bundle: dict, k: int = 3) -> list[dict]:
    """Nearest CWRU recordings to this signal in standardised feature space (case-based evidence)."""
    prof = bundle.get("case_profiles")
    if not prof:
        return []
    q = ((X - prof["mu"]) / prof["sd"]).mean(axis=0)
    d = np.linalg.norm(prof["vectors"] - q, axis=1) / np.sqrt(len(q))
    out = []
    for i in np.argsort(d)[:k]:
        lab = prof["label"][i]
        out.append({"cwru_file": f"{int(prof['file_id'][i])}.mat", "label": lab, "fault_type": cwru.fault_type_of(lab),
                    "load_hp": int(prof["load_hp"][i]), "rpm": int(prof["rpm"][i]),
                    "similarity": round(float(np.exp(-d[i])), 3)})
    return out


def classify_signal(signal: np.ndarray, fs: int | None = config.FS, override_label: str | None = None,
                    model: str = "rf") -> ClassifyResult:
    """Classify a raw vibration signal (any length >= 2048 samples at 12 kHz).

    Every 2048-sample window is classified; calibrated probabilities are averaged
    over windows. `model` = "rf" (feature RandomForest) or "cnn" (spectrogram CNN).
    `override_label` is used only by the wrong-ML stress test to simulate a faulty classifier.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    x = to_target_fs(np.asarray(signal, dtype=np.float64).ravel(), fs)
    if not np.all(np.isfinite(x)):
        x = x[np.isfinite(x)]
    if len(x) < config.WINDOW:
        raise ValueError(
            f"Signal too short: {len(x)} samples at {config.FS} Hz; need at least {config.WINDOW}."
        )
    b = load_bundle(str(ACTIVE_MODEL_PATH))
    windows = sliding_windows(x, b["window"], b["overlap"])
    X = window_features(windows)  # hand-crafted features: RF input and anomaly detector input
    if model == "cnn":
        from . import cnn

        crossload = str(ACTIVE_MODEL_PATH) == str(CROSSLOAD_MODEL_PATH)
        proba, labels = cnn.predict_proba(windows.astype(np.float32),
                                          str(cnn.CNN_CROSSLOAD_PATH if crossload else cnn.CNN_PATH))
        model_name = "Spectrogram CNN (temperature-scaled)"
    else:
        proba = b["model"].predict_proba(X)
        labels = list(b["labels"])
        model_name = "RandomForest (isotonic-calibrated)"
    mean_p = proba.mean(axis=0)
    votes = np.bincount(proba.argmax(axis=1), minlength=len(labels))
    label = labels[int(mean_p.argmax())]
    conf = float(mean_p.max())

    raw = -b["iso"].score_samples(X)
    span = max(b["iso_q99"] - b["iso_q50"], 1e-9)
    anomaly = float(np.clip((np.median(raw) - b["iso_q50"]) / span, 0.0, 1.0))

    if override_label is not None:
        if override_label not in labels:
            raise ValueError(f"Unknown label {override_label}")
        label = override_label
        conf = float(mean_p[labels.index(label)]) if mean_p[labels.index(label)] > 0.5 else 0.85

    return ClassifyResult(
        label=label,
        fault_type=cwru.fault_type_of(label),
        fault_size_in=cwru.fault_size_of(label),
        confidence=round(conf, 4),
        class_probabilities={lab: round(float(p), 4) for lab, p in zip(labels, mean_p)},
        window_votes={lab: int(v) for lab, v in zip(labels, votes) if v},
        n_windows=int(len(X)),
        anomaly_score=round(anomaly, 3),
        low_confidence=conf < config.CONFIDENCE_THRESHOLD,
        key_features=summary_features(x),
        model_name=model_name,
        similar_cases=similar_cases(X, b),
    )
