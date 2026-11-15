"""Spectrogram CNN classifier (deep-learning counterpart of the RandomForest).

Pipeline: 2048-sample window -> log-power STFT image (64 x 64) -> small CNN -> class probabilities.
Probabilities are calibrated with temperature scaling on a held-out validation part.
Optional training augmentation (random white noise 5-30 dB SNR + time shift) makes the network
robust to noisy sensors; its effect is measured separately in the experiments.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from . import cwru
from .features import sliding_windows
from .spectrogram import IMG_SHAPE, window_spectrograms

CNN_PATH = config.MODELS_DIR / "cnn_classifier.pt"
CNN_CROSSLOAD_PATH = config.MODELS_DIR / "cnn_classifier_crossload.pt"
WINDOWS_CACHE = config.PROCESSED_DIR / "windows.npz"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(n_classes: int):
    import torch.nn as nn

    def block(cin, cout, pool=True):
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return layers

    return nn.Sequential(
        *block(1, 16), *block(16, 32), *block(32, 64), *block(64, 64, pool=False),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes),
    )


# ------------------------------------------------------------------ data
def load_windows(use_cache: bool = True, verbose: bool = True):
    """Raw 2048-sample windows of all CWRU recordings (same windowing as the RandomForest)."""
    if use_cache and WINDOWS_CACHE.exists():
        z = np.load(WINDOWS_CACHE, allow_pickle=False)
        return z["X"], z["label"].astype(str), z["load"], z["file_id"], z["pos"]
    xs, labels, loads, files, pos = [], [], [], [], []
    for c in cwru.available():
        w = sliding_windows(cwru.load_file(c)).astype(np.float32)
        xs.append(w)
        labels += [c.label] * len(w)
        loads += [c.load_hp] * len(w)
        files += [c.number] * len(w)
        pos.append(np.linspace(0, 1, len(w), dtype=np.float32))  # relative position inside the recording
    if not xs:
        raise FileNotFoundError("No CWRU data. Run: python scripts/download_cwru.py")
    X = np.concatenate(xs)
    out = (X, np.array(labels), np.array(loads), np.array(files), np.concatenate(pos))
    np.savez_compressed(WINDOWS_CACHE, X=out[0], label=out[1], load=out[2], file_id=out[3], pos=out[4])
    if verbose:
        print(f"Cached {len(X)} windows -> {WINDOWS_CACHE}")
    return out


def _augment(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = w.copy()
    shift = rng.integers(0, w.shape[1], size=len(w))
    for i, s in enumerate(shift):
        w[i] = np.roll(w[i], s)
    noisy = rng.random(len(w)) < 0.5
    if noisy.any():
        snr_db = rng.uniform(5, 30, size=noisy.sum())
        p = np.mean(w[noisy] ** 2, axis=1)
        w[noisy] += rng.standard_normal(w[noisy].shape).astype(np.float32) * np.sqrt(p / 10 ** (snr_db / 10))[:, None]
    return w


# ------------------------------------------------------------------ training
def train_cnn(X: np.ndarray, y: np.ndarray, n_classes: int, val_mask: np.ndarray | None = None,
              epochs: int = 14, augment: bool = True, seed: int = config.RANDOM_STATE, verbose: bool = True):
    """Train on raw windows X (n, 2048) with integer labels y. Returns (model, temperature, history)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = get_device()
    if val_mask is None:
        val_mask = np.zeros(len(X), dtype=bool)
    Xtr, ytr = X[~val_mask], y[~val_mask]
    model = build_model(n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Xtr) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    # class weights against imbalance (Normal has more windows)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        tot, n = 0.0, 0
        t0 = time.time()
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            wb = _augment(Xtr[idx], rng) if augment else Xtr[idx]
            xb = torch.from_numpy(window_spectrograms(wb)).to(dev)
            yb = torch.from_numpy(ytr[idx].astype(np.int64)).to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        hist.append(round(tot / n, 4))
        if verbose:
            print(f"    epoch {ep + 1:2d}/{epochs}  loss {tot / n:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    T = 1.0
    if val_mask.any():
        # calibrate on noise-augmented validation windows: clean CWRU validation data is ~100 % correct,
        # which makes temperature scaling degenerate (it would only sharpen the probabilities)
        Xv = np.concatenate([X[val_mask], _augment(X[val_mask], np.random.default_rng(seed + 7))])
        T = fit_temperature(model, Xv, np.concatenate([y[val_mask]] * 2))
    return model, T, hist


def logits_of(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(window_spectrograms(X[i:i + batch])).to(dev)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(model, Xv: np.ndarray, yv: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): pick T minimising validation negative log-likelihood.

    Safety rule: T >= 1, i.e. calibration may soften but never sharpen the probabilities. On near-perfect
    CWRU validation data the NLL optimum is T -> 0 (always ~100 % confident), which would silence the
    copilot's low-confidence warning on unfamiliar machines.
    """
    z = logits_of(model, Xv)
    best_T, best_nll = 1.0, np.inf
    for T in np.exp(np.linspace(0.0, np.log(10), 80)):
        p = softmax(z, T)
        nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return round(best_T, 4)


def save(model, T: float, labels: list[str], path: Path, meta: dict | None = None) -> None:
    import torch

    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "temperature": T,
                "labels": labels, "img_shape": IMG_SHAPE, "meta": meta or {}}, path)


@lru_cache(maxsize=2)
def load(path: str = str(CNN_PATH)):
    import torch

    if not Path(path).exists():
        raise FileNotFoundError(f"CNN model not found at {path}. Run: python scripts/train_cnn.py")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(len(ck["labels"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, float(ck["temperature"]), list(ck["labels"]), ck.get("meta", {})


def predict_proba(X: np.ndarray, path: str = str(CNN_PATH)) -> tuple[np.ndarray, list[str]]:
    model, T, labels, _ = load(path)
    return softmax(logits_of(model, X), T), labels


def write_meta(path: Path, info: dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
