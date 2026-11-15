"""Spectrogram-CNN experiments, run on exactly the same splits as the RandomForest (train.py).

Outputs: reports/cnn_metrics.json, reports/cnn_results.md (incl. RF vs CNN table) and figures
cm_cnn_cross_load.png, cm_cnn_cross_severity.png, noise_rf_vs_cnn.png, spectrogram_examples.png.
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from .. import config  # noqa: E402
from . import cnn, cwru  # noqa: E402
from .features import sliding_windows, window_features  # noqa: E402
from .spectrogram import window_spectrograms  # noqa: E402
from .train import CHART_DATA, brier, expected_calibration_error, make_rf, plot_confusion  # noqa: E402

METRICS_PATH = config.REPORTS_DIR / "cnn_metrics.json"
TYPES = ["Normal", "Ball", "Inner race", "Outer race"]


def _scores(y, p) -> dict:
    return {"accuracy": round(float(accuracy_score(y, p)), 4), "macro_f1": round(float(f1_score(y, p, average="macro")), 4)}


def plot_examples(X, labels, path) -> None:
    """Dark-themed gallery: one CNN input image per class (matches the app's dashboard)."""
    bg, fg = "#0b1630", "#e6eefc"
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.4), facecolor=bg)
    for ax, lab in zip(axes.ravel(), cwru.LABELS):
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            ax.axis("off")
            continue
        i = int(idx[len(idx) // 2])
        img = window_spectrograms(X[i:i + 1])[0, 0]
        ax.imshow(img, origin="lower", aspect="auto", cmap="magma", extent=[0, 2048 / config.FS * 1000, 0, 6000])
        ax.set_facecolor(bg)
        ax.set_title(lab, fontsize=10, color=fg)
        ax.set_xlabel("ms", fontsize=8, color=fg)
        ax.set_ylabel("Hz", fontsize=8, color=fg)
        ax.tick_params(labelsize=7, colors=fg)
        for sp in ax.spines.values():
            sp.set_color("#2a3f6e")
    fig.suptitle("CNN input: standardised log-power spectrogram of one 2048-sample window per class", color=fg)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=bg)
    plt.close(fig)


def run(epochs: int = 14, verbose: bool = True) -> dict:
    if not cnn.torch_available():
        raise ImportError("PyTorch is required: pip install torch (it is installed with sentence-transformers)")
    t0 = time.time()
    X, lab, loads, files, pos = cnn.load_windows(verbose=verbose)
    labels = list(cwru.LABELS)
    labels_sorted = sorted(labels)  # same order as the RandomForest LabelEncoder
    y = np.array([labels_sorted.index(v) for v in lab])
    res: dict = {"n_windows": int(len(X)), "epochs": epochs, "device": str(cnn.get_device()),
                 "architecture": "4 conv blocks (16-32-64-64) + global avg pool + dropout + linear, ~61k params",
                 "input": "64x64 standardised log-power STFT (nperseg 128, 75% overlap)"}
    plot_examples(X, lab, config.FIGURES_DIR / "spectrogram_examples.png")

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # 1) random split -------------------------------------------------------------
    log("[1/5] random 80/20 split")
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    m, T, _ = cnn.train_cnn(X[tr], y[tr], len(labels), epochs=epochs, verbose=verbose)
    p = cnn.softmax(cnn.logits_of(m, X[te]), T).argmax(1)
    res["random_split"] = _scores(y[te], p)

    # 2) cross-load, with and without augmentation ----------------------------------
    trl = np.isin(loads, config.TRAIN_LOADS)
    tel = np.isin(loads, config.TEST_LOADS)
    val = trl & (pos > 0.85)  # last 15 % of each training recording -> temperature scaling
    idx_tr = np.where(trl)[0]
    cl_models = {}
    for aug in (False, True):
        tag = "cross_load_aug" if aug else "cross_load_noaug"
        log(f"[{2 + aug}/5] cross-load (train 0-2 HP, test 3 HP), augmentation={aug}")
        m, T, _ = cnn.train_cnn(X[idx_tr], y[idx_tr], len(labels), val_mask=val[idx_tr], epochs=epochs,
                                augment=aug, verbose=verbose)
        z = cnn.logits_of(m, X[tel])
        p_raw, p_cal = cnn.softmax(z, 1.0), cnn.softmax(z, T)
        pred = p_cal.argmax(1)
        to_type = np.array([cwru.fault_type_of(v) for v in labels_sorted])
        res[tag] = {**_scores(y[tel], pred),
                    "fault_type_accuracy": round(float((to_type[pred] == to_type[y[tel]]).mean()), 4),
                    "temperature": T,
                    "ece_raw": round(expected_calibration_error(p_raw, y[tel]), 4),
                    "ece_temperature_scaled": round(expected_calibration_error(p_cal, y[tel]), 4),
                    "brier_temperature_scaled": round(brier(p_cal, y[tel]), 4)}
        cl_models[aug] = (m, T)
        if aug:
            plot_confusion(y[tel], pred, labels_sorted, "Spectrogram CNN - cross-load (train 0-2 HP, test 3 HP)",
                           config.FIGURES_DIR / "cm_cnn_cross_load.png")
            cnn.save(m, T, labels_sorted, cnn.CNN_CROSSLOAD_PATH, {"trained_on": "loads 0-2 HP", "augment": True})

    # 3) cross-severity (4 fault types, unseen 0.021 in) ------------------------------
    log("[4/5] cross-severity (unseen 0.021 in defects)")
    ft = np.array([cwru.fault_type_of(v) for v in lab])
    yt = np.array([TYPES.index(v) for v in ft])
    sev_tr = np.array([(v == "Normal" and ld in (0, 1)) or v.endswith(("007", "014")) for v, ld in zip(lab, loads)])
    sev_te = np.array([(v == "Normal" and ld in (2, 3)) or v.endswith("021") for v, ld in zip(lab, loads)])
    m, T, _ = cnn.train_cnn(X[sev_tr], yt[sev_tr], 4, epochs=epochs, verbose=verbose)
    p = cnn.softmax(cnn.logits_of(m, X[sev_te]), T).argmax(1)
    res["cross_severity"] = {**_scores(yt[sev_te], p),
                             "per_class_recall": {t: round(float((p[yt[sev_te] == i] == i).mean()), 3)
                                                  for i, t in enumerate(TYPES)}}
    plot_confusion(yt[sev_te], p, TYPES, "Spectrogram CNN - cross-severity (unseen 0.021 in)",
                   config.FIGURES_DIR / "cm_cnn_cross_severity.png")

    # 4) noise robustness (same protocol as the RandomForest) ---------------------------
    # fairness control: the RandomForest trained with the SAME noise augmentation as the CNN
    log("      RandomForest with the same noise augmentation (fairness control)")
    aug_rng = np.random.default_rng(config.RANDOM_STATE + 1)
    Xa = cnn._augment(X[idx_tr], aug_rng)
    rf_aug = make_rf().fit(np.vstack([window_features(X[idx_tr]), window_features(Xa)]), np.concatenate([y[idx_tr]] * 2))
    rng = np.random.default_rng(config.RANDOM_STATE)
    test_files = [c for c in cwru.available() if c.load_hp in config.TEST_LOADS]
    noise = {"noaug": {}, "aug": {}, "rf_aug": {}}
    for snr_db in (None, 20, 10, 5, 0):
        key = "clean" if snr_db is None else f"{snr_db}dB"
        wins, ys = [], []
        for c in test_files:
            x = cwru.load_file(c)
            if snr_db is not None:
                x = x + rng.normal(0, np.sqrt(np.mean(x**2) / 10 ** (snr_db / 10)), size=len(x))
            w = sliding_windows(x).astype(np.float32)
            wins.append(w)
            ys += [labels_sorted.index(c.label)] * len(w)
        W, ys = np.concatenate(wins), np.array(ys)
        noise["rf_aug"][key] = round(float(accuracy_score(ys, rf_aug.predict(window_features(W)))), 4)
        if snr_db == 10:  # calibration under realistic noise, where errors actually occur
            m_a, T_a = cl_models[True]
            z10 = cnn.logits_of(m_a, W)
            res["cross_load_aug"]["ece_10dB_raw"] = round(expected_calibration_error(cnn.softmax(z10, 1.0), ys), 4)
            res["cross_load_aug"]["ece_10dB_temperature_scaled"] = round(
                expected_calibration_error(cnn.softmax(z10, T_a), ys), 4)
        for aug, (m, T) in cl_models.items():
            pr = cnn.softmax(cnn.logits_of(m, W), T).argmax(1)
            noise["aug" if aug else "noaug"][key] = round(float(accuracy_score(ys, pr)), 4)
    res["noise_robustness_cross_load"] = noise

    # 5) deployable model: all loads, augmentation, temperature scaling ------------------
    log("[5/5] final model on all loads")
    valall = pos > 0.85
    m, T, hist = cnn.train_cnn(X, y, len(labels), val_mask=valall, epochs=epochs, verbose=verbose)
    cnn.save(m, T, labels_sorted, cnn.CNN_PATH, {"trained_on": "loads 0-3 HP", "augment": True, "loss": hist})
    cnn.load.cache_clear()
    res["train_seconds"] = round(time.time() - t0, 1)
    res["chart_data"] = {k: v for k, v in CHART_DATA.items() if k.startswith("cm_cnn")}
    METRICS_PATH.write_text(json.dumps(res, indent=2))
    write_comparison(res)
    if verbose:
        print((config.REPORTS_DIR / "cnn_results.md").read_text())
    return res


def write_comparison(c: dict) -> None:
    rf_path = config.REPORTS_DIR / "ml_metrics.json"
    rf = json.loads(rf_path.read_text()) if rf_path.exists() else {}
    noise_rf = rf.get("noise_robustness_cross_load", {})

    def g(d, *keys):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return float("nan")
            d = d[k]
        return d

    rows = [
        ("Random 80/20 split (10 classes)", g(rf, "random_split", "RandomForest", "accuracy"), c["random_split"]["accuracy"]),
        ("Cross-load 0-2 → 3 HP (10 classes)", g(rf, "cross_load", "RandomForest", "accuracy"), c["cross_load_aug"]["accuracy"]),
        ("Cross-load macro-F1", g(rf, "cross_load", "RandomForest", "macro_f1"), c["cross_load_aug"]["macro_f1"]),
        ("Cross-severity, unseen 0.021\" (4 types)", g(rf, "cross_severity", "accuracy"), c["cross_severity"]["accuracy"]),
    ]
    for k in ("20dB", "10dB", "5dB", "0dB"):
        rows.append((f"Cross-load + {k} noise", noise_rf.get(k, float("nan")), c["noise_robustness_cross_load"]["aug"][k]))
    lines = ["# RandomForest vs Spectrogram CNN (auto-generated by `python scripts/train_cnn.py`)", "",
             f"CNN: {c['architecture']}; input {c['input']}; {c['epochs']} epochs on {c['device']}.", "",
             "| Test | RandomForest (features) | Spectrogram CNN |", "|---|---|---|"]
    for name, a, b in rows:
        lines.append(f"| {name} | {a:.4f} | {b:.4f} |")
    na, nn_ = c["noise_robustness_cross_load"]["aug"], c["noise_robustness_cross_load"]["noaug"]
    ra = c["noise_robustness_cross_load"].get("rf_aug", {})
    lines += ["", "## Is it the CNN or the augmentation? (cross-load test with added noise)", "",
              "Both models trained without and with the same noise augmentation (random 5-30 dB white noise + shift).", "",
              "| Added noise | RF | RF + augmentation | CNN | CNN + augmentation |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {noise_rf.get(k, float('nan')):.4f} | {ra.get(k, float('nan')):.4f} | {nn_[k]:.4f} | {na[k]:.4f} |"
              for k in na]
    cs = c["cross_severity"]["per_class_recall"]
    lines += ["", "## Cross-severity recall per fault type (CNN)", "", "| Fault type | Recall |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in cs.items()]
    cl = c["cross_load_aug"]
    lines += ["", "## CNN calibration (temperature scaling, fitted on noise-augmented validation windows)", "",
              f"Temperature T = {cl['temperature']} (constrained to T >= 1: calibration may soften, never sharpen, "
              "so the low-confidence warning still works on unfamiliar data).", "",
              "| Test set | ECE before (T = 1) | ECE after |", "|---|---|---|",
              f"| Cross-load, clean | {cl['ece_raw']:.4f} | {cl['ece_temperature_scaled']:.4f} |",
              f"| Cross-load, 10 dB noise | {cl.get('ece_10dB_raw', float('nan')):.4f} | "
              f"{cl.get('ece_10dB_temperature_scaled', float('nan')):.4f} |", "",
              "Figures: `reports/figures/spectrogram_examples.png`, `cm_cnn_cross_load.png`, "
              "`cm_cnn_cross_severity.png`, `noise_rf_vs_cnn.png`."]
    (config.REPORTS_DIR / "cnn_results.md").write_text("\n".join(lines) + "\n")

    if noise_rf:
        keys = list(na.keys())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(keys, [noise_rf.get(k, np.nan) for k in keys], "o-", label="RandomForest (features)", color="#2f6f9f")
        ax.plot(keys, [nn_[k] for k in keys], "s--", label="CNN (no augmentation)", color="#8e44ad")
        ax.plot(keys, [na[k] for k in keys], "^-", label="CNN (+noise augmentation)", color="#c0392b")
        if ra:
            ax.plot(keys, [ra.get(k, np.nan) for k in keys], "d:", label="RF (+noise augmentation)", color="#27ae60")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy (3 HP test)")
        ax.set_xlabel("Added white noise (SNR)")
        ax.set_title("Noise robustness: RandomForest vs Spectrogram CNN")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "noise_rf_vs_cnn.png", dpi=150)
        plt.close(fig)
