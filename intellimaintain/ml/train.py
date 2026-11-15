"""Train and evaluate the fault classifier.

Experiments (all saved to reports/):
  1. Random window split (80/20, stratified)  -> optimistic: overlapping windows
     from the same recording land in both train and test (data leakage).
  2. Cross-load split: train 0-2 HP, test 3 HP -> honest generalisation test.
  3. Leave-one-load-out (4 folds)              -> robustness across all loads.
  4. RandomForest vs XGBoost, raw vs calibrated probabilities (ECE, Brier).
  5. Physics (envelope spectrum) check accuracy per recording.
The deployed model is a calibrated RandomForest trained on all loads plus an
Isolation Forest anomaly detector fitted on healthy windows only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from .. import config  # noqa: E402
from . import cwru  # noqa: E402
from .dataset import build_feature_table  # noqa: E402
from .features import FEATURE_NAMES  # noqa: E402

MODEL_PATH = config.MODELS_DIR / "classifier.joblib"
CROSSLOAD_MODEL_PATH = config.MODELS_DIR / "classifier_crossload.joblib"  # trained on 0-2 HP only (for evaluation)
METRICS_PATH = config.REPORTS_DIR / "ml_metrics.json"


def make_rf(n_estimators: int = 300) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=config.RANDOM_STATE,
    )


def make_xgb():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
        random_state=config.RANDOM_STATE, eval_metric="mlogloss",
    )


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier(proba: np.ndarray, y_true: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y_true]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def _scores(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "n_test": int(len(y_true)),
        "per_class": classification_report(
            y_true, y_pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0
        ),
    }


# raw numbers behind every figure, saved in the metrics JSON so the app can draw interactive charts
CHART_DATA: dict = {}


def plot_confusion(y_true, y_pred, labels, title, path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    CHART_DATA[Path(path).stem] = {"title": title, "labels": list(labels), "matrix": cm.astype(int).tolist()}
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(probas: dict[str, np.ndarray], y_true, path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    bins = np.linspace(0, 1, 11)
    CHART_DATA["reliability"] = {}
    for name, proba in probas.items():
        conf, correct = proba.max(1), (proba.argmax(1) == y_true)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, "o-", label=f"{name} (ECE={expected_calibration_error(proba, y_true):.3f})")
        CHART_DATA["reliability"][name] = {"confidence": [round(float(v), 4) for v in xs],
                                           "accuracy": [round(float(v), 4) for v in ys],
                                           "ece": round(expected_calibration_error(proba, y_true), 4)}
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability diagram (cross-load test, 3 HP)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(model: RandomForestClassifier, path) -> None:
    imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values().tail(15)
    CHART_DATA["feature_importance"] = {k: round(float(v), 5) for k, v in imp.items()}
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color="#2f6f9f")
    ax.set_title("Top-15 feature importances (RandomForest)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def physics_accuracy(verbose: bool = True) -> dict:
    """How often the envelope-spectrum check alone identifies the fault type (per recording)."""
    from .physics import physics_check

    rows = []
    for c in cwru.available():
        x = cwru.load_file(c)
        ff = physics_check(x, c.rpm)
        truth = cwru.fault_type_of(c.label)
        pred = ff.physics_suggests or "Normal"
        rows.append({"file": c.number, "label": c.label, "truth": truth, "physics": pred,
                     **{f"snr_{k}": v for k, v in ff.envelope_match.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(config.REPORTS_DIR / "physics_check.csv", index=False)
    per_type = df.assign(ok=df.truth == df.physics).groupby("truth")["ok"].mean().round(3).to_dict()
    out = {"overall_accuracy": round(float((df.truth == df.physics).mean()), 3), "per_fault_type": per_type}
    if verbose:
        print(f"Physics envelope check: {out}")
    return out


def case_profiles(df: pd.DataFrame, mask: np.ndarray | None = None) -> dict:
    """Mean standardised feature vector of every CWRU recording ("case library" for similar-case lookup)."""
    d = df if mask is None else df[mask]
    F = d[FEATURE_NAMES].to_numpy()
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    prof = d.assign(**{c: (d[c] - m) / s_ for c, m, s_ in zip(FEATURE_NAMES, mu, sd)}) \
        .groupby("file_id").agg({**{c: "mean" for c in FEATURE_NAMES}, "label": "first", "load_hp": "first",
                                 "rpm": "first"})
    return {"mu": mu, "sd": sd, "vectors": prof[FEATURE_NAMES].to_numpy(),
            "file_id": prof.index.to_numpy().astype(int), "label": prof["label"].tolist(),
            "load_hp": prof["load_hp"].astype(int).tolist(), "rpm": prof["rpm"].astype(int).tolist()}


def run(verbose: bool = True) -> dict:
    CHART_DATA.clear()
    t0 = time.time()
    df = build_feature_table(verbose=verbose)
    le = LabelEncoder().fit(cwru.LABELS)
    labels = list(le.classes_)
    X = df[FEATURE_NAMES].to_numpy()
    y = le.transform(df["label"])
    loads = df["load_hp"].to_numpy()
    results: dict = {"n_windows": int(len(df)), "n_files": int(df.file_id.nunique()),
                     "window": config.WINDOW, "overlap": config.OVERLAP, "labels": labels}

    # 1) random split -----------------------------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    rf = make_rf().fit(Xtr, ytr)
    pred = rf.predict(Xte)
    results["random_split"] = {"RandomForest": _scores(yte, pred, labels)}
    plot_confusion(yte, pred, labels, "RandomForest - random 80/20 split", config.FIGURES_DIR / "cm_random_split.png")

    # 2) cross-load split -------------------------------------------------------
    tr = np.isin(loads, config.TRAIN_LOADS)
    te = np.isin(loads, config.TEST_LOADS)
    rf_cl = make_rf().fit(X[tr], y[tr])
    pred_cl = rf_cl.predict(X[te])
    results["cross_load"] = {"RandomForest": _scores(y[te], pred_cl, labels)}
    plot_confusion(y[te], pred_cl, labels, "RandomForest - cross-load (train 0-2 HP, test 3 HP)",
                   config.FIGURES_DIR / "cm_cross_load.png")
    plot_importance(rf_cl, config.FIGURES_DIR / "feature_importance.png")

    # fault-type level (4 classes) accuracy on the cross-load test
    to_type = np.array([cwru.fault_type_of(lab) for lab in labels])
    results["cross_load"]["RandomForest"]["fault_type_accuracy"] = round(
        float((to_type[pred_cl] == to_type[y[te]]).mean()), 4)

    # 3) XGBoost comparison --------------------------------------------------------
    xgb = make_xgb()
    if xgb is not None:
        xgb.fit(Xtr, ytr)
        results["random_split"]["XGBoost"] = _scores(yte, xgb.predict(Xte), labels)
        xgb_cl = make_xgb().fit(X[tr], y[tr])
        p = xgb_cl.predict(X[te])
        results["cross_load"]["XGBoost"] = _scores(y[te], p, labels)
        results["cross_load"]["XGBoost"]["fault_type_accuracy"] = round(float((to_type[p] == to_type[y[te]]).mean()), 4)

    # 4) calibration (folds grouped by load -> no leakage between overlapping windows)
    tr_idx = np.where(tr)[0]
    tr_loads = loads[tr]
    folds = [(np.where(tr_loads != g)[0], np.where(tr_loads == g)[0]) for g in config.TRAIN_LOADS]
    cal = CalibratedClassifierCV(make_rf(200), method="isotonic", cv=folds).fit(X[tr_idx], y[tr_idx])
    p_raw, p_cal = rf_cl.predict_proba(X[te]), cal.predict_proba(X[te])
    results["calibration_cross_load"] = {
        "raw": {"ece": round(expected_calibration_error(p_raw, y[te]), 4), "brier": round(brier(p_raw, y[te]), 4)},
        "isotonic": {"ece": round(expected_calibration_error(p_cal, y[te]), 4), "brier": round(brier(p_cal, y[te]), 4),
                     "accuracy": round(float((p_cal.argmax(1) == y[te]).mean()), 4)},
    }
    plot_reliability({"RF raw": p_raw, "RF isotonic": p_cal}, y[te], config.FIGURES_DIR / "reliability.png")

    # 5) leave-one-load-out -------------------------------------------------------
    lolo = {}
    for g in sorted(set(loads)):
        m = make_rf(200).fit(X[loads != g], y[loads != g])
        pg = m.predict(X[loads == g])
        lolo[f"test_{g}HP"] = {"accuracy": round(float(accuracy_score(y[loads == g], pg)), 4),
                               "macro_f1": round(float(f1_score(y[loads == g], pg, average="macro")), 4)}
    lolo["mean_accuracy"] = round(float(np.mean([v["accuracy"] for v in lolo.values()])), 4)
    results["leave_one_load_out"] = lolo

    # 5b) cross-severity: 4-class fault TYPE, train on 0.007"/0.014", test on unseen 0.021"
    ftype = df["fault_type"].to_numpy()
    lab = df["label"].to_numpy()
    sev_train = np.array([(l_ == "Normal" and ld in (0, 1)) or l_.endswith(("007", "014"))
                          for l_, ld in zip(lab, loads)])
    sev_test = np.array([(l_ == "Normal" and ld in (2, 3)) or l_.endswith("021") for l_, ld in zip(lab, loads)])
    m_sev = make_rf().fit(X[sev_train], ftype[sev_train])
    p_sev = m_sev.predict(X[sev_test])
    type_labels = ["Normal", "Ball", "Inner race", "Outer race"]
    results["cross_severity"] = {
        "description": "4-class fault type; train 0.007/0.014 in (+Normal 0-1 HP), test unseen 0.021 in (+Normal 2-3 HP)",
        "accuracy": round(float(accuracy_score(ftype[sev_test], p_sev)), 4),
        "macro_f1": round(float(f1_score(ftype[sev_test], p_sev, average="macro")), 4),
        "per_class_recall": {t: round(float((p_sev[ftype[sev_test] == t] == t).mean()), 3) for t in type_labels},
    }
    enc = {t: i for i, t in enumerate(type_labels)}
    plot_confusion(np.array([enc[v] for v in ftype[sev_test]]), np.array([enc[v] for v in p_sev]), type_labels,
                   "Cross-severity (unseen 0.021 in defects)", config.FIGURES_DIR / "cm_cross_severity.png")

    # 5c) noise robustness of the cross-load model (white noise added to raw 3 HP signals)
    from .features import signal_features

    rng = np.random.default_rng(config.RANDOM_STATE)
    test_files = [c for c in cwru.available() if c.load_hp in config.TEST_LOADS]
    noise = {}
    for snr_db in (None, 20, 10, 5, 0):
        yt, yp = [], []
        for c in test_files:
            x = cwru.load_file(c)
            if snr_db is not None:
                p_sig = np.mean(x**2)
                x = x + rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), size=len(x))
            f = signal_features(x)[FEATURE_NAMES].to_numpy()
            yp.extend(rf_cl.predict(f))
            yt.extend([le.transform([c.label])[0]] * len(f))
        noise["clean" if snr_db is None else f"{snr_db}dB"] = round(float(accuracy_score(yt, yp)), 4)
    results["noise_robustness_cross_load"] = noise
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(list(noise.keys()), list(noise.values()), "o-", color="#c0392b")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (3 HP test)")
    ax.set_xlabel("Signal-to-noise ratio of added white noise")
    ax.set_title("Noise robustness of the cross-load RandomForest")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "noise_robustness.png", dpi=150)
    plt.close(fig)

    # 6) physics cross-check --------------------------------------------------------
    results["physics_envelope_check"] = physics_accuracy(verbose)

    # 7) deployable model: calibrated RF on all loads + anomaly detector ------------
    all_folds = [(np.where(loads != g)[0], np.where(loads == g)[0]) for g in sorted(set(loads))]
    final = CalibratedClassifierCV(make_rf(300), method="isotonic", cv=all_folds).fit(X, y)
    normal = df["label"].to_numpy() == "Normal"
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=config.RANDOM_STATE).fit(X[normal])
    raw_normal = -iso.score_samples(X[normal])
    bundle = {
        "model": final, "labels": labels, "features": FEATURE_NAMES, "iso": iso,
        "iso_q50": float(np.quantile(raw_normal, 0.50)), "iso_q99": float(np.quantile(raw_normal, 0.99)),
        "fs": config.FS, "window": config.WINDOW, "overlap": config.OVERLAP,
        "trained_on": "CWRU 12k drive-end, loads 0-3 HP, 10 classes",
        "sklearn_version": __import__("sklearn").__version__,
        "case_profiles": case_profiles(df),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    # evaluation bundle: same pipeline but trained on 0-2 HP only, so the 3 HP test recordings are unseen
    normal_tr = normal & tr
    iso_cl = IsolationForest(n_estimators=300, random_state=config.RANDOM_STATE).fit(X[normal_tr])
    raw_cl = -iso_cl.score_samples(X[normal_tr])
    joblib.dump({**bundle, "model": cal, "iso": iso_cl, "iso_q50": float(np.quantile(raw_cl, 0.5)),
                 "iso_q99": float(np.quantile(raw_cl, 0.99)), "trained_on": "CWRU loads 0-2 HP only",
                 "case_profiles": case_profiles(df, tr)},  # no 3 HP cases -> no test leakage
                CROSSLOAD_MODEL_PATH, compress=3)
    results["model_path"] = str(MODEL_PATH.relative_to(config.ROOT))
    results["train_seconds"] = round(time.time() - t0, 1)
    results["chart_data"] = dict(CHART_DATA)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    write_markdown_summary(results)
    if verbose:
        print_summary(results)
    return results


def print_summary(r: dict) -> None:
    print("\n=== ML results ===")
    for split in ("random_split", "cross_load"):
        for model, s in r[split].items():
            extra = f"  fault-type acc={s['fault_type_accuracy']:.3f}" if "fault_type_accuracy" in s else ""
            print(f"{split:<13} {model:<12} acc={s['accuracy']:.4f}  macroF1={s['macro_f1']:.4f}{extra}")
    print("Leave-one-load-out:", r["leave_one_load_out"])
    print("Cross-severity:", {k: v for k, v in r["cross_severity"].items() if k != "description"})
    print("Noise robustness:", r["noise_robustness_cross_load"])
    print("Calibration (cross-load):", r["calibration_cross_load"])
    print(f"Saved model -> {r['model_path']}  ({r['train_seconds']} s)")


def write_markdown_summary(r: dict) -> None:
    lines = ["# ML results (auto-generated by `python scripts/train_ml.py`)", "",
             f"Windows: {r['n_windows']} from {r['n_files']} recordings, "
             f"window={r['window']} samples, overlap={int(r['overlap'] * 100)} %", "",
             "| Split | Model | Accuracy | Macro-F1 | Fault-type acc. |", "|---|---|---|---|---|"]
    for split, name in (("random_split", "Random 80/20"), ("cross_load", "Cross-load 0-2 -> 3 HP")):
        for model, s in r[split].items():
            fta = f"{s['fault_type_accuracy']:.4f}" if "fault_type_accuracy" in s else "-"
            lines.append(f"| {name} | {model} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} | {fta} |")
    lines += ["", "## Leave-one-load-out (RandomForest)", "", "| Test load | Accuracy | Macro-F1 |", "|---|---|---|"]
    for k, v in r["leave_one_load_out"].items():
        if k != "mean_accuracy":
            lines.append(f"| {k} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} |")
    lines += [f"| **mean** | **{r['leave_one_load_out']['mean_accuracy']:.4f}** | |", "",
              "## Calibration on the cross-load test", "",
              "| Probabilities | ECE (lower is better) | Brier |", "|---|---|---|"]
    for k, v in r["calibration_cross_load"].items():
        lines.append(f"| {k} | {v['ece']:.4f} | {v['brier']:.4f} |")
    cs = r["cross_severity"]
    lines += ["", "## Cross-severity generalisation (harder test)", "", cs["description"], "",
              f"Accuracy {cs['accuracy']:.4f}, macro-F1 {cs['macro_f1']:.4f}", "",
              "| Fault type | Recall |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in cs["per_class_recall"].items()]
    lines += ["", "## Noise robustness (cross-load model, 3 HP test)", "",
              "| Added noise | Accuracy |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in r["noise_robustness_cross_load"].items()]
    pe = r["physics_envelope_check"]
    lines += ["", "## Physics cross-check (envelope spectrum only, no ML)", "",
              f"Overall per-recording accuracy: {pe['overall_accuracy']:.3f}", "",
              "| Fault type | Accuracy |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in pe["per_fault_type"].items()]
    lines += ["", "Figures: `reports/figures/cm_random_split.png`, `cm_cross_load.png`, "
              "`reliability.png`, `feature_importance.png`."]
    (config.REPORTS_DIR / "ml_results.md").write_text("\n".join(lines) + "\n")
