"""Precision, recall, F1 and accuracy for every part of IntelliMaintain, in one table.

Usage: python scripts/compute_metrics.py          (needs the CWRU data in data/raw, ~2-4 minutes)
Writes reports/metrics_summary.md and reports/metrics_summary.json.

The experiments use the same splits, seeds and models as intellimaintain/ml/train.py; this script only adds the
precision / recall / F1 that train.py does not store for the hard tests (unseen defect size, noise, physics).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intellimaintain import config  # noqa: E402
from intellimaintain.ml import cwru  # noqa: E402
from intellimaintain.ml.features import FEATURE_NAMES, signal_features  # noqa: E402
from intellimaintain.ml.train import build_feature_table, make_rf  # noqa: E402


def prf(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4), "precision": round(float(p), 4),
            "recall": round(float(r), 4), "f1": round(float(f), 4), "weighted_f1": round(float(fw), 4),
            "n": int(len(y_true))}


def per_class(y_true, y_pred, labels) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {lab: {"precision": round(float(a), 3), "recall": round(float(b), 3), "f1": round(float(c), 3),
                  "support": int(d)} for lab, a, b, c, d in zip(labels, p, r, f, s)}


def main():
    out: dict = {}
    ml = json.loads((config.REPORTS_DIR / "ml_metrics.json").read_text())
    cnn = json.loads((config.REPORTS_DIR / "cnn_metrics.json").read_text())

    def stored(d):
        m = d["per_class"]["macro avg"]
        return {"accuracy": d["accuracy"], "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
                "f1": round(m["f1-score"], 4), "weighted_f1": round(d["per_class"]["weighted avg"]["f1-score"], 4),
                "n": d["n_test"]}

    out["RF random split (10 classes)"] = stored(ml["random_split"]["RandomForest"])
    out["XGBoost random split (10 classes)"] = stored(ml["random_split"]["XGBoost"])
    out["RF cross-load, unseen 3 HP (10 classes)"] = stored(ml["cross_load"]["RandomForest"])
    out["XGBoost cross-load, unseen 3 HP (10 classes)"] = stored(ml["cross_load"]["XGBoost"])

    # ---- recompute the hard tests with precision / recall
    df = build_feature_table(verbose=False)
    X = df[FEATURE_NAMES].to_numpy()
    lab = df["label"].to_numpy()
    ftype = df["fault_type"].to_numpy()
    loads = df["load_hp"].to_numpy()

    sev_train = np.array([(l_ == "Normal" and ld in (0, 1)) or l_.endswith(("007", "014"))
                          for l_, ld in zip(lab, loads)])
    sev_test = np.array([(l_ == "Normal" and ld in (2, 3)) or l_.endswith("021") for l_, ld in zip(lab, loads)])
    p_sev = make_rf().fit(X[sev_train], ftype[sev_train]).predict(X[sev_test])
    out["RF unseen defect size 0.021 in (4 fault types)"] = prf(ftype[sev_test], p_sev)
    out["_per_class_unseen_defect_size"] = per_class(ftype[sev_test], p_sev,
                                                     ["Normal", "Ball", "Inner race", "Outer race"])

    tr = np.isin(loads, config.TRAIN_LOADS)
    rf_cl = make_rf().fit(X[tr], lab[tr])
    rng = np.random.default_rng(config.RANDOM_STATE)
    test_files = [c for c in cwru.available() if c.load_hp in config.TEST_LOADS]
    for snr_db in (10, 5, 0):
        yt, yp = [], []
        for c in test_files:
            x = cwru.load_file(c)
            x = x + rng.normal(0, np.sqrt(np.mean(x**2) / 10 ** (snr_db / 10)), size=len(x))
            f = signal_features(x)[FEATURE_NAMES].to_numpy()
            yp.extend(rf_cl.predict(f))
            yt.extend([c.label] * len(f))
        out[f"RF cross-load + {snr_db} dB noise (10 classes)"] = prf(yt, yp)

    # ---- CNN (stored by train_cnn.py: accuracy and macro-F1 only)
    for key, name in (("random_split", "CNN random split (10 classes)"),
                      ("cross_load_aug", "CNN cross-load, unseen 3 HP (10 classes)"),
                      ("cross_severity", "CNN unseen defect size 0.021 in (4 fault types)")):
        d = cnn[key]
        out[name] = {"accuracy": d["accuracy"], "f1": d.get("macro_f1")}
        if d["accuracy"] == 1.0:  # every window correct -> precision and recall are 1 as well
            out[name].update(precision=1.0, recall=1.0)

    # ---- physics check alone (per recording, 4 fault types)
    phys = pd.read_csv(config.REPORTS_DIR / "physics_check.csv")
    out["Physics envelope check alone (40 recordings)"] = prf(phys["truth"], phys["physics"])
    out["_per_class_physics"] = per_class(phys["truth"], phys["physics"],
                                          ["Normal", "Ball", "Inner race", "Outer race"])

    (config.REPORTS_DIR / "metrics_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
