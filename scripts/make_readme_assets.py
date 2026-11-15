"""Generate the figures used in README.md (docs/assets/*.png) from the project's own data and results.

Usage: python scripts/make_readme_assets.py
Every number in the figures comes from data/samples/ and reports/ - nothing is typed in by hand.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intellimaintain import config  # noqa: E402
from intellimaintain.ml.features import summary_features  # noqa: E402  (kurtosis etc. of a whole signal)
from intellimaintain.ml.physics import defect_frequencies, envelope_spectrum  # noqa: E402
from intellimaintain.ml.signal_io import load_signal  # noqa: E402

OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BG, PANEL, GRID, TEXT, MUTED = "#0b1730", "#0f1f3d", "#1f3460", "#e6edf7", "#93a4c3"
CYAN, BLUE, ORANGE, RED, GREEN, VIOLET = "#22d3ee", "#2f8cff", "#f59e0b", "#f43f5e", "#22c55e", "#a78bfa"
CLASS_COLOR = {"Normal": GREEN, "Inner race": RED, "Outer race": ORANGE, "Ball": VIOLET}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "grid.color": GRID, "grid.alpha": .7,
    "font.size": 10.5, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": TEXT,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

SAMPLES = [("Normal", "normal_3hp"), ("Inner race", "inner_race_007_3hp"),
           ("Outer race", "outer_race_007_3hp"), ("Ball", "ball_021_3hp")]


def _sig(stem):
    s = load_signal(config.SAMPLES_DIR / f"{stem}.csv")
    return s.x, s.fs or config.FS, s.rpm or 1730.0


def signals_figure():
    """01 - raw vibration of the four bearing conditions."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True)
    for ax, (name, stem) in zip(axes, SAMPLES):
        x, fs, _ = _sig(stem)
        n = int(0.12 * fs)
        t = np.arange(n) / fs * 1000
        ax.plot(t, x[:n], lw=.8, color=CLASS_COLOR[name])
        k = summary_features(x)["kurtosis"]
        ax.set_ylabel("g")
        ax.grid(True, axis="y")
        ax.text(0.005, 0.86, name, transform=ax.transAxes, fontsize=12, fontweight="bold", color=CLASS_COLOR[name])
        ax.text(0.995, 0.86, f"kurtosis {k:.1f}", transform=ax.transAxes, ha="right", color=TEXT, fontsize=10.5,
                bbox=dict(boxstyle="round,pad=.3", fc=BG, ec=GRID))
    axes[-1].set_xlabel("time [ms]")
    fig.suptitle("Vibration of a healthy bearing and the three fault types (CWRU, 3 HP, 1730 rpm)",
                 fontsize=13, fontweight="bold", x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "signals.png", dpi=150)
    plt.close(fig)


def envelope_figure():
    """02 - the physics check: envelope spectra with the bearing defect frequencies marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (name, stem, key, label) in zip(axes, [
            ("Inner race", "inner_race_007_3hp", "bpfi_hz", "BPFI"),
            ("Outer race", "outer_race_007_3hp", "bpfo_hz", "BPFO")]):
        x, fs, rpm = _sig(stem)
        f, a = envelope_spectrum(x[: 10 * fs], fs)
        m = f <= 520
        a = a[m] / a[m].max()
        ax.fill_between(f[m], a, color=BLUE, alpha=.25, lw=0)
        ax.plot(f[m], a, color=CYAN, lw=1)
        d = defect_frequencies(rpm)
        f0, fr = d[key], rpm / 60
        for h in (1, 2, 3):
            if h * f0 < 520:
                ax.axvline(h * f0, color=CLASS_COLOR[name], ls="--", lw=1.2)
                ax.text(h * f0, 1.04, f"{h}x {label}" if h > 1 else f"{label} {f0:.0f} Hz", ha="center",
                        color=CLASS_COLOR[name], fontsize=9.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=.25", fc=PANEL, ec=CLASS_COLOR[name], lw=.8))
        if name == "Inner race":
            for sb in (-fr, fr):
                ax.axvline(f0 + sb, color=MUTED, ls=":", lw=1)
            ax.text(f0 + fr + 4, .82, "shaft-speed\nsidebands", color=MUTED, fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_xlim(0, 520)
        ax.set_title(f"{name} fault: envelope spectrum", loc="left")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("normalised amplitude")
        ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "envelope.png", dpi=150)
    plt.close(fig)


def model_figure():
    """03 - RandomForest vs spectrogram CNN on the hard tests."""
    ml = json.loads((config.REPORTS_DIR / "ml_metrics.json").read_text())
    cnn = json.loads((config.REPORTS_DIR / "cnn_metrics.json").read_text())
    nr = cnn["noise_robustness_cross_load"]

    def get(d, *keys):
        for k in keys:
            d = d[k]
        return float(d)

    tests = ["Cross-load\n(unseen 3 HP)", "Unseen defect\nsize (0.021\")", "+10 dB\nnoise", "+5 dB\nnoise",
             "+0 dB\nnoise"]
    rfn = ml["noise_robustness_cross_load"]
    rf = [get(ml, "cross_load", "RandomForest", "accuracy"), get(ml, "cross_severity", "accuracy"),
          float(rfn["10dB"]), float(rfn["5dB"]), float(rfn["0dB"])]
    rf_aug = [nr["rf_aug"].get("clean", rf[0]), np.nan, nr["rf_aug"]["10dB"], nr["rf_aug"]["5dB"],
              nr["rf_aug"]["0dB"]]
    cnn_aug = [get(cnn, "cross_load_aug", "accuracy"), get(cnn, "cross_severity", "accuracy"), nr["aug"]["10dB"],
               nr["aug"]["5dB"], nr["aug"]["0dB"]]
    x = np.arange(len(tests))
    w = .26
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, (vals, lab, col) in enumerate([(rf, "RandomForest", BLUE), (rf_aug, "RandomForest + noise augmentation",
                                                                         CYAN), (cnn_aug, "Spectrogram CNN + augmentation",
                                                                                 VIOLET)]):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, .02, "not\ntested", ha="center", fontsize=8, color=MUTED)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.0%}", ha="center", fontsize=8.5, color=TEXT)
    ax.set_xticks(x, tests)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("accuracy")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", ncol=3, fontsize=9.5, bbox_to_anchor=(1, 1.12))
    ax.set_title("Where the models hold up - and where they break", loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(OUT / "model_results.png", dpi=150)
    plt.close(fig)
    return ml, cnn


def copilot_figure():
    """04 - what ML output and retrieval each add to the copilot (2 x 3 ablation)."""
    runs = sorted((config.REPORTS_DIR / "eval").glob("*/ablation_summary.csv"),
                  key=lambda p: (not p.parent.name.startswith("gemini"), p.parent.name))
    df = pd.read_csv(runs[0]).set_index("config")
    order = ["noML+none", "noML+basic", "noML+improved", "ML+none", "ML+basic", "ML+improved"]
    labels = ["LLM only", "basic RAG", "improved RAG", "ML only", "ML + basic RAG", "ML + improved RAG\n(full system)"]
    df = df.loc[[o for o in order if o in df.index]]
    metrics = [("fault_correct", "Fault type correct", ORANGE), ("key_fact_recall", "Key facts in the answer", CYAN),
               ("citation_coverage", "Sentences with a citation", VIOLET)]
    x = np.arange(len(df))
    w = .26
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, (col, lab, c) in enumerate(metrics):
        vals = df[col].fillna(0).values
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c, zorder=2)
    ax.axvspan(len(df) - 1.5, len(df) - .5, color=GREEN, alpha=.10, lw=0, zorder=0)
    ax.set_xticks(x, labels[: len(df)])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncol=3, fontsize=9.5, bbox_to_anchor=(0, 1.12))
    llm = "Gemini" if runs[0].parent.name.startswith("gemini") else "offline generator"
    ax.set_title(f"What ML and retrieval each add to the copilot (20 test cases, {llm})", loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(OUT / "copilot_results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    signals_figure()
    envelope_figure()
    model_figure()
    copilot_figure()
    print(f"Figures written to {OUT}")
