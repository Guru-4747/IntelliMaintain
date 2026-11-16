"""Short code snippets for the viva. Run all: python scripts/viva_snippets.py
Or copy one block at a time into a Python shell / Jupyter cell (start every session with BLOCK 0).
Part A uses the project's own functions. Part B writes the same things by hand (numpy / scipy / sklearn),
for when an examiner asks "show it without your files".
"""
# ============================================================ BLOCK 0 - setup (always run first)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

# ============================================================ PART A - with the project's functions

# ---- A1. Import the CWRU dataset (list all 40 recordings, load one)
from intellimaintain.ml import cwru

files = cwru.available()                          # recordings found in data/raw
print(len(files), "recordings")
c = cwru.BY_NUMBER[108]                           # 108.mat = inner race, 0.007 in, 3 HP
x = cwru.load_file(c)                             # numpy array, 12 kHz, already resampled
print(c.label, c.load_hp, "HP", c.rpm, "rpm", len(x), "samples")

# ---- A2. Import the test cases
from intellimaintain.evaluation.evaluate import case_signal, load_cases

cases = load_cases()                              # eval/test_cases.csv -> DataFrame
print(cases[["case_id", "fault_type", "cwru_file", "rpm"]].head())

# ---- A3. Import a demo sample or any uploaded file (CSV / TXT / MAT / NPY)
from intellimaintain.ml.signal_io import load_signal

s = load_signal(ROOT / "data/samples/outer_race_007_3hp.csv")
print(len(s.x), "samples at", s.fs, "Hz,", s.rpm, "rpm")

# ---- A4. Classify a signal
from intellimaintain.ml.classifier import classify_signal

r = classify_signal(x)
print(r.label, r.fault_type, round(r.confidence, 3))

# ---- A5. Physics check (defect frequencies + envelope spectrum)
from intellimaintain.ml.physics import physics_check

ff = physics_check(x, rpm=c.rpm)
print(round(ff.bpfi_hz, 1), "Hz BPFI ->", ff.physics_suggests)

# ---- A6. Search the knowledge base
from intellimaintain.rag.retriever import retrieve_evidence

for e in retrieve_evidence("how to fix an inner race fault", top_k=3):
    print(e.source, "page", e.page, round(e.score, 3))

# ---- A7. Ask the copilot (offline here; use LLM("gemini") with a key)
from intellimaintain.agent.agent import DiagnosisAgent
from intellimaintain.agent.llm import LLM

agent = DiagnosisAgent(LLM("offline"))
d = agent.run("What is wrong with this bearing?", signal=x, rpm=c.rpm)
print(d.fault_type, "|", d.answer[:150])

# ---- A8. Run all test cases and score the fault type
rows = []
for _, row in cases.iterrows():
    sig, rpm = case_signal(row)                   # None for knowledge-only questions
    if sig is None:
        continue
    d = agent.run(row["question"], signal=sig, rpm=rpm)
    rows.append({"case": row["case_id"], "truth": row["fault_type"], "pred": d.fault_type})
res = pd.DataFrame(rows)
print(res, "\naccuracy:", (res.truth == res.pred).mean())

# ============================================================ PART B - by hand (not using the project files)

# ---- B1. RMS, peak, crest factor, kurtosis of a signal
from scipy.stats import kurtosis

rms = np.sqrt(np.mean(x ** 2))
peak = np.max(np.abs(x))
crest = peak / rms
kurt = kurtosis(x, fisher=False)                  # healthy, Gaussian-like signal ~ 3
print(f"RMS {rms:.3f}  peak {peak:.3f}  crest {crest:.2f}  kurtosis {kurt:.2f}")

# ---- B2. Cut a signal into 2048-sample windows with 50% overlap
win, step = 2048, 1024
windows = np.array([x[i:i + win] for i in range(0, len(x) - win + 1, step)])
print(windows.shape)                              # (number of windows, 2048)

# ---- B3. Bearing defect frequencies (SKF 6205: 9 balls, d = 0.3126 in, D = 1.537 in)
n, d_b, D, rpm = 9, 0.3126, 1.537, 1730
fr = rpm / 60                                     # shaft frequency in Hz
ratio = d_b / D
bpfo = n / 2 * fr * (1 - ratio)
bpfi = n / 2 * fr * (1 + ratio)
bsf = D / (2 * d_b) * fr * (1 - ratio ** 2)
ftf = fr / 2 * (1 - ratio)
print(f"BPFO {bpfo:.1f}  BPFI {bpfi:.1f}  BSF {bsf:.1f}  2xBSF {2 * bsf:.1f}  FTF {ftf:.1f} Hz")

# ---- B4. FFT and envelope spectrum (band-pass -> Hilbert envelope -> FFT)
from scipy.signal import butter, filtfilt, hilbert

fs = 12000
b, a = butter(4, [2000, 5000], btype="band", fs=fs)
env = np.abs(hilbert(filtfilt(b, a, x)))          # outline of the impact bursts
env = env - env.mean()
freqs = np.fft.rfftfreq(len(env), 1 / fs)
amp = np.abs(np.fft.rfft(env))
band = (freqs > 20) & (freqs < 500)
print("strongest envelope peak:", round(freqs[band][np.argmax(amp[band])], 1), "Hz  (BPFI =", round(bpfi, 1), ")")

# ---- B5. Train a RandomForest from scratch: train 0-2 HP, test unseen 3 HP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from intellimaintain.ml.dataset import build_feature_table
from intellimaintain.ml.features import FEATURE_NAMES

df = build_feature_table(verbose=False)           # 4643 windows x 29 features
train, test = df[df.load_hp < 3], df[df.load_hp == 3]
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(train[FEATURE_NAMES], train["label"])
pred = rf.predict(test[FEATURE_NAMES])
print(classification_report(test["label"], pred, digits=3))   # precision, recall, F1 per class

# ---- B6. Precision, recall and F1 by hand from a confusion matrix
labels = sorted(df["label"].unique())
cm = confusion_matrix(test["label"], pred, labels=labels)
tp = np.diag(cm)
precision = tp / cm.sum(axis=0).clip(min=1)       # column = predicted
recall = tp / cm.sum(axis=1).clip(min=1)          # row = true
f1 = 2 * precision * recall / (precision + recall + 1e-12)
print(pd.DataFrame({"precision": precision, "recall": recall, "f1": f1}, index=labels).round(3))
print("accuracy:", tp.sum() / cm.sum())

# ---- B7. Most important features
imp = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
print(imp.head(5).round(3))

# ---- B8. Plot a signal and its envelope spectrum
import matplotlib

matplotlib.use("Agg")                              # remove this line in Jupyter to show the plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(9, 5))
t = np.arange(1200) / fs * 1000
ax[0].plot(t, x[:1200])
ax[0].set_xlabel("time [ms]")
ax[1].plot(freqs[band], amp[band])
ax[1].axvline(bpfi, color="r", ls="--", label="BPFI")
ax[1].set_xlabel("frequency [Hz]")
ax[1].legend()
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/viva_plot.png")
print("saved reports/figures/viva_plot.png")
