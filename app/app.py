"""IntelliMaintain - Streamlit dashboard.

Run:  streamlit run app/app.py
Panels: signal upload -> prediction -> evidence -> cited explanation -> human review (accept / reject / edit).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme  # noqa: E402  (dark industrial look: CSS + Plotly template)
import results_dashboard  # noqa: E402  (Model & evaluation results tab)

from intellimaintain import config  # noqa: E402
from intellimaintain.agent.agent import DiagnosisAgent  # noqa: E402
from intellimaintain.agent.llm import DEFAULT_MODELS, LLM, SUGGESTED_GEMINI_MODELS, gemini_key_available  # noqa: E402
from intellimaintain.ml.classifier import cnn_ready  # noqa: E402
from intellimaintain.ml.physics import defect_frequencies, envelope_spectrum  # noqa: E402
from intellimaintain.ml.spectrogram import signal_spectrogram  # noqa: E402
from intellimaintain.ml.signal_io import load_signal, load_signal_csv  # noqa: E402
from intellimaintain.schemas import Diagnosis  # noqa: E402
from intellimaintain.security import redact  # noqa: E402

MAX_LLM_RUNS = int(os.getenv("IM_MAX_LLM_RUNS_PER_SESSION", "30"))

st.set_page_config(page_title="IntelliMaintain", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)


def _secrets_file_present() -> bool:
    """True only when a secrets.toml exists in one of the places Streamlit reads (Streamlit Cloud writes the
    app's Secrets there). Checked on disk, without touching st.secrets: some Streamlit versions show a red
    "No secrets files found" banner as soon as st.secrets is accessed and no file exists."""
    if getattr(st.secrets, "_secrets", None):  # already loaded (cloud / test harness) - no parsing triggered
        return True
    candidates = [Path.home() / ".streamlit" / "secrets.toml", ROOT / ".streamlit" / "secrets.toml",
                  Path.cwd() / ".streamlit" / "secrets.toml"]
    try:
        configured = st.get_option("secrets.files")  # newer Streamlit: the exact list it reads
        candidates += [Path(os.path.expanduser(str(f))) for f in (configured or [])]
    except Exception:  # noqa: BLE001 - option not available in older versions
        pass
    return any(c.is_file() for c in candidates)


def load_cloud_secrets() -> None:
    """Online (Streamlit Community Cloud): copy GEMINI_API_KEY etc. from st.secrets into the environment.
    Locally the key comes from .env, so nothing happens when there is no secrets file."""
    if not _secrets_file_present():
        return
    try:
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "IM_LLM_MODEL", "IM_LLM_RPM"):
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:  # noqa: BLE001
        pass


load_cloud_secrets()

FAULT_COLOR = theme.FAULT_COLOR
CONSISTENCY_ICON = {"consistent": "", "conflict": "", "inconclusive": "", "not_run": ""}


# ------------------------------------------------------------------ helpers
def system_status() -> dict:
    return {
        "model": (config.MODELS_DIR / "classifier.joblib").exists(),
        "index": (config.INDEX_DIR / "faiss.index").exists(),
    }


def read_upload(uploaded) -> tuple[np.ndarray, int | None, float | None, str]:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".csv", ".txt")):
        sig = load_signal_csv(data)
    else:
        tmp = config.PROCESSED_DIR / f"_upload_{Path(name).name}"
        tmp.write_bytes(data)
        try:
            sig = load_signal(tmp, name)
        finally:
            tmp.unlink(missing_ok=True)
    return sig.x, sig.fs, sig.rpm, sig.column


def plot_time(x: np.ndarray, fs: int) -> go.Figure:
    n = min(len(x), int(0.25 * fs))
    t = np.arange(n) / fs
    fig = go.Figure(go.Scatter(x=t, y=x[:n], mode="lines", line=dict(width=1.2, color=theme.CYAN)))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Time waveform (first 0.25 s)",
                      xaxis_title="Time [s]", yaxis_title="Acceleration [g]")
    return fig


def plot_fft(x: np.ndarray, fs: int) -> go.Figure:
    seg = x[: min(len(x), 2 * fs)] - np.mean(x[: min(len(x), 2 * fs)])
    amp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / len(seg)
    f = np.fft.rfftfreq(len(seg), 1 / fs)
    fig = go.Figure(go.Scatter(x=f, y=amp, mode="lines", line=dict(width=1.2, color=theme.CYAN),
                               fill="tozeroy", fillcolor="rgba(34,211,238,0.12)"))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="FFT spectrum",
                      xaxis_title="Frequency [Hz]", yaxis_title="Amplitude")
    return fig


def plot_spectrogram(x: np.ndarray, fs: int) -> go.Figure:
    t, f, db = signal_spectrogram(x[: min(len(x), 3 * fs)], fs)
    fig = go.Figure(go.Heatmap(x=t, y=f / 1000, z=db, colorscale="Turbo", zmin=np.percentile(db, 5),
                               zmax=np.percentile(db, 99.5), colorbar=dict(title="dB")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title="Spectrogram — the time-frequency image the CNN classifies",
                      xaxis_title="Time [s]", yaxis_title="Frequency [kHz]")
    return fig


def plot_envelope(x: np.ndarray, fs: int, rpm: float) -> go.Figure:
    f, a = envelope_spectrum(x[: min(len(x), 10 * fs)], fs)
    m = f <= 600
    fig = go.Figure(go.Scatter(x=f[m], y=a[m], mode="lines", line=dict(width=1.3, color=theme.BLUE),
                               fill="tozeroy", fillcolor="rgba(47,140,255,0.15)", name="envelope"))
    df = defect_frequencies(rpm)
    for key, label, color in (("bpfo_hz", "BPFO", theme.ORANGE), ("bpfi_hz", "BPFI", theme.RED),
                              ("bsf_hz", "2×BSF", theme.VIOLET)):
        f0 = df[key] * (2 if key == "bsf_hz" else 1)
        for k in (1, 2, 3):
            if k * f0 < 600:
                pos = {"BPFO": "top left", "BPFI": "top right", "2×BSF": "bottom right"}[label]
                label_args = dict(annotation_text=label, annotation_font_size=10, annotation_position=pos,
                                  annotation_font_color=color) if k == 1 else {}
                fig.add_vline(x=k * f0, line=dict(color=color, dash="dot", width=1), **label_args)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      title=f"Envelope spectrum · defect frequencies @ {rpm:.0f} rpm",
                      xaxis_title="Frequency [Hz]", yaxis_title="Envelope amplitude")
    return fig


def save_review(d: Diagnosis, status: str, reviewer: str, comment: str, edited: str) -> None:
    new = not config.REVIEW_LOG.exists()
    with open(config.REVIEW_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "reviewer", "status", "question", "fault_type", "ml_label", "ml_confidence",
                        "physics_consistency", "config", "llm", "citation_coverage", "comment", "final_text"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), reviewer, status, d.question, d.fault_type,
                    d.ml_result.label if d.ml_result else "", d.ml_result.confidence if d.ml_result else "",
                    d.fault_frequencies.consistency if d.fault_frequencies else "", d.config_name,
                    f"{d.llm_provider}:{d.llm_model}", d.citation_coverage, comment, edited])


def report_markdown(d: Diagnosis) -> str:
    lines = [f"# IntelliMaintain diagnosis report", "", f"*Generated {d.created_at} — advisory output, "
             f"review status: **{d.review_status}***", "", f"**Question:** {d.question}", "",
             f"## Result: {d.fault_type}", "", d.summary, ""]
    if d.ml_result:
        lines += [f"- ML: {d.ml_result.label}, confidence {d.ml_result.confidence:.2f}, "
                  f"anomaly score {d.ml_result.anomaly_score:.2f}"]
    if d.fault_frequencies:
        ff = d.fault_frequencies
        lines += [f"- Physics: BPFO {ff.bpfo_hz:.1f} Hz, BPFI {ff.bpfi_hz:.1f} Hz, BSF {ff.bsf_hz:.1f} Hz — "
                  f"{ff.consistency}: {ff.consistency_note}"]
    lines += ["", "## Explanation", "", d.explanation]
    if d.fault_mechanism:
        lines += ["", "## How this fault occurs", "", d.fault_mechanism]
    lines += ["", "## Recommended actions", ""]
    lines += [f"{i}. {a}" for i, a in enumerate(d.recommended_actions, 1)] or ["(none)"]
    if d.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in d.warnings]
    lines += ["", "## Evidence", ""]
    lines += [f"[{i}] {e.source}, {e.url}" if e.url else f"[{i}] {e.source}, page {e.page}"
              for i, e in enumerate(d.evidence, 1)]
    return "\n".join(lines)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown(theme.brand_html(), unsafe_allow_html=True)
status = system_status()
if not status["model"] or not status["index"]:
    st.sidebar.error("Setup incomplete. Run `python scripts/setup_all.py` first.")

st.sidebar.subheader("LLM")
providers = ["gemini", "offline"]
default_provider = "gemini" if gemini_key_available() else "offline"
provider = st.sidebar.selectbox("Provider", providers, index=providers.index(default_provider),
                                help="gemini = Google Gemini (free API key). offline = no LLM: extractive, "
                                     "citation-safe answers (also used automatically if Gemini fails).")
if provider == "gemini":
    models = list(SUGGESTED_GEMINI_MODELS)
    env_model = os.getenv("IM_LLM_MODEL", "")
    if env_model and env_model not in models:
        models.insert(0, env_model)
    model = st.sidebar.selectbox("Gemini model", models, index=models.index(env_model) if env_model in models else 0,
                                 help="flash-lite = fastest, highest free quota · 3.8-flash = best answers")
    # The key is NEVER typed into or shown by the interface: it is read privately from .env (laptop)
    # or from the app's Secrets (online). The page only shows whether one is configured.
    if gemini_key_available():
        st.sidebar.markdown("Gemini key: **configured** (kept private on the server)")
    else:
        st.sidebar.warning("Gemini key not configured. Add GEMINI_API_KEY to the .env file (laptop) or to "
                           "Settings → Secrets (online). Until then the offline generator answers.")
else:
    model = DEFAULT_MODELS["offline"]
if st.sidebar.button("Test LLM connection"):
    ok, msg = LLM(provider, model).health_check()
    (st.sidebar.success if ok else st.sidebar.error)(redact(msg))

st.sidebar.subheader("Agent")
use_ml = st.sidebar.toggle("Use ML + physics tools", value=True)
retrieval = st.sidebar.selectbox("Retrieval", ["improved", "basic", "none"],
                                 help="improved = BGE dense + BM25 + rank fusion (+ reranker); basic = dense only")
planner = st.sidebar.selectbox("Planner", ["fixed", "llm"],
                               help="fixed = deterministic tool order; llm = the LLM chooses tools (bounded)")
max_steps = st.sidebar.slider("Max tool calls", 1, 8, config.MAX_AGENT_STEPS)
WEB_LABELS = {"auto": "auto — when the documents are not enough", "always": "always — add web sources",
              "off": "off — documents only"}
web_mode = st.sidebar.selectbox("Online search", list(WEB_LABELS), format_func=WEB_LABELS.get,
                                index=list(WEB_LABELS).index(os.getenv("IM_WEB_SEARCH", "auto").lower()
                                                             if os.getenv("IM_WEB_SEARCH", "auto").lower()
                                                             in WEB_LABELS else "auto"),
                                help="Searches Wikipedia for extra evidence. Web passages are cited and verified "
                                     "like the manuals; without internet the app simply uses the documents.")
CLASSIFIERS = {"RandomForest (features)": "rf", "Spectrogram CNN (image)": "cnn"}
classifier_options = list(CLASSIFIERS) if cnn_ready() else list(CLASSIFIERS)[:1]
classifier_label = st.sidebar.selectbox("Classifier", classifier_options,
                                        help="Spectrogram CNN appears after: python scripts/train_cnn.py")
classifier = CLASSIFIERS[classifier_label]
st.sidebar.markdown(theme.status_html(status["model"], status["index"]), unsafe_allow_html=True)

# ------------------------------------------------------------------ main
st.markdown(theme.hero_html(), unsafe_allow_html=True)

tab_diag, tab_results, tab_reviews, tab_about = st.tabs(
    ["Diagnose", "Model & evaluation results", "Review log", "About"])

with tab_diag:
    c1, c2 = st.columns([2, 1])
    with c1:
        source = st.radio("Signal source", ["Demo sample", "Upload file", "No signal (question only)"],
                          horizontal=True)
        x = fs = rpm_meta = None
        if source == "Demo sample":
            samples = sorted(config.SAMPLES_DIR.glob("*.csv"))
            if samples:
                names = [p.stem.replace("_", " ") for p in samples]  # plain strings: works on all Streamlit versions
                pick = st.selectbox("Sample", names)
                sig = load_signal(samples[names.index(pick)])
                x, fs, rpm_meta = sig.x, sig.fs, sig.rpm
                st.caption(" · ".join(f"{k}={v}" for k, v in sig.meta.items()))
            else:
                st.warning("No demo samples. Run `python scripts/make_samples.py`.")
        elif source == "Upload file":
            up = st.file_uploader("Vibration signal (.csv, .txt, .mat, .npy)", type=["csv", "txt", "mat", "npy"])
            if up is not None:
                try:
                    x, fs, rpm_meta, col = read_upload(up)
                    st.caption(f"Loaded column '{col}' with {len(x):,} samples.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read the file: {e}")
    with c2:
        fs_in = st.number_input("Sampling rate [Hz]", min_value=1000, max_value=200000,
                                value=int(fs or config.FS), step=1000, disabled=source.startswith("No"))
        rpm_in = st.number_input("Shaft speed [rpm]", min_value=100.0, max_value=20000.0,
                                 value=float(rpm_meta or 1772.0), step=1.0, disabled=source.startswith("No"))

    if x is not None:
        with st.expander("Signal preview", expanded=False):
            st.plotly_chart(plot_time(x, int(fs_in)), theme=None)

    # ---------------- chat with the copilot
    # The signal is classified once; every chat turn reuses that diagnosis and only searches and answers again.
    sig_key = None
    if x is not None:
        xa = np.asarray(x, dtype=np.float64)
        sig_key = (len(xa), float(np.round(xa[:4096].sum(), 6)), int(fs_in), float(rpm_in), classifier, use_ml)
    if st.session_state.get("chat_key") != sig_key:  # new signal (or none): start a new conversation
        st.session_state["chat_key"] = sig_key
        st.session_state["chat"] = []
        st.session_state.pop("diag", None)
        st.session_state.pop("diag_cache", None)
    chat: list[dict] = st.session_state.setdefault("chat", [])

    st.subheader("Chat with the copilot")
    st.caption("Ask anything in your own words — about this bearing, its data, the CWRU dataset, the models or "
               "maintenance. Follow-up questions like “why?” or “can I keep running it?” work too."
               + (" A signal is loaded, so answers use its diagnosis." if x is not None else
                  " No signal loaded: answers come from the documents" + (" and online search." if web_mode != "off"
                                                                         else ".")))
    ready = status["model"] and status["index"]
    b1, b2, _ = st.columns([1.3, 1, 4])
    run_full = b1.button("Run diagnosis", type="primary", disabled=not ready or x is None,
                         help="Full diagnosis of the loaded signal (you can also just type a question below)")
    if b2.button("New chat", disabled=not chat):
        st.session_state["chat"] = []
        st.session_state.pop("diag", None)
        st.rerun()

    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("note"):
                st.caption(msg["note"])

    prompt = st.chat_input("Ask the copilot… e.g. “what is wrong with this bearing?”, “what grease should I use?”",
                           disabled=not ready)
    if run_full:
        prompt = "What is the condition of this bearing, why, and what maintenance action is recommended?"
    if prompt:
        if x is not None and len(x) < config.WINDOW * max(1, int(fs_in) // config.FS):
            st.error(f"Signal too short: need at least {config.WINDOW} samples at 12 kHz.")
        else:
            history = [{"role": m["role"], "content": m["content"]} for m in chat]
            chat.append({"role": "user", "content": prompt})
            with st.spinner("Thinking: diagnosis → documents → (web) → answer ..."):
                try:
                    llm = LLM(provider, model, max_retries=2)  # keep the UI responsive
                    if provider == "gemini" and st.session_state.get("llm_runs", 0) >= MAX_LLM_RUNS:
                        llm = LLM("offline")  # protect the owner's free quota from heavy use of a public link
                        st.info(f"Gemini limit of {MAX_LLM_RUNS} answers per session reached - "
                                "using the offline generator.")
                    elif provider == "gemini":
                        st.session_state["llm_runs"] = st.session_state.get("llm_runs", 0) + 1
                    agent = DiagnosisAgent(llm=llm, use_ml=use_ml, retrieval=retrieval, planner=planner,
                                           max_steps=max_steps, classifier=classifier, web=web_mode)
                    cache = st.session_state.get("diag_cache") or {}
                    d = agent.run(prompt, signal=x, fs=int(fs_in) if x is not None else None,
                                  rpm=float(rpm_in) if x is not None else None, history=history,
                                  ml=cache.get("ml"), ff=cache.get("ff"))
                    if x is not None and d.ml_result is not None:
                        st.session_state["diag_cache"] = {"ml": d.ml_result, "ff": d.fault_frequencies}
                    note = f"{'Gemini, verified' if d.answer_source.endswith('(verified)') else d.answer_source}"
                    if any(e.url for e in d.evidence):
                        note += " · includes online sources"
                    if d.warnings:
                        note += f" · {len(d.warnings)} note(s) below"
                    reply = d.answer or d.summary
                    if d.fault_mechanism and d.fault_type != "Normal" and d.question_focus in ("diagnosis",
                                                                                               "mechanism"):
                        reply += "\n\n**How this fault occurs:** " + d.fault_mechanism
                    chat.append({"role": "assistant", "content": reply, "note": note})
                    if d.answer_source != "assistant":
                        st.session_state["diag"] = d
                        st.session_state["diag_signal"] = (x, int(fs_in), float(rpm_in)) if x is not None else None
                        st.session_state.pop("review_saved", None)
                except Exception as e:  # noqa: BLE001 - show a short, key-free message (no traceback)
                    chat.append({"role": "assistant", "content": redact(f"Sorry, that failed: {type(e).__name__}: {e}")})
            st.rerun()

    d: Diagnosis | None = st.session_state.get("diag")
    if d is not None:
        st.divider()
        st.caption(f"Details of the latest answer — question: “{d.question}”")
        # ---------------- prediction panel
        st.subheader("1 · Prediction")
        m1, m2, m3, m4 = st.columns(4)
        color = FAULT_COLOR.get(d.fault_type, theme.MUTED)
        fault_icon = ""
        m1.markdown(theme.card_html(fault_icon, "Fault type", d.fault_type,
                                    "Advisory — human review required", color=color), unsafe_allow_html=True)
        if d.ml_result:
            conf = d.ml_result.confidence
            conf_color = theme.GREEN if conf >= 0.85 else theme.ORANGE if conf >= config.CONFIDENCE_THRESHOLD \
                else theme.RED
            m2.markdown(theme.card_html("", "ML class / confidence", f"{d.ml_result.label} · {conf:.0%}",
                                        "Calibrated probability", color=theme.TEXT, bar=conf, bar_color=conf_color),
                        unsafe_allow_html=True)
            an = d.ml_result.anomaly_score
            m3.markdown(theme.card_html("", "Anomaly score", f"{an:.2f}", "0 = healthy · 1 = anomalous",
                                        color=theme.GREEN if an < 0.3 else theme.ORANGE if an < 0.7 else theme.RED,
                                        bar=an, bar_color=theme.CYAN), unsafe_allow_html=True)
        else:
            m2.markdown(theme.card_html("", "ML class", "not used",
                                        "ML tools switched off" if not use_ml else "No vibration signal given",
                                        color=theme.MUTED),
                        unsafe_allow_html=True)
        if d.fault_frequencies and d.fault_frequencies.envelope_match:
            ff = d.fault_frequencies
            m4.markdown(theme.card_html(CONSISTENCY_ICON[ff.consistency], "Physics cross-check", ff.consistency,
                                        f"Envelope points to: {ff.physics_suggests or 'no clear line'}",
                                        color=theme.STATUS_COLOR.get(ff.consistency, theme.TEXT)),
                        unsafe_allow_html=True)
        if d.ml_result:
            pc1, pc2 = st.columns([3, 2])
            probs = pd.Series(d.ml_result.class_probabilities).sort_values()
            fig = go.Figure(go.Bar(x=probs.values, y=probs.index, orientation="h",
                                   marker=dict(color=probs.values, colorscale=[[0, "#1b3a6b"], [1, theme.CYAN]])))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Calibrated class probabilities",
                              xaxis_range=[0, 1])
            pc1.plotly_chart(fig, theme=None)
            pc2.markdown("**Signal indicators**")
            pc2.dataframe(pd.DataFrame(d.ml_result.key_features.items(), columns=["indicator", "value"]),
                          hide_index=True)
            if d.ml_result.similar_cases:
                pc2.markdown("**Most similar recordings in the CWRU dataset**")
                pc2.dataframe(pd.DataFrame(d.ml_result.similar_cases)[
                    ["cwru_file", "label", "load_hp", "similarity"]], hide_index=True)
            ref = d.ml_result.cwru_reference
            if ref:
                rows = []
                for key, name in (("rms_g", "RMS [g]"), ("kurtosis", "Kurtosis"), ("crest_factor", "Crest factor")):
                    if key in ref and key in d.ml_result.key_features:
                        lo, hi = ref[key]
                        own = d.ml_result.key_features[key]
                        rows.append({"indicator": name, "this signal": own, f"CWRU {ref['cwru_class']}": f"{lo} – {hi}",
                                     "": "inside" if lo <= own <= hi else "above" if own > hi else "below"})
                if rows:
                    pc2.markdown(f"**Compared with CWRU class {ref['cwru_class']}** ({ref['n_recordings']} recordings)")
                    pc2.dataframe(pd.DataFrame(rows), hide_index=True)
            if d.fault_frequencies:
                ff = d.fault_frequencies
                pc2.markdown(f"**Defect frequencies @ {ff.rpm:.0f} rpm**  \nBPFO {ff.bpfo_hz:.1f} Hz · "
                             f"BPFI {ff.bpfi_hz:.1f} Hz · BSF {ff.bsf_hz:.1f} Hz · FTF {ff.ftf_hz:.1f} Hz")
        sigdata = st.session_state.get("diag_signal")
        if sigdata is not None:
            sx, sfs, srpm = sigdata
            from intellimaintain.ml.classifier import to_target_fs
            sx12 = to_target_fs(sx, sfs)
            g1, g2 = st.columns(2)
            g1.plotly_chart(plot_fft(sx12, config.FS), theme=None)
            if len(sx12) >= 4096:
                g2.plotly_chart(plot_envelope(sx12, config.FS, srpm), theme=None)
            st.plotly_chart(plot_spectrogram(sx12, config.FS), theme=None)

        # ---------------- explanation panel
        st.subheader("2 · Full explanation (cited)")
        from intellimaintain.agent.nlu import understand

        nq = understand(d.question, has_signal=d.ml_result is not None)
        if nq.corrections or nq.added_terms:
            fixes = ", ".join(f"{k} → {v}" for k, v in nq.corrections.items())
            st.caption(f"Understood your question as: “{nq.cleaned}”" + (f" (corrected: {fixes})" if fixes else "")
                       + f" · intent: {nq.intent}")
        if d.answer_source.endswith("(verified)"):
            st.success(f"Written by Gemini and verified against the data and documents — "
                       f"{d.grounding_score:.0%} of its sentences matched their cited source; the fault type comes "
                       f"from the classifier.")
        elif "fallback" in d.answer_source:
            st.info("Gemini was unavailable or its answer could not be verified, so the offline, "
                    "data-verified answer is shown.")
        if d.insufficient_evidence:
            st.warning("Insufficient evidence — the copilot will not guess. " + d.summary)
        else:
            st.info(d.summary)
        for w in d.warnings:
            st.warning(w)
        st.markdown(d.explanation or "_No explanation generated._")
        if d.fault_mechanism and d.fault_type != "Normal":
            st.markdown("**How this fault occurs**")
            st.markdown(d.fault_mechanism)
        if d.recommended_actions:
            st.markdown("**Recommended actions**")
            st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(d.recommended_actions, 1)))
        st.caption(f"Answer source: {d.answer_source} · grounding {d.grounding_score:.0%} · "
                   f"Citation coverage {d.citation_coverage:.0%} · configuration {d.config_name} · "
                   f"LLM {d.llm_provider}:{d.llm_model} · online search {d.web_search} · [ML] = classifier, "
                   f"[PHYS] = physics check, [n] = evidence passage n")

        # ---------------- evidence panel
        st.subheader("3 · Evidence")
        if not d.evidence:
            st.caption("No documents retrieved (retrieval disabled or nothing relevant).")
        for i, e in enumerate(d.evidence, 1):
            if e.url:
                with st.expander(f"[{i}] {e.source} — online source (score {e.score:.2f}, web)"):
                    st.markdown(f"[Open source]({e.url}) · Wikipedia text, CC BY-SA licence")
                    st.text(e.text)
                continue
            with st.expander(f"[{i}] {e.title or e.source} — {e.source}, page/section {e.page} "
                             f"(score {e.score:.3f}, {e.retriever})"):
                st.text(e.text)

        with st.expander("Agent trace (tool calls)"):
            st.dataframe(pd.DataFrame([t.model_dump() for t in d.tool_trace]), hide_index=True)

        # ---------------- human review
        st.subheader("4 · Human review")
        rc1, rc2 = st.columns([1, 2])
        reviewer = rc1.text_input("Reviewer name", value="engineer")
        decision = rc1.radio("Decision", ["accepted", "rejected", "edited"], horizontal=False)
        comment = rc2.text_input("Comment / reason")
        edited = rc2.text_area("Final text (edit if needed)",
                               value=d.summary + "\n\n" + "\n".join(f"- {a}" for a in d.recommended_actions),
                               height=160)
        if st.button("Save review"):
            d.review_status = decision
            save_review(d, decision, reviewer, comment, edited)
            st.session_state["review_saved"] = decision
        if st.session_state.get("review_saved"):
            st.success(f"Review saved as '{st.session_state['review_saved']}' in data/reviews.csv")
        dc1, dc2 = st.columns(2)
        dc1.download_button("Download diagnosis (JSON)", d.model_dump_json(indent=2), file_name="diagnosis.json",
                            mime="application/json")
        dc2.download_button("Download report (Markdown)", report_markdown(d), file_name="diagnosis_report.md",
                            mime="text/markdown")

with tab_results:
    results_dashboard.render()

with tab_reviews:
    if config.REVIEW_LOG.exists():
        rv = pd.read_csv(config.REVIEW_LOG)
        st.dataframe(rv, hide_index=True)
        st.caption(f"{len(rv)} reviews · acceptance rate "
                   f"{(rv['status'] == 'accepted').mean():.0%}")
    else:
        st.caption("No reviews saved yet.")

with tab_about:
    meta_path = config.INDEX_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    st.markdown(f"""
**IntelliMaintain** combines four layers:

1. **ML layer** – 2048-sample windows (50 % overlap), time + FFT-band features, isotonic-calibrated RandomForest
   (10 CWRU classes), Isolation-Forest anomaly score.
2. **Physics layer** – bearing defect frequencies (BPFO/BPFI/BSF/FTF) and envelope-spectrum check that can
   *contradict* the ML model.
3. **RAG layer** – {meta.get('n_chunks', '?')} chunks from {len(meta.get('sources', []))} documents, embedder
   `{meta.get('embed_model', '?')}`, FAISS + BM25 with reciprocal-rank fusion and optional cross-encoder reranking.
4. **Agent + LLM** – max {config.MAX_AGENT_STEPS} calls to 4 approved tools (classifier, physics check, document
   retrieval, optional online search); grounded prompt, mandatory citations, automatic citation verification,
   a cited "how this fault occurs" explanation, "insufficient evidence" fallback, human review.

Output is **advisory** and must be reviewed by a qualified engineer.
""")
