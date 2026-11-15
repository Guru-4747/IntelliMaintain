"""'Model & evaluation results' tab: a dashboard of interactive charts and tables built from the real result
files (reports/ml_metrics.json, reports/cnn_metrics.json, reports/eval/<run>/*.csv).

Every panel degrades gracefully: if a result file or a field is missing (older run), the panel shows the saved
PNG figure or a hint telling which script produces it - the tab never crashes.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from intellimaintain import config

R = config.REPORTS_DIR
FIG = config.FIGURES_DIR
PALETTE = ["#3b82f6", "#f59e0b", "#22c55e", "#ef4444", "#a78bfa", "#8d5a4a"]  # 6 ablation configs
H = 300  # default chart height


# ------------------------------------------------------------------ helpers
def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:  # noqa: BLE001
        return None


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}".rstrip("0").rstrip(".") if isinstance(v, float) and v != int(v) else f"{v:g}"
    return html.escape(str(v))


def card_title(title: str, subtitle: str = "") -> None:
    sub = f"<div class='im-card-sub2'>{html.escape(subtitle)}</div>" if subtitle else ""
    st.markdown(f"<div class='im-card-title'>{html.escape(title)}</div>{sub}", unsafe_allow_html=True)


def table(df: pd.DataFrame, digits: int = 3, highlight_row: str | None = None) -> None:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cls = " class='im-hl'" if highlight_row is not None and str(r.iloc[0]) == highlight_row else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{fmt(v, digits)}</td>" for v in r.values) + "</tr>")
    st.markdown(f"<div class='im-tablewrap'><table class='im-table'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


def layout(fig: go.Figure, height: int = H, legend_top: bool = True, legend_bottom: bool = False) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=8, r=8, t=36 if legend_top and not legend_bottom else 8, b=8),
                      showlegend=legend_top or legend_bottom)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", y=-0.32, x=0, font=dict(size=10)))
    elif legend_top:
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)))
    return fig


_CHART_N = [0]


def chart(fig: go.Figure) -> None:
    # every chart gets its own key: two charts with identical data (e.g. two perfect
    # confusion matrices) would otherwise collide with StreamlitDuplicateElementId
    _CHART_N[0] += 1
    try:
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, key=f"im_res_chart_{_CHART_N[0]}")
    except TypeError:  # very old Streamlit without `key`: make the figures differ instead
        fig.update_layout(meta=f"im_res_chart_{_CHART_N[0]}")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def image_or_hint(name: str, hint: str) -> None:
    p = FIG / name
    if p.exists():
        st.image(str(p))
    else:
        st.caption(f"Run `{hint}` to create this panel.")


def heatmap(cm: dict, height: int = H) -> go.Figure:
    labels, m = cm["labels"], cm["matrix"]
    rows = [sum(r) or 1 for r in m]
    norm = [[v / rows[i] for v in r] for i, r in enumerate(m)]
    text = [[str(v) if v else "" for v in r] for r in m]
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#0b1a33"], [0.5, "#1d4ed8"], [1, "#60a5fa"]], zmin=0, zmax=1,
                               colorbar=dict(thickness=8, len=0.9),
                               hovertemplate="true %{y}<br>predicted %{x}<br>count %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="True", tickfont=dict(size=9))
    fig.update_xaxes(title="Predicted", tickfont=dict(size=9), tickangle=-40)
    return layout(fig, height, legend_top=False)


def eval_runs() -> list[Path]:
    root = R / "eval"
    runs = [p for p in root.glob("*") if (p / "ablation_summary.csv").exists()] if root.exists() else []
    return sorted(runs, key=lambda p: (p / "ablation_summary.csv").stat().st_mtime, reverse=True)


# ------------------------------------------------------------------ main
def render() -> None:
    _CHART_N[0] = 0
    ml = _json(R / "ml_metrics.json")
    cnn = _json(R / "cnn_metrics.json")
    runs = eval_runs()
    if not ml and not runs:
        st.info("No results yet. Run `python scripts/train_ml.py` and `python scripts/run_eval.py`.")
        return

    top = st.columns([3, 1])
    run = None
    if runs:
        names = [p.name for p in runs]
        default = names.index("offline") if "offline" in names else 0
        pick = top[1].selectbox("Evaluation run", names, index=default,
                                help="Folders in reports/eval/ - one per LLM / classifier (e.g. offline, "
                                     "offline_cnn, gemini_...)")
        run = runs[names.index(pick)]
    top[0].markdown("<div class='im-section'>Model &amp; evaluation results</div>"
                    "<div class='im-card-sub2'>All numbers are read from the result files produced by "
                    "train_ml.py, train_cnn.py and run_eval.py.</div>", unsafe_allow_html=True)

    abl = _csv(run / "ablation_summary.csv") if run else None
    stress = _csv(run / "stress_test.csv") if run else None
    retr = _csv(run / "retrieval_benchmark.csv") if run else None
    casual = _csv(run / "casual_questions_results.csv") if run else None

    # ---------------- KPI strip
    full = abl.set_index("config").loc["ML+improved"] if abl is not None and "ML+improved" in set(abl.config) else None
    wrong = stress[(stress.injected_error.astype(str) == "True") & (stress.policy == "abstain")] \
        if stress is not None else None
    kpis = [
        ("🎯", "Cross-load accuracy (RF)", g(ml, "cross_load", "RandomForest", "accuracy"), "train 0-2 HP → test 3 HP"),
        ("🔁", "Leave-one-load-out", g(ml, "leave_one_load_out", "mean_accuracy"), "mean of 4 folds"),
        ("📏", "Calibration error (ECE)", g(ml, "calibration_cross_load", "isotonic", "ece"), "lower is better"),
        ("🧠", "CNN accuracy @ 5 dB noise", g(cnn, "noise_robustness_cross_load", "aug", "5dB"), "with augmentation"),
        ("📚", "Key-fact recall (full system)", None if full is None else float(full["key_fact_recall"]),
         "ML + improved RAG"),
        ("🛡️", "Wrong ML labels flagged", None if wrong is None or wrong.empty else float(wrong.flagged.mean()),
         "stress test"),
    ]
    cols = st.columns(len(kpis))
    for c, (icon, label, val, sub) in zip(cols, kpis):
        txt = "–" if val is None else (f"{val:.3f}" if "ECE" in label else f"{val:.1%}")
        c.markdown(theme.card_html(icon, label, txt, sub, color=theme.CYAN), unsafe_allow_html=True)

    # ---------------- row 1: ablation chart + table
    if abl is not None:
        c1, c2 = st.columns([1.15, 1])
        with c1.container(border=True):
            card_title("2 × 3 ablation", "(no ML / ML) × (no retrieval / basic RAG / improved RAG)")
            metrics = ["fault_correct", "key_fact_recall", "citation_coverage", "faithfulness"]
            fig = go.Figure()
            for i, (_, r) in enumerate(abl.iterrows()):
                fig.add_bar(name=r["config"], x=[m.replace("_", " ") for m in metrics],
                            y=[r.get(m) for m in metrics], marker_color=PALETTE[i % len(PALETTE)])
            fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
            chart(layout(fig))
        with c2.container(border=True):
            card_title("Ablation table", "mean over the test cases of eval/test_cases.csv")
            cols_ = [c for c in ["config", "fault_correct", "key_fact_recall", "citation_coverage", "faithfulness",
                                 "context_precision", "context_recall", "insufficient_ok", "grounding"] if c in abl]
            table(abl[cols_], highlight_row="ML+improved")

    # ---------------- row 2: stress test
    if wrong is not None and not wrong.empty:
        c1, c2 = st.columns([1, 1.15])
        with c1.container(border=True):
            card_title("Wrong-ML stress test", "injected wrong labels, by true fault type")
            gdf = wrong.groupby("truth")[["flagged", "conflict_flag", "error_propagated"]].mean()
            fig = go.Figure()
            for col, name, colr in (("flagged", "flagged for review", theme.BLUE),
                                    ("conflict_flag", "physics conflict", theme.RED),
                                    ("error_propagated", "wrong label propagated", "#94a3b8")):
                fig.add_bar(name=name, x=list(gdf.index), y=list(gdf[col]), marker_color=colr)
            fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
            chart(layout(fig))
        with c2.container(border=True):
            card_title("Stress-test detail",
                       f"{len(wrong)} wrong labels · {wrong.flagged.mean():.1%} flagged · "
                       f"physics conflict {wrong.conflict_flag.mean():.1%}")
            t1 = gdf.round(3).reset_index().rename(columns={"truth": "true fault"})
            table(t1)
            pol = stress.groupby(["policy", "injected_error"])[["correct", "error_propagated", "abstained",
                                                                "flagged"]].mean().round(3).reset_index()
            pol["injected_error"] = pol["injected_error"].astype(str).map({"True": "wrong ML label",
                                                                           "False": "true ML label"})
            table(pol)
            st.caption("abstain = answer 'Unknown – verify' when physics contradicts the ML label; "
                       "trust_ml = always report the ML label.")

    # ---------------- row 3: retrieval / informal questions / C0 vs C1
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        card_title("Retrieval benchmark", "knowledge questions")
        if retr is not None:
            table(retr.round(3))
        else:
            st.caption("Run `python scripts/run_eval.py`.")
    with c2.container(border=True):
        card_title("Informal questions", "typos, short forms, everyday words")
        if casual is not None and "hit@5_raw" in casual:
            s = {k: casual[k].dropna().mean() for k in ("hit@5_raw", "hit@5_normalised", "top1_raw",
                                                       "top1_normalised")}
            st.markdown(f"<div class='im-card-sub2'>{len(casual)} chat-style questions · intent recognised "
                        f"{casual.intent_ok.mean():.0%}</div>", unsafe_allow_html=True)
            table(pd.DataFrame({"Retrieval": ["Relevant document in top 5", "Relevant document ranked first"],
                                "Without language layer": [f"{s['hit@5_raw']:.0%}", f"{s['top1_raw']:.0%}"],
                                "With language layer": [f"{s['hit@5_normalised']:.0%}",
                                                        f"{s['top1_normalised']:.0%}"]}))
        else:
            st.caption("Run `python scripts/run_eval.py` (needs the latest version).")
    with c3.container(border=True):
        card_title("Preliminary result: C0 vs C1", "LLM-only vs basic RAG")
        if abl is not None and {"noML+none", "noML+basic"} <= set(abl.config):
            c0c1 = abl[abl.config.isin(["noML+none", "noML+basic"])].copy()
            c0c1["config"] = c0c1["config"].map({"noML+none": "C0 (LLM only)", "noML+basic": "C1 (basic RAG)"})
            cols_ = [c for c in ["config", "fault_correct", "key_fact_recall", "citation_coverage", "faithfulness",
                                 "context_recall"] if c in c0c1]
            table(c0c1[cols_])
        else:
            st.caption("Run `python scripts/run_eval.py`.")

    # ---------------- row 4: RF vs CNN / spectrograms / noise / feature importance
    c1, c2, c3, c4 = st.columns([1.05, 1, 1.05, 1])
    with c1.container(border=True):
        card_title("RandomForest vs Spectrogram CNN")
        if cnn:
            nr = g(ml, "noise_robustness_cross_load", default={})
            rows = [("Random 80/20 split", g(ml, "random_split", "RandomForest", "accuracy"),
                     g(cnn, "random_split", "accuracy")),
                    ("Cross-load 0-2 → 3 HP", g(ml, "cross_load", "RandomForest", "accuracy"),
                     g(cnn, "cross_load_aug", "accuracy")),
                    ("Cross-severity (unseen 0.021\")", g(ml, "cross_severity", "accuracy"),
                     g(cnn, "cross_severity", "accuracy"))]
            for k in ("20dB", "10dB", "5dB", "0dB"):
                rows.append((f"Cross-load + {k} noise", nr.get(k), g(cnn, "noise_robustness_cross_load", "aug", k)))
            table(pd.DataFrame(rows, columns=["Test", "RF", "CNN"]), digits=4)
        else:
            st.caption("Run `python scripts/train_cnn.py`.")
    with c2.container(border=True):
        card_title("Spectrogram examples", "CNN input, one window per class")
        image_or_hint("spectrogram_examples.png", "python scripts/train_cnn.py")
    with c3.container(border=True):
        card_title("Noise robustness", "accuracy on the 3 HP test with added noise")
        nr = g(ml, "noise_robustness_cross_load", default={})
        series = [("RF (features)", nr, theme.BLUE, "solid"),
                  ("RF + noise augmentation", g(cnn, "noise_robustness_cross_load", "rf_aug", default={}),
                   theme.GREEN, "dot"),
                  ("CNN (no augmentation)", g(cnn, "noise_robustness_cross_load", "noaug", default={}),
                   theme.VIOLET, "dash"),
                  ("CNN + noise augmentation", g(cnn, "noise_robustness_cross_load", "aug", default={}),
                   theme.RED, "solid")]
        fig = go.Figure()
        for name, d, colr, dash in series:
            if d:
                keys = [k for k in ("clean", "20dB", "10dB", "5dB", "0dB") if k in d]
                fig.add_scatter(x=keys, y=[d[k] for k in keys], name=name, mode="lines+markers",
                                line=dict(color=colr, dash=dash, width=2))
        fig.update_layout(yaxis_range=[0, 1.05])
        chart(layout(fig, height=340, legend_top=False, legend_bottom=True))
    with c4.container(border=True):
        card_title("Top-15 feature importances", "RandomForest")
        fi = g(ml, "chart_data", "feature_importance")
        if fi:
            fig = go.Figure(go.Bar(x=list(fi.values()), y=list(fi.keys()), orientation="h",
                                   marker=dict(color=list(fi.values()),
                                               colorscale=[[0, "#1e3a8a"], [1, theme.CYAN]])))
            chart(layout(fig, legend_top=False))
        else:
            image_or_hint("feature_importance.png", "python scripts/train_ml.py")

    # ---------------- row 5: confusion matrices / reliability / calibration / noise table
    c1, c2, c3, c4, c5 = st.columns([1.1, 1, 1, 0.8, 0.8])
    for col, key, title, png, script in (
            (c1, "cm_cnn_cross_load", "Spectrogram CNN – cross-load", "cm_cnn_cross_load.png", "train_cnn.py"),
            (c2, "cm_cnn_cross_severity", "Spectrogram CNN – cross-severity", "cm_cnn_cross_severity.png",
             "train_cnn.py")):
        with col.container(border=True):
            card_title(title)
            cm = g(cnn, "chart_data", key)
            if cm:
                chart(heatmap(cm))
            else:
                image_or_hint(png, f"python scripts/{script}")
    with c3.container(border=True):
        card_title("Reliability diagram", "cross-load test, 3 HP")
        rel = g(ml, "chart_data", "reliability")
        if rel:
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(color="#64748b", dash="dash"))
            for (name, d), colr in zip(rel.items(), (theme.BLUE, theme.ORANGE)):
                fig.add_scatter(x=d["confidence"], y=d["accuracy"], mode="lines+markers",
                                name=f"{name} (ECE={d['ece']:.3f})", line=dict(color=colr))
            fig.update_traces(marker=dict(size=9), selector=dict(mode="lines+markers"))
            fig.update_layout(xaxis_title="predicted confidence", yaxis_title="observed accuracy",
                              xaxis_range=[0, 1.02], yaxis_range=[0, 1.05])
            chart(layout(fig, height=340, legend_top=False, legend_bottom=True))
        else:
            image_or_hint("reliability.png", "python scripts/train_ml.py")
    with c4.container(border=True):
        card_title("Calibration", "cross-load test")
        cal = g(ml, "calibration_cross_load", default={})
        if cal:
            table(pd.DataFrame([(k, v.get("ece"), v.get("brier")) for k, v in cal.items()],
                               columns=["Probabilities", "ECE", "Brier"]), digits=4)
    with c5.container(border=True):
        card_title("Noise robustness", "RF, cross-load")
        nr = g(ml, "noise_robustness_cross_load", default={})
        if nr:
            table(pd.DataFrame(list(nr.items()), columns=["Added noise", "Accuracy"]), digits=4)

    # ---------------- row 6: more RF results
    c1, c2, c3, c4 = st.columns([1.1, 1, 0.9, 0.9])
    for col, key, title in ((c1, "cm_cross_load", "RandomForest – cross-load"),
                            (c2, "cm_cross_severity", "RandomForest – cross-severity")):
        with col.container(border=True):
            card_title(title)
            cm = g(ml, "chart_data", key)
            if cm:
                chart(heatmap(cm))
            else:
                image_or_hint(f"{key}.png", "python scripts/train_ml.py")
    with c3.container(border=True):
        card_title("Accuracy by split")
        rows = []
        short = {"RandomForest": "RF", "XGBoost": "XGB"}
        for split, nm in (("random_split", "Random"), ("cross_load", "Cross-load")):
            for model, sc in g(ml, split, default={}).items():
                rows.append((f"{nm} · {short.get(model, model)}", sc.get("accuracy"), sc.get("macro_f1")))
        lolo = g(ml, "leave_one_load_out", default={})
        for k, v in lolo.items():
            if isinstance(v, dict):
                rows.append((f"LOLO · {k.replace('test_', '')}", v.get("accuracy"), v.get("macro_f1")))
        if rows:
            table(pd.DataFrame(rows, columns=["Split · model", "Accuracy", "Macro-F1"]), digits=4)
    with c4.container(border=True):
        card_title("Physics check alone", "envelope spectrum, per recording")
        pe = g(ml, "physics_envelope_check", default={})
        if pe:
            st.markdown(theme.card_html("🔬", "Overall accuracy", f"{pe.get('overall_accuracy', 0):.0%}",
                                        "no ML involved", color=theme.CYAN), unsafe_allow_html=True)
            table(pd.DataFrame(list(pe.get("per_fault_type", {}).items()), columns=["Fault type", "Accuracy"]))
