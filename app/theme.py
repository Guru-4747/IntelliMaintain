"""Visual theme for the IntelliMaintain dashboard: dark navy / electric-blue industrial style.

Only presentation lives here (CSS, HTML snippets, Plotly template). The app logic is unchanged.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------------ palette
BG = "#040b1a"
PANEL = "#0a1730"
PANEL_2 = "#0d1f40"
BORDER = "rgba(64, 140, 255, 0.28)"
BLUE = "#2f8cff"
CYAN = "#22d3ee"
TEXT = "#e6eefc"
MUTED = "#8ea3c7"
GREEN = "#22c55e"
RED = "#ef4444"
ORANGE = "#f59e0b"
VIOLET = "#a78bfa"

FAULT_COLOR = {"Normal": GREEN, "Inner race": RED, "Outer race": ORANGE, "Ball": VIOLET, "Unknown": MUTED}
STATUS_COLOR = {"consistent": GREEN, "conflict": RED, "inconclusive": ORANGE, "not_run": MUTED}

# ------------------------------------------------------------------ Plotly template
pio.templates["intellimaintain"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif", size=12),
    title=dict(font=dict(size=15, color=TEXT)),
    colorway=[CYAN, BLUE, VIOLET, ORANGE, GREEN, RED],
    xaxis=dict(gridcolor="rgba(120,160,255,0.10)", zerolinecolor="rgba(120,160,255,0.15)",
               linecolor="rgba(120,160,255,0.25)"),
    yaxis=dict(gridcolor="rgba(120,160,255,0.10)", zerolinecolor="rgba(120,160,255,0.15)",
               linecolor="rgba(120,160,255,0.25)"),
    hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BLUE, font=dict(color=TEXT)),
))
pio.templates.default = "plotly_dark+intellimaintain"

# ------------------------------------------------------------------ CSS
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}

.stApp {{
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(47,140,255,0.18), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(34,211,238,0.10), transparent 60%),
    linear-gradient(180deg, #050d1f 0%, {BG} 100%);
  color: {TEXT};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}

/* ---------- sidebar */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #071530 0%, #040c1d 100%);
  border-right: 1px solid {BORDER};
  box-shadow: 4px 0 30px rgba(47,140,255,0.08);
}}
section[data-testid="stSidebar"] h3 {{
  font-size: 0.78rem; letter-spacing: .12em; text-transform: uppercase; color: {MUTED};
  margin-top: 1.1rem; border: none; padding-left: 0;
}}
.im-brand {{ display:flex; align-items:center; gap:.7rem; margin:.2rem 0 .3rem 0; }}
.im-brand-logo {{
  width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background: radial-gradient(circle at 30% 30%, rgba(47,140,255,.45), rgba(47,140,255,.08));
  border:1px solid {BORDER}; box-shadow: 0 0 18px rgba(47,140,255,.45); font-size:1.5rem;
}}
.im-brand-name {{ font-size:1.25rem; font-weight:700; color:{TEXT}; line-height:1.1; }}
.im-brand-sub {{ font-size:.78rem; color:{MUTED}; }}
.im-status {{
  margin-top:.8rem; padding:.7rem .9rem; border-radius:14px; background:rgba(13,31,64,.7);
  border:1px solid {BORDER}; font-size:.85rem;
}}
.im-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.45rem;
  box-shadow:0 0 10px currentColor; vertical-align:middle; }}
.im-tagline {{ color:{MUTED}; font-size:.78rem; margin-top:1rem; letter-spacing:.04em; }}

/* ---------- hero banner */
.im-hero {{
  position:relative; overflow:hidden; border-radius:20px; padding:1.5rem 1.8rem; margin-bottom:1.2rem;
  background:
    linear-gradient(100deg, rgba(10,23,48,.96) 35%, rgba(10,23,48,.55) 70%, rgba(47,140,255,.18)),
    repeating-linear-gradient(90deg, rgba(47,140,255,.07) 0 2px, transparent 2px 46px);
  border:1px solid {BORDER}; box-shadow: 0 0 40px rgba(47,140,255,.12) inset, 0 10px 30px rgba(0,0,0,.35);
}}
.im-hero::after {{
  content:""; position:absolute; right:-60px; top:-80px; width:420px; height:260px;
  background: radial-gradient(closest-side, rgba(34,211,238,.35), transparent);
  filter: blur(10px);
}}
.im-hero-small {{ color:{MUTED}; font-size:1rem; }}
.im-hero-big {{ color:{TEXT}; font-size:1.9rem; font-weight:700; margin:.15rem 0 .35rem 0; }}
.im-hero-desc {{ color:{MUTED}; font-size:.92rem; max-width:860px; }}
.im-chip {{ display:inline-block; margin:.6rem .4rem 0 0; padding:.22rem .7rem; border-radius:999px;
  font-size:.75rem; color:{TEXT}; background:rgba(47,140,255,.14); border:1px solid {BORDER}; }}

/* ---------- KPI cards */
.im-card {{
  border-radius:16px; padding:1rem 1.1rem; min-height:112px; height:100%;
  background: linear-gradient(145deg, rgba(16,36,74,.92), rgba(7,18,40,.95));
  border:1px solid {BORDER}; box-shadow: 0 0 24px rgba(47,140,255,.10);
}}
.im-card-label {{ color:{MUTED}; font-size:.82rem; display:flex; align-items:center; gap:.45rem; }}
.im-card-icon {{ font-size:1.1rem; }}
.im-card-value {{ font-size:1.6rem; font-weight:700; margin-top:.25rem; line-height:1.2; }}
.im-card-sub {{ font-size:.8rem; color:{MUTED}; margin-top:.2rem; }}
.im-bar {{ height:6px; border-radius:6px; background:rgba(120,160,255,.15); margin-top:.55rem; overflow:hidden; }}
.im-bar > div {{ height:100%; border-radius:6px; }}

/* ---------- headings */
h1, h2, h3 {{ color:{TEXT} !important; }}
.stMarkdown h3, div[data-testid="stHeadingWithActionElements"] h3 {{
  border-left:4px solid {BLUE}; padding-left:.65rem; font-size:1.2rem;
  text-shadow: 0 0 18px rgba(47,140,255,.35);
}}

/* ---------- tabs as pills (works for old baseweb and new react-aria tab markup) */
div[data-baseweb="tab-list"], .stTabs [role="tablist"] {{ gap:.45rem; border-bottom:none !important; }}
button[data-baseweb="tab"], .stTabs [role="tab"] {{
  background:rgba(13,31,64,.7); border:1px solid {BORDER}; border-radius:12px; padding:.45rem 1rem;
  color:{MUTED}; cursor:pointer;
}}
button[data-baseweb="tab"][aria-selected="true"], .stTabs [role="tab"][aria-selected="true"] {{
  background:linear-gradient(135deg, {BLUE}, #1b5fd1); color:white; border-color:transparent;
  box-shadow:0 0 18px rgba(47,140,255,.45);
}}
.stTabs [role="tab"][aria-selected="true"] p, .stTabs [role="tab"][aria-selected="true"] * {{ color:white !important; }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display:none; }}
.stTabs [role="tablist"] + div, .stTabs [role="tablist"]::after {{ border:none; }}

/* ---------- buttons */
.stButton > button, .stDownloadButton > button {{
  border-radius:12px; border:1px solid {BORDER}; background:rgba(13,31,64,.8); color:{TEXT};
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:{BLUE}; color:white;
  box-shadow:0 0 14px rgba(47,140,255,.35); }}
.stButton > button[kind="primary"] {{
  background:linear-gradient(135deg, {BLUE}, #1b5fd1); border:none; color:white; font-weight:600;
  box-shadow:0 0 22px rgba(47,140,255,.45);
}}

/* ---------- inputs, expanders, alerts, tables, charts */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea, .stNumberInput input {{
  background:rgba(10,23,48,.9) !important; border-color:{BORDER} !important; border-radius:10px !important;
}}
div[data-testid="stExpander"] details {{
  background:rgba(10,23,48,.75); border:1px solid {BORDER}; border-radius:14px;
}}
div[data-testid="stAlert"] {{ border-radius:14px; }}
div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"], div[data-testid="stImage"] img {{
  border-radius:14px;
}}
div[data-testid="stPlotlyChart"] {{
  background: linear-gradient(145deg, rgba(16,36,74,.75), rgba(7,18,40,.85));
  border:1px solid {BORDER}; padding:.4rem; box-shadow:0 0 20px rgba(47,140,255,.08);
}}
div[data-testid="stFileUploaderDropzone"] {{ background:rgba(10,23,48,.8); border:1px dashed {BLUE};
  border-radius:14px; }}
hr {{ border-color:{BORDER} !important; }}

/* ---------- dashboard cards (bordered containers): old + new Streamlit DOM */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .im-card-title) {{
  background: linear-gradient(150deg, rgba(20,34,78,.92), rgba(8,16,40,.96)) !important;
  border: 1px solid rgba(124,108,255,.35) !important; border-radius: 16px !important;
  box-shadow: 0 0 26px rgba(99,102,241,.12), inset 0 1px 0 rgba(255,255,255,.03);
  padding: .55rem .8rem .7rem .8rem;
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stPlotlyChart"],
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .im-card-title)
  div[data-testid="stPlotlyChart"] {{ background: none !important; border: none !important;
  box-shadow: none !important; padding: 0 !important; }}
.im-card-title {{ font-weight: 700; font-size: 1.0rem; color: {TEXT}; border-left: 3px solid #8b5cf6;
  padding-left: .55rem; margin: .1rem 0 .15rem 0; text-shadow: 0 0 14px rgba(139,92,246,.35); }}
.im-card-sub2 {{ color: {MUTED}; font-size: .78rem; margin: 0 0 .35rem .1rem; }}
.im-section {{ font-size: 1.35rem; font-weight: 700; color: {TEXT}; margin-top: .3rem; }}
.im-tablewrap {{ overflow-x: auto; margin: .25rem 0 .45rem 0; }}
.im-table {{ width: 100%; border-collapse: collapse; font-size: .78rem; color: {TEXT}; }}
.im-table th {{ text-align: left; color: #c4b5fd; font-weight: 600; padding: .32rem .45rem;
  border-bottom: 1px solid rgba(139,92,246,.45); background: rgba(139,92,246,.10); white-space: nowrap; }}
.im-table td {{ padding: .28rem .45rem; border-bottom: 1px solid rgba(120,160,255,.10); }}
.im-table tr:hover td {{ background: rgba(59,130,246,.08); }}
.im-table tr.im-hl td {{ background: rgba(34,211,238,.10); font-weight: 600; }}
</style>
"""


def esc(text) -> str:
    return html.escape(str(text))


def brand_html() -> str:
    return (
        "<div class='im-brand'><div class='im-brand-logo'>⚙️</div><div>"
        "<div class='im-brand-name'>IntelliMaintain</div>"
        "<div class='im-brand-sub'>AI-Powered Predictive Maintenance</div></div></div>"
    )


def status_html(model_ok: bool, index_ok: bool) -> str:
    ok = model_ok and index_ok
    color = GREEN if ok else RED
    rows = (f"<div style='margin-top:.35rem;color:{MUTED};font-size:.78rem'>"
            f"Classifier {'✅' if model_ok else '❌'} &nbsp;·&nbsp; Knowledge index {'✅' if index_ok else '❌'}</div>")
    return (f"<div class='im-status'><span class='im-dot' style='color:{color};background:{color}'></span>"
            f"<b>System Status</b><div style='color:{color};font-size:.8rem;margin-left:1.3rem'>"
            f"{'All systems online' if ok else 'Setup incomplete'}</div>{rows}</div>"
            "<div class='im-tagline'>Predict • Prevent • Perform</div>")


def hero_html() -> str:
    chips = "".join(f"<span class='im-chip'>{c}</span>" for c in
                    ("ML fault classifier", "Physics cross-check", "Hybrid RAG", "LLM agent", "Human review"))
    return (
        "<div class='im-hero'><div class='im-hero-small'>Welcome back,</div>"
        "<div class='im-hero-big'>Let’s keep your machines healthy</div>"
        "<div class='im-hero-desc'>IntelliMaintain — Bearing Fault Diagnosis Copilot. ML fault classifier + physics "
        "cross-check + hybrid RAG over manuals and maintenance logs → cited, advisory explanation for human review."
        f"</div>{chips}</div>"
    )


def card_html(icon: str, label: str, value: str, sub: str = "", color: str = TEXT,
              bar: float | None = None, bar_color: str | None = None) -> str:
    bar_html = ""
    if bar is not None:
        pct = max(0.0, min(1.0, float(bar))) * 100
        bar_html = (f"<div class='im-bar'><div style='width:{pct:.0f}%;background:{bar_color or color};"
                    f"box-shadow:0 0 10px {bar_color or color}'></div></div>")
    return (
        f"<div class='im-card'><div class='im-card-label'><span class='im-card-icon'>{icon}</span>{esc(label)}</div>"
        f"<div class='im-card-value' style='color:{color}'>{esc(value)}</div>"
        f"<div class='im-card-sub'>{esc(sub)}</div>{bar_html}</div>"
    )
