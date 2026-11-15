"""Generate the README banner and flow diagrams (docs/assets/banner.svg, system.svg, rag.svg).

Usage: python scripts/make_readme_svgs.py
"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import math, random
from pathlib import Path
OUT = Path("docs/assets")
FONT = "font-family=\"'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif\""

# ---------------------------------------------------------------- banner
random.seed(3)
pts = []
for i in range(0, 1201, 3):
    t = i / 1200
    base = 6 * math.sin(i / 9) + random.uniform(-4, 4)
    burst = 0
    for c in range(80, 1200, 95):  # periodic impacts, like a bearing defect
        d = i - c
        if 0 <= d < 40:
            burst += 34 * math.exp(-d / 9) * math.sin(d * 1.6)
    pts.append(f"{i},{278 + base + burst:.1f}")
wave = " ".join(pts)
pills = [("Machine Learning", "#2f8cff"), ("Physics Validation", "#22c55e"), ("Hybrid RAG", "#a78bfa"),
         ("AI Agent", "#f59e0b"), ("Chat Copilot", "#22d3ee")]
x = 60
pill_svg = ""
for label, c in pills:
    w = 16 + len(label) * 8.6
    pill_svg += (f'<rect x="{x}" y="196" width="{w:.0f}" height="30" rx="15" fill="{c}" fill-opacity=".14" '
                 f'stroke="{c}" stroke-opacity=".55"/><text x="{x + w / 2:.0f}" y="216" text-anchor="middle" '
                 f'font-size="14.5" fill="{c}" {FONT} font-weight="600">{label}</text>')
    x += w + 12
banner = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360" viewBox="0 0 1200 360">
<defs>
 <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1730"/><stop offset="1" stop-color="#101f45"/></linearGradient>
 <linearGradient id="wv" x1="0" x2="1"><stop offset="0" stop-color="#22d3ee" stop-opacity=".05"/><stop offset=".5" stop-color="#22d3ee" stop-opacity=".9"/><stop offset="1" stop-color="#2f8cff" stop-opacity=".25"/></linearGradient>
 <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1f3460" stroke-width=".6" stroke-opacity=".55"/></pattern>
 <radialGradient id="glow" cx=".85" cy=".2" r=".6"><stop offset="0" stop-color="#2f8cff" stop-opacity=".28"/><stop offset="1" stop-color="#2f8cff" stop-opacity="0"/></radialGradient>
</defs>
<rect width="1200" height="360" rx="22" fill="url(#bg)"/>
<rect width="1200" height="360" rx="22" fill="url(#grid)"/>
<rect width="1200" height="360" rx="22" fill="url(#glow)"/>
<polyline points="{wave}" fill="none" stroke="url(#wv)" stroke-width="2"/>
<g transform="translate(60,52)">
 <rect width="58" height="58" rx="14" fill="#2f8cff" fill-opacity=".18" stroke="#2f8cff" stroke-opacity=".6"/>
 <text x="29" y="38" text-anchor="middle" font-size="22" font-weight="800" fill="#e6edf7" {FONT}>IM</text>
</g>
<text x="134" y="96" font-size="48" font-weight="800" fill="#e6edf7" {FONT}>IntelliMaintain</text>
<text x="60" y="148" font-size="26" font-weight="600" fill="#22d3ee" {FONT}>AI-Powered Predictive Maintenance Copilot</text>
<text x="60" y="180" font-size="18" fill="#93a4c3" {FONT}>Detect.  Verify.  Explain.  Prevent.</text>
{pill_svg}
<text x="1140" y="342" text-anchor="end" font-size="13" fill="#93a4c3" {FONT}>An evidence-grounded industrial bearing diagnosis system · CWRU bearing data</text>
</svg>'''
(OUT / "banner.svg").write_text(banner)

# ---------------------------------------------------------------- boxes helper
def box(x, y, w, h, title, sub, color, num=None):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{color}" fill-opacity=".10" '
         f'stroke="{color}" stroke-opacity=".7" stroke-width="1.4"/>')
    ty = y + h / 2 - (8 if sub else -6) + (9 if num else 0)
    if num:
        s += f'<text x="{x + 14}" y="{y + 22}" font-size="12" font-weight="700" fill="{color}" {FONT}>{num}</text>'
    s += (f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="16" font-weight="700" fill="#e6edf7" '
          f'{FONT}>{title}</text>')
    for k, line in enumerate(sub.split("|") if sub else []):
        s += (f'<text x="{x + w / 2}" y="{ty + 22 + 17 * k}" text-anchor="middle" font-size="12.5" fill="#93a4c3" '
              f'{FONT}>{line}</text>')
    return s


def arrow(x1, y1, x2, y2, color="#4b6aa8"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" '
            f'marker-end="url(#ah)"/>')


DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto"><path d="M0 0L10 5L0 10z" fill="#4b6aa8"/></marker></defs>')

# ---------------------------------------------------------------- whole system
W, H = 1200, 420
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS,
     f'<rect width="{W}" height="{H}" rx="20" fill="#0b1730"/>',
     f'<text x="40" y="44" font-size="20" font-weight="800" fill="#e6edf7" {FONT}>How a question is answered</text>',
     f'<text x="40" y="68" font-size="13.5" fill="#93a4c3" {FONT}>Bounded agent: at most 6 calls to 4 approved tools, '
     f'then a grounded answer that is checked sentence by sentence</text>']
s.append(box(40, 106, 176, 96, "Vibration signal", "CSV · TXT · MAT · NPY|any sampling rate", "#22d3ee", "INPUT"))
s.append(box(40, 250, 176, 96, "Your question", "typed like a chat|typos &amp; follow-ups OK", "#22d3ee", "INPUT"))
s.append(box(262, 88, 216, 84, "ML classifier", "RandomForest or CNN", "#2f8cff", "TOOL 1"))
s.append(box(262, 182, 216, 84, "Physics check", "BPFO · BPFI · BSF envelope", "#22c55e", "TOOL 2"))
s.append(box(262, 276, 216, 84, "Document search", "BGE + BM25 + rerank", "#a78bfa", "TOOL 3"))
s.append(f'<text x="370" y="384" text-anchor="middle" font-size="12.5" fill="#f59e0b" {FONT}>+ TOOL 4: online search (Wikipedia), optional</text>')
s.append(box(530, 146, 210, 150, "Bounded agent", "plans the tool calls|reuses the diagnosis|in a chat|max 6 steps", "#f59e0b", "AGENT"))
s.append(box(796, 100, 190, 100, "Gemini answer", "fault type locked to|ML + physics", "#f59e0b", "LLM"))
s.append(box(796, 240, 190, 100, "Verifier", "every sentence must match|its cited source", "#f43f5e", "CHECK"))
s.append(box(1018, 140, 156, 160, "Cited answer", "[ML] [PHYS] [1] [2]|how the fault occurs|human review", "#22c55e", "OUTPUT"))
s += [arrow(216, 154, 260, 132), arrow(216, 154, 260, 222), arrow(216, 298, 260, 316),
      arrow(478, 130, 528, 192), arrow(478, 224, 528, 222), arrow(478, 318, 528, 252),
      arrow(740, 200, 794, 152), arrow(891, 200, 891, 238), arrow(986, 290, 1016, 262),
      f'<path d="M796 310 C 760 340, 700 340, 660 298" fill="none" stroke="#4b6aa8" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ah)"/>',
      f'<text x="690" y="362" font-size="11.5" fill="#93a4c3" {FONT}>fails check → verified offline answer</text>',
      '</svg>']
(OUT / "system.svg").write_text("\n".join(s))

# ---------------------------------------------------------------- RAG / evidence card
W, H = 1200, 300
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS,
     f'<rect width="{W}" height="{H}" rx="20" fill="#0b1730"/>']
items = [("Knowledge base", "13 guides + SKF/NSK|manuals + work orders", "#a78bfa"),
         ("Chunking", "~600 tokens|page-tagged", "#a78bfa"),
         ("Hybrid search", "BGE + FAISS dense|BM25 keywords", "#2f8cff"),
         ("Fusion + rerank", "reciprocal rank fusion|cross-encoder", "#2f8cff"),
         ("Grounded answer", "only from evidence|cites [1] [2] [ML]", "#f59e0b"),
         ("Verified", "unsupported sentences|are removed", "#22c55e")]
x0, w, gap = 30, 166, 31
for i, (t, sub, c) in enumerate(items):
    x = x0 + i * (w + gap)
    s.append(box(x, 90, w, 120, t, sub, c, f"{i + 1:02d}"))
    if i:
        s.append(arrow(x - gap + 2, 150, x - 3, 150))
s.append(f'<text x="30" y="52" font-size="20" font-weight="800" fill="#e6edf7" {FONT}>Evidence-based answers</text>')
s.append(f'<text x="30" y="250" font-size="13.5" fill="#93a4c3" {FONT}>Optional online search adds Wikipedia passages '
         f'(numbered after the documents, linked, verified the same way). If nothing relevant is found, the copilot '
         f'says so instead of guessing.</text>')
s.append('</svg>')
(OUT / "rag.svg").write_text("\n".join(s))
print("ok")

# ---------------------------------------------------------------- compact 3 x 2 version for the README card
W, H = 640, 470
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS,
     f'<rect width="{W}" height="{H}" rx="18" fill="#0b1730"/>',
     f'<text x="26" y="42" font-size="20" font-weight="800" fill="#e6edf7" {FONT}>Evidence-based answers</text>']
bw, bh, gx = 176, 128, 32
for i, (t, sub, c) in enumerate(items):
    row, col = divmod(i, 3)
    if row == 1:
        col = 2 - col  # snake layout: the second row runs right to left
    x, y = 26 + col * (bw + gx), 70 + row * (bh + 70)
    s.append(box(x, y, bw, bh, t, sub, c, f"{i + 1:02d}"))
    if i in (1, 2):
        s.append(arrow(x - gx + 2, y + bh / 2, x - 3, y + bh / 2))
    if i == 3:
        s.append(arrow(x + bw / 2, y - 68, x + bw / 2, y - 3))
    if i in (4, 5):
        s.append(arrow(x + bw + gx - 2, y + bh / 2, x + bw + 3, y + bh / 2))
s.append(f'<text x="26" y="{H - 34}" font-size="13" fill="#93a4c3" {FONT}>Optional online search adds linked Wikipedia '
         f'passages, verified the same way.</text>')
s.append(f'<text x="26" y="{H - 14}" font-size="13" fill="#93a4c3" {FONT}>If nothing relevant is found, the copilot '
         f'says so instead of guessing.</text>')
s.append('</svg>')
(OUT / "rag_card.svg").write_text("\n".join(s))
print("ok card")
