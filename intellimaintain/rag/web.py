"""Online evidence: search Wikipedia and return the most relevant passages as citable evidence.

Why Wikipedia: it is free, needs no API key, has stable links, a clear licence (CC BY-SA) and good articles on
bearings, vibration analysis, fatigue and lubrication. The passages are cited and verified exactly like the
maintenance documents, so the LLM still cannot add facts that are not in a retrieved source.

Only `requests` is needed. Every failure raises WebSearchError; the agent turns it into a warning and carries on
with the local documents, so the app works the same without internet.
"""
from __future__ import annotations

import hashlib
import re
import urllib.parse

from .. import config
from ..schemas import EvidenceChunk

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "IntelliMaintain/1.0 (B.Tech final-year project; bearing maintenance copilot)"
_MIN_PARA_CHARS = 120
_MAX_CHUNK_WORDS = 160


class WebSearchError(RuntimeError):
    pass


def _get(params: dict, timeout: int) -> dict:
    import requests

    try:
        r = requests.get(WIKI_API, params={**params, "format": "json", "formatversion": 2}, timeout=timeout,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001 - offline, blocked, timeout, bad JSON ...
        raise WebSearchError(f"online search unavailable ({type(e).__name__})") from e


def web_query(question: str, fault_type: str | None = None, max_terms: int = 8) -> str:
    """Short keyword query: search engines work better with a few key terms than with a whole sentence."""
    from .retriever import tokenize

    terms = []
    for w in tokenize(question):
        if w not in terms and len(w) > 2:
            terms.append(w)
    q = " ".join(terms[:max_terms])
    if fault_type and fault_type not in ("Normal", "Unknown") and fault_type.lower() not in q:
        q = f"{fault_type.lower()} bearing fault {q}"
    elif "bearing" not in q and re.search(r"\b(race|ball|bpf|bsf|spall|brinell|kurtosis|envelope)\w*", q):
        q = f"bearing {q}"
    return q.strip() or question[:100]


def _paragraph_chunks(text: str) -> list[str]:
    """Split an article into passages of whole paragraphs, at most ~160 words each."""
    chunks, cur = [], []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if len(para) < _MIN_PARA_CHARS or para.startswith("=="):
            continue
        words = para.split()
        if len(words) > _MAX_CHUNK_WORDS:  # long paragraph: cut at sentence ends
            sent, n = [], 0
            for s in re.split(r"(?<=[.!?])\s+", para):
                if sent and n + len(s.split()) > _MAX_CHUNK_WORDS:
                    chunks.append(" ".join(sent))
                    sent, n = [], 0
                sent.append(s)
                n += len(s.split())
            if sent:
                chunks.append(" ".join(sent))
            continue
        if cur and sum(len(c.split()) for c in cur) + len(words) > _MAX_CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(para)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _overlap_score(q_tokens: set[str], doc_tokens: list[str]) -> float:
    """Distinct query terms found in the passage, plus a small bonus for repeated mentions.
    (BM25 is not used here: with only a handful of passages its IDF term becomes zero.)"""
    if not q_tokens or not doc_tokens:
        return 0.0
    hits = [t for t in doc_tokens if t in q_tokens]
    return len(set(hits)) + 0.05 * len(hits)


def web_search(query: str, top_k: int = config.WEB_TOP_K, timeout: int = config.WEB_TIMEOUT,
               n_pages: int = 4) -> list[EvidenceChunk]:
    """Search Wikipedia for `query` and return the top_k passages that share the most terms with the query."""
    from .retriever import tokenize

    if not query.strip():
        return []
    hits = _get({"action": "query", "list": "search", "srsearch": query, "srlimit": n_pages}, timeout)
    titles = [h["title"] for h in (hits.get("query") or {}).get("search", [])]
    if not titles:
        return []
    pages = _get({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                  "titles": "|".join(titles)}, timeout)
    passages: list[tuple[str, str]] = []
    for pg in (pages.get("query") or {}).get("pages", []):
        for chunk in _paragraph_chunks(pg.get("extract") or ""):
            passages.append((pg.get("title", ""), chunk))
    if not passages:
        return []
    q_tokens = set(tokenize(query))
    scores = [_overlap_score(q_tokens, tokenize(f"{t} {p}")) for t, p in passages]
    order = sorted(range(len(passages)), key=lambda i: -scores[i])
    out: list[EvidenceChunk] = []
    for i in order:
        if scores[i] <= 0 or len(out) >= top_k:
            break
        title, text = passages[i]
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        h = hashlib.md5(f"{title}|{text[:60]}".encode()).hexdigest()[:8]
        out.append(EvidenceChunk(chunk_id=f"web-{h}", source=f"Wikipedia: {title}", page=1, text=text,
                                 score=round(float(scores[i]), 3), retriever="web", title=title, url=url))
    return out
