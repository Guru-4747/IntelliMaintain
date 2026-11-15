"""Citation verification (governance layer): every sentence must carry a valid citation."""
from __future__ import annotations

import re

CITE_RE = re.compile(r"\[(\d+|ML|PHYS)\]", re.I)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # keep citations attached to the sentence they follow: "... end. [2] Next" -> "... end [2]."
    text = re.sub(r"\.\s*((?:\[(?:\d+|ML|PHYS)\]\s*)+)", r" \1. ", text, flags=re.I)
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 3]


def citations_in(text: str) -> list[str]:
    return [c.upper() for c in CITE_RE.findall(text or "")]


def clean_invalid_citations(text: str, n_evidence: int, has_ml: bool, has_phys: bool) -> tuple[str, list[str]]:
    """Remove citations that point to non-existent sources; return cleaned text + removed tags."""
    removed: list[str] = []

    def repl(m):
        tag = m.group(1).upper()
        ok = (tag.isdigit() and 1 <= int(tag) <= n_evidence) or (tag == "ML" and has_ml) or (tag == "PHYS" and has_phys)
        if not ok:
            removed.append(f"[{tag}]")
            return ""
        return f"[{tag}]"

    return CITE_RE.sub(repl, text or ""), removed


def citation_coverage(texts: list[str]) -> float:
    sents = [s for t in texts for s in split_sentences(t)]
    if not sents:
        return 0.0
    return round(sum(1 for s in sents if CITE_RE.search(s)) / len(sents), 3)


_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def _norm_tokens(text: str) -> set[str]:
    from ..rag.retriever import tokenize

    return {t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokenize(text)}


def numbers_in(text: str) -> set[str]:
    return set(_NUM_RE.findall(CITE_RE.sub(" ", text or "")))


def number_supported(num: str, context_numbers: set[str]) -> bool:
    if num in context_numbers:
        return True
    try:
        v = float(num)
    except ValueError:
        return False
    if v <= 10 and float(v).is_integer():  # small counts / list numbers ("2 harmonics", "step 3")
        return True
    for c in context_numbers:
        try:
            cv = float(c)
        except ValueError:
            continue
        if cv and abs(cv - v) / abs(cv) <= 0.01:  # tolerate rounding of the last digit
            return True
        if cv and 0 < cv <= 1 and abs(cv * 100 - v) <= 1.0:  # "0.97" written as "97 %"
            return True
    return False


def verify_sentences(sentences: list[str], sources: dict[str, str], min_overlap: float = 0.3):
    """Split sentences into (verified, rejected) against the text of the sources they cite.

    A sentence is verified when it cites at least one existing source, at least `min_overlap` of its content
    words appear in the cited sources, and every number in it appears in the cited sources.
    """
    src_tokens = {k: _norm_tokens(v) for k, v in sources.items()}
    src_numbers = {k: numbers_in(v) | set(_NUM_RE.findall(v)) for k, v in sources.items()}
    ok, bad = [], []
    for sent in sentences:
        tags = [t.upper() for t in CITE_RE.findall(sent)]
        tags = [t for t in tags if t in sources]
        words = _norm_tokens(CITE_RE.sub(" ", sent))
        if not tags or not words:
            bad.append((sent, "no valid citation"))
            continue
        cited_words = set().union(*(src_tokens[t] for t in tags))
        overlap = len(words & cited_words) / len(words)
        cited_nums = set().union(*(src_numbers[t] for t in tags))
        missing = [n for n in numbers_in(sent) if not number_supported(n, cited_nums)]
        if overlap < min_overlap:
            bad.append((sent, f"only {overlap:.0%} of its words appear in the cited source"))
        elif missing:
            bad.append((sent, f"number(s) {', '.join(missing)} not found in the cited source"))
        else:
            ok.append(sent)
    return ok, bad
