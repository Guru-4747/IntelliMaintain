"""Parse knowledge-base documents into page-tagged text and split into chunks.

* PDF  -> one "page" per physical page (PyMuPDF), so citations point to real page numbers.
* .md / .txt -> one "page" per top-level section (# heading), cited as section numbers.
Chunks never cross a page boundary, so every chunk has exactly one (source, page) citation.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Approximate sub-word token count (close to BERT/BGE word-piece counts for English)."""
    return int(len(_TOKEN_RE.findall(text)) * 1.1)


@dataclass
class Page:
    source: str
    page: int
    text: str
    title: str = ""


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int
    text: str
    title: str = ""


def _clean(text: str) -> str:
    text = text.replace("­", "")              # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path) -> list[Page]:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(path) as doc:
        title = (doc.metadata or {}).get("title") or path.stem
        for i, pg in enumerate(doc, start=1):
            txt = _clean(pg.get_text("text"))
            # re-flow PDF lines into paragraphs (keep list items / bullets on their own line)
            txt = re.sub(r"(?<![.!?:])\n(?!\s*(?:\d+[.)]|[-\u2022*]))", " ", txt)
            if len(txt) > 40:  # skip blank/figure-only pages
                pages.append(Page(path.name, i, txt, title))
    return pages


def parse_markdown(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = path.stem
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, flags=re.S)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), flags=re.M)
        if t:
            title = t.group(1).strip()
        raw = raw[m.end():]
    sections = re.split(r"(?m)^#\s+", raw)
    pages = []
    for sec in sections:
        sec = _clean(sec)
        if len(sec) > 40:
            pages.append(Page(path.name, len(pages) + 1, sec, title))
    return pages


def load_corpus(corpus_dir: Path = config.CORPUS_DIR) -> list[Page]:
    pages: list[Page] = []
    files = sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in (".pdf", ".md", ".txt"))
    for f in files:
        if f.name.lower() in ("readme.md", "sources.md"):
            continue
        try:
            pages.extend(parse_pdf(f) if f.suffix.lower() == ".pdf" else parse_markdown(f))
        except Exception as e:  # noqa: BLE001 - a broken PDF must not stop indexing
            print(f"[warn] could not parse {f.name}: {e}")
    return pages


def _split_sentences(text: str, max_unit_tokens: int = 200) -> list[str]:
    """Split into paragraph / list-item units; only very long paragraphs are split into sentences.

    Keeping list items whole preserves procedures like "1. Verify ... 2. Increase ..." intact.
    """
    units: list[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        if count_tokens(line) <= max_unit_tokens:
            units.append(line)
        else:
            units.extend(p.strip() for p in re.split(r"(?<=[^0-9][.!?])\s+", line) if p.strip())
    return units


def chunk_pages(pages: list[Page], chunk_tokens: int = config.CHUNK_TOKENS,
                overlap_tokens: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Sentence-aware sliding chunker: ~chunk_tokens per chunk, ~overlap_tokens carried over."""
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    for pg in pages:
        sents = _split_sentences(pg.text)
        cur: list[str] = []
        cur_tok = 0
        idx = 0

        def flush():
            nonlocal idx
            text = "\n".join(cur).strip()
            if text:
                h = hashlib.md5(f"{pg.source}|{pg.page}|{idx}|{text[:50]}".encode()).hexdigest()[:8]
                chunks.append(Chunk(f"{Path(pg.source).stem}-p{pg.page}-c{idx}-{h}", pg.source, pg.page, text, pg.title))
                idx += 1

        for s in sents:
            t = count_tokens(s)
            if cur and cur_tok + t > chunk_tokens:
                flush()
                # carry the last sentences as overlap
                carry, carry_tok = [], 0
                for prev in reversed(cur):
                    pt = count_tokens(prev)
                    if carry_tok + pt > overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carry_tok += pt
                cur, cur_tok = carry, carry_tok
            cur.append(s)
            cur_tok += t
        flush()
    return chunks
