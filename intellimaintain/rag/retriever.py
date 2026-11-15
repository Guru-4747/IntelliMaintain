"""Hybrid retrieval: dense (BGE + FAISS) + sparse (BM25) + reciprocal rank fusion + reranker.

Retrieval modes (used by the 2x3 ablation study):
  none      -> no retrieval (LLM-only baseline, C0)
  basic     -> dense FAISS search only                      (C1: basic RAG)
  improved  -> dense + BM25 fused with RRF, then cross-encoder rerank   (C2: improved RAG)
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from ..schemas import EvidenceChunk
from .embed import BGEEmbedder, LSAEmbedder, get_embedder
from .ingest import Chunk, chunk_pages, load_corpus

_WORD = re.compile(r"[a-z0-9]+")
# Probe questions used to calibrate the "off-topic" similarity threshold for whichever embedder built the index.
OFF_TOPIC_PROBES = [
    "how to cook pasta at home", "who won the cricket world cup", "best smartphone under 20000 rupees",
    "write a poem about the moon", "latest movie reviews this week", "how to learn guitar chords",
    "what is the capital of australia", "tips for a job interview",
]
ON_TOPIC_PROBES = [
    "outer race bearing fault symptoms", "how often should bearings be relubricated",
    "what does high kurtosis mean in vibration analysis", "safety steps before replacing a bearing",
    "inner race defect frequency", "bearing failure modes",
]


def calibrate_threshold(embedder, index) -> dict:
    """Threshold halfway between the best-matching off-topic and the worst on-topic probe question."""
    def top1(q):
        scores, _ = index.search(embedder.embed_query(q), 1)
        return float(scores[0][0])

    if getattr(embedder, "kind", "") != "bge":
        return {}  # sparse LSA vectors: a single shared word can score high, so keep the fixed threshold
    off = [top1(q) for q in OFF_TOPIC_PROBES]
    on = [top1(q) for q in ON_TOPIC_PROBES]
    hi_off, lo_on = max(off), min(on)
    thr = (hi_off + lo_on) / 2 if lo_on > hi_off else hi_off + 0.02
    thr = min(max(thr, 0.45), 0.70)  # stay in a sensible range for BGE cosine scores
    return {"min_relevance": round(thr, 4), "probe_off_topic_max": round(hi_off, 4),
            "probe_on_topic_min": round(lo_on, 4)}


_STOP = set("""a an the and or of to in on for with by is are was were be been at as it its this that these those
from into than then so such can may should must not no do does did how what which who when where why""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _reused_chunks(index_dir: Path, present_sources: set[str]) -> list[Chunk]:
    """Chunks of downloaded PDFs that are in the current index but not on this computer (PDFs are not in git)."""
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        old = [Chunk(**json.loads(line)) for line in f if line.strip()]
    return [c for c in old if c.source.lower().endswith(".pdf") and c.source not in present_sources]


def build_index(corpus_dir: Path = config.CORPUS_DIR, index_dir: Path = config.INDEX_DIR,
                prefer: str = "bge", verbose: bool = True, reuse_missing_pdfs: bool = False) -> dict:
    t0 = time.time()
    pages = load_corpus(corpus_dir)
    if not pages:
        raise FileNotFoundError(f"No .pdf/.md/.txt documents found in {corpus_dir}")
    chunks = chunk_pages(pages)
    n_pages = len(pages)
    if reuse_missing_pdfs:
        reused = _reused_chunks(index_dir, {p.source for p in pages})
        if verbose and reused:
            print(f"Reusing {len(reused)} chunks of {len({c.source for c in reused})} PDFs that are indexed but "
                  "not present in data/corpus on this computer.")
        chunks += reused
        n_pages += len({(c.source, c.page) for c in reused})
    texts = [f"{c.title}\n{c.text}" for c in chunks]
    emb = get_embedder(prefer, corpus_texts=texts, verbose=verbose)
    if verbose:
        print(f"Embedding {len(chunks)} chunks from {len(pages)} pages with {emb.model_name} ...", flush=True)
    vecs = np.ascontiguousarray(emb.embed_documents(texts), dtype=np.float32)
    import faiss  # imported after PyTorch on purpose (avoids an OpenMP crash on macOS)

    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine similarity on normalised vectors
    index.add(vecs)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    emb.save(index_dir)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    meta = {
        "embedder": emb.kind, "embed_model": emb.model_name, "dim": int(vecs.shape[1]),
        "n_chunks": len(chunks), "n_pages": n_pages,
        "sources": sorted({c.source for c in chunks}),
        "chunk_tokens": config.CHUNK_TOKENS, "chunk_overlap": config.CHUNK_OVERLAP,
        **calibrate_threshold(emb, index),
        "build_seconds": round(time.time() - t0, 1),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"Indexed {len(chunks)} chunks from {len(meta['sources'])} documents "
              f"({n_pages} pages) with {emb.model_name} -> {index_dir}")
    load_retriever.cache_clear()
    return meta


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        from rank_bm25 import BM25Okapi

        if not (index_dir / "faiss.index").exists():
            raise FileNotFoundError(f"No RAG index in {index_dir}. Run: python scripts/build_index.py")
        self.meta = json.loads((index_dir / "meta.json").read_text())
        # load the embedder (PyTorch) before faiss - avoids an OpenMP clash on macOS
        if self.meta["embedder"] == "bge":
            self.embedder = BGEEmbedder(self.meta["embed_model"])
        else:
            self.embedder = LSAEmbedder.load(index_dir)
        import faiss

        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        self.bm25 = BM25Okapi([tokenize(f"{c.title} {c.text}") for c in self.chunks])
        self._reranker = None
        self._reranker_failed = False

    # ---------------------------------------------------------------- components
    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.embed_query(query)
        scores, idx = self.index.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]

    @property
    def reranker(self):
        if config.RERANK_MODEL.strip().lower() in ("", "none", "off"):
            return None
        if self._reranker is None and not self._reranker_failed:
            try:
                from sentence_transformers import CrossEncoder

                from .embed import load_hf_model

                self._reranker = load_hf_model(CrossEncoder, config.RERANK_MODEL)
            except Exception:  # noqa: BLE001 - optional component
                self._reranker_failed = True
        return self._reranker

    # ---------------------------------------------------------------- public API
    def search(self, query: str, top_k: int = config.TOP_K, mode: str = "improved") -> list[EvidenceChunk]:
        if mode == "none" or not query.strip():
            return []
        if mode == "basic":
            hits = self.dense(query, top_k)
            return [self._to_evidence(i, s, "dense") for i, s in hits]
        if mode not in ("improved", "hybrid", "bm25"):
            raise ValueError(f"Unknown retrieval mode '{mode}'")
        if mode == "bm25":
            return [self._to_evidence(i, s, "bm25") for i, s in self.sparse(query, top_k)]

        pool = max(20, top_k * 4)
        dense, sparse = self.dense(query, pool), self.sparse(query, pool)
        fused: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        cand = sorted(fused, key=fused.get, reverse=True)[:pool]
        dense_score = dict(dense)
        if mode == "improved" and self.reranker is not None:
            pairs = [(query, self.chunks[i].text) for i in cand]
            rr = self.reranker.predict(pairs)
            order = np.argsort(-np.asarray(rr))[:top_k]
            return [self._to_evidence(cand[j], float(rr[j]), "rerank") for j in order]
        # no cross-encoder available: keep RRF order, report dense similarity as score
        out = []
        for i in cand[:top_k]:
            s = dense_score.get(i)
            if s is None:
                s = float(np.ravel(self.embedder.embed_query(query) @ self.index.reconstruct(i))[0])
            out.append(self._to_evidence(i, s, "hybrid"))
        return out

    def _to_evidence(self, i: int, score: float, retriever: str) -> EvidenceChunk:
        c = self.chunks[i]
        return EvidenceChunk(chunk_id=c.chunk_id, source=c.source, page=c.page, text=c.text,
                             title=c.title, score=round(score, 4), retriever=retriever)

    def relevance(self, query: str, chunks: list[EvidenceChunk]) -> float:
        """Max dense cosine similarity between the query and the evidence (0..1)."""
        if not chunks:
            return 0.0
        q = self.embedder.embed_query(query)[0]
        idx = {c.chunk_id: n for n, c in enumerate(self.chunks)}
        sims = [float(q @ self.index.reconstruct(idx[c.chunk_id])) for c in chunks if c.chunk_id in idx]
        return max(sims) if sims else 0.0

    @property
    def min_relevance(self) -> float:
        """Below this similarity the evidence is considered off-topic -> 'insufficient evidence'."""
        if "min_relevance" in self.meta:  # calibrated when the index was built
            return float(self.meta["min_relevance"])
        return 0.55 if self.meta["embedder"] == "bge" else 0.12  # older indexes without calibration


@lru_cache(maxsize=1)
def load_retriever() -> Retriever:
    return Retriever()


def retrieve_evidence(query: str, top_k: int = config.TOP_K, mode: str = "improved") -> list[EvidenceChunk]:
    """P2 contract: return the top-k evidence chunks (with source + page) for a query."""
    return load_retriever().search(query, top_k=top_k, mode=mode)
