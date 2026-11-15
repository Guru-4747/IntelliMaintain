import pytest

from conftest import needs_index
from intellimaintain.rag.ingest import Page, chunk_pages, count_tokens, parse_markdown, parse_pdf
from intellimaintain.schemas import EvidenceChunk


def test_chunker_respects_size_overlap_and_pages():
    para = " ".join(f"Sentence number {i} talks about bearing kurtosis and envelope analysis." for i in range(200))
    pages = [Page("doc.pdf", 1, para, "Doc"), Page("doc.pdf", 2, "Short page about BPFO harmonics.", "Doc")]
    chunks = chunk_pages(pages, chunk_tokens=600, overlap_tokens=100)
    assert len(chunks) > 3
    assert all(count_tokens(c.text) <= 700 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}           # never crosses a page
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert set(first[-20:]) & set(second[:40])          # overlap carried over
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunker_keeps_list_items_whole():
    text = "Actions:\n1. Verify the finding. Repeat the measurement.\n2. Increase monitoring frequency."
    chunks = chunk_pages([Page("a.md", 1, text)], 600, 100)
    assert "1. Verify the finding. Repeat the measurement." in chunks[0].text


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_pages([Page("a", 1, "x")], chunk_tokens=100, overlap_tokens=100)


def test_pdf_parsing_keeps_page_numbers(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(3):
        p = doc.new_page()
        p.insert_text((72, 72), f"Page {i + 1}: outer race fault inspection procedure step text for testing.")
    path = tmp_path / "manual.pdf"
    doc.save(path)
    pages = parse_pdf(path)
    assert [p.page for p in pages] == [1, 2, 3] and pages[0].source == "manual.pdf"


def test_markdown_sections_are_pages(tmp_path):
    f = tmp_path / "kb.md"
    f.write_text("---\ntitle: KB\n---\n# 1. One\n\nFirst section text is long enough to be kept.\n\n# 2. Two\n\n"
                 "Second section text is also long enough to be kept.\n")
    pages = parse_markdown(f)
    assert [p.page for p in pages] == [1, 2] and pages[0].title == "KB"


@needs_index
@pytest.mark.parametrize("mode", ["basic", "improved", "bm25"])
def test_retrieve_evidence_contract(mode):
    from intellimaintain.rag.retriever import retrieve_evidence

    ev = retrieve_evidence("outer race fault high kurtosis inspection action", top_k=5, mode=mode)
    assert 0 < len(ev) <= 5 and all(isinstance(e, EvidenceChunk) for e in ev)
    assert all(e.page >= 1 and e.source for e in ev)
    assert any("outer" in e.source for e in ev[:3])


@needs_index
def test_retrieval_modes_and_relevance():
    from intellimaintain.rag.retriever import load_retriever

    r = load_retriever()
    assert r.search("anything", mode="none") == []
    with pytest.raises(ValueError):
        r.search("x", mode="bogus")
    on = r.relevance("lockout tagout before bearing replacement", r.search("lockout tagout before bearing replacement"))
    off = r.relevance("best chicken biryani recipe", r.search("best chicken biryani recipe"))
    assert on > r.min_relevance > off


def test_lsa_embedder_roundtrip(tmp_path):
    from intellimaintain.rag.embed import LSAEmbedder

    texts = ["outer race fault bpfo", "inner race fault bpfi", "lubrication grease interval", "lockout tagout"]
    e = LSAEmbedder(8).fit(texts)
    e.save(tmp_path)
    e2 = LSAEmbedder.load(tmp_path)
    q = e2.embed_query("grease relubrication")
    assert q.shape[0] == 1 and (e2.embed_documents(texts) @ q[0]).argmax() == 2


def test_bge_embedder_code_path(monkeypatch):
    """BGE path with a fake SentenceTransformer (no download needed): checks prefix + normalisation."""
    import sys
    import types

    import numpy as np

    calls = {}

    class FakeST:
        def __init__(self, name, device=None):
            calls["name"] = name

        def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
            calls.setdefault("texts", []).extend(texts)
            v = np.ones((len(texts), 4))
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    from intellimaintain import config
    from intellimaintain.rag.embed import BGEEmbedder

    e = BGEEmbedder()
    assert calls["name"] == config.EMBED_MODEL
    assert e.embed_documents(["a", "b"]).shape == (2, 4)
    e.embed_query("bearing")
    assert calls["texts"][-1].startswith(config.BGE_QUERY_PREFIX)


def test_models_load_from_local_cache_without_internet(monkeypatch):
    """No internet: the first load fails, the retry with HF_HUB_OFFLINE=1 uses the cached copy."""
    import os

    from intellimaintain.rag.embed import load_hf_model

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = []

    class FakeModel:
        def __init__(self, name, device=None):
            calls.append(os.environ.get("HF_HUB_OFFLINE"))
            if os.environ.get("HF_HUB_OFFLINE") != "1":
                raise OSError("no internet")

    assert isinstance(load_hf_model(FakeModel, "BAAI/bge-small-en-v1.5"), FakeModel)
    assert calls == [None, "1"]
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
