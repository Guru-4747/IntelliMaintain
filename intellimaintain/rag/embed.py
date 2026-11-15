"""Text embedders.

Primary: BGE (BAAI/bge-small-en-v1.5) through sentence-transformers.
Fallback: TF-IDF + truncated SVD (latent semantic analysis). The fallback keeps the
whole system runnable on machines without internet/GPU; the index records which
embedder built it so query and document vectors always match.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .. import config


def load_hf_model(cls, name: str):
    """Load a Hugging Face model; without internet, fall back to the copy cached on this computer."""
    import os

    try:
        return cls(name, device="cpu")
    except Exception:  # noqa: BLE001 - typically no internet: retry from the local cache only
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return cls(name, device="cpu")


class BGEEmbedder:
    kind = "bge"

    def __init__(self, model_name: str = config.EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.model = load_hf_model(SentenceTransformer, model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return np.asarray(v, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = config.BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        v = self.model.encode([prefix + text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def save(self, index_dir: Path) -> None:  # nothing to persist; model reloads by name
        pass


class LSAEmbedder:
    kind = "lsa"
    model_name = "tfidf-lsa-256"

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.vec = None
        self.svd = None

    def fit(self, texts: list[str]) -> LSAEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=config.RANDOM_STATE).fit(X)
        return self

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = self.svd.transform(self.vec.transform(texts)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])

    def save(self, index_dir: Path) -> None:
        joblib.dump({"vec": self.vec, "svd": self.svd}, index_dir / "lsa.joblib")

    @classmethod
    def load(cls, index_dir: Path) -> LSAEmbedder:
        d = joblib.load(index_dir / "lsa.joblib")
        e = cls()
        e.vec, e.svd = d["vec"], d["svd"]
        return e


def get_embedder(prefer: str = "bge", corpus_texts: list[str] | None = None, verbose: bool = True):
    """Return a BGE embedder when available, otherwise a fitted LSA embedder."""
    if prefer == "bge":
        try:
            return BGEEmbedder()
        except Exception as e:  # noqa: BLE001 - missing package, no internet, etc.
            if verbose:
                print(f"[info] BGE embedder unavailable ({type(e).__name__}: {str(e)[:120]}); "
                      "falling back to TF-IDF/LSA. Install sentence-transformers for BGE.")
    if corpus_texts is None:
        raise ValueError("LSA embedder needs corpus_texts to fit")
    return LSAEmbedder().fit(corpus_texts)
