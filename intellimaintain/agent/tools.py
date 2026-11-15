"""The three approved agent tools. The agent can call nothing else (bounded autonomy)."""
from __future__ import annotations

import numpy as np

from .. import config
from ..ml.classifier import classify_signal as _classify
from ..ml.physics import cross_check, defect_frequencies, physics_check
from ..rag.retriever import retrieve_evidence as _retrieve
from ..schemas import ClassifyResult, EvidenceChunk, FaultFrequencies


def classify_signal(signal: np.ndarray, fs: int | None = config.FS,
                    override_label: str | None = None, model: str = "rf") -> ClassifyResult:
    """Tool 1 - ML fault classifier (P1): model = "rf" (features) or "cnn" (spectrogram CNN)."""
    return _classify(signal, fs=fs, override_label=override_label, model=model)


def fault_frequency_calc(rpm: float, signal: np.ndarray | None = None,
                         ml_fault_type: str | None = None) -> FaultFrequencies:
    """Tool 2 - defect frequencies; with a signal also the envelope physics cross-check."""
    if signal is None or len(signal) < 4096:
        f = defect_frequencies(rpm)
        return FaultFrequencies(rpm=float(rpm), bearing=config.BEARING_6205["name"],
                                **{k: round(v, 3) for k, v in f.items()})
    ff = physics_check(signal, rpm)
    if ml_fault_type:
        ff.consistency, ff.consistency_note = cross_check(ml_fault_type, ff)
    return ff


def retrieve_evidence(query: str, top_k: int = config.TOP_K, mode: str = "improved") -> list[EvidenceChunk]:
    """Tool 3 - hybrid retrieval over manuals, guides and maintenance logs (P2)."""
    return _retrieve(query, top_k=top_k, mode=mode)


TOOL_NAMES = ("classify_signal", "fault_frequency_calc", "retrieve_evidence")
