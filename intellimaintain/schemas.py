"""Shared data contracts between the ML, RAG, agent and app layers.

All team members code against these classes, so P1/P2/P3/P4 can work in
parallel and plug together without surprises.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

FaultType = Literal["Normal", "Inner race", "Outer race", "Ball"]


class ClassifyResult(BaseModel):
    """Output of the ML layer (P1) for one uploaded signal."""

    label: str = Field(..., description="Fine class, e.g. 'IR_014' or 'Normal'")
    fault_type: FaultType
    fault_size_in: Optional[float] = Field(None, description="Defect diameter in inches (0.007/0.014/0.021)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Mean calibrated probability of the predicted class")
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    window_votes: dict[str, int] = Field(default_factory=dict, description="How many windows voted for each class")
    n_windows: int = 0
    anomaly_score: float = Field(0.0, description="0 = looks healthy, 1 = strongly anomalous (Isolation Forest)")
    low_confidence: bool = False
    key_features: dict[str, float] = Field(default_factory=dict, description="RMS, kurtosis, crest factor ... of the signal")
    model_name: str = "RandomForest"
    similar_cases: list[dict[str, Any]] = Field(
        default_factory=list, description="Most similar recordings in the CWRU training dataset (case-based evidence)")
    cwru_reference: dict[str, Any] = Field(
        default_factory=dict, description="Range of RMS / kurtosis / crest factor of the predicted class in CWRU")


class EvidenceChunk(BaseModel):
    """One retrieved passage from the knowledge base (P2)."""

    chunk_id: str
    source: str = Field(..., description="File name of the source document")
    page: int = Field(..., ge=1, description="1-based page number (section number for .md files)")
    text: str
    score: float = 0.0
    retriever: str = Field("hybrid", description="dense | bm25 | hybrid | rerank | web")
    title: str = ""
    url: str = Field("", description="Link to the online source (web evidence only)")


class FaultFrequencies(BaseModel):
    """Bearing characteristic frequencies (output of fault_frequency_calc)."""

    rpm: float
    shaft_hz: float
    bpfo_hz: float
    bpfi_hz: float
    bsf_hz: float
    ftf_hz: float
    bearing: str = ""
    envelope_match: dict[str, float] = Field(
        default_factory=dict,
        description="Relative envelope-spectrum energy at each defect frequency (physics cross-check)",
    )
    physics_suggests: Optional[str] = Field(None, description="Fault type the physics check points to, if any")
    consistency: Literal["consistent", "conflict", "inconclusive", "not_run"] = "not_run"
    consistency_note: str = ""


class ToolCall(BaseModel):
    step: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    ok: bool = True


class Diagnosis(BaseModel):
    """Final structured answer produced by the agent (P3) and shown in the app (P4)."""

    question: str
    fault_type: str = Field(..., description="Final fault type or 'Unknown'")
    confidence: float = 0.0
    summary: str = ""
    answer: str = Field("", description="Direct, conversational reply to the question that was asked (cited)")
    question_focus: str = Field("diagnosis", description="What the question asks about (see nlu.FOCUS_PATTERNS)")
    explanation: str = Field("", description="Cited explanation; every claim ends with [n]")
    fault_mechanism: str = Field("", description="Cited, step-by-step explanation of how the fault develops and "
                                                 "why it produces the observed vibration")
    recommended_actions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    ml_result: Optional[ClassifyResult] = None
    fault_frequencies: Optional[FaultFrequencies] = None
    insufficient_evidence: bool = False
    warnings: list[str] = Field(default_factory=list)
    citation_coverage: float = Field(0.0, description="Share of explanation sentences carrying a valid citation")
    grounding_score: float = Field(1.0, description="Share of LLM sentences verified against their cited source")
    answer_source: str = Field("offline", description="offline | gemini (verified) | gemini -> offline fallback")
    tool_trace: list[ToolCall] = Field(default_factory=list)
    llm_provider: str = "offline"
    llm_model: str = ""
    config_name: str = Field("full", description="Ablation configuration that produced this diagnosis")
    web_search: str = Field("off", description="Online evidence setting: off | auto | always")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    review_status: Literal["pending", "accepted", "rejected", "edited"] = "pending"
