"""Fixed JSON schemas for the local VeriRay REST API (Blueprint 26 Nov §3, §5 module 8)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AllowedVerdict = Literal["MODEL-SUPPORTED", "NOT SUPPORTED", "SHORTCUT RISK", "INDETERMINATE"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    offline_only: Literal[True] = True
    reader_loaded: bool
    checkpoint_id: str | None = None
    reason_code: str | None = None


class VerifiedSentenceSchema(BaseModel):
    text: str
    evidence_ids: list[str]
    verified: bool


class VerifyResponse(BaseModel):
    sample_name: str
    finding: Literal["pneumothorax"] = "pneumothorax"
    verdict: AllowedVerdict
    reason_codes: list[str]
    user_facing_text: str
    disclaimer: str
    model_score_raw: float | None = None
    checkpoint_id: str | None = None
    preprocessing_id: str | None = None
    gate_status: str | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    gate_results: dict[str, bool] = Field(default_factory=dict)
    evidence_items: dict[str, Any] = Field(default_factory=dict)
    verified_explanation: list[VerifiedSentenceSchema] = Field(default_factory=list)
    run_id: str | None = None
    latencies_ms: dict[str, float] = Field(default_factory=dict)


class BatchVerifyResponse(BaseModel):
    total_cases: int
    verdict_counts: dict[str, int]
    results: list[VerifyResponse]
    total_elapsed_ms: float


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    verdict: AllowedVerdict | None = None
    reason_codes: list[str] = Field(default_factory=list)
    evidence_items: dict[str, Any] = Field(default_factory=dict)


class ChatResponseSchema(BaseModel):
    allowed: bool
    answer: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    refusal_reason_codes: list[str] = Field(default_factory=list)
