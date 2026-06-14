from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


class Decision(StrEnum):
    APPROVE = "APPROVE"
    DECLINE = "DECLINE"
    ESCALATE = "ESCALATE"


class EscalationCategory(StrEnum):
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    INSUFFICIENT_GROUNDING = "INSUFFICIENT_GROUNDING"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NOVEL_PATTERN = "NOVEL_PATTERN"
    POLICY_REQUIRED = "POLICY_REQUIRED"


class SourceType(StrEnum):
    RULE = "rule"
    CASE = "case"


class CitedSource(BaseModel):
    type: SourceType
    id: str
    excerpt: str


class Signals(BaseModel):
    user_history_flag: bool
    ip_risk_flag: bool
    device_fingerprint_flag: bool
    blacklist_hit: bool
    velocity_flag: bool


class Verdict(BaseModel):
    transaction_id: str
    decision: Decision
    risk_score: float  # 0.0 – 1.0
    reasoning: str
    cited_sources: list[CitedSource]
    signals: Signals
    escalation_reason: str | None = None
    escalation_category: EscalationCategory | None = None
    latency_ms: int
    model: str
    tool_calls: int
    langfuse_trace_id: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    # Denormalized from Transaction for the persistence layer (velocity tracking).
    user_id: str | None = None
    amount: float | None = None
    currency: str | None = None
    ip_address: str | None = None
    card_bin: str | None = None
    device_id: str | None = None


class SSEEvent(BaseModel):
    type: Literal["start", "tool_start", "tool_complete", "verdict", "error", "done"]
    data: dict[str, Any]
