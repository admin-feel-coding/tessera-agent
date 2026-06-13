from enum import StrEnum

from pydantic import BaseModel


class Decision(StrEnum):
    APPROVE = "APPROVE"
    DECLINE = "DECLINE"
    ESCALATE = "ESCALATE"


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
    latency_ms: int
    model: str
    tool_calls: int
    langfuse_trace_id: str = ""
