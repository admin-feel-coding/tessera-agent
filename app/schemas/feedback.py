from pydantic import BaseModel

from app.schemas.verdict import Decision, Verdict


class FeedbackPayload(BaseModel):
    transaction_id: str
    corrected_decision: Decision
    analyst_note: str
    decisive_signals: list[str]
    original_verdict: Verdict
