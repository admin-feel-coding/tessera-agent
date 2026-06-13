from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.verdict import Decision, Signals, Verdict

client = TestClient(app)

_SAMPLE_TRANSACTION = {
    "transaction_id": "txn_grounding_001",
    "user_id": "user_001",
    "amount": 20.00,
    "currency": "USD",
    "merchant_category": "food",
    "ip_address": "1.2.3.4",
    "device_id": "device_abc",
    "card_bin": "411111",
    "email": "test@example.com",
}

_APPROVE_NO_SOURCES = Verdict(
    transaction_id="txn_grounding_001",
    decision=Decision.APPROVE,
    risk_score=0.10,
    reasoning="No signals detected.",
    cited_sources=[],
    signals=Signals(
        user_history_flag=False,
        ip_risk_flag=False,
        device_fingerprint_flag=False,
        blacklist_hit=False,
        velocity_flag=False,
    ),
    escalation_reason=None,
    latency_ms=5,
    model="mock-runner-v1",
    tool_calls=4,
)


def test_grounding_override_forces_escalate() -> None:
    # Patch _run_mock so the runner returns APPROVE with no cited_sources.
    # The top-level analyze() still runs _apply_grounding_override — that is what we are testing.
    with patch(
        "app.services.analyze._run_mock",
        new=AsyncMock(return_value=_APPROVE_NO_SOURCES),
    ):
        response = client.post(
            "/analyze",
            json=_SAMPLE_TRANSACTION,
            headers={"X-Internal-Key": "test-secret"},
        )

    assert response.status_code == 200
    verdict = Verdict.model_validate(response.json())
    assert verdict.decision == Decision.ESCALATE
