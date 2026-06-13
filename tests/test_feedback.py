from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_SAMPLE_FEEDBACK = {
    "transaction_id": "txn_test_001",
    "corrected_decision": "APPROVE",
    "analyst_note": "Transaction is legitimate — user confirmed purchase.",
    "decisive_signals": ["user_history_flag"],
    "original_verdict": {
        "transaction_id": "txn_test_001",
        "decision": "ESCALATE",
        "risk_score": 0.65,
        "reasoning": "Multiple risk signals present.",
        "cited_sources": [
            {
                "type": "rule",
                "id": "MULTI_SIGNAL_RISK",
                "excerpt": "Multiple risk signals present; escalating for human review.",
            }
        ],
        "signals": {
            "user_history_flag": True,
            "ip_risk_flag": True,
            "device_fingerprint_flag": False,
            "blacklist_hit": False,
            "velocity_flag": False,
        },
        "escalation_reason": "Multiple risk signals detected; human review required.",
        "latency_ms": 120,
        "model": "mock-runner-v1",
        "tool_calls": 4,
        "langfuse_trace_id": "",
    },
}


def test_feedback_returns_202() -> None:
    response = client.post(
        "/feedback",
        json=_SAMPLE_FEEDBACK,
        headers={"X-Internal-Key": "test-secret"},
    )
    assert response.status_code == 202
    assert response.json() == {"accepted": True}
