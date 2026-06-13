from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_SAMPLE_FEEDBACK = {
    "transaction_id": "txn_feedback_persist_001",
    "corrected_decision": "APPROVE",
    "analyst_note": "Transaction is legitimate — user confirmed purchase.",
    "decisive_signals": ["user_history_flag"],
    "original_verdict": {
        "transaction_id": "txn_feedback_persist_001",
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


def test_feedback_persists_case() -> None:
    save_mock = AsyncMock(
        return_value={"id": "case_new_001", "transaction_id": "txn_feedback_persist_001"}
    )

    with patch(
        "app.clients.tessera_data.TesseraDataClient.save_case",
        new=save_mock,
    ):
        response = client.post(
            "/feedback",
            json=_SAMPLE_FEEDBACK,
            headers={"X-Internal-Key": "test-secret"},
        )

    assert response.status_code == 202
    assert response.json() == {"accepted": True}

    save_mock.assert_called_once()
    call_kwargs = save_mock.call_args

    # save_case is called with (case_data, trace_id=None) — first positional arg is case_data
    case_data = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("case_data")
    assert case_data is not None
    assert case_data["transaction_id"] == "txn_feedback_persist_001"
    assert case_data["decision"] == "APPROVE"
    assert len(case_data["embedding"]) == 1536
