from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.verdict import Decision, SourceType, Verdict

client = TestClient(app)

_SAMPLE_TRANSACTION = {
    "transaction_id": "txn_similar_001",
    "user_id": "user_001",
    "amount": 45.00,
    "currency": "USD",
    "merchant_category": "electronics",
    "ip_address": "1.2.3.4",
    "device_id": "device_abc",
    "card_bin": "411111",
    "email": "test@example.com",
}

_SAFE_HISTORY = {
    "transaction_count": 5,
    "avg_amount": 40.0,
    "countries": ["US"],
    "last_txn_at": None,
    "high_velocity": False,
}
_SAFE_IP = {"risk_score": 0.1, "is_vpn": False, "country": "US"}
_SAFE_DEVICE = {"suspicious": False, "user_count": 1, "first_seen": None}
_SAFE_BLACKLIST = {"match": False, "kind": None, "reason": None}
_SIMILAR_CASES = {
    "cases": [
        {
            "id": "case_099",
            "reasoning": "Same device fingerprint and merchant category, analyst approved.",
            "similarity": 0.91,
        }
    ]
}


def test_similar_case_appears_in_cited_sources() -> None:
    with (
        patch(
            "app.clients.tessera_data.TesseraDataClient.get_user_history",
            new_callable=AsyncMock,
            return_value=_SAFE_HISTORY,
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.get_ip_risk",
            new_callable=AsyncMock,
            return_value=_SAFE_IP,
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.get_device_fingerprint",
            new_callable=AsyncMock,
            return_value=_SAFE_DEVICE,
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.check_blacklist",
            new_callable=AsyncMock,
            return_value=_SAFE_BLACKLIST,
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.search_similar_cases",
            new_callable=AsyncMock,
            return_value=_SIMILAR_CASES,
        ),
    ):
        response = client.post(
            "/analyze",
            json=_SAMPLE_TRANSACTION,
            headers={"X-Internal-Key": "test-secret"},
        )

    assert response.status_code == 200
    verdict = Verdict.model_validate(response.json())

    case_sources = [s for s in verdict.cited_sources if s.type == SourceType.CASE]
    assert len(case_sources) >= 1, "Expected at least one case cited source"
    assert case_sources[0].id == "case_099"
    assert verdict.decision != Decision.ESCALATE or len(verdict.cited_sources) >= 1
