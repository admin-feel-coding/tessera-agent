from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_SAMPLE_TRANSACTION = {
    "transaction_id": "txn_auth_001",
    "user_id": "user_001",
    "amount": 10.00,
}


def test_analyze_without_key_returns_401() -> None:
    response = client.post("/analyze", json=_SAMPLE_TRANSACTION)
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"


def test_analyze_with_wrong_key_returns_401() -> None:
    response = client.post(
        "/analyze",
        json=_SAMPLE_TRANSACTION,
        headers={"X-Internal-Key": "wrong-key"},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
