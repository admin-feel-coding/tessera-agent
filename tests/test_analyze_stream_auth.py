"""Auth guard tests for POST /analyze/stream."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

_SAMPLE_TRANSACTION = {
    "transaction_id": "txn_auth_stream_001",
    "user_id": "user_001",
    "amount": 10.00,
}


@pytest.mark.asyncio
async def test_stream_without_key_returns_401() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/analyze/stream", json=_SAMPLE_TRANSACTION)
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_stream_with_wrong_key_returns_401() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze/stream",
            json=_SAMPLE_TRANSACTION,
            headers={"X-Internal-Key": "wrong-key"},
        )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
