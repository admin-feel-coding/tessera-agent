"""Tests for POST /analyze/stream (mock runner path)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.verdict import Verdict

_SAMPLE_TRANSACTION = {
    "transaction_id": "txn_stream_001",
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
_EMPTY_SIMILAR = {"cases": []}


def _parse_sse_frames(raw: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of {type, data} dicts."""
    frames = []
    current_type: str | None = None
    for line in raw.decode().splitlines():
        if line.startswith("event: "):
            current_type = line[len("event: ") :]
        elif line.startswith("data: ") and current_type is not None:
            frames.append({"type": current_type, "data": json.loads(line[len("data: ") :])})
            current_type = None
    return frames


@pytest.mark.asyncio
async def test_stream_mock_event_sequence() -> None:
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
            return_value=_EMPTY_SIMILAR,
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.save_verdict",
            new_callable=AsyncMock,
            return_value={"id": "v1", "transaction_id": "txn_stream_001"},
        ),
        # Skip the 0.6s sleep so tests run fast
        patch("app.services.analyze.asyncio.sleep", new_callable=AsyncMock),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            async with ac.stream(
                "POST",
                "/analyze/stream",
                json=_SAMPLE_TRANSACTION,
                headers={"X-Internal-Key": "test-secret"},
            ) as response:
                assert response.status_code == 200
                raw = await response.aread()

    frames = _parse_sse_frames(raw)
    types = [f["type"] for f in frames]

    # First event must be start
    assert types[0] == "start"

    # Exactly 5 tool_start and 5 tool_complete
    assert types.count("tool_start") == 5
    assert types.count("tool_complete") == 5

    # Last two events are verdict then done
    assert types[-2] == "verdict"
    assert types[-1] == "done"

    # Verdict payload is schema-valid
    verdict_frame = next(f for f in frames if f["type"] == "verdict")
    verdict = Verdict.model_validate(verdict_frame["data"])
    assert verdict.transaction_id == "txn_stream_001"
    assert len(verdict.cited_sources) >= 1 or verdict.decision == "ESCALATE"

    # Start payload fields
    start_data = frames[0]["data"]
    assert start_data["mode"] == "mock"
    assert start_data["model"] == "mock-runner-v1"

    # tool_start/tool_complete pairs carry name field
    tool_start_names = [f["data"]["name"] for f in frames if f["type"] == "tool_start"]
    assert "get_user_history" in tool_start_names
    assert "check_blacklist" in tool_start_names
    assert "search_similar_cases" in tool_start_names

    tool_complete_frames = [f for f in frames if f["type"] == "tool_complete"]
    for tc in tool_complete_frames:
        assert "duration_ms" in tc["data"]
        assert "summary" in tc["data"]
        assert isinstance(tc["data"]["duration_ms"], int)
