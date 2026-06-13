import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.transaction import Transaction
from app.schemas.verdict import Decision
from app.services.analyze import analyze

_SAMPLE_TXN = Transaction(
    transaction_id="txn_timeout",
    user_id="u1",
    amount=100,
    currency="USD",
    merchant_category="test",
    email="x@x.com",
)

_PERSIST_TXN = Transaction(
    transaction_id="txn_persist_fail",
    user_id="u1",
    amount=100,
    currency="USD",
    merchant_category="test",
    email="x@x.com",
)

_SAFE_HISTORY = {
    "transaction_count": 2,
    "avg_amount": 50.0,
    "countries": ["US"],
    "last_txn_at": None,
    "high_velocity": False,
}
_SAFE_IP = {"risk_score": 0.1, "is_vpn": False, "country": "US"}
_SAFE_DEVICE = {"suspicious": False, "user_count": 1, "first_seen": None}
_SAFE_BLACKLIST = {"match": False, "kind": None, "reason": None}
_EMPTY_SIMILAR = {"cases": []}


@pytest.mark.asyncio
async def test_analyze_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """When analysis times out, analyze() returns an ESCALATE verdict."""

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(999)

    monkeypatch.setattr("app.services.analyze._run_mock", slow_run)
    monkeypatch.setattr("app.config.settings.analyze_timeout_seconds", 0.01)

    with patch(
        "app.services.analyze._persist_verdict",
        new=AsyncMock(),
    ):
        verdict = await analyze(_SAMPLE_TXN)

    assert verdict.decision == Decision.ESCALATE
    assert verdict.escalation_reason is not None
    assert "timed out" in verdict.escalation_reason.lower()


@pytest.mark.asyncio
async def test_verdict_persistence_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """If save_verdict fails, analyze() still returns a valid verdict."""
    with (
        patch(
            "app.clients.tessera_data.TesseraDataClient.get_user_history",
            new=AsyncMock(return_value=_SAFE_HISTORY),
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.get_ip_risk",
            new=AsyncMock(return_value=_SAFE_IP),
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.get_device_fingerprint",
            new=AsyncMock(return_value=_SAFE_DEVICE),
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.check_blacklist",
            new=AsyncMock(return_value=_SAFE_BLACKLIST),
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.search_similar_cases",
            new=AsyncMock(return_value=_EMPTY_SIMILAR),
        ),
        patch(
            "app.clients.tessera_data.TesseraDataClient.save_verdict",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ),
    ):
        verdict = await analyze(_PERSIST_TXN)

    assert verdict.decision in (Decision.APPROVE, Decision.DECLINE, Decision.ESCALATE)
