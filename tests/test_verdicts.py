from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_list_verdicts_requires_auth() -> None:
    r = client.get("/verdicts")
    assert r.status_code == 401


def test_list_verdicts_returns_list() -> None:
    with patch(
        "app.clients.tessera_data.TesseraDataClient.list_verdicts",
        new=AsyncMock(return_value={"verdicts": [], "total": 0}),
    ):
        r = client.get("/verdicts", headers={"X-Internal-Key": "test-secret"})

    assert r.status_code == 200
    body = r.json()
    assert "verdicts" in body
    assert "total" in body


def test_list_verdicts_passes_pagination() -> None:
    mock = AsyncMock(return_value={"verdicts": [], "total": 0})
    with patch(
        "app.clients.tessera_data.TesseraDataClient.list_verdicts",
        new=mock,
    ):
        r = client.get(
            "/verdicts?limit=10&offset=20",
            headers={"X-Internal-Key": "test-secret"},
        )

    assert r.status_code == 200
    mock.assert_called_once_with(limit=10, offset=20)
