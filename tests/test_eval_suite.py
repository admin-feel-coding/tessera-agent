import httpx
import pytest


def _tessera_data_reachable() -> bool:
    try:
        r = httpx.get("http://localhost:8002/health", timeout=1.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not _tessera_data_reachable(),
    reason="tessera-data not running — start: cd tessera-data && go run ./cmd/server",
)
@pytest.mark.asyncio
async def test_eval_match_rate() -> None:
    from evals.runner import run_eval
    from evals.scorer import compute_metrics

    results = await run_eval()
    metrics = compute_metrics(results)
    assert metrics["match_rate"] >= 0.80, f"Match rate {metrics['match_rate']:.1%} < 80%"
