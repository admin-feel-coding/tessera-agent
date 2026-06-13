"""Eval runner — loads the golden dataset and calls analyze() for each case.

Run with: uv run python -m evals
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from app.schemas.transaction import Transaction
from app.services.analyze import analyze

_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


async def run_eval(dataset_path: Path | None = None) -> list[dict]:
    """Run eval against all cases in golden_dataset.json.

    Returns a list of result dicts with keys:
        case_id, expected, actual, match, latency_ms, error
    """
    path = dataset_path or _DATASET_PATH
    cases: list[dict] = json.loads(path.read_text())

    results: list[dict] = []

    for case in cases:
        case_id: str = case["id"]
        expected: str = case["expected_decision"]
        txn_data: dict = case["transaction"]

        start = time.monotonic()
        error: str | None = None
        actual: str = "ESCALATE"

        try:
            transaction = Transaction(**txn_data)
            verdict = await analyze(transaction)
            actual = str(verdict.decision)
            latency_ms = verdict.latency_ms
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.monotonic() - start) * 1000)
            error = str(exc)

        results.append(
            {
                "case_id": case_id,
                "expected": expected,
                "actual": actual,
                "match": actual == expected,
                "latency_ms": latency_ms,
                "error": error,
            }
        )

    return results


if __name__ == "__main__":
    asyncio.run(run_eval())
