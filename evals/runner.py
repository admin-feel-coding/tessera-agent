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


async def run_eval(
    dataset_path: Path | None = None,
    repeats: int = 1,
) -> list[dict]:
    """Run eval against all cases in golden_dataset.json.

    When repeats == 1, each result has keys:
        case_id, expected, actual, match, latency_ms, error

    When repeats > 1, each result also has:
        runs: list of {actual, match, latency_ms}
        actual: majority-or-first-run value (first matching run, else first run)
        match: True if ANY run matched
        latency_ms: average across all runs
    """
    path = dataset_path or _DATASET_PATH
    cases: list[dict] = json.loads(path.read_text())

    results: list[dict] = []

    for case in cases:
        case_id: str = case["id"]
        expected: str = case["expected_decision"]
        txn_data: dict = case["transaction"]

        if repeats == 1:
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
        else:
            runs: list[dict] = []
            errors: list[str | None] = []

            for _ in range(repeats):
                start = time.monotonic()
                run_error: str | None = None
                run_actual: str = "ESCALATE"
                run_latency: int = 0

                try:
                    transaction = Transaction(**txn_data)
                    verdict = await analyze(transaction)
                    run_actual = str(verdict.decision)
                    run_latency = verdict.latency_ms
                except Exception as exc:  # noqa: BLE001
                    run_latency = int((time.monotonic() - start) * 1000)
                    run_error = str(exc)

                run_match = run_actual == expected
                runs.append(
                    {
                        "actual": run_actual,
                        "match": run_match,
                        "latency_ms": run_latency,
                    }
                )
                errors.append(run_error)

            any_match = any(r["match"] for r in runs)
            # actual: first matching run's value for backwards compat; fall back to first run
            actual = next((r["actual"] for r in runs if r["match"]), runs[0]["actual"])
            avg_latency = int(sum(r["latency_ms"] for r in runs) / len(runs))
            # surface first non-None error if all errored, else None
            first_error = next((e for e in errors if e is not None), None)

            results.append(
                {
                    "case_id": case_id,
                    "expected": expected,
                    "actual": actual,
                    "match": any_match,
                    "latency_ms": avg_latency,
                    "error": first_error,
                    "runs": runs,
                }
            )

    return results


if __name__ == "__main__":
    asyncio.run(run_eval())
