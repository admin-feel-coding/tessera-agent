"""CLI entry point for the Tessera eval suite.

Usage:
    uv run python -m evals
    uv run python -m evals --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.runner import run_eval
from evals.scorer import compute_metrics

_RESULTS_DIR = Path(__file__).parent / "results"


def _print_table(metrics: dict, results: list[dict]) -> None:
    total = metrics["total"]
    correct = metrics["correct"]
    match_rate = metrics["match_rate"]
    avg_lat = metrics["avg_latency_ms"]
    p95_lat = metrics["p95_latency_ms"]
    errors = metrics["errors"]
    by_decision = metrics["by_decision"]

    print()
    print("=" * 60)
    print("  Tessera Eval Suite — Results")
    print("=" * 60)
    print(f"  Total cases      : {total}")
    print(f"  Correct          : {correct}")
    print(f"  Match rate       : {match_rate:.1%}")
    print(f"  Avg latency      : {avg_lat:.0f} ms")
    print(f"  P95 latency      : {p95_lat:.0f} ms")
    print(f"  Errors           : {errors}")

    if "pass_at_1" in metrics:
        k = metrics["pass_k"]
        print(f"  pass@1           : {metrics['pass_at_1']:.1%}  (any run correct, k={k})")
        print(f"  pass@k           : {metrics['pass_at_k']:.1%}  (all runs correct, k={k})")

    print()
    print("  By decision:")
    print(f"  {'Decision':<12} {'Expected':>10} {'Correct':>10} {'Accuracy':>10}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")
    for decision, counts in by_decision.items():
        expected_count = counts["expected"]
        correct_count = counts["correct"]
        accuracy = correct_count / expected_count if expected_count > 0 else 0.0
        print(f"  {decision:<12} {expected_count:>10} {correct_count:>10} {accuracy:>9.1%}")
    print()

    failures = [r for r in results if not r["match"]]
    if failures:
        print(f"  Failures ({len(failures)}):")
        print(f"  {'Case ID':<20} {'Expected':<12} {'Actual':<12} {'Error'}")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 30}")
        for r in failures:
            error_str = (r["error"] or "")[:40]
            print(f"  {r['case_id']:<20} {r['expected']:<12} {r['actual']:<12} {error_str}")
        print()

    threshold = 0.80
    status = "PASSED" if match_rate >= threshold else "FAILED"
    print(f"  Gate (>= {threshold:.0%}): {status}")
    print("=" * 60)
    print()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Tessera eval suite")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help="Run each case N times for pass^k measurement (default: 1)",
    )
    args = parser.parse_args()

    print("Running eval suite against golden_dataset.json ...")
    if args.repeats > 1:
        print(f"  repeats={args.repeats} — pass^k mode enabled")

    results = await run_eval(repeats=args.repeats)
    metrics = compute_metrics(results)

    _print_table(metrics, results)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "run_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "results": results,
    }
    latest_path = _RESULTS_DIR / "latest.json"
    latest_path.write_text(json.dumps(output, indent=2))
    print(f"Full results saved to: {latest_path}")

    if metrics["match_rate"] < 0.80:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
