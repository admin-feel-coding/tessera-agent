"""Scorer — computes aggregate metrics from eval run results."""

from __future__ import annotations

_DECISIONS = ("APPROVE", "DECLINE", "ESCALATE")


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from eval run results.

    Returns:
        {
            "total": int,
            "correct": int,
            "match_rate": float,       # 0.0–1.0
            "avg_latency_ms": float,
            "p95_latency_ms": float,
            "errors": int,
            "by_decision": {
                "APPROVE":  {"expected": int, "correct": int, "precision": float, "recall": float},
                "DECLINE":  {"expected": int, "correct": int, "precision": float, "recall": float},
                "ESCALATE": {"expected": int, "correct": int, "precision": float, "recall": float},
            },
            "confusion_matrix": {expected: {actual: count}},
        }
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "match_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "errors": 0,
            "by_decision": {
                d: {"expected": 0, "correct": 0, "precision": 0.0, "recall": 0.0}
                for d in _DECISIONS
            },
            "confusion_matrix": {},
        }

    correct = sum(1 for r in results if r["match"])
    errors = sum(1 for r in results if r["error"] is not None)

    latencies = sorted(r["latency_ms"] for r in results)
    avg_latency_ms = sum(latencies) / total
    p95_index = max(0, int(total * 0.95) - 1)
    p95_latency_ms = float(latencies[p95_index])

    by_decision: dict[str, dict] = {d: {"expected": 0, "correct": 0} for d in _DECISIONS}

    # predicted_d = number of results where actual == d
    predicted: dict[str, int] = {d: 0 for d in _DECISIONS}

    confusion: dict[str, dict[str, int]] = {}

    for result in results:
        expected = result["expected"]
        actual = result["actual"]

        # by_decision counts (keyed on expected label)
        if expected in by_decision:
            by_decision[expected]["expected"] += 1
            if result["match"]:
                by_decision[expected]["correct"] += 1

        # predicted counts (keyed on actual label)
        if actual in predicted:
            predicted[actual] += 1

        # confusion matrix
        confusion.setdefault(expected, {})
        confusion[expected][actual] = confusion[expected].get(actual, 0) + 1

    # attach precision / recall to each decision bucket
    for d in _DECISIONS:
        correct_d = by_decision[d]["correct"]
        expected_d = by_decision[d]["expected"]
        predicted_d = predicted[d]

        by_decision[d]["precision"] = correct_d / predicted_d if predicted_d else 0.0
        by_decision[d]["recall"] = correct_d / expected_d if expected_d else 0.0

    metrics: dict = {
        "total": total,
        "correct": correct,
        "match_rate": correct / total,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "errors": errors,
        "by_decision": by_decision,
        "confusion_matrix": confusion,
    }

    # Pass^k computation (when runs data is available)
    cases_with_runs = [r for r in results if "runs" in r]
    if cases_with_runs:
        k = len(cases_with_runs[0]["runs"])
        n = len(cases_with_runs)
        pass_at_1 = sum(any(run["match"] for run in r["runs"]) for r in cases_with_runs) / n
        pass_at_k = sum(all(run["match"] for run in r["runs"]) for r in cases_with_runs) / n
        metrics["pass_at_1"] = round(pass_at_1, 4)
        metrics["pass_at_k"] = round(pass_at_k, 4)
        metrics["pass_k"] = k  # so the consumer knows what k was

    return metrics
