"""Scorer — computes aggregate metrics from eval run results."""

from __future__ import annotations


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
                "APPROVE":  {"expected": int, "correct": int},
                "DECLINE":  {"expected": int, "correct": int},
                "ESCALATE": {"expected": int, "correct": int},
            }
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
                "APPROVE": {"expected": 0, "correct": 0},
                "DECLINE": {"expected": 0, "correct": 0},
                "ESCALATE": {"expected": 0, "correct": 0},
            },
        }

    correct = sum(1 for r in results if r["match"])
    errors = sum(1 for r in results if r["error"] is not None)

    latencies = sorted(r["latency_ms"] for r in results)
    avg_latency_ms = sum(latencies) / total
    p95_index = max(0, int(total * 0.95) - 1)
    p95_latency_ms = float(latencies[p95_index])

    by_decision: dict[str, dict[str, int]] = {
        "APPROVE": {"expected": 0, "correct": 0},
        "DECLINE": {"expected": 0, "correct": 0},
        "ESCALATE": {"expected": 0, "correct": 0},
    }

    for result in results:
        expected = result["expected"]
        if expected in by_decision:
            by_decision[expected]["expected"] += 1
            if result["match"]:
                by_decision[expected]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "match_rate": correct / total,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "errors": errors,
        "by_decision": by_decision,
    }
