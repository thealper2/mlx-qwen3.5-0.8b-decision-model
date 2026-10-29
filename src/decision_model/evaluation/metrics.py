"""Classification metrics for routing results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from decision_model.task import ROUTES


def _rate(n: int | float, d: int) -> float | None:
    return round(n / d, 4) if d else None


def expected_calibration_error(results: Sequence[dict[str, Any]], bins: int = 10) -> float | None:
    """ECE over results that carry a ``confidence``: mean |accuracy − confidence| per bin."""
    scored = [r for r in results if r.get("confidence") is not None]
    if not scored:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [
            r for r in scored if lo < r["confidence"] <= hi or (b == 0 and r["confidence"] == 0)
        ]
        if in_bin:
            acc = sum(r["correct"] for r in in_bin) / len(in_bin)
            conf = sum(r["confidence"] for r in in_bin) / len(in_bin)
            total += len(in_bin) / len(scored) * abs(acc - conf)
    return round(total, 4)


def aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy, per-route recall, confusion matrix, invalid answers, calibration, latency."""
    n = len(results)
    correct = sum(r["correct"] for r in results)
    labels = list(ROUTES)
    per_route = {}
    confusion = {gold: {**dict.fromkeys(labels, 0), "invalid": 0} for gold in labels}
    for r in results:
        confusion[r["label"]][r["prediction"] if r["prediction"] in ROUTES else "invalid"] += 1
    for gold in labels:
        rows = [r for r in results if r["label"] == gold]
        per_route[gold] = {
            "n": len(rows),
            "accuracy": _rate(sum(r["correct"] for r in rows), len(rows)),
        }
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    confidences = [r["confidence"] for r in results if r.get("confidence") is not None]
    return {
        "examples": n,
        "correct": correct,
        "accuracy": _rate(correct, n),
        "invalid_answers": sum(r["prediction"] not in ROUTES for r in results),
        "per_route": per_route,
        "confusion": confusion,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "expected_calibration_error": expected_calibration_error(results),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }
