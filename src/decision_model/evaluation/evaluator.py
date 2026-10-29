"""Evaluation runners for the decision head (System 1) and the chat baseline (System 2).

Models are loaded once by the caller and passed in as callables, so the scoring logic is
testable without MLX and never reloads weights per example.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from decision_model.evaluation.metrics import aggregate
from decision_model.task import QUESTION_ID, ROUTES, build_request, parse_chat_reply
from decision_model.training.artifacts import read_json, utc_now, write_json

log = logging.getLogger(__name__)

# requests -> (answers per request, wall seconds for the batch)
DecideFn = Callable[[list[dict[str, Any]]], tuple[list[dict[str, dict[str, Any]]], float]]
# complaint -> (raw reply, latency seconds)
ChatFn = Callable[[str], tuple[str, float]]
LATEST = "latest.json"


def evaluate_decisions(
    rows: Sequence[dict[str, Any]], decide: DecideFn, batch_size: int
) -> list[dict[str, Any]]:
    """Score the head on every row with all 9 routes offered, ``batch_size`` at a time."""
    results = []
    for start in range(0, len(rows), batch_size):
        part = rows[start : start + batch_size]
        answers, seconds = decide([build_request(r["complaint"]) for r in part])
        for row, answer in zip(part, answers, strict=True):
            a = answer[QUESTION_ID]
            results.append(
                {
                    "id": row["id"],
                    "label": row["product"],
                    "prediction": a["choice"],
                    "confidence": a["confidence"],
                    "probabilities": a["probabilities"],
                    "correct": a["choice"] == row["product"],
                    "latency_ms": round(1000 * seconds / len(part), 1),
                }
            )
        done = sum(r["correct"] for r in results)
        log.info("%d/%d evaluated, %d correct", len(results), len(rows), done)
    return results


def evaluate_chat(rows: Sequence[dict[str, Any]], chat: ChatFn) -> list[dict[str, Any]]:
    """Score the base model's chat replies; a reply without a valid route is wrong."""
    results = []
    for i, row in enumerate(rows, start=1):
        try:
            raw, seconds = chat(row["complaint"])
            error = None
        except Exception as exc:  # one failure must not abort the run
            raw, seconds, error = "", None, f"{type(exc).__name__}: {exc}"
        prediction = parse_chat_reply(raw)
        results.append(
            {
                "id": row["id"],
                "label": row["product"],
                "raw_output": raw,
                "prediction": prediction,
                "confidence": None,
                "correct": prediction == row["product"],
                "generation_error": error,
                "latency_ms": None if seconds is None else round(1000 * seconds, 1),
            }
        )
        if i % 50 == 0 or i == len(rows):
            log.info(
                "%d/%d evaluated, %d correct", i, len(rows), sum(r["correct"] for r in results)
            )
    return results


def build_result(label: str, results: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable evaluation record."""
    return {
        "status": "measured",
        "label": label,
        "evaluated_at": utc_now(),
        **meta,
        "metrics": aggregate(results),
        "examples": results,
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Readable report for one evaluation record."""
    m = result["metrics"]

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}%"

    def na(v: float | None, suffix: str = "") -> str:
        return "n/a" if v is None else f"{v}{suffix}"

    run = (result.get("run") or {}).get("run_id") or "none (base model)"
    lines = [
        f"# Evaluation: {result['label']}",
        "",
        f"- Mode: {result.get('mode')}",
        f"- Model: `{result.get('model', {}).get('repo_id')}` @ "
        f"`{(result.get('model', {}).get('revision') or 'main')[:12]}`",
        f"- Training run: {run}",
        f"- Test file: `{result.get('test_file')}` ({m['examples']} complaints, all 9 routes)",
        f"- Evaluated: {result['evaluated_at']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {m['correct']} / {m['examples']} ({pct(m['accuracy'])}) |",
        f"| Invalid answers | {m['invalid_answers']} |",
        f"| Mean confidence | {na(m['mean_confidence'])} |",
        f"| Expected calibration error | {na(m['expected_calibration_error'])} |",
        f"| Mean latency | {na(m['mean_latency_ms'], ' ms')} |",
        "",
        "| Route | n | Accuracy |",
        "|---|---|---|",
        *(f"| {k} | {v['n']} | {pct(v['accuracy'])} |" for k, v in m["per_route"].items()),
        "",
        "Confusion matrix (rows = correct route, columns = answer):",
        "",
        "| | " + " | ".join([*ROUTES, "invalid"]) + " |",
        "|---" * (len(ROUTES) + 2) + "|",
        *(
            f"| {gold} | " + " | ".join(str(row[c]) for c in [*ROUTES, "invalid"]) + " |"
            for gold, row in m["confusion"].items()
        ),
    ]
    return "\n".join(lines) + "\n"


def save_result(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write ``<label>/<timestamp>.json|.md`` and refresh ``<label>/latest.json``."""
    folder = output_dir / result["label"]
    stamp = result["evaluated_at"].replace(":", "").replace("-", "").replace("+0000", "Z")
    json_path = folder / f"{stamp}.json"
    md_path = folder / f"{stamp}.md"
    write_json(json_path, result)
    md_path.write_text(render_markdown(result), encoding="utf-8")
    write_json(folder / LATEST, result)
    (folder / "latest.md").write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def load_latest(output_dir: Path, label: str) -> dict[str, Any] | None:
    """Most recent evaluation record for ``label``, or None if never evaluated."""
    return read_json(output_dir / label / LATEST)
