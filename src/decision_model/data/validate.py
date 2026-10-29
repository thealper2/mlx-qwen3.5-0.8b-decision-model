"""Row-level and file-level validation of the prepared data."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decision_model.task import QUESTION_ID, ROUTES

MAX_REPORTED_ERRORS = 20


def row_problems(row: Any) -> list[str]:
    """Problems with one ``{id, complaint, product}`` row; empty when usable."""
    if not isinstance(row, dict):
        return [f"row is {type(row).__name__}, expected an object"]
    problems = []
    for key in ("id", "complaint"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            problems.append(f"'{key}' must be a non-empty string")
    if row.get("product") not in ROUTES:
        problems.append(f"'product' {row.get('product')!r} is not one of the 9 routes")
    return problems


def request_line_problems(obj: Any) -> list[str]:
    """Problems with one training-stream line ``{id, product, request}``."""
    problems = row_problems({"complaint": "x", **obj} if isinstance(obj, dict) else obj)
    if problems:
        return problems
    request = obj.get("request")
    try:
        question = request["questions"][QUESTION_ID]
        criteria = question["criteria"]
        complaint = request["state"]["complaint"]
    except (TypeError, KeyError):
        return ["'request' must hold state.complaint and questions.product.criteria"]
    if question.get("type") != "choice":
        problems.append("question type must be 'choice'")
    if not isinstance(complaint, str) or not complaint.strip():
        problems.append("request state.complaint must be a non-empty string")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        problems.append("criteria must offer at least two routes")
    elif obj["product"] not in criteria:
        problems.append("the correct route is missing from the offered criteria")
    elif any(k not in ROUTES or criteria[k] != ROUTES[k] for k in criteria):
        problems.append("criteria must be a subset of the 9 routes with their descriptions")
    return problems


@dataclass
class FileReport:
    path: Path
    lines: int = 0
    errors: list[str] = field(default_factory=list)
    error_count: int = 0

    @property
    def ok(self) -> bool:
        return self.error_count == 0 and self.lines > 0

    def add(self, line_no: int, problem: str) -> None:
        self.error_count += 1
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(f"line {line_no}: {problem}")


def validate_jsonl(path: Path, check: Callable[[Any], list[str]]) -> FileReport:
    """Validate every line of a JSONL file with ``check``."""
    report = FileReport(path)
    if not path.is_file():
        report.add(0, "file does not exist")
        return report
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            report.lines += 1
            if not line.strip():
                report.add(line_no, "blank line")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.add(line_no, f"invalid JSON ({exc.msg})")
                continue
            for problem in check(obj):
                report.add(line_no, problem)
    if report.lines == 0:
        report.add(0, "file is empty")
    return report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def overlap_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    """Shared ids and identical complaint texts between each pair of splits."""
    names = list(splits)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = {r["id"] for r in splits[a]} & {r["id"] for r in splits[b]}
            texts = {r["complaint"] for r in splits[a]} & {r["complaint"] for r in splits[b]}
            out[f"{a}/{b}"] = {"ids": len(ids), "texts": len(texts)}
    return out
