from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from conftest import fake_splits
from decision_model.cli import validate_prepared
from decision_model.data.prepare import (
    prepare_dataset,
    prepare_smoke_subset,
    sha256_file,
    training_stream,
)
from decision_model.data.validate import (
    overlap_report,
    read_jsonl,
    request_line_problems,
    row_problems,
    validate_jsonl,
)
from decision_model.task import QUESTION_ID, ROUTES, build_request


def test_training_stream_follows_its_definition():
    train = fake_splits()["train"]
    rng = random.Random(7)

    def subset(label: str) -> dict:
        if rng.random() < 0.5:
            others = [k for k in ROUTES if k != label]
            keep = {label, *rng.sample(others, rng.randint(1, len(others)))}
            return {k: v for k, v in ROUTES.items() if k in keep}
        return dict(ROUTES)

    subset(train[0]["product"])  # the discarded draw before sampling
    rows = rng.sample(train, 20)
    expected = [build_request(r["complaint"], subset(r["product"])) for r in rows]
    stream = training_stream(train, 7, 20)
    assert [s["request"] for s in stream] == expected
    assert [s["id"] for s in stream] == [r["id"] for r in rows]
    assert stream != training_stream(train, 8, 20)
    assert all(request_line_problems(s) == [] for s in stream)


def test_prepare_writes_files_and_manifest(project):
    manifest = prepare_dataset(fake_splits(), project.data, project.root)
    out = project.data.output_dir
    assert sorted(p.name for p in out.iterdir()) == [
        "manifest.json",
        "test.jsonl",
        "train.jsonl",
        "train_requests.jsonl",
        "validation.jsonl",
    ]
    assert read_jsonl(out / "test.jsonl") == fake_splits()["test"]
    assert len(read_jsonl(out / "train_requests.jsonl")) == 16
    assert manifest["files"]["train_requests"]["path"] == "data/processed/train_requests.jsonl"
    assert manifest["overlap"]["train/test"] == {"ids": 0, "texts": 0}
    assert validate_prepared(project, smoke=False) == []


def test_prepare_rejects_wrong_counts_and_bad_rows(project):
    splits = fake_splits()
    splits["test"] = splits["test"][:-1]
    with pytest.raises(ValueError, match="expected 18"):
        prepare_dataset(splits, project.data, project.root)
    splits = fake_splits()
    splits["train"][3]["product"] = "insurance"
    with pytest.raises(ValueError, match="invalid rows"):
        prepare_dataset(splits, project.data, project.root)


def test_dry_run_writes_nothing(project):
    manifest = prepare_dataset(fake_splits(), project.data, project.root, dry_run=True)
    assert manifest["files"] == {} and not project.data.output_dir.exists()


def test_smoke_subset_is_a_head_and_leaves_canonical_untouched(prepared_project):
    d = prepared_project.data
    canonical = [d.output_dir / "train_requests.jsonl", d.output_dir / "test.jsonl"]
    before = [sha256_file(p) for p in canonical]
    prepare_smoke_subset(d, prepared_project.root)
    assert [sha256_file(p) for p in canonical] == before
    assert read_jsonl(d.smoke_dir / "train_requests.jsonl") == read_jsonl(canonical[0])[:2]
    assert read_jsonl(d.smoke_dir / "test.jsonl") == read_jsonl(canonical[1])[:4]
    assert validate_prepared(prepared_project, smoke=True) == []


def test_validation_detects_leakage(prepared_project):
    out = prepared_project.data.output_dir
    test_row = read_jsonl(out / "test.jsonl")[0]
    with (out / "train.jsonl").open("a") as fh:
        fh.write(json.dumps(test_row) + "\n")
    problems = validate_prepared(prepared_project, smoke=False)
    assert any("overlap" in p for p in problems)


def test_row_and_request_validation():
    good = {"id": "a", "complaint": "text", "product": "mortgage"}
    assert row_problems(good) == []
    assert row_problems({**good, "product": "insurance"})[0].startswith("'product'")
    assert row_problems({**good, "complaint": " "}) == ["'complaint' must be a non-empty string"]
    line = {"id": "a", "product": "mortgage", "request": build_request("text")}
    assert request_line_problems(line) == []
    no_label = {
        **line,
        "request": build_request(
            "t", {"credit_card": ROUTES["credit_card"], "bank_account": ROUTES["bank_account"]}
        ),
    }
    assert request_line_problems(no_label) == [
        "the correct route is missing from the offered criteria"
    ]
    edited = build_request("t")
    edited["questions"][QUESTION_ID]["criteria"]["mortgage"] = "houses"
    assert "subset of the 9 routes" in request_line_problems({**line, "request": edited})[0]
    assert request_line_problems({**line, "request": {}})[0].startswith("'request' must hold")


def test_validate_jsonl_and_overlap(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"id": "a", "complaint": "t", "product": "mortgage"}\n{oops\n\n')
    report = validate_jsonl(path, row_problems)
    assert report.lines == 3 and report.error_count == 2 and not report.ok
    rows = [{"id": "1", "complaint": "same"}]
    assert overlap_report({"a": rows, "b": rows}) == {"a/b": {"ids": 1, "texts": 1}}
