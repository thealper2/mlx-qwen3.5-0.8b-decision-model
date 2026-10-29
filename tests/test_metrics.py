from __future__ import annotations

import pytest

from conftest import fake_splits
from decision_model.evaluation.evaluator import (
    build_result,
    evaluate_chat,
    evaluate_decisions,
    load_latest,
    render_markdown,
    save_result,
)
from decision_model.evaluation.metrics import aggregate, expected_calibration_error
from decision_model.task import QUESTION_ID, ROUTES

ROWS = fake_splits()["test"][:6]


def _answer(choice: str, confidence: float) -> dict:
    return {
        QUESTION_ID: {
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": {k: 0.0 for k in ROUTES},
        }
    }


def test_decision_evaluation_batches_and_scores():
    calls = []

    def decide(requests):
        calls.append(len(requests))
        for req in requests:  # every request offers all 9 routes
            assert req["questions"][QUESTION_ID]["criteria"] == ROUTES
        answers = []
        for req in requests:
            row = next(r for r in ROWS if r["complaint"] == req["state"]["complaint"])
            wrong = "mortgage" if row["product"] != "mortgage" else "credit_card"
            answers.append(_answer(row["product"] if row["id"] != "test-1" else wrong, 0.8))
        return answers, 0.4

    results = evaluate_decisions(ROWS, decide, batch_size=4)
    assert calls == [4, 2]
    m = aggregate(results)
    assert m["correct"] == 5 and m["accuracy"] == pytest.approx(0.8333)
    assert m["invalid_answers"] == 0
    assert results[0]["latency_ms"] == 100.0  # 0.4 s for 4 requests
    assert m["confusion"]["debt_collection"]["mortgage"] == 1  # test-1 is debt_collection


def test_chat_evaluation_counts_invalid_and_failures():
    replies = iter(
        [
            '{"product": "credit_reporting"}',
            "I think it is about a loan.",
            '{"product": "not_a_route"}',
        ]
    )

    def chat(complaint):
        if "complaint 3" in complaint:
            raise RuntimeError("boom")
        return next(replies, '{"product": "mortgage"}'), 0.25

    results = evaluate_chat(ROWS[:5], chat)
    m = aggregate(results)
    assert [r["prediction"] for r in results] == ["credit_reporting", None, None, None, "mortgage"]
    assert m["correct"] == 2 and m["invalid_answers"] == 3
    assert results[3]["generation_error"] == "RuntimeError: boom"
    assert m["mean_confidence"] is None and m["expected_calibration_error"] is None
    assert m["confusion"]["debt_collection"]["invalid"] == 1


def test_expected_calibration_error():
    perfect = [{"confidence": 1.0, "correct": True}, {"confidence": 1.0, "correct": True}]
    assert expected_calibration_error(perfect) == 0.0
    overconfident = [{"confidence": 0.9, "correct": False}] * 4
    assert expected_calibration_error(overconfident) == pytest.approx(0.9)
    assert expected_calibration_error([{"confidence": None, "correct": True}]) is None


def test_reports_round_trip(tmp_path):
    results = evaluate_chat(ROWS[:2], lambda c: ('{"product": "credit_reporting"}', 0.1))
    record = build_result("base-chat", results, {"mode": "chat", "model": {"repo_id": "x/y"}})
    save_result(record, tmp_path)
    assert load_latest(tmp_path, "base-chat")["metrics"]["examples"] == 2
    md = render_markdown(record)
    assert "| Accuracy | 1 / 2 (50.0%) |" in md and "Confusion matrix" in md
    assert load_latest(tmp_path, "decision") is None
