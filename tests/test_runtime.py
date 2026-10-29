"""Runtime (MLX) tests: prompt encoding, head shapes, padding invariance, answers.

These need MLX and are skipped on other platforms. They use a tiny head and a toy
character tokenizer, so no model is downloaded.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mlx.core", reason="needs MLX (Apple Silicon)")
pytestmark = pytest.mark.macos

import mlx.core as mx  # noqa: E402

from decision_model.runtime import (  # noqa: E402
    PREFIX,
    SUFFIX,
    SYSTEM_PROMPT,
    JointSchemaHead,
    choice_answer,
    choice_options,
    collate,
    encode_request,
    render,
)
from decision_model.task import ROUTES, build_request  # noqa: E402


def toy_tokenize(text: str) -> list[int]:
    return [ord(c) % 200 for c in text]


def test_prompt_layout_and_spans():
    req = build_request(
        "card charged twice", {"credit_card": ROUTES["credit_card"], "mortgage": ROUTES["mortgage"]}
    )
    enc = encode_request(toy_tokenize, req)
    prompt = (
        PREFIX.format(system=SYSTEM_PROMPT) + render(req["state"]) + "\n\nSCHEMA FIELDS:\n"
        "\nFIELD 1\nID: product\nTYPE: choice\nINSTRUCTION: "
        + req["questions"]["product"]["instructions"]
        + "\nALLOWED OPTIONS:\n"
        + "OPTION 1: "
        + render({"option_id": "credit_card", "description": ROUTES["credit_card"]})
        + "\n"
        + "OPTION 2: "
        + render({"option_id": "mortgage", "description": ROUTES["mortgage"]})
        + "\n"
        + "END FIELD\n"
        + SUFFIX
    )
    assert list(enc.input_ids) == toy_tokenize(prompt)
    q = enc.questions[0]
    assert q.option_ids == ("credit_card", "mortgage")  # sorted by id
    s0, s1 = q.option_spans[1]
    assert enc.input_ids[s0:s1] == tuple(
        toy_tokenize(
            json.dumps(
                {"description": ROUTES["mortgage"], "option_id": "mortgage"}, separators=(",", ":")
            )
        )
    )
    q0, q1 = q.question_span
    assert enc.input_ids[q0:q1] == tuple(toy_tokenize(req["questions"]["product"]["instructions"]))


def test_state_is_truncated_never_the_schema():
    req = build_request("x" * 5000)
    enc = encode_request(toy_tokenize, req, max_length=3000)
    assert len(enc.input_ids) == 3000
    full = encode_request(toy_tokenize, req, max_length=100_000)
    assert len(enc.questions[0].option_spans) == len(full.questions[0].option_spans) == 9
    with pytest.raises(ValueError, match="schema alone"):
        encode_request(toy_tokenize, req, max_length=500)


def test_only_choice_questions():
    with pytest.raises(ValueError, match="only 'choice'"):
        choice_options({"type": "noul", "criteria": {}})
    with pytest.raises(ValueError, match="at least two"):
        choice_options({"type": "choice", "criteria": {"a": "x"}})


def _tiny_head() -> JointSchemaHead:
    mx.random.seed(0)
    return JointSchemaHead(
        hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=12
    )


def test_head_logits_per_option_and_padding_invariance():
    head = _tiny_head()
    requests = [
        build_request("a b c"),
        build_request(
            "a much longer complaint text here",
            {"credit_card": ROUTES["credit_card"], "mortgage": ROUTES["mortgage"]},
        ),
    ]
    records = [encode_request(toy_tokenize, r) for r in requests]
    ids, pad = collate(records, pad_id=0)
    vocab = mx.random.normal((200, 16))
    hidden = mx.random.normal((2, ids.shape[1], 16))
    logits = [
        head.record_logits(hidden[b], ids[b], pad[b], rec, vocab) for b, rec in enumerate(records)
    ]
    assert [lg[0].shape for lg in logits] == [(9,), (2,)]
    # The same record alone (no padding) gives the same logits as inside the padded batch.
    n = len(records[0].input_ids)
    alone = head.record_logits(
        hidden[0, :n], ids[0, :n], mx.zeros((n,), dtype=mx.bool_), records[0], vocab
    )
    assert mx.allclose(alone[0], logits[0][0], atol=1e-5)
    answer = choice_answer(records[1].questions[0].option_ids, logits[1][0])
    assert answer["choice"] in ("credit_card", "mortgage")
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_head_weights_round_trip(tmp_path):
    head = _tiny_head()
    path = tmp_path / "head.safetensors"
    head.save_weights(str(path))
    other = JointSchemaHead(
        hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=12
    )
    other.load_weights(str(path))
    from mlx.utils import tree_flatten

    a, b = dict(tree_flatten(head.parameters())), dict(tree_flatten(other.parameters()))
    assert a.keys() == b.keys() and all(mx.array_equal(a[k], b[k]) for k in a)
