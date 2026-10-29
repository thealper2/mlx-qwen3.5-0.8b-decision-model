from __future__ import annotations

import json
import random

import pytest

from decision_model.task import (
    CHAT_SYSTEM,
    INSTRUCTIONS,
    QUESTION_ID,
    ROUTES,
    build_request,
    chat_messages,
    option_subset,
    parse_chat_reply,
)


def test_routes_are_the_nine_products_in_order():
    assert list(ROUTES) == [
        "credit_reporting",
        "debt_collection",
        "credit_card",
        "bank_account",
        "mortgage",
        "money_transfer",
        "vehicle_loan",
        "student_loan",
        "personal_loan",
    ]
    assert ROUTES["money_transfer"] == "money transfers, wire, payment apps, virtual currency"


def test_build_request_layout():
    req = build_request("my card was charged twice")
    assert req["state"] == {"complaint": "my card was charged twice"}
    q = req["questions"][QUESTION_ID]
    assert q == {"type": "choice", "instructions": INSTRUCTIONS, "criteria": ROUTES}
    assert q["criteria"] is not ROUTES  # a copy, never the shared constant
    sub = build_request("x", {"mortgage": ROUTES["mortgage"], "credit_card": ROUTES["credit_card"]})
    assert list(sub["questions"][QUESTION_ID]["criteria"]) == ["mortgage", "credit_card"]


def test_option_subset_rules():
    rng = random.Random(0)
    subsets = [option_subset("mortgage", rng) for _ in range(2000)]
    assert all("mortgage" in s and len(s) >= 2 for s in subsets)
    assert all(list(s) == [k for k in ROUTES if k in s] for s in subsets)  # route order kept
    reduced = sum(len(s) < 9 for s in subsets) / len(subsets)
    assert 0.4 < reduced < 0.5  # p = 0.5 of subsetting, and a subset can still hold all 9


def test_option_subset_draw_order_is_fixed():
    rng = random.Random(3)
    first = rng.random()
    rng = random.Random(3)
    result = option_subset("debt_collection", rng)
    assert (len(result) == 9) == (first >= 0.5)


def test_chat_messages():
    system, user = chat_messages("late fee on my loan")
    assert system == {"role": "system", "content": CHAT_SYSTEM}
    body = json.loads(user["content"])
    assert set(body) == {"state", "questions"}
    assert body["questions"][QUESTION_ID]["criteria"] == ROUTES


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"product": "mortgage"}', "mortgage"),
        ('Answer: {"product": "nope"}', None),
        ('<think>{"product": "mortgage"}</think>\n{"product": "bank_account"}', "bank_account"),
        ("no json at all", None),
        ('{"a": 1} {"product": "credit_card"}', "credit_card"),
        ('{"product": "credit_card"} {broken', "credit_card"),
        ('{"product": "mortgage"} {"other": 2}', None),  # the last parseable object decides
    ],
)
def test_parse_chat_reply(reply, expected):
    assert parse_chat_reply(reply) == expected
