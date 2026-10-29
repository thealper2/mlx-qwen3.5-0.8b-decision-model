"""The task: route a consumer complaint to one of 9 financial products.

``ROUTES`` are the allowed options with their descriptions. ``build_request`` writes the
decision request (a ``state`` plus typed ``questions``) that the decision model reads.
``chat_messages`` / ``parse_chat_reply`` ask an ordinary chat model the same question and
read its JSON reply — the generative baseline.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

QUESTION_ID = "product"
ROUTES: dict[str, str] = {
    "credit_reporting": "credit reports, credit bureaus, credit repair, identity on report",
    "debt_collection": "debt collectors, collection calls, debts owed",
    "credit_card": "credit cards and prepaid cards",
    "bank_account": "checking or savings accounts",
    "mortgage": "home loans and mortgages",
    "money_transfer": "money transfers, wire, payment apps, virtual currency",
    "vehicle_loan": "car loans or vehicle leases",
    "student_loan": "student loans",
    "personal_loan": "payday, title, personal or installment loans",
}
INSTRUCTIONS = "Which financial product is the consumer's `complaint` about?"
CHAT_SYSTEM = (
    "You are a decision model. You get a JSON request with a `state` and typed `questions`. "
    "Answer every question with exactly one of its criteria keys. "
    'Reply with JSON only, like {"product": "<key>"}.'
)
_OBJECT = re.compile(r"\{[^{}]*\}")


def build_request(complaint: str, routes: dict[str, str] | None = None) -> dict[str, Any]:
    """Decision request: the complaint as state, one ``choice`` question over ``routes``."""
    return {
        "model": "decision-model",
        "state": {"complaint": complaint},
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": dict(ROUTES if routes is None else routes),
            }
        },
    }


def option_subset(label: str, rng: random.Random) -> dict[str, str]:
    """Training augmentation: with probability 0.5 keep only some routes (always ``label``).

    Draw order (fixed, part of the data definition): ``rng.random()``; if below 0.5, keep
    ``rng.randint(1, 8)`` of the other routes chosen with ``rng.sample``.
    """
    if rng.random() >= 0.5:
        return dict(ROUTES)
    others = [k for k in ROUTES if k != label]
    keep = {label, *rng.sample(others, rng.randint(1, len(others)))}
    return {k: v for k, v in ROUTES.items() if k in keep}


def chat_messages(complaint: str) -> list[dict[str, str]]:
    """The same request for a chat model: system instruction + the request as JSON."""
    req = build_request(complaint)
    body = {"state": req["state"], "questions": req["questions"]}
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": json.dumps(body)},
    ]


def parse_chat_reply(reply: str) -> str | None:
    """The route a chat model chose, or None when its reply holds no valid choice.

    Looks after the last ``</think>``, takes the last flat ``{...}`` that parses as JSON,
    and accepts its ``product`` only if it is one of the routes.
    """
    for candidate in reversed(_OBJECT.findall(reply.split("</think>")[-1])):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        product = obj.get(QUESTION_ID) if isinstance(obj, dict) else None
        return product if product in ROUTES else None
    return None
