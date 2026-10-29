"""Model loading and the two ways of answering: System 1 (head) and System 2 (chat).

MLX imports happen inside the functions so the rest of the package (and the unit tests)
work on machines without MLX.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model.task import chat_messages

log = logging.getLogger(__name__)


class InferenceUnavailable(RuntimeError):
    """MLX / MLX-LM cannot be used on this machine or the artifacts are missing."""


@dataclass(frozen=True)
class ChatReply:
    raw: str
    latency_s: float


def require_mlx() -> None:
    """Fail early with a clear message when MLX-LM cannot run here."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise InferenceUnavailable("MLX-LM needs macOS on Apple Silicon (arm64).")
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise InferenceUnavailable("mlx-lm is not installed; run `make install`.") from exc


def check_run_dir(run_dir: Path) -> None:
    """Raise when ``run_dir`` does not hold a trained adapter + head."""
    from decision_model.training.artifacts import RUN_FILES

    missing = [n for n in RUN_FILES if not (run_dir / n).is_file()]
    if missing:
        raise InferenceUnavailable(
            f"{run_dir} is not a trained run (missing {', '.join(missing)}). Train one first "
            "(`make train-smoke` or `make train`)."
        )


def load_decision_model(repo_id: str, revision: str | None, run_dir: Path) -> tuple[Any, Any]:
    """Base model + LoRA adapter + head from a training run directory, in eval mode."""
    require_mlx()
    check_run_dir(run_dir)
    from decision_model.runtime import load_release

    log.info("Loading %s @ %s + run %s", repo_id, (revision or "main")[:12], run_dir)
    return load_release(repo_id, revision, run_dir)


def load_base_model(repo_id: str, revision: str | None) -> tuple[Any, Any]:
    """The base chat model alone (no adapter, no head)."""
    require_mlx()
    from mlx_lm import load

    log.info("Loading %s @ %s (base model, no adapter)", repo_id, (revision or "main")[:12])
    loaded = load(repo_id, revision=revision)
    return loaded[0], loaded[1]


def decide_batch(
    model: Any, tokenizer: Any, requests: list[dict[str, Any]], max_length: int
) -> tuple[list[dict[str, dict[str, Any]]], float]:
    """Answers for several requests in one forward pass, plus the wall time."""
    import mlx.core as mx

    from decision_model.runtime import choice_answer, collate, encode_request, tokenizer_fn

    tokenize = tokenizer_fn(tokenizer)
    records = [encode_request(tokenize, r, max_length) for r in requests]
    start = time.perf_counter()
    ids, pad = collate(records, tokenizer.pad_token_id)
    logits = model(ids, pad, records)
    mx.eval(logits)
    answers = [
        {
            q.question_id: choice_answer(q.option_ids, lg)
            for q, lg in zip(rec.questions, rec_logits, strict=True)
        }
        for rec, rec_logits in zip(records, logits, strict=True)
    ]
    return answers, time.perf_counter() - start


def chat_reply(
    model: Any, tokenizer: Any, complaint: str, max_tokens: int, enable_thinking: bool = False
) -> ChatReply:
    """The base model answering the same request as a chat (greedy)."""
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        chat_messages(complaint),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    start = time.perf_counter()
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return ChatReply(raw=raw, latency_s=round(time.perf_counter() - start, 3))


def peak_memory_gb() -> float | None:
    """Peak MLX memory of this process so far, if MLX is available."""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return round(mx.get_peak_memory() / 1e9, 2)
