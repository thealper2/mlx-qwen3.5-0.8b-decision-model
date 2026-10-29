"""The MLX training loop: LoRA on the language model + the joint schema head, jointly.

MLX-LM's own trainer only optimises next-token prediction, so this loop is custom. It
reuses MLX-LM for everything it can: model loading, LoRA layers
(``linear_to_lora_layers``), gradient checkpointing and the adapter file format, so the
saved adapter loads with ``mlx_lm.load(..., adapter_path=...)``.

One update = ``grad_accumulation_steps`` micro-batches of ``batch_size`` consecutive
requests from the ordered training stream. Loss = mean cross-entropy of the correct
option. Gradients are averaged over micro-batches, clipped to ``max_grad_norm`` (global
norm over LoRA + head), then applied by AdamW with separate learning rates for LoRA and
head parameters and the schedule ``lr x min(1, (step+1)/warmup) x 0.5(1+cos(pi step/steps))``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from decision_model.config import ProjectConfig, TrainingProfile
from decision_model.task import QUESTION_ID

log = logging.getLogger(__name__)
ADAPTER_WEIGHTS = "adapters.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
HEAD_WEIGHTS = "head.safetensors"  # same names as in runtime.py
HEAD_CONFIG = "head_config.json"


def lr_schedule(base: float, warmup: int, steps: int) -> Callable[[Any], Any]:
    """Linear warmup multiplied by a half-cosine over all ``steps`` (0-based step)."""
    import mlx.core as mx

    def schedule(step: Any) -> Any:
        s = mx.array(step).astype(mx.float32)
        warm = mx.minimum(1.0, (s + 1) / warmup) if warmup > 0 else 1.0
        return base * warm * 0.5 * (1 + mx.cos(math.pi * s / steps))

    return schedule


def schedule_value(base: float, warmup: int, steps: int, step: int) -> float:
    """Plain-Python twin of :func:`lr_schedule`, for logs and tests."""
    warm = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
    return base * warm * 0.5 * (1 + math.cos(math.pi * step / steps))


def adapter_config(project: ProjectConfig) -> dict[str, Any]:
    """``adapter_config.json`` in MLX-LM's format, so ``mlx_lm.load`` can apply the LoRA."""
    lora = project.training.lora
    return {
        "fine_tune_type": "lora",
        "num_layers": -1,
        "lora_parameters": {
            "rank": lora.rank,
            "scale": lora.scale,
            "dropout": lora.dropout,
            "keys": list(lora.keys),
        },
        "base_model": project.training.model.repo_id,
        "base_model_revision": project.training.model.revision,
    }


def build_model(project: ProjectConfig, seed: int) -> tuple[Any, Any]:
    """Base model (frozen) + fresh LoRA layers + a fresh head, seeded."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from decision_model.runtime import DecisionModel, JointSchemaHead

    m = project.training.model
    log.info("Loading %s @ %s", m.repo_id, (m.revision or "main")[:12])
    backbone, tokenizer = load(m.repo_id, revision=m.revision)[:2]
    mx.random.seed(seed)
    backbone.freeze()
    linear_to_lora_layers(backbone, -1, adapter_config(project)["lora_parameters"])
    head = JointSchemaHead(**project.training.head.to_dict())
    return DecisionModel(backbone, head), tokenizer


def count_parameters(tree: Any) -> int:
    from mlx.utils import tree_flatten

    flat: Any = tree_flatten(tree)
    return sum(int(v.size) for _, v in flat)


def save_weights(model: Any, project: ProjectConfig, folder: Path) -> None:
    """LoRA adapter (MLX-LM format) + head weights + both configs."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    folder.mkdir(parents=True, exist_ok=True)
    adapters = dict(tree_flatten(model.backbone.trainable_parameters()))
    mx.save_safetensors(str(folder / ADAPTER_WEIGHTS), adapters)
    (folder / ADAPTER_CONFIG).write_text(json.dumps(adapter_config(project), indent=2) + "\n")
    model.head.save_weights(str(folder / HEAD_WEIGHTS))
    (folder / HEAD_CONFIG).write_text(json.dumps(project.training.head.to_dict(), indent=2) + "\n")


def target_index(record: Any, label: str) -> int:
    question = next(q for q in record.questions if q.question_id == QUESTION_ID)
    return question.option_ids.index(label)


def train(
    project: ProjectConfig,
    profile: TrainingProfile,
    stream: list[dict[str, Any]],
    output_dir: Path,
    on_log: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the training loop and save the weights. Returns metrics for the manifest."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from mlx_lm.tuner.trainer import grad_checkpoint

    from decision_model.runtime import collate, encode_request, tokenizer_fn

    model, tokenizer = build_model(project, profile.seed)
    tokenize = tokenizer_fn(tokenizer)
    pad_id = tokenizer.pad_token_id
    max_length = project.training.max_length
    if profile.grad_checkpoint:
        grad_checkpoint(model.text_model().layers[0])
    trainable = {
        "lora": count_parameters(model.backbone.trainable_parameters()),
        "head": count_parameters(model.head.trainable_parameters()),
        "total_backbone": count_parameters(model.backbone.parameters()),
    }
    log.info(
        "Trainable parameters: LoRA %.2fM, head %.2fM (backbone %.1fM)",
        trainable["lora"] / 1e6,
        trainable["head"] / 1e6,
        trainable["total_backbone"] / 1e6,
    )

    def adamw(base_lr: float) -> optim.Optimizer:
        return optim.AdamW(
            learning_rate=lr_schedule(base_lr, profile.warmup_steps, profile.steps),
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=profile.weight_decay,
            bias_correction=True,
        )

    optimizer = optim.MultiOptimizer(
        [adamw(profile.lr_lora), adamw(profile.lr_head)],
        [lambda path, _: path.startswith("backbone.")],
    )

    def loss_fn(mdl: Any, ids: Any, pad: Any, records: list, targets: list[int]) -> Any:
        logits = mdl(ids, pad, records)
        losses = [mx.logsumexp(lg[0]) - lg[0][t] for lg, t in zip(logits, targets, strict=True)]
        return mx.mean(mx.stack(losses))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    model.train()
    mx.reset_peak_memory()
    history: list[dict[str, Any]] = []
    recent: list[float] = []
    start = time.perf_counter()
    per_update = profile.examples_per_update
    tokens = 0
    for step in range(profile.steps):
        tic = time.perf_counter()
        grads = None
        step_loss = 0.0
        for micro in range(profile.grad_accumulation_steps):
            lo = step * per_update + micro * profile.batch_size
            part = stream[lo : lo + profile.batch_size]
            records = [encode_request(tokenize, p["request"], max_length) for p in part]
            targets = [target_index(r, p["product"]) for r, p in zip(records, part, strict=True)]
            ids, pad = collate(records, pad_id)
            tokens += sum(len(r.input_ids) for r in records)
            loss, g = loss_and_grad(model, ids, pad, records, targets)
            grads = g if grads is None else tree_map(lambda a, b: a + b, grads, g)
            mx.eval(loss, grads)
            step_loss += float(loss) / profile.grad_accumulation_steps
        if profile.grad_accumulation_steps > 1:
            grads = tree_map(lambda x: x / profile.grad_accumulation_steps, grads)
        grads, grad_norm = optim.clip_grad_norm(grads, profile.max_grad_norm)
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        recent = [*recent[-9:], step_loss]
        entry = {
            "step": step + 1,
            "loss": round(step_loss, 4),
            "loss_avg10": round(sum(recent) / len(recent), 4),
            "grad_norm": round(float(grad_norm), 4),
            "lr_lora": schedule_value(profile.lr_lora, profile.warmup_steps, profile.steps, step),
            "seconds": round(time.perf_counter() - tic, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
        history.append(entry)
        if (step + 1) % profile.log_every == 0 or step + 1 == profile.steps:
            log.info(
                "step %d/%d loss %.4f (avg10 %.4f) grad-norm %.3f lr %.2e %.1fs peak %.2f GB",
                entry["step"],
                profile.steps,
                entry["loss"],
                entry["loss_avg10"],
                entry["grad_norm"],
                entry["lr_lora"],
                entry["seconds"],
                entry["peak_memory_gb"],
            )
            if on_log:
                on_log(entry)
        if profile.save_every and (step + 1) % profile.save_every == 0 and step + 1 < profile.steps:
            save_weights(model, project, output_dir / "checkpoints" / f"step_{step + 1:04d}")
    wall = time.perf_counter() - start
    save_weights(model, project, output_dir)
    return {
        "history": history,
        "final_loss_avg10": history[-1]["loss_avg10"] if history else None,
        "trainable_parameters": trainable,
        "train_wall_time_s": round(wall, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "tokens": tokens,
        "requests_used": profile.requests_needed,
    }
