"""Guarded training entry point.

``plan_training`` performs every check without side effects (it is the dry run).
``execute_training`` runs the plan; it is only reached through an explicit
``--execute`` flag and never from tests, smoke checks or other commands.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decision_model.config import ProjectConfig, TrainingProfile
from decision_model.data.validate import read_jsonl, request_line_problems, validate_jsonl
from decision_model.training.artifacts import (
    RUN_MANIFEST,
    TRAIN_LOG,
    environment_info,
    git_commit,
    new_run_id,
    run_problems,
    utc_now,
    write_json,
)
from decision_model.utils.memory import free_disk_gb, memory_snapshot
from decision_model.utils.paths import display, resolve

log = logging.getLogger(__name__)

MODEL_WEIGHTS_GB = 1.75  # the pinned base checkpoint (1,746,942,600 bytes, text + vision)


class TrainingBlocked(RuntimeError):
    """The training plan failed a pre-flight check."""


@dataclass
class TrainingPlan:
    profile: TrainingProfile
    requests_file: Path
    output_dir: Path
    available_requests: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def plan_training(
    project: ProjectConfig, profile_name: str, output_dir: str | None = None
) -> TrainingPlan:
    """Validate everything a training run needs, without changing anything on disk."""
    root = project.root
    profile = project.training.profile(profile_name)
    requests_file = resolve(profile.requests_file, root)
    out = resolve(output_dir or profile.output_dir, root)
    problems: list[str] = []
    warnings: list[str] = []

    report = validate_jsonl(requests_file, request_line_problems)
    if not report.ok:
        problems.append(
            f"{display(requests_file, root)}: {report.error_count} problem(s), e.g. "
            f"{'; '.join(report.errors[:3])}"
        )
    elif report.lines < profile.requests_needed:
        problems.append(
            f"{display(requests_file, root)} has {report.lines} requests; the profile needs "
            f"{profile.requests_needed} ({profile.steps} updates x {profile.examples_per_update})"
        )
    existing = [p for p in ("head.safetensors", RUN_MANIFEST) if (out / p).exists()]
    if existing:
        problems.append(
            f"{display(out, root)} already holds a trained run ({', '.join(existing)}). "
            "Runs are never overwritten: pass --output-dir <new dir> or move it"
            + (
                " (the disposable smoke run is removed by `make clean-artifacts`)."
                if profile.kind == "smoke"
                else "."
            )
        )
    disk = free_disk_gb(out)
    if disk < project.training.safety.min_free_disk_gb:
        problems.append(
            f"Only {disk:.1f} GB free disk; need >= {project.training.safety.min_free_disk_gb} GB"
        )
    mem = memory_snapshot()
    if mem.total_gb is not None and mem.total_gb <= 8.5 and profile.batch_size > 1:
        warnings.append(
            f"batch_size={profile.batch_size} on an 8 GB machine: measured peaks are ~3.5-4.3 GB "
            "for one request and ~6.5 GB for two; prefer the m1_air_8gb profile (batch 1)."
        )
    if profile.kind == "full":
        warnings.append(
            f"Slow: ~10-20 s per request on an M1 (MLX-LM trains Qwen3.5's linear-attention "
            f"layers without their fused kernel); {profile.requests_needed} requests = hours."
        )
    return TrainingPlan(profile, requests_file, out, report.lines, problems, warnings)


def describe_plan(plan: TrainingPlan, project: ProjectConfig) -> str:
    """Human-readable summary of a plan (used for dry runs and before real runs)."""
    p = plan.profile
    t = project.training
    lines = [
        f"Training profile : {p.name} ({p.kind})",
        f"Base model       : {t.model.repo_id} @ {t.model.revision}",
        f"Requests         : {display(plan.requests_file, project.root)} "
        f"({plan.available_requests} available, {p.requests_needed} used in order)",
        f"Output           : {display(plan.output_dir, project.root)}",
        f"LoRA             : rank {t.lora.rank}, alpha {t.lora.alpha:g} (scale {t.lora.scale:g}), "
        f"dropout {t.lora.dropout:g}, all blocks: {', '.join(t.lora.keys)}",
        f"Head             : {json.dumps(t.head.to_dict())}",
        f"Optimisation     : AdamW (weight decay {p.weight_decay:g}), lr LoRA {p.lr_lora:g} / "
        f"head {p.lr_head:g}, warmup {p.warmup_steps} x cosine over {p.steps} updates, "
        f"grad-norm clip {p.max_grad_norm:g}",
        f"Updates          : {p.steps} x (batch {p.batch_size} x accumulate "
        f"{p.grad_accumulation_steps}) = {p.examples_per_update} requests per update",
        f"Memory           : gradient checkpointing {p.grad_checkpoint}; base weights "
        f"~{MODEL_WEIGHTS_GB} GB on disk",
    ]
    lines += [f"WARNING: {w}" for w in plan.warnings]
    lines += [f"BLOCKED: {pr}" for pr in plan.problems]
    return "\n".join(lines)


def execute_training(plan: TrainingPlan, project: ProjectConfig) -> dict[str, Any]:
    """Run the training loop for an approved plan and write the run manifest."""
    from decision_model.training.loop import train

    if not plan.ok:
        raise TrainingBlocked("; ".join(plan.problems))
    profile = plan.profile
    root = project.root
    if profile.kind == "full":
        log.warning(
            "Starting FULL training (%s): %d updates of %d requests. This can take a "
            "long time and most of the machine's memory. Ctrl-C stops it.",
            profile.name,
            profile.steps,
            profile.examples_per_update,
        )
    stream = read_jsonl(plan.requests_file)[: profile.requests_needed]
    out = plan.output_dir
    out.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(profile.name)
    started = utc_now()
    log_path = out / TRAIN_LOG

    def on_log(entry: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    status, metrics, error = "failed", {}, None
    try:
        metrics = train(project, profile, stream, out, on_log)
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:  # recorded in the manifest, then re-raised
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        missing = run_problems(out)
        if status == "completed" and missing:
            status = "failed"
        t = project.training
        manifest = {
            "run_id": run_id,
            "status": status,
            "error": error,
            "started_at": started,
            "finished_at": utc_now(),
            "profile": profile.to_dict(),
            "base_model": {"repo_id": t.model.repo_id, "revision": t.model.revision},
            "lora": {
                "rank": t.lora.rank,
                "alpha": t.lora.alpha,
                "scale": t.lora.scale,
                "dropout": t.lora.dropout,
                "keys": t.lora.keys,
            },
            "head": t.head.to_dict(),
            "max_length": t.max_length,
            "dataset": {
                "repo_id": project.data.dataset.repo_id,
                "revision": project.data.dataset.revision,
                "requests_file": display(plan.requests_file, root),
            },
            "artifacts": {"run_dir": display(out, root), "missing": missing},
            "metrics": {k: v for k, v in metrics.items() if k != "history"},
            "history": metrics.get("history", []),
            "environment": environment_info(),
            "git_commit": git_commit(root),
        }
        write_json(out / RUN_MANIFEST, manifest)
        write_json(root / "reports" / "experiments" / f"{run_id}.json", manifest)
        log.info("Run %s %s. Manifest: %s", run_id, status, display(out / RUN_MANIFEST, root))
    return manifest
