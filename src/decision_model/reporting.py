"""Experiment summary built only from recorded run manifests and evaluation records."""

from __future__ import annotations

from pathlib import Path

from decision_model.config import ProjectConfig
from decision_model.evaluation.evaluator import load_latest
from decision_model.training.artifacts import RUN_MANIFEST, read_json, utc_now
from decision_model.utils.paths import resolve

LABELS = ("base-chat", "decision", "base-chat-smoke", "decision-smoke")


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_experiment_report(project: ProjectConfig) -> str:
    """Markdown summary of every training profile's run and the latest evaluations."""
    m = project.training.model
    d = project.data.dataset
    lines = [
        "# Experiment report",
        "",
        f"Generated {utc_now()} from recorded outputs only.",
        "",
        f"- Base model: `{m.repo_id}` @ `{m.revision}`",
        f"- Dataset: `{d.repo_id}` @ `{d.revision}`",
        "",
        "## Training runs",
        "",
        "| Profile | Status | Run id | Updates | Final loss (avg of 10) | Wall time | "
        "Peak memory |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, profile in sorted(project.training.profiles.items()):
        run = read_json(resolve(profile.output_dir, project.root) / RUN_MANIFEST)
        if run is None:
            lines.append(f"| {name} | not run | – | {profile.steps} | – | – | – |")
            continue
        mt = run.get("metrics", {})

        def fmt(key: str, suffix: str = "", mt: dict = mt) -> str:
            value = mt.get(key)
            return "n/a" if value is None else f"{value}{suffix}"

        lines.append(
            f"| {name} | {run.get('status')} | {run.get('run_id')} | {profile.steps} | "
            f"{fmt('final_loss_avg10')} | {fmt('train_wall_time_s', ' s')} | "
            f"{fmt('peak_memory_gb', ' GB')} |"
        )
    lines += [
        "",
        "## Evaluations (latest per label)",
        "",
        "| Label | Training run | Complaints | Accuracy | Invalid | Mean latency | Date |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        ev = load_latest(project.evaluation.output_dir, label)
        if ev is None:
            lines.append(f"| {label} | – | – | not evaluated | – | – | – |")
            continue
        mt = ev["metrics"]
        run_id = (ev.get("run") or {}).get("run_id") or "–"
        latency = "n/a" if mt["mean_latency_ms"] is None else f"{mt['mean_latency_ms']} ms"
        lines.append(
            f"| {label} | {run_id} | {mt['examples']} | {mt['correct']} "
            f"({_pct(mt['accuracy'])}) | {mt['invalid_answers']} | {latency} | "
            f"{ev['evaluated_at'][:10]} |"
        )
    lines += [
        "",
        "Smoke labels use a handful of complaints and a 2-update model; they only show that the "
        "pipeline runs and say nothing about model quality.",
    ]
    return "\n".join(lines) + "\n"


def write_experiment_report(project: ProjectConfig) -> Path:
    path = project.root / "reports" / "experiments" / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_experiment_report(project), encoding="utf-8")
    return path
