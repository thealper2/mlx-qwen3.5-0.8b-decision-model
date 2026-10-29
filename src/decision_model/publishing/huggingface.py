"""Opt-in upload of a trained decision model (LoRA adapter + head) to the Hugging Face Hub.

``plan_upload`` and ``stage_upload`` never touch the network. ``publish`` uploads only
when called with ``execute=True`` (``make publish``); otherwise it is a dry run.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from decision_model.config import ConfigError, ProjectConfig, validate_hf_id
from decision_model.training.artifacts import RUN_FILES, RUN_MANIFEST, read_json
from decision_model.utils.paths import display, is_within

log = logging.getLogger(__name__)

CARD_NAME = "README.md"
RUNTIME_NAME = "decision_model_runtime.py"
_FORBIDDEN_NAMES = re.compile(r"(^\.env|\.pem$|\.key$|token|secret|credential)", re.I)
_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".md", ".txt", ".py"}


class PublishBlocked(RuntimeError):
    """The upload plan failed a safety check."""


@dataclass
class UploadPlan:
    repo_id: str | None
    private: bool
    run_dir: Path
    card_path: Path
    staging_dir: Path
    files: list[tuple[Path, str]] = field(default_factory=list)  # (source, name in repo)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def plan_upload(
    project: ProjectConfig, repo_id: str | None = None, run_dir: Path | None = None
) -> UploadPlan:
    """Decide what would be uploaded and check every safeguard (no side effects)."""
    pub = project.publishing
    repo = repo_id or pub.repo_id
    adir = run_dir or pub.run_dir
    plan = UploadPlan(repo, pub.private, adir, pub.model_card_path, pub.staging_dir)
    if not repo:
        plan.problems.append(
            "No destination: set repo_id in configs/publishing.yaml or pass "
            "--repo-id <namespace>/<name>"
        )
    else:
        try:
            validate_hf_id(repo, "repo_id")
        except ConfigError as exc:
            plan.problems.append(str(exc))
    if not is_within(pub.staging_dir, project.root / "artifacts"):
        plan.problems.append("staging_dir must be inside artifacts/ (it is cleared on each run)")

    if not adir.is_dir():
        plan.problems.append(f"Run directory {display(adir, project.root)} does not exist")
        return plan
    run = read_json(adir / RUN_MANIFEST)
    if run is None:
        plan.problems.append(
            f"{RUN_MANIFEST} missing: only runs trained by this project "
            "(with recorded provenance) can be published"
        )
    else:
        if run.get("status") != "completed":
            plan.problems.append(f"Training run status is '{run.get('status')}', not 'completed'")
        if run.get("profile", {}).get("kind") == "smoke":
            plan.problems.append("This is a smoke-test run (2 updates); it is not publishable")
    for name in pub.include:
        src = adir / name
        if _FORBIDDEN_NAMES.search(name):
            plan.problems.append(f"Refusing to upload a file named like a secret: {name}")
        elif src.is_file():
            plan.files.append((src, name))
        elif name in RUN_FILES:
            plan.problems.append(f"Required run file missing: {name}")
    # The self-contained runtime, so the release loads without this project.
    plan.files.append((Path(__file__).resolve().parent.parent / "runtime.py", RUNTIME_NAME))
    if pub.model_card_path.is_file():
        plan.files.append((pub.model_card_path, CARD_NAME))
    else:
        plan.problems.append(
            f"Model card {display(pub.model_card_path, project.root)} not found: run `make card`"
        )
    return plan


def _sanitize(obj: Any, replacements: dict[str, str]) -> Any:
    """Replace local absolute paths in config values with portable identifiers."""
    if isinstance(obj, dict):
        return {
            k: (
                replacements[k]
                if k in replacements and isinstance(v, str)
                else _sanitize(v, replacements)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize(v, replacements) for v in obj]
    if isinstance(obj, str) and obj.startswith(str(Path.home())):
        return "<local path removed>"
    return obj


def stage_upload(plan: UploadPlan, project: ProjectConfig) -> list[Path]:
    """Copy the planned files into a clean staging folder, removing local paths.

    JSON/YAML files are rewritten so any ``model`` value is the Hub model id and any path
    under the home directory is removed; the staged files are then scanned for leaks.
    Returns the staged files.
    """
    if not plan.ok:
        raise PublishBlocked("; ".join(plan.problems))
    model = project.training.model
    model_ref = model.repo_id
    if plan.staging_dir.exists():
        if not is_within(plan.staging_dir, project.root / "artifacts"):
            raise PublishBlocked("Refusing to clear a staging dir outside artifacts/")
        shutil.rmtree(plan.staging_dir)
    plan.staging_dir.mkdir(parents=True)
    staged = []
    for src, name in plan.files:
        dest = plan.staging_dir / name
        if name.endswith(".json") and name != CARD_NAME:
            data = json.loads(src.read_text(encoding="utf-8"))
            data = _sanitize(data, {"model": model_ref})
            dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        elif name.endswith((".yaml", ".yml")):
            data = yaml.safe_load(src.read_text(encoding="utf-8"))
            data = _sanitize(data, {"model": model_ref})
            dest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            shutil.copy2(src, dest)
        staged.append(dest)
    leaks = scan_for_leaks(staged)
    if leaks:
        raise PublishBlocked("Staged files still contain sensitive data: " + "; ".join(leaks))
    return staged


def scan_for_leaks(paths: list[Path]) -> list[str]:
    """Token-like strings or home-directory paths in staged text files."""
    home = str(Path.home())
    found = []
    for path in paths:
        if path.suffix not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _TOKEN_PATTERN.search(text):
            found.append(f"{path.name}: looks like a Hugging Face token")
        if home and home != "/" and home in text:
            found.append(f"{path.name}: contains a local home-directory path")
    return found


def describe_upload(plan: UploadPlan, project: ProjectConfig) -> str:
    lines = [
        f"Destination : https://huggingface.co/{plan.repo_id or '<unset>'} "
        f"({'private' if plan.private else 'PUBLIC'})",
        f"Run         : {display(plan.run_dir, project.root)}",
        "Files       :",
    ]
    for src, name in plan.files:
        lines.append(
            f"  {name:<22} {src.stat().st_size / 1e6:8.2f} MB  <- {display(src, project.root)}"
        )
    lines.append("Stays local : training logs, intermediate checkpoints, data, .env, caches")
    lines += [f"BLOCKED: {p}" for p in plan.problems]
    return "\n".join(lines)


def publish(plan: UploadPlan, project: ProjectConfig, execute: bool = False) -> str | None:
    """Stage and (only with ``execute=True``) upload. Returns the commit URL or None."""
    staged = stage_upload(plan, project)
    log.info("Staged %d files in %s", len(staged), display(plan.staging_dir, project.root))
    if not execute:
        log.info("Dry run: nothing uploaded. Use `make publish` to upload.")
        return None
    from huggingface_hub import HfApi, get_token

    if get_token() is None:
        raise PublishBlocked(
            "Not logged in to Hugging Face. Run `uv run hf auth login` "
            "(token with write access) and retry."
        )
    assert plan.repo_id is not None
    api = HfApi()
    api.create_repo(plan.repo_id, repo_type="model", private=plan.private, exist_ok=True)
    info = api.upload_folder(
        repo_id=plan.repo_id,
        repo_type="model",
        folder_path=plan.staging_dir,
        commit_message=project.publishing.commit_message,
        allow_patterns=[p.name for p in staged],
    )
    url = str(getattr(info, "commit_url", info))
    log.info("Uploaded: %s", url)
    return url
