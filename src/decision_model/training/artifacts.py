"""Run manifests, environment provenance and run-directory checks."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

RUN_MANIFEST = "run_manifest.json"
TRAIN_LOG = "train_log.jsonl"
RUN_FILES = ("adapters.safetensors", "adapter_config.json", "head.safetensors", "head_config.json")
TRACKED_PACKAGES = (
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "huggingface-hub",
    "transformers",
    "pyarrow",
    "pyyaml",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id(label: str) -> str:
    """``<label>-YYYYmmdd-HHMMSS`` in UTC."""
    return f"{label}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that define the training/inference stack."""
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _sysctl(key: str) -> str | None:
    try:
        res = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() or None


def environment_info() -> dict[str, Any]:
    """Python, OS and hardware description for manifests (no user-identifying data)."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "packages": package_versions(),
    }
    if platform.system() == "Darwin":
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        mem = _sysctl("hw.memsize")
        info["memory_gb"] = round(int(mem) / 1024**3) if mem and mem.isdigit() else None
    return info


def git_commit(root: Path) -> str | None:
    """Current commit hash (with ``-dirty`` when the tree has changes), if in a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def run_problems(run_dir: Path) -> list[str]:
    """Missing or empty run files; empty list when the run directory looks complete."""
    problems = []
    for name in RUN_FILES:
        path = run_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif path.stat().st_size == 0:
            problems.append(f"{name} is empty")
    return problems


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    """Parsed JSON object, or None when the file does not exist."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
