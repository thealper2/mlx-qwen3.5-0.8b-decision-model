"""Repository-root discovery and path resolution."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "DECISION_MODEL_ROOT"
_MARKERS = ("pyproject.toml", "configs")


def find_repo_root(start: Path | None = None) -> Path:
    """Return the project root.

    Uses ``$DECISION_MODEL_ROOT`` when set, otherwise walks up from ``start`` (default: the
    current directory) until a directory containing both ``pyproject.toml`` and
    ``configs/`` is found.
    """
    env = os.environ.get(ROOT_ENV)
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{ROOT_ENV}={env} is not a directory")
        return root
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / m).exists() for m in _MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not find the project root above {here}. Run from inside the repository "
        f"or set {ROOT_ENV}."
    )


def resolve(path: str | Path, root: Path) -> Path:
    """Resolve ``path`` relative to ``root`` unless it is already absolute."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (root / p)


def is_within(path: Path, parent: Path) -> bool:
    """True when ``path`` is ``parent`` or lies inside it (after resolving symlinks)."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def display(path: Path, root: Path) -> str:
    """Path relative to ``root`` when possible, for logs and reports."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
