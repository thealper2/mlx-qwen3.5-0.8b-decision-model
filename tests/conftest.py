"""Shared fixtures: a throw-away project root with small synthetic splits.

Nothing here touches the network, the dataset, a model or training.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from decision_model.config import ProjectConfig, load_project_config
from decision_model.task import ROUTES

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZES = {"train": 27, "validation": 9, "test": 18}


def fake_splits() -> dict[str, list[dict[str, Any]]]:
    """Splits shaped like the real dataset, every route present, no overlap."""
    routes = list(ROUTES)
    out = {}
    for split, n in SIZES.items():
        out[split] = [
            {
                "id": f"{split}-{i}",
                "complaint": f"{split} complaint {i} about {routes[i % 9]}",
                "product": routes[i % 9],
            }
            for i in range(n)
        ]
    return out


def make_project(root: Path) -> ProjectConfig:
    """Copy the real configs into ``root`` with small sizes, then load them."""
    cfg_dir = root / "configs"
    shutil.copytree(REPO_ROOT / "configs", cfg_dir)
    (root / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    data = yaml.safe_load((cfg_dir / "data.yaml").read_text())
    data["dataset"]["expected_rows"] = dict(SIZES)
    data["training_requests"]["num_requests"] = 16
    data["smoke"].update(train_size=2, test_size=4)
    (cfg_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    training = yaml.safe_load((cfg_dir / "training.yaml").read_text())
    for profile in training["profiles"].values():  # 16 requests: 2 updates of 8
        if profile["kind"] == "full":
            profile["steps"] = 2
            profile["warmup_steps"] = 1
    (cfg_dir / "training.yaml").write_text(yaml.safe_dump(training, sort_keys=False))
    return load_project_config(root=root)


@pytest.fixture
def project(tmp_path: Path) -> ProjectConfig:
    return make_project(tmp_path)


@pytest.fixture
def prepared_project(tmp_path: Path) -> ProjectConfig:
    """Project with canonical + smoke data prepared from the fake splits."""
    from decision_model.data.prepare import prepare_dataset, prepare_smoke_subset

    proj = make_project(tmp_path)
    prepare_dataset(fake_splits(), proj.data, proj.root)
    prepare_smoke_subset(proj.data, proj.root)
    return proj


@pytest.fixture
def repo_root_env(monkeypatch: pytest.MonkeyPatch):
    """Point the CLI at a temporary project root."""

    def _set(root: Path) -> None:
        monkeypatch.setenv("DECISION_MODEL_ROOT", str(root))

    return _set
