from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT
from decision_model.config import (
    ConfigError,
    load_project_config,
    parse_training_config,
    read_yaml,
)


def test_real_configs_keep_experiment_identity():
    cfg = load_project_config(root=REPO_ROOT)
    assert cfg.training.model.repo_id == "Qwen/Qwen3.5-0.8B"
    assert cfg.training.model.revision == "2fc06364715b967f1860aea9cf38778875588b17"
    d = cfg.data.dataset
    assert d.repo_id == "kgptalkie/cfpb-complaint-routing"
    assert d.revision == "3ec70ba3690c26b8d9b117629fa3803ce4e4b84f"
    assert d.expected_rows == {"train": 2700, "validation": 270, "test": 826}
    assert (cfg.data.requests.seed, cfg.data.requests.num_requests) == (7, 800)


def test_reference_recipe():
    t = load_project_config(root=REPO_ROOT).training
    p = t.profile("reference")
    assert (p.steps, p.batch_size, p.grad_accumulation_steps) == (100, 8, 1)
    assert (p.lr_lora, p.lr_head, p.weight_decay) == (2e-4, 1e-3, 0.01)
    assert (p.warmup_steps, p.max_grad_norm, p.seed) == (20, 1.0, 7)
    assert p.grad_checkpoint
    assert (t.lora.rank, t.lora.alpha, t.lora.dropout, t.lora.scale) == (16, 32, 0.05, 2.0)
    assert set(t.lora.keys) == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    }
    assert t.head.to_dict() == {
        "hidden_size": 1024,
        "width": 512,
        "routing_layers": 1,
        "layers": 2,
        "heads": 8,
        "feedforward": 2048,
    }
    assert t.max_length == 4096


def test_m1_profile_uses_the_same_requests_per_update():
    t = load_project_config(root=REPO_ROOT).training
    ref, m1 = t.profile("reference"), t.profile("m1_air_8gb")
    assert m1.requests_needed == ref.requests_needed == 800
    assert m1.examples_per_update == ref.examples_per_update == 8
    assert (m1.steps, m1.warmup_steps, m1.lr_lora, m1.lr_head) == (
        ref.steps,
        ref.warmup_steps,
        ref.lr_lora,
        ref.lr_head,
    )
    assert m1.batch_size == 1


def _training_raw() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs/training.yaml").read_text())


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("reference", "lr", 1e-4, "unknown key"),
        ("reference", "batch_size", "8", "batch_size must be an integer"),
        ("reference", "warmup_steps", 500, "warmup_steps cannot exceed steps"),
        ("reference", "lr_head", 0.5, "implausible"),
        ("smoke", "steps", 100, "smoke profile has steps=100"),
    ],
)
def test_invalid_profiles(section, key, value, message):
    raw = _training_raw()
    raw["profiles"][section][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_training_config(raw)


def test_invalid_model_lora_and_head():
    raw = _training_raw()
    raw["model"]["revision"] = "main"
    with pytest.raises(ConfigError, match="40-character commit hash"):
        parse_training_config(raw)
    raw = _training_raw()
    raw["lora"]["keys"] = []
    with pytest.raises(ConfigError, match="keys must not be empty"):
        parse_training_config(raw)
    raw = _training_raw()
    raw["head"]["heads"] = 7
    with pytest.raises(ConfigError, match="divisible"):
        parse_training_config(raw)
    raw = _training_raw()
    del raw["profiles"]["reference"]
    with pytest.raises(ConfigError, match="reference"):
        parse_training_config(raw)


def test_paths_resolve_relative_to_root(project):
    assert project.data.output_dir == project.root / "data/processed"
    assert project.publishing.run_dir == project.root / "artifacts/runs/m1_air_8gb"


def test_data_config_errors(tmp_path: Path):
    from decision_model.config import parse_data_config

    raw = yaml.safe_load((REPO_ROOT / "configs/data.yaml").read_text())
    bad = {**raw, "dataset": {**raw["dataset"], "files": {"train": "x.parquet"}}}
    with pytest.raises(ConfigError, match="must map exactly"):
        parse_data_config(bad, tmp_path)
    bad = {**raw, "training_requests": {"seed": 7, "num_requests": 5000}}
    with pytest.raises(ConfigError, match="exceeds the training rows"):
        parse_data_config(bad, tmp_path)
    bad = {**raw, "smoke": {"dir": "data/processed", "train_size": 1, "test_size": 1}}
    with pytest.raises(ConfigError, match=r"smoke\.dir must differ"):
        parse_data_config(bad, tmp_path)


def test_missing_and_invalid_files(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        read_yaml(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        read_yaml(bad)


def test_publishing_defaults_are_safe(project):
    assert project.publishing.repo_id is None
    assert project.publishing.private is True
