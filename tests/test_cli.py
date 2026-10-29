from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_model import cli
from decision_model.publishing import huggingface
from decision_model.training import trainer
from decision_model.training.artifacts import RUN_MANIFEST, write_json


def _forbid(*_args, **_kwargs):
    raise AssertionError("must not be called")


def test_parser_validates_arguments():
    parse = cli.build_parser().parse_args
    with pytest.raises(SystemExit):
        parse(["train"])
    with pytest.raises(SystemExit):
        parse(["evaluate", "--mode", "generate"])
    with pytest.raises(SystemExit):
        parse(["chat", "--complaint", "x", "--max-tokens", "0"])
    with pytest.raises(SystemExit):
        parse(["decide"])


def test_config_check(project, repo_root_env, capsys):
    repo_root_env(project.root)
    assert cli.main(["config-check"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3.5-0.8B" in out and "Configuration OK" in out


def test_train_without_execute_is_a_dry_run(prepared_project, repo_root_env, monkeypatch, capsys):
    repo_root_env(prepared_project.root)
    monkeypatch.setattr(trainer, "execute_training", _forbid)
    assert cli.main(["train", "--profile", "reference"]) == 0
    out = capsys.readouterr().out
    assert "Dry run only" in out and "lr LoRA 0.0002 / head 0.001" in out
    assert not (prepared_project.root / "artifacts/runs/reference").exists()


def test_train_blocked_without_data(project, repo_root_env, monkeypatch):
    repo_root_env(project.root)
    monkeypatch.setattr(trainer, "execute_training", _forbid)
    assert cli.main(["train", "--profile", "smoke", "--execute"]) == 1


def test_execute_flag_reaches_training(prepared_project, repo_root_env, monkeypatch):
    repo_root_env(prepared_project.root)
    calls = []
    monkeypatch.setattr(
        trainer,
        "execute_training",
        lambda plan, project: calls.append(plan.profile.name) or {"status": "completed"},
    )
    assert cli.main(["train", "--profile", "smoke", "--execute"]) == 0
    assert calls == ["smoke"]


def test_decide_rejects_unknown_options_before_loading(
    prepared_project, repo_root_env, monkeypatch
):
    repo_root_env(prepared_project.root)
    import decision_model.inference as inference

    monkeypatch.setattr(inference, "load_decision_model", _forbid)
    assert cli.main(["decide", "--complaint", "x", "--options", "mortgage,insurance"]) == 2
    assert cli.main(["decide", "--complaint", "x", "--options", "mortgage"]) == 2


def _fake_run(root: Path, kind: str = "full", status: str = "completed") -> Path:
    run = root / "artifacts/runs/m1_air_8gb"
    run.mkdir(parents=True)
    for name in ("adapters.safetensors", "head.safetensors"):
        (run / name).write_bytes(b"\0" * 8)
    write_json(
        run / "adapter_config.json",
        {"model": "/Users/someone/.cache/huggingface/hub/x", "num_layers": -1},
    )
    write_json(run / "head_config.json", {"width": 512})
    write_json(run / RUN_MANIFEST, {"run_id": "r1", "status": status, "profile": {"kind": kind}})
    (run / ".env").write_text("HF_TOKEN=hf_" + "a" * 30)
    return run


def test_publish_requires_destination(prepared_project, repo_root_env, capsys):
    repo_root_env(prepared_project.root)
    _fake_run(prepared_project.root)
    assert cli.main(["card"]) == 0
    assert cli.main(["publish"]) == 1
    assert "No destination" in capsys.readouterr().out


def test_publish_dry_run_stages_without_uploading(
    prepared_project, repo_root_env, monkeypatch, capsys
):
    repo_root_env(prepared_project.root)
    _fake_run(prepared_project.root)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _forbid)
    assert cli.main(["card"]) == 0
    assert cli.main(["publish", "--repo-id", "me/router"]) == 0
    assert "Dry run OK" in capsys.readouterr().out
    staged = prepared_project.publishing.staging_dir
    assert sorted(p.name for p in staged.iterdir()) == [
        "README.md",
        "adapter_config.json",
        "adapters.safetensors",
        "decision_model_runtime.py",
        "head.safetensors",
        "head_config.json",
        "run_manifest.json",
    ]
    config = json.loads((staged / "adapter_config.json").read_text())
    assert config["model"] == "Qwen/Qwen3.5-0.8B"  # local path removed
    runtime = (staged / "decision_model_runtime.py").read_text()
    assert "import decision_model" not in runtime and "from decision_model" not in runtime


def test_publish_execute_requires_login(prepared_project, repo_root_env, monkeypatch):
    repo_root_env(prepared_project.root)
    _fake_run(prepared_project.root)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", _forbid)
    assert cli.main(["card"]) == 0
    assert cli.main(["publish", "--repo-id", "me/router", "--execute"]) == 1


@pytest.mark.parametrize(("kind", "status"), [("smoke", "completed"), ("full", "interrupted")])
def test_smoke_or_unfinished_runs_are_not_publishable(
    prepared_project, repo_root_env, kind, status
):
    repo_root_env(prepared_project.root)
    _fake_run(prepared_project.root, kind=kind, status=status)
    assert cli.main(["card"]) == 0
    assert not huggingface.plan_upload(prepared_project, "me/router").ok


def test_report_without_runs(prepared_project, repo_root_env, capsys):
    repo_root_env(prepared_project.root)
    assert cli.main(["report"]) == 0
    out = capsys.readouterr().out
    assert "| reference | not run |" in out and "| decision | – | – | not evaluated |" in out
