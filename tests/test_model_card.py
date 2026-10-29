from __future__ import annotations

import re

import pytest

from conftest import fake_splits
from decision_model.evaluation.evaluator import build_result, evaluate_chat, save_result
from decision_model.publishing.model_card import (
    PENDING,
    REQUIRED_SECTIONS,
    collect_inputs,
    parse_front_matter,
    render_card,
    runtime_constant,
    training_status,
    write_card,
)
from decision_model.training.artifacts import RUN_MANIFEST, write_json

RUN_ID = "m1_air_8gb-20260101-000000"


def _manifest(kind: str = "full") -> dict:
    return {
        "run_id": RUN_ID,
        "status": "completed",
        "profile": {
            "name": "m1_air_8gb",
            "kind": kind,
            "steps": 100,
            "batch_size": 1,
            "grad_accumulation_steps": 8,
            "lr_lora": 2e-4,
            "lr_head": 1e-3,
            "weight_decay": 0.01,
            "warmup_steps": 20,
            "max_grad_norm": 1.0,
            "grad_checkpoint": True,
            "seed": 7,
        },
        "metrics": {
            "final_loss_avg10": 0.4321,
            "train_wall_time_s": 9876.5,
            "peak_memory_gb": 4.3,
            "trainable_parameters": {
                "lora": 10223616,
                "head": 15763460,
                "total_backbone": 762616640,
            },
        },
        "environment": {
            "chip": "Apple M1",
            "memory_gb": 8,
            "os_version": "27.0",
            "python": "3.12.15",
            "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2"},
        },
    }


def _record(project, label: str, run_id: str | None, correct: int) -> None:
    rows = fake_splits()["test"][:4]
    replies = iter(
        [f'{{"product": "{r["product"]}"}}' if i < correct else "?" for i, r in enumerate(rows)]
    )
    results = evaluate_chat(rows, lambda c: (next(replies), 0.1))
    meta = {
        "model": {
            "repo_id": project.training.model.repo_id,
            "revision": project.training.model.revision,
        },
        "run": {"run_id": run_id} if run_id else None,
    }
    save_result(build_result(label, results, meta), project.evaluation.output_dir)


def test_untrained_card_has_all_sections_and_no_numbers(project):
    card = render_card(collect_inputs(project))
    for section in REQUIRED_SECTIONS:
        assert section in card
    assert "**Training status: not trained.**" in card
    assert card.count(PENDING) == 2
    rows = [ln for ln in card.splitlines() if PENDING in ln]
    assert len(rows) == 2 and not any(re.search(r"\d+\.\d%", ln) for ln in rows)
    assert "| Final training loss (mean of last 10 updates) | not measured |" in card
    assert "| Hardware / OS | not recorded (no training run yet) |" in card


def test_metadata_is_valid_and_derived_from_config(project):
    meta = parse_front_matter(render_card(collect_inputs(project)))
    assert meta["base_model"] == "Qwen/Qwen3.5-0.8B"
    assert meta["datasets"] == ["kgptalkie/cfpb-complaint-routing"]
    assert meta["library_name"] == "mlx"
    assert meta["license"] == project.publishing.adapter_license


def test_card_credits_the_head_design_and_explains_artifacts(project):
    card = render_card(collect_inputs(project))
    assert "Clef" in card and "not affiliated" in card
    assert "not a standalone model" in card
    assert runtime_constant("SYSTEM_PROMPT") in card


def test_usage_example_uses_the_shipped_runtime(project):
    card = render_card(collect_inputs(project, repo_id="me/router"))
    usage = card.split("## How to use")[1].split("## Limitations")[0]
    code = re.findall(r"```python\n(.*?)```", usage, re.S)[0]
    compile(code, "usage", "exec")  # valid Python
    assert 'snapshot_download("me/router")' in code
    assert "from decision_model_runtime import decide, load_release" in code
    assert f'"{project.training.model.revision}"' in code
    assert "transformers" not in code and "torch" not in code


def test_trained_card_reports_recorded_values_only(project):
    write_json(project.publishing.run_dir / RUN_MANIFEST, _manifest())
    _record(project, "base-chat", None, correct=1)
    _record(project, "decision", RUN_ID, correct=3)
    card = render_card(collect_inputs(project))
    assert "**Training status: completed.**" in card
    assert "| Final training loss (mean of last 10 updates) | 0.4321 |" in card
    assert "LoRA 10,223,616, head 15,763,460" in card
    assert "1 / 4 (25.0%)" in card and "3 / 4 (75.0%)" in card


def test_decision_eval_from_another_run_is_ignored(project):
    write_json(project.publishing.run_dir / RUN_MANIFEST, _manifest())
    _record(project, "decision", "other-run", correct=4)
    inputs = collect_inputs(project)
    assert inputs.decision_eval is None
    assert "4 / 4 (100.0%)" not in render_card(inputs)


def test_training_status_values():
    assert training_status(None) == "not trained"
    assert training_status(_manifest("smoke")) == "smoke test only"
    assert training_status({**_manifest(), "status": "interrupted"}) == "interrupted"


def test_write_card_never_overwrites_project_readme(project):
    inputs = collect_inputs(project)
    with pytest.raises(ValueError, match="Refusing"):
        write_card(inputs, project.root / "README.md")
