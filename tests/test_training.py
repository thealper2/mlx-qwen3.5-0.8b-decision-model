from __future__ import annotations

import math

import pytest

from decision_model.training import trainer
from decision_model.training.loop import adapter_config, schedule_value


def test_schedule_matches_the_recipe():
    # lr x min(1, (step+1)/20) x 0.5 (1 + cos(pi step / 100)), step counted from 0
    assert schedule_value(2e-4, 20, 100, 0) == pytest.approx(2e-4 / 20)
    assert schedule_value(2e-4, 20, 100, 19) == pytest.approx(
        2e-4 * 0.5 * (1 + math.cos(math.pi * 19 / 100))
    )
    assert schedule_value(1e-3, 20, 100, 50) == pytest.approx(1e-3 * 0.5)
    assert schedule_value(1e-3, 20, 100, 99) < 1e-6
    assert schedule_value(1e-3, 0, 10, 0) == pytest.approx(1e-3)


def test_adapter_config_is_mlx_lm_compatible(project):
    cfg = adapter_config(project)
    assert cfg["fine_tune_type"] == "lora" and cfg["num_layers"] == -1
    assert cfg["lora_parameters"] == {
        "rank": 16,
        "scale": 2.0,
        "dropout": 0.05,
        "keys": list(project.training.lora.keys),
    }
    assert cfg["base_model_revision"] == project.training.model.revision


def test_plan_is_side_effect_free_and_reports_problems(project):
    plan = trainer.plan_training(project, "reference")
    assert not plan.ok and "problem" in plan.problems[0]  # nothing prepared yet
    assert not plan.output_dir.exists()


def test_plan_needs_enough_requests(prepared_project):
    assert trainer.plan_training(prepared_project, "reference").ok  # 2 x 8 = 16 available
    plan = trainer.plan_training(prepared_project, "smoke")
    assert plan.ok and plan.available_requests == 2
    assert "Updates          : 2 x (batch 1 x accumulate 1)" in trainer.describe_plan(
        plan, prepared_project
    )


def test_existing_run_is_never_overwritten(prepared_project):
    plan = trainer.plan_training(prepared_project, "smoke")
    plan.output_dir.mkdir(parents=True)
    (plan.output_dir / "head.safetensors").write_bytes(b"x")
    blocked = trainer.plan_training(prepared_project, "smoke")
    assert not blocked.ok and "never overwritten" in blocked.problems[0]
    with pytest.raises(trainer.TrainingBlocked):
        trainer.execute_training(blocked, prepared_project)


@pytest.mark.macos
def test_mlx_schedule_equals_python_twin():
    pytest.importorskip("mlx.core", reason="needs MLX (Apple Silicon)")
    from decision_model.training.loop import lr_schedule

    fn = lr_schedule(2e-4, 20, 100)
    for step in (0, 5, 19, 20, 60, 99):
        assert float(fn(step)) == pytest.approx(
            schedule_value(2e-4, 20, 100, step), rel=1e-5, abs=1e-10
        )
