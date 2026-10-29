"""Bounded end-to-end pipeline checks (``make smoke-test``).

Never trains and never uploads: the training entry point is replaced by a function that
raises, and the publication check stages a synthetic run without network access. The only
network use is downloading the ~1 MB dataset if it is not cached. Pass ``--with-model`` to
also load the base model, run the untrained head once and ask the chat baseline once.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

from decision_model.cli import validate_prepared
from decision_model.config import load_project_config
from decision_model.data.download import load_splits
from decision_model.data.prepare import prepare_dataset, prepare_smoke_subset, training_stream
from decision_model.data.validate import read_jsonl
from decision_model.evaluation.evaluator import evaluate_chat, evaluate_decisions
from decision_model.evaluation.metrics import aggregate
from decision_model.logging_config import setup_logging
from decision_model.publishing import huggingface
from decision_model.publishing.model_card import (
    REQUIRED_SECTIONS,
    collect_inputs,
    parse_front_matter,
    write_card,
)
from decision_model.task import QUESTION_ID, ROUTES
from decision_model.training import trainer
from decision_model.training.artifacts import write_json

Result = tuple[str, str]  # (status, note)


def _forbid_training(*_: object, **__: object) -> None:
    raise AssertionError("smoke test attempted to start training")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-model", action="store_true", help="also load the base model (~2-3 GB RAM)"
    )
    args = parser.parse_args()
    setup_logging()
    trainer.execute_training = _forbid_training  # type: ignore[assignment]
    project = load_project_config()
    root = project.root
    results: list[tuple[str, str, str]] = []

    def check(name: str, fn: Callable[[], Result]) -> None:
        try:
            status, note = fn()
        except Exception as exc:
            traceback.print_exc()
            status, note = "FAIL", f"{type(exc).__name__}: {exc}"
        results.append((name, status, note))
        print(f"[{status}] {name}: {note}", flush=True)

    def data() -> Result:
        d = project.data
        if not (d.output_dir / "manifest.json").is_file():
            prepare_dataset(load_splits(d.dataset), d, root)
            note = "prepared canonical data"
        else:
            note = "canonical prepared data present"
        prepare_smoke_subset(d, root)
        problems = validate_prepared(project, smoke=False)
        problems += validate_prepared(project, smoke=True)
        if problems:
            return "FAIL", "; ".join(problems[:3])
        return "PASS", note + "; files valid, no overlap between splits"

    def stream() -> Result:
        d = project.data
        train = read_jsonl(d.output_dir / "train.jsonl")
        regenerated = training_stream(train, d.requests.seed, d.requests.num_requests)
        on_disk = read_jsonl(d.output_dir / "train_requests.jsonl")
        if regenerated != on_disk:
            return "FAIL", "training stream on disk differs from its definition"
        return "PASS", f"{len(on_disk)} requests reproduce exactly from seed {d.requests.seed}"

    def dry_runs() -> Result:
        notes = []
        for name in sorted(project.training.profiles):
            plan = trainer.plan_training(project, name)
            data_problems = [p for p in plan.problems if "already holds" not in p]
            if data_problems:
                return "FAIL", f"{name}: {data_problems[0]}"
            notes.append(f"{name}={plan.profile.steps}x{plan.profile.examples_per_update}")
        return "PASS", ", ".join(notes) + " (planned only)"

    def evaluation() -> Result:
        rows = read_jsonl(project.evaluation.smoke_test_file)
        labels = {r["complaint"]: r["product"] for r in rows}

        def oracle(requests: list[dict]) -> tuple[list[dict], float]:
            answers = []
            for req in requests:
                gold = labels[req["state"]["complaint"]]
                probs = {k: (0.9 if k == gold else 0.1 / (len(ROUTES) - 1)) for k in ROUTES}
                answers.append(
                    {
                        QUESTION_ID: {
                            "type": "choice",
                            "choice": gold,
                            "confidence": 0.9,
                            "probabilities": probs,
                        }
                    }
                )
            return answers, 0.0

        good = aggregate(evaluate_decisions(rows, oracle, batch_size=3))
        bad = aggregate(evaluate_chat(rows, lambda c: ("I am not sure.", 0.0)))
        if (
            good["correct"] != len(rows)
            or bad["correct"] != 0
            or bad["invalid_answers"] != len(rows)
        ):
            return "FAIL", f"oracle={good['correct']} chat-invalid={bad['invalid_answers']}"
        return "PASS", f"oracle {len(rows)}/{len(rows)}, unparseable chat 0 (no model)"

    def card() -> Result:
        inputs = collect_inputs(project)
        out = write_card(inputs, project.publishing.model_card_path)
        text = out.read_text()
        meta = parse_front_matter(text)
        missing = [s for s in REQUIRED_SECTIONS if s not in text]
        if missing or meta["base_model"] != project.training.model.repo_id:
            return "FAIL", f"missing sections {missing}"
        return "PASS", f"{len(text):,} chars, metadata valid, status line present"

    def publish() -> Result:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            for name in ("adapters.safetensors", "head.safetensors"):
                (run / name).write_bytes(b"\0" * 16)
            write_json(
                run / "adapter_config.json",
                {"model": str(Path.home() / "cache" / "snapshot"), "num_layers": -1},
            )
            write_json(run / "head_config.json", project.training.head.to_dict())
            write_json(
                run / "run_manifest.json",
                {"run_id": "synthetic", "status": "completed", "profile": {"kind": "full"}},
            )
            staging = root / "artifacts" / "exports" / "smoke_test_upload"
            pub = dataclasses.replace(project.publishing, staging_dir=staging)
            proj = dataclasses.replace(project, publishing=pub)
            plan = huggingface.plan_upload(proj, "smoke-test/not-a-real-repo", run)
            if not plan.ok:
                return "FAIL", "; ".join(plan.problems)
            try:
                url = huggingface.publish(plan, proj, execute=False)
                staged = json.loads((staging / "adapter_config.json").read_text())
                runtime = (staging / huggingface.RUNTIME_NAME).is_file()
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        if url is not None or staged["model"] != project.training.model.repo_id or not runtime:
            return "FAIL", "dry run uploaded, kept local paths, or missed the runtime"
        return "PASS", f"{len(plan.files)} files staged incl. runtime, nothing uploaded"

    def model() -> Result:
        if not args.with_model:
            return "SKIP", "pass --with-model (or run `make chat-smoke` / `make decide-smoke`)"
        from decision_model.inference import chat_reply, load_base_model, peak_memory_gb
        from decision_model.task import parse_chat_reply

        m = project.training.model
        mdl, tok = load_base_model(m.repo_id, m.revision)
        complaint = (root / "examples" / "complaint.txt").read_text().strip()
        reply = chat_reply(mdl, tok, complaint, max_tokens=16)
        return "PASS", (
            f"chat answered {parse_chat_reply(reply.raw)!r} in {reply.latency_s}s, "
            f"peak {peak_memory_gb()} GB"
        )

    check("data pipeline", data)
    check("training stream", stream)
    check("training dry runs", dry_runs)
    check("evaluation pipeline", evaluation)
    check("model card", card)
    check("publish dry run", publish)
    check("model load + chat", model)

    print("\n| Check | Status | Notes |\n|---|---|---|")
    for name, status, note in results:
        print(f"| {name} | {status} | {note} |")
    return 1 if any(status == "FAIL" for _, status, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
