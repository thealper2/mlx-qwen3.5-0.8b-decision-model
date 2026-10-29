"""Command-line interface: ``decision-model <command>``.

Commands that can be expensive or have outside effects need an explicit flag:
``train --execute`` starts training, ``publish --execute`` uploads. Without the flag
both are dry runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from collections import Counter
from importlib import util as import_util
from pathlib import Path
from typing import Any

from decision_model import __version__
from decision_model.config import SPLITS, ConfigError, ProjectConfig, load_project_config
from decision_model.logging_config import setup_logging
from decision_model.utils.paths import display, resolve

log = logging.getLogger("decision_model")


# --------------------------------------------------------------------------- environment


def cmd_check_env(project: ProjectConfig, args: argparse.Namespace) -> int:
    """Report Python/OS/hardware/package/resource status; never loads a model."""
    from decision_model.training.artifacts import environment_info
    from decision_model.utils.memory import free_disk_gb, memory_snapshot

    env = environment_info()
    mem = memory_snapshot()
    apple_silicon = env["system"] == "Darwin" and env["machine"] == "arm64"
    disk = free_disk_gb(project.root)
    rows: list[tuple[str, str, bool | None]] = [
        ("Python", env["python"], sys.version_info >= (3, 11)),
        ("OS", f"{env['system']} {env['os_version']}", env["system"] == "Darwin"),
        ("Apple Silicon", str(env.get("chip") or env["machine"]), apple_silicon),
        ("Unified memory", f"{mem.total_gb:.1f} GB" if mem.total_gb else "unknown", None),
        (
            "Approx. available",
            f"{mem.available_gb:.1f} GB (changes constantly)" if mem.available_gb else "unknown",
            None,
        ),
        ("Free disk", f"{disk:.1f} GB", disk >= project.training.safety.min_free_disk_gb),
    ]
    required = {"huggingface-hub", "pyyaml", "pyarrow"} | (
        {"mlx", "mlx-lm"} if apple_silicon else set()
    )
    for pkg, version in env["packages"].items():
        rows.append((pkg, version or "not installed", bool(version) if pkg in required else None))
    mlx_device = None
    if apple_silicon and import_util.find_spec("mlx") is not None:
        try:
            import mlx.core as mx

            mlx_device = str(mx.default_device())
        except Exception as exc:  # report, do not crash the check
            mlx_device = f"error: {exc}"
    rows.append(
        (
            "MLX device",
            mlx_device or "unavailable",
            None if not apple_silicon else bool(mlx_device and mlx_device.startswith("Device(gpu")),
        )
    )
    rows.append(("Configuration", "valid (all four files)", True))
    token_present = bool(os.environ.get("HF_TOKEN"))
    try:
        from huggingface_hub import get_token

        token_present = token_present or get_token() is not None
    except ImportError:
        pass
    rows.append(
        (
            "Hugging Face login",
            "found (needed only for publishing)"
            if token_present
            else "none (fine: only `make publish` needs it)",
            None,
        )
    )
    failed = False
    for name, value, ok in rows:
        mark = "OK  " if ok else ("FAIL" if ok is False else "INFO")
        failed |= ok is False
        print(f"[{mark}] {name:<18} {value}")
    print("\nMemory figures are approximate; MLX's usable share changes while it runs.")
    if mem.total_gb and mem.total_gb <= 8.5:
        print("8 GB machine detected: use PROFILE=m1_air_8gb for full training.")
    return 1 if failed else 0


def cmd_config_check(project: ProjectConfig, args: argparse.Namespace) -> int:
    """Print the validated configuration (loading it already validated it)."""
    t = project.training
    d = project.data.dataset
    print(f"Model   : {t.model.repo_id} @ {t.model.revision}")
    rows = ", ".join(f"{k} {v}" for k, v in d.expected_rows.items())
    print(f"Dataset : {d.repo_id} @ {d.revision} ({rows})")
    print(f"Head    : {json.dumps(t.head.to_dict())}")
    print(
        f"LoRA    : rank {t.lora.rank}, alpha {t.lora.alpha:g}, dropout {t.lora.dropout:g}, "
        f"{len(t.lora.keys)} projection types"
    )
    names = [args.profile] if args.profile else sorted(t.profiles)
    for name in names:
        p = t.profile(name)
        print(
            f"Profile {name:<11}: kind={p.kind} steps={p.steps} batch={p.batch_size} "
            f"accum={p.grad_accumulation_steps} lr={p.lr_lora:g}/{p.lr_head:g} "
            f"warmup={p.warmup_steps} clip={p.max_grad_norm:g} -> {p.output_dir}"
        )
    print("Configuration OK")
    return 0


# --------------------------------------------------------------------------- data


def _splits(project: ProjectConfig, offline: bool) -> dict[str, list[dict[str, Any]]]:
    from decision_model.data.download import load_splits

    return load_splits(project.data.dataset, local_files_only=offline)


def cmd_data_download(project: ProjectConfig, args: argparse.Namespace) -> int:
    splits = _splits(project, args.offline)
    print(
        f"{project.data.dataset.repo_id}: "
        + ", ".join(f"{k} {len(v)} rows" for k, v in splits.items())
    )
    return 0


def cmd_data_inspect(project: ProjectConfig, args: argparse.Namespace) -> int:
    splits = _splits(project, args.offline)
    summary = {}
    for name, rows in splits.items():
        chars = sorted(len(r["complaint"]) for r in rows)
        summary[name] = {
            "rows": len(rows),
            "labels": dict(sorted(Counter(r["product"] for r in rows).items())),
            "duplicate_ids": len(rows) - len({r["id"] for r in rows}),
            "duplicate_texts": len(rows) - len({r["complaint"] for r in rows}),
            "complaint_chars": {
                "min": chars[0],
                "median": statistics.median(chars),
                "max": chars[-1],
            },
            "example": {k: rows[0][k] for k in ("id", "product", "complaint")},
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_data_prepare(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.data.prepare import prepare_dataset

    manifest = prepare_dataset(
        _splits(project, args.offline), project.data, project.root, dry_run=args.dry_run
    )
    print(json.dumps({k: manifest[k] for k in ("rows", "overlap", "training_stream")}, indent=2))
    for name, info in manifest["files"].items():
        print(f"{name:<15} {info['lines']:>5} lines -> {info['path']}")
    return 0


def cmd_data_smoke(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.data.prepare import prepare_smoke_subset

    manifest = prepare_smoke_subset(project.data, project.root)
    for name, info in manifest["files"].items():
        print(f"{name:<15} {info['lines']:>3} lines -> {info['path']}")
    return 0


def validate_prepared(project: ProjectConfig, smoke: bool) -> list[str]:
    """Validate prepared files; returns a list of problems (empty = valid)."""
    from decision_model.data.prepare import REQUESTS_FILE
    from decision_model.data.validate import (
        overlap_report,
        read_jsonl,
        request_line_problems,
        row_problems,
        validate_jsonl,
    )

    d = project.data
    folder = d.smoke_dir if smoke else d.output_dir
    split_names = ["test"] if smoke else list(SPLITS)
    reports = [validate_jsonl(folder / f"{s}.jsonl", row_problems) for s in split_names]
    reports.append(validate_jsonl(folder / REQUESTS_FILE, request_line_problems))
    problems = []
    for r in reports:
        status = "OK" if r.ok else f"{r.error_count} problem(s)"
        print(f"{display(r.path, project.root):<38} {r.lines:>5} lines  {status}")
        problems += [f"{display(r.path, project.root)} {e}" for e in r.errors]
    if problems:
        return problems
    if smoke:
        return problems
    splits = {s: read_jsonl(folder / f"{s}.jsonl") for s in SPLITS}
    overlap = overlap_report(splits)
    print(f"Overlap between splits (ids, identical texts): {overlap}")
    if any(v["ids"] or v["texts"] for v in overlap.values()):
        problems.append(f"splits overlap: {overlap}")
    stream = read_jsonl(folder / REQUESTS_FILE)
    test_ids = {r["id"] for r in splits["test"]}
    if any(s["id"] in test_ids for s in stream):
        problems.append("the training stream contains test complaints")
    for s in SPLITS:
        if len(splits[s]) != d.dataset.expected_rows[s]:
            problems.append(f"{s}: {len(splits[s])} rows, expected {d.dataset.expected_rows[s]}")
    if len(stream) != d.requests.num_requests:
        problems.append(
            f"training stream has {len(stream)} requests, expected {d.requests.num_requests}"
        )
    return problems


def cmd_data_validate(project: ProjectConfig, args: argparse.Namespace) -> int:
    problems = validate_prepared(project, args.smoke)
    for p in problems:
        print(f"PROBLEM: {p}")
    print("Data valid" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


# --------------------------------------------------------------------------- training


def cmd_train(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.training.trainer import describe_plan, execute_training, plan_training

    plan = plan_training(project, args.profile, args.output_dir)
    print(describe_plan(plan, project), flush=True)
    if not plan.ok:
        print("\nTraining blocked; fix the problems above.")
        return 1
    if not args.execute:
        print(
            "\nDry run only: no model was loaded and nothing was trained. Add --execute to train."
        )
        return 0
    manifest = execute_training(plan, project)
    return 0 if manifest["status"] == "completed" else 1


# --------------------------------------------------------------------------- inference


def _run_dir(project: ProjectConfig, value: str | None) -> Path:
    return resolve(value, project.root) if value else project.publishing.run_dir


def cmd_decide(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.inference import decide_batch, load_decision_model, peak_memory_gb
    from decision_model.task import ROUTES, build_request

    routes = None
    if args.options:
        keys = [k.strip() for k in args.options.split(",") if k.strip()]
        unknown = [k for k in keys if k not in ROUTES]
        if unknown or len(keys) < 2:
            raise ConfigError(f"--options needs at least two of {list(ROUTES)}; got {keys}")
        routes = {k: ROUTES[k] for k in keys}
    m = project.training.model
    run_dir = _run_dir(project, args.run_dir)
    model, tok = load_decision_model(m.repo_id, m.revision, run_dir)
    request = build_request(args.complaint, routes)
    answers, seconds = decide_batch(model, tok, [request], project.training.max_length)
    response = {
        "model": request["model"],
        "answers": answers[0],
        "usage": {"output_tokens": 0},
        "latency_ms": round(1000 * seconds, 1),
        "peak_memory_gb": peak_memory_gb(),
        "run_dir": display(run_dir, project.root),
    }
    text = json.dumps(response, indent=2)
    if args.output:
        out = resolve(args.output, project.root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_chat(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.inference import chat_reply, load_base_model
    from decision_model.task import parse_chat_reply

    m = project.training.model
    model, tok = load_base_model(m.repo_id, m.revision)
    reply = chat_reply(
        model, tok, args.complaint, args.max_tokens, project.evaluation.enable_thinking
    )
    print(
        json.dumps(
            {
                "mode": "base model, chat (System 2)",
                "raw_output": reply.raw,
                "product": parse_chat_reply(reply.raw),
                "latency_ms": round(1000 * reply.latency_s, 1),
            },
            indent=2,
        )
    )
    return 0


def cmd_evaluate(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.data.prepare import sha256_file
    from decision_model.data.validate import read_jsonl
    from decision_model.evaluation.evaluator import (
        build_result,
        evaluate_chat,
        evaluate_decisions,
        save_result,
    )
    from decision_model.inference import (
        chat_reply,
        decide_batch,
        load_base_model,
        load_decision_model,
        peak_memory_gb,
    )
    from decision_model.training.artifacts import RUN_MANIFEST, environment_info, read_json

    ev = project.evaluation
    test_file = ev.smoke_test_file if args.smoke else ev.test_file
    if not test_file.is_file():
        raise FileNotFoundError(
            f"{display(test_file, project.root)} not found; run "
            f"`make {'data-smoke' if args.smoke else 'data-prepare'}`"
        )
    limit = args.max_examples or (ev.smoke_max_examples if args.smoke else ev.max_examples)
    rows = read_jsonl(test_file)
    rows = rows[:limit] if limit else rows
    m = project.training.model
    meta: dict[str, Any] = {
        "model": {"repo_id": m.repo_id, "revision": m.revision},
        "test_file": display(test_file, project.root),
        "test_file_sha256": sha256_file(test_file),
    }
    if args.mode == "decision":
        run_dir = _run_dir(project, args.run_dir)
        model, tok = load_decision_model(m.repo_id, m.revision, run_dir)
        run = read_json(run_dir / RUN_MANIFEST) or {}
        meta["mode"] = "System 1: joint schema head, one forward pass, no generation"
        meta["run"] = {
            "dir": display(run_dir, project.root),
            "run_id": run.get("run_id"),
            "profile": run.get("profile", {}).get("name"),
        }
        results = evaluate_decisions(
            rows,
            lambda reqs: decide_batch(model, tok, reqs, project.training.max_length),
            ev.batch_size,
        )
        label = "decision"
    else:
        model, tok = load_base_model(m.repo_id, m.revision)
        meta["mode"] = (
            f"System 2: base model as chat, greedy, max_tokens={ev.chat_max_tokens}, "
            f"thinking={'on' if ev.enable_thinking else 'off'}"
        )
        meta["run"] = None

        def chat(complaint: str) -> tuple[str, float]:
            reply = chat_reply(model, tok, complaint, ev.chat_max_tokens, ev.enable_thinking)
            return reply.raw, reply.latency_s

        results = evaluate_chat(rows, chat)
        label = "base-chat"
    label += "-smoke" if args.smoke else ""
    meta["peak_memory_gb"] = peak_memory_gb()
    meta["environment"] = environment_info()
    result = build_result(label, results, meta)
    json_path, md_path = save_result(result, ev.output_dir)
    mt = result["metrics"]
    print(
        f"{label}: accuracy {mt['correct']}/{mt['examples']} ({mt['accuracy']}), "
        f"invalid {mt['invalid_answers']}, mean latency {mt['mean_latency_ms']} ms"
    )
    print(f"Results: {display(json_path, project.root)}, {display(md_path, project.root)}")
    return 0


def cmd_report(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.reporting import write_experiment_report

    path = write_experiment_report(project)
    print(path.read_text())
    print(f"Written to {display(path, project.root)}")
    return 0


# --------------------------------------------------------------------------- publishing


def cmd_card(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.publishing.model_card import collect_inputs, training_status, write_card

    run_dir = resolve(args.run_dir, project.root) if args.run_dir else None
    inputs = collect_inputs(project, run_dir, args.repo_id)
    out = resolve(args.output, project.root) if args.output else project.publishing.model_card_path
    write_card(inputs, out)
    print(
        f"Model card written to {display(out, project.root)} "
        f"(training status: {training_status(inputs.run)}; chat baseline: "
        f"{'yes' if inputs.chat_eval else 'pending'}; decision eval: "
        f"{'yes' if inputs.decision_eval else 'pending'})"
    )
    return 0


def cmd_publish(project: ProjectConfig, args: argparse.Namespace) -> int:
    from decision_model.publishing.huggingface import (
        PublishBlocked,
        describe_upload,
        plan_upload,
        publish,
    )

    run_dir = resolve(args.run_dir, project.root) if args.run_dir else None
    plan = plan_upload(project, args.repo_id, run_dir)
    print(describe_upload(plan, project))
    if not plan.ok:
        print("\nPublication blocked; nothing was uploaded.")
        return 1
    try:
        url = publish(plan, project, execute=args.execute)
    except PublishBlocked as exc:
        print(f"BLOCKED: {exc}")
        return 1
    print(f"Uploaded: {url}" if url else "\nDry run OK: staged locally, nothing uploaded.")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decision-model",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--config-dir", type=Path, help="alternative configs/ directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-env", help="check Python, macOS, MLX, packages, memory, disk")
    p = sub.add_parser("config-check", help="validate configs and print training profiles")
    p.add_argument("--profile")

    data = sub.add_parser("data", help="dataset commands").add_subparsers(
        dest="data_command", required=True
    )
    for name, text in (
        ("download", "download (or reuse cached) dataset files"),
        ("inspect", "summary statistics of the three splits"),
        ("prepare", "write splits and the ordered training-request stream"),
    ):
        p = data.add_parser(name, help=text)
        p.add_argument("--offline", action="store_true", help="use the local cache only")
        if name == "prepare":
            p.add_argument("--dry-run", action="store_true", help="compute stats, write nothing")
    data.add_parser("smoke", help="write the tiny smoke subset (canonical data untouched)")
    p = data.add_parser("validate", help="validate prepared data files")
    p.add_argument("--smoke", action="store_true", help="validate the smoke subset instead")

    p = sub.add_parser("train", help="plan (default) or run LoRA + head training")
    p.add_argument("--profile", required=True, help="training profile from training.yaml")
    p.add_argument("--output-dir", help="override the profile's run directory")
    p.add_argument(
        "--execute", action="store_true", help="actually train (without it this is a dry run)"
    )

    p = sub.add_parser("decide", help="route one complaint with the trained decision model")
    p.add_argument("--complaint", required=True)
    p.add_argument("--options", help="comma-separated subset of routes to choose from")
    p.add_argument("--run-dir", help="training run directory (default: publishing.run_dir)")
    p.add_argument("--output", help="also write the response as JSON to this file")

    p = sub.add_parser("chat", help="ask the base model the same question as a chat")
    p.add_argument("--complaint", required=True)
    p.add_argument("--max-tokens", type=_positive_int, default=16)

    p = sub.add_parser("evaluate", help="evaluate the decision model or the chat baseline")
    p.add_argument("--mode", choices=("decision", "chat"), required=True)
    p.add_argument("--run-dir", help="training run directory (decision mode)")
    p.add_argument("--smoke", action="store_true", help="tiny deterministic subset")
    p.add_argument("--max-examples", type=_positive_int)

    sub.add_parser("report", help="summarise recorded training runs and evaluations")

    p = sub.add_parser("card", help="generate the Hugging Face model card locally")
    p.add_argument("--run-dir")
    p.add_argument("--repo-id")
    p.add_argument("--output", help="output path (default: publishing.model_card_path)")

    p = sub.add_parser("publish", help="dry run (default) or upload the run to the Hub")
    p.add_argument("--repo-id", help="override publishing.repo_id")
    p.add_argument("--run-dir")
    p.add_argument("--execute", action="store_true", help="actually upload")
    return parser


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


_COMMANDS = {
    "check-env": cmd_check_env,
    "config-check": cmd_config_check,
    "train": cmd_train,
    "decide": cmd_decide,
    "chat": cmd_chat,
    "evaluate": cmd_evaluate,
    "report": cmd_report,
    "card": cmd_card,
    "publish": cmd_publish,
}
_DATA_COMMANDS = {
    "download": cmd_data_download,
    "inspect": cmd_data_inspect,
    "prepare": cmd_data_prepare,
    "smoke": cmd_data_smoke,
    "validate": cmd_data_validate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        project = load_project_config(config_dir=args.config_dir)
        handler = (
            _DATA_COMMANDS[args.data_command] if args.command == "data" else _COMMANDS[args.command]
        )
        return handler(project, args)
    except (ConfigError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    except (RuntimeError, ValueError) as exc:  # data, training, inference, publishing errors
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
