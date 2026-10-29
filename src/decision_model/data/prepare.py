"""Write the prepared splits and the ordered training-request stream.

Outputs (canonical, ``data/processed``)::

    train.jsonl / validation.jsonl / test.jsonl   {id, complaint, product}, file order
    train_requests.jsonl                          the ordered training stream (see below)
    manifest.json                                 provenance, counts, overlap, file hashes

The training stream is the exact sequence of decision requests the trainer consumes:
``random.Random(seed)``; one augmentation draw on the first training row (discarded);
``rng.sample(train, num_requests)``; then one option-subset draw per sampled row, in order.
Every profile reads the same stream in the same order, so all profiles see the same
requests in the same updates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_model.config import SPLITS, DataConfig
from decision_model.data.validate import overlap_report, read_jsonl, row_problems
from decision_model.task import build_request, option_subset
from decision_model.utils.paths import display

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
REQUESTS_FILE = "train_requests.jsonl"


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL atomically (temp file + rename). Returns the line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def training_stream(train: list[dict[str, Any]], seed: int, n: int) -> list[dict[str, Any]]:
    """The ordered training requests (definition in the module docstring)."""
    rng = random.Random(seed)
    option_subset(train[0]["product"], rng)  # one draw before sampling, discarded
    rows = rng.sample(train, n)
    return [
        {
            "id": r["id"],
            "product": r["product"],
            "request": build_request(r["complaint"], option_subset(r["product"], rng)),
        }
        for r in rows
    ]


def prepare_dataset(
    splits: dict[str, list[dict[str, Any]]], cfg: DataConfig, root: Path, dry_run: bool = False
) -> dict[str, Any]:
    """Validate the splits, build the training stream and write everything."""
    for name in SPLITS:
        bad = [i for i, r in enumerate(splits[name]) if row_problems(r)]
        if bad:
            raise ValueError(
                f"{name}: {len(bad)} invalid rows, e.g. row {bad[0]}: "
                f"{row_problems(splits[name][bad[0]])}"
            )
        if len(splits[name]) != cfg.dataset.expected_rows[name]:
            raise ValueError(
                f"{name} has {len(splits[name])} rows, expected "
                f"{cfg.dataset.expected_rows[name]}; check the revision"
            )
    stream = training_stream(splits["train"], cfg.requests.seed, cfg.requests.num_requests)
    subset_share = sum(
        len(s["request"]["questions"]["product"]["criteria"]) < 9 for s in stream
    ) / len(stream)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "repo_id": cfg.dataset.repo_id,
            "revision": cfg.dataset.revision,
            "license": cfg.dataset.license,
        },
        "rows": {n: len(splits[n]) for n in SPLITS},
        "labels": {
            n: dict(sorted(Counter(r["product"] for r in splits[n]).items())) for n in SPLITS
        },
        "overlap": overlap_report(splits),
        "training_stream": {
            "seed": cfg.requests.seed,
            "requests": len(stream),
            "unique_rows": len({s["id"] for s in stream}),
            "share_with_option_subset": round(subset_share, 4),
        },
        "files": {},
    }
    if dry_run:
        log.info("Dry run: nothing written")
        return manifest
    out = cfg.output_dir
    targets = {n: out / f"{n}.jsonl" for n in SPLITS}
    for name, path in targets.items():
        write_jsonl(path, splits[name])
    targets["train_requests"] = out / REQUESTS_FILE
    write_jsonl(targets["train_requests"], stream)
    sizes = {**{n: len(splits[n]) for n in SPLITS}, "train_requests": len(stream)}
    manifest["files"] = {
        name: {"path": display(path, root), "sha256": sha256_file(path), "lines": sizes[name]}
        for name, path in targets.items()
    }
    _write_json(out / MANIFEST, manifest)
    return manifest


def prepare_smoke_subset(cfg: DataConfig, root: Path) -> dict[str, Any]:
    """The head of the training stream and of the test split, written to the smoke dir."""
    sources = {
        "train_requests": cfg.output_dir / REQUESTS_FILE,
        "test": cfg.output_dir / "test.jsonl",
    }
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Canonical prepared data not found ({', '.join(missing)}). Run `make data-prepare`."
        )
    sizes = {"train_requests": cfg.smoke.train_size, "test": cfg.smoke.test_size}
    files = {}
    for name, src in sources.items():
        rows = read_jsonl(src)[: sizes[name]]
        target = cfg.smoke_dir / src.name
        write_jsonl(target, rows)
        files[name] = {"path": display(target, root), "lines": len(rows)}
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": "smoke-subset",
        "derived_from": display(cfg.output_dir / MANIFEST, root),
        "files": files,
    }
    _write_json(cfg.smoke_dir / MANIFEST, manifest)
    return manifest


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
