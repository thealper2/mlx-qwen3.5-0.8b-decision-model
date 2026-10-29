# Troubleshooting

Start with `make check-env`; it reports most environment problems in one go.

## Installation and dependencies

**`uv: command not found`** — install uv (`brew install uv`), then `make install`.

**`uv sync --frozen` fails because the lock is out of date** — `pyproject.toml` was edited
without re-locking. Run `uv lock`, review the diff of `uv.lock`, commit it.

**`mlx-lm is not installed` / `MLX-LM needs macOS on Apple Silicon`** — MLX only ships
wheels for Apple Silicon macOS. Elsewhere the package installs without MLX so most unit
tests run, but training, inference and evaluation are unavailable by design.

## Hugging Face access

**`The dataset is not in the local Hugging Face cache…`** — you used `--offline` or
`HF_HUB_OFFLINE=1` before the first download, or the Hub is unreachable. Run
`make data-download` with network access.

**Rate-limit warnings ("unauthenticated requests")** — harmless; `uv run hf auth login`
with a read token raises the limits.

## Data

**`… has N rows, expected …`** — the dataset revision in `configs/data.yaml` was changed.
Restore it, or update `expected_rows` deliberately (that is a new experiment).

**`training stream on disk differs from its definition`** (smoke test) — `task.py`,
`data/prepare.py` or the seed changed after `make data-prepare`. Re-run it and treat the
result as a new experiment.

**`Canonical prepared data not found`** — run `make data-prepare` before `make data-smoke`.

## Training

**`… already holds a trained run`** — runs are never overwritten. Use
`--output-dir artifacts/runs/<new-name>` or move the old directory. For the disposable
smoke run: `make clean-artifacts`.

**`[scaled_dot_product_attention] the vector kernels do not produce a logsumexp`** — some
code path uses MLX's fused attention with very short queries during training. The head in
`runtime.py` uses explicit softmax attention for this reason; keep it that way.

**Training is very slow** — expected: MLX-LM trains Qwen3.5's linear-attention layers
without their fused kernel (≈10–20 s per request on an M1). If it is much slower, the
machine is swapping: quit other apps, check Activity Monitor → Memory, keep batch size 1
and gradient checkpointing on.

**Memory pressure / `[METAL] Insufficient Memory`** — gradient checkpointing must stay on
(without it one request needs ~12 GB). Use `PROFILE=m1_air_8gb`.

**A run was stopped** — it is marked `interrupted`; checkpoints in
`artifacts/runs/<profile>/checkpoints/` can be evaluated with `--run-dir`, but training
cannot resume from them. Start a new run in a new directory.

**Disk space** — the base model needs ≈1.75 GB in the Hugging Face cache; each checkpoint
≈104 MB (10 for a full `m1_air_8gb` run).

## Inference and evaluation

**`… is not a trained run (missing …)`** — train first, or point `RUN_DIR=` / `--run-dir`
at a directory written by `decision-model train` (or one of its checkpoints).

**`--options needs at least two of …`** — `decide --options` takes a comma-separated list of
route ids, at least two.

**Chat baseline answers look truncated** — 16 new tokens is the baseline's definition; the
JSON answer fits in about 8.

## Publishing

**`No destination`** — set `repo_id` in `configs/publishing.yaml` or pass `REPO_ID=…`.

**`This is a smoke-test run`** — by design; publish a full training run.

**`Not logged in to Hugging Face`** — `uv run hf auth login` with a write token.

**`Staged files still contain sensitive data`** — a staged text file contains a token-like
string or your home-directory path. Inspect `artifacts/exports/hf_upload/`; nothing was
uploaded.
