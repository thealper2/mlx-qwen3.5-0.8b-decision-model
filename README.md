# Qwen3.5-0.8B complaint-routing decision model with MLX

A **System 1 decision model** built on **`Qwen/Qwen3.5-0.8B`**: it reads a consumer complaint
and a typed question with its allowed options **once** — one forward pass, nothing
generated — and returns the chosen financial product with a probability for every option.
A LoRA adapter and a small joint schema head are trained together on Apple Silicon with
MLX. The baseline is the same base model answering as a chat (System 2).

The repository is an installable Python package with a CLI, YAML configs, a Makefile,
offline tests and an opt-in Hugging Face publishing workflow, built and tested for a
**MacBook Air M1 with 8 GB** of unified memory.

> **Status.** The pipeline is complete and verified with bounded smoke tests (including a
> real 2-update training run). No full training run or full evaluation has been performed
> yet, so there are no quality results in this repository.

## The experiment

| | |
|---|---|
| Task | complaint → one of `credit_reporting`, `debt_collection`, `credit_card`, `bank_account`, `mortgage`, `money_transfer`, `vehicle_loan`, `student_loan`, `personal_loan` |
| Base model | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` (Apache-2.0) |
| Data | `kgptalkie/cfpb-complaint-routing` @ `3ec70ba3690c26b8d9b117629fa3803ce4e4b84f` (CC0-1.0): 2,700 / 270 / 826 |
| Model | LoRA r 16 / alpha 32 on every attention, MLP and linear-attention projection + joint schema head (15.8M) |
| Training | 100 updates of 8 requests (half with a random subset of options), AdamW lr 2e-4 / 1e-3, warmup × cosine, clip 1.0 |
| Metric | accuracy on 826 test complaints (all 9 options), plus calibration and the chat baseline |

The head follows the design of Cloudflare's Clef joint schema model (Apache-2.0), scaled
down and text-only; this is an independent MLX implementation, not affiliated with
Cloudflare. Full definition and every implementation decision:
[docs/experiment_specification.md](docs/experiment_specification.md).

## Requirements

* macOS on Apple Silicon (M1 or later) for training, inference and evaluation. Most unit
  tests also run on Linux/Intel, without MLX.
* [uv](https://docs.astral.sh/uv/) (`brew install uv`). It installs Python 3.12.
* ≈ 2 GB disk for the model cache, plus ~1 GB for a full run with checkpoints.

### 8 GB caveats

Measured on an M1 Air 8 GB:

| Operation | Peak MLX memory | Speed |
|---|---|---|
| Training, batch 1, gradient checkpointing | 3.5–4.3 GB | ≈10–20 s per request |
| Training, batch 2 | 6.5 GB | — |
| Training, batch 1, no checkpointing | 12.6 GB (swaps) | — |
| Decision evaluation (forward only, batch 8) | 2.4 GB | ≈0.6 s per complaint |

Use the `m1_air_8gb` profile (batch 1 × 8 accumulated). A full run (800 requests) is
estimated at 2–5 hours and has not been measured; quit other apps first. MLX-LM trains
Qwen3.5's linear-attention layers without their fused kernel, which is why training is
slow.

## Quick start (no training)

```bash
make install          # .venv from uv.lock; downloads no model
make check-env        # Python, macOS, MLX, memory, disk, configs
make test             # offline unit tests
make data-prepare     # download the pinned parquet files (~1 MB), build the training stream
make data-validate
make smoke-test       # bounded end-to-end checks; never trains or uploads
make train-dry-run    # show the full training plan of the m1_air_8gb profile
```

`make help` lists every target.

## Workflow

```bash
make evaluate-chat                       # System 2 baseline on 826 complaints
make train PROFILE=m1_air_8gb            # full training — hours on an M1
make evaluate                            # System 1 decision model on 826 complaints
make report                              # reports/experiments/summary.md
```

Smoke versions: `make train-smoke` (2 real updates, ~3.5–4.3 GB), `make decide-smoke
RUN_DIR=artifacts/runs/smoke`, `make chat-smoke`, `make evaluate-smoke`. A passing smoke
test says nothing about model quality.

### Inference

```bash
uv run decision-model decide --complaint "My card was charged twice for one purchase."
uv run decision-model decide --complaint "…" --options credit_card,bank_account,money_transfer
uv run decision-model chat --complaint "…"        # the base model as a chat (System 2)
```

`decide` returns the choice, its confidence and the probability of every offered option;
`--run-dir` selects the trained run (default `artifacts/runs/m1_air_8gb`).

### Model card and Hugging Face

```bash
make card
make publish-dry-run REPO_ID=<you>/<repo>
uv run hf auth login
make publish REPO_ID=<you>/<repo>
```

A release holds the LoRA adapter, the head and `decision_model_runtime.py`, a
self-contained loader. See [docs/publishing.md](docs/publishing.md).

## Repository layout

```
configs/      data.yaml, training.yaml (reference / m1_air_8gb / smoke), evaluation.yaml, publishing.yaml
src/decision_model/
  cli.py config.py task.py runtime.py inference.py logging_config.py reporting.py
  data/        download, prepare (splits + training stream), validate
  training/    loop (MLX training loop), trainer (plan / execute), artifacts (manifests)
  evaluation/  evaluator (decision + chat), metrics
  publishing/  model_card, huggingface
  utils/       paths, memory
examples/     complaint.txt
scripts/smoke_test.py
tests/        offline pytest suite (+ macOS-only MLX tests)
docs/         experiment_specification, architecture, data, training, evaluation, publishing, troubleshooting
```

Generated, git-ignored: `data/processed`, `data/smoke`, `artifacts/`, `reports/evaluation`,
`reports/experiments`. See [docs/architecture.md](docs/architecture.md).

## Development

```bash
make lint && make typecheck && make test
```

CI (`.github/workflows/tests.yml`) runs lint, types and tests on Linux without MLX and the
full suite on macOS; it never downloads a model or trains.

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).

## License

Code: MIT ([LICENSE](LICENSE)). The base model (Apache-2.0), the data (CC0-1.0, from the
US CFPB Consumer Complaint Database) and the head design (Cloudflare Clef, Apache-2.0) keep
their own terms — see [CITATION.cff](CITATION.cff).
