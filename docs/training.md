# Training

LoRA on the language model and a freshly initialised joint schema head are trained
together with this project's MLX loop (`src/decision_model/training/loop.py`). MLX-LM's
own trainer is not used because it only optimises next-token prediction; MLX-LM still
provides model loading, the LoRA layers, gradient checkpointing and the adapter format.

## What is trained

* **LoRA** (10.2M parameters): rank 16, scale 2.0 (alpha 32 / rank 16), dropout 0.05 on
  every attention, MLP and linear-attention projection of all 24 blocks. The 762.6M base
  weights stay frozen.
* **Joint schema head** (15.8M parameters, float32): see `runtime.py` and
  `docs/experiment_specification.md`.

Loss: cross-entropy of the correct option among the offered ones. Optimizer: AdamW with
bias correction, weight decay 0.01, lr 2e-4 for LoRA and 1e-3 for the head, schedule
`lr × min(1, (step+1)/20) × ½(1+cos(π·step/100))`, gradients averaged over accumulated
micro-batches and clipped to global norm 1.0.

## Profiles (`configs/training.yaml`)

| | `reference` | `m1_air_8gb` | `smoke` |
|---|---|---|---|
| Purpose | the reference recipe | full training on an 8 GB M1 Air | prove the loop, measure memory |
| Requests | `data/processed/train_requests.jsonl` | same | `data/smoke/train_requests.jsonl` |
| Updates × batch × accumulation | 100 × 8 × 1 | 100 × 1 × 8 | 2 × 1 × 1 |
| Requests used | 800 | 800 (same order) | 2 |
| Warmup | 20 | 20 | 1 |
| Checkpoints | final only | every 10 updates + final | final only |
| Seed | 7 | 7 | 7 |
| Output | `artifacts/runs/reference` | `artifacts/runs/m1_air_8gb` | `artifacts/runs/smoke` |
| Validated on M1 8 GB | **no** | **no** (not run yet) | **yes**, see below |

## Commands

| Command | What happens |
|---|---|
| `make train-config-check` | validates all configs and plans the `reference` profile |
| `make train-dry-run PROFILE=…` | full plan; loads and trains nothing |
| `make smoke-test` | bounded checks; plans every profile, never trains |
| `make train-smoke` | real 2-update run on 2 requests |
| `make train PROFILE=m1_air_8gb` | **full training** (default profile) |

Without `--execute` the CLI only plans. A plan is blocked when the request stream is
invalid or too short for the profile, the run directory already holds a run (runs are never
overwritten; use `--output-dir`), or free disk is below `safety.min_free_disk_gb`.

## What a run leaves behind

`artifacts/runs/<profile>/`: `adapters.safetensors` (≈41 MB) + `adapter_config.json` (MLX-LM
format, so `mlx_lm.load(base, adapter_path=run_dir)` applies the LoRA), `head.safetensors`
(≈63 MB) + `head_config.json`, `train_log.jsonl` (one line per logged update), optional
`checkpoints/step_NNNN/`, and `run_manifest.json`: run id, status, timestamps, profile,
model and data provenance, LoRA and head configuration, loss / gradient norm / learning
rate / time / peak memory per update, trainable parameter counts, environment, package
versions, git commit. A copy goes to `reports/experiments/<run_id>.json`. Ctrl-C marks the
run `interrupted`.

## Measured on the target machine (MacBook Air M1, 8 GB, macOS 27.0)

| Measurement | Result |
|---|---|
| `make train-smoke` (2 updates of 1 request, ≈290 tokens) | completed; 7.9 s and 16.2 s per update; **peak MLX memory 4.3 GB**; 24 s training wall time |
| One request (246 tokens), forward + backward, **without** gradient checkpointing | 12.6 GB peak (the machine swaps heavily) |
| Same with gradient checkpointing | 3.5 GB peak |
| Two requests (batch 2, 341 tokens) with checkpointing | 6.5 GB peak |
| Forward pass only (evaluation mode), batch of 8 | ≈0.6 s per complaint, 2.4 GB peak |

Why training is slow: in training mode MLX-LM computes Qwen3.5's linear-attention layers
step by step instead of with their fused Metal kernel (`use_kernel=not self.training`),
because the fused kernel has no gradient. At ~10–20 s per request, the 800 requests of a
full `m1_air_8gb` run are **an estimated 2–5 hours** — not measured, and much longer if the
machine starts swapping.

## Memory guidance for 8 GB

* Use `m1_air_8gb` (batch 1). Batch 2 already peaks at 6.5 GB; the `reference` profile
  (batch 8) does not fit in 8 GB.
* Keep gradient checkpointing on.
* Quit other applications; never run evaluation or another model process at the same time.
* The run checkpoints every 10 updates; if it is stopped, those weights remain usable for
  evaluation (`--run-dir artifacts/runs/m1_air_8gb/checkpoints/step_NNNN`), but a stopped
  run cannot be resumed.
