# Architecture

A small Python package (`src/decision_model`) with one CLI (`decision-model`), four YAML
config files and a Makefile on top. MLX-LM loads the base model and provides LoRA layers;
this project adds the decision head, the training loop, evaluation and publishing.

## Data flow

```
Hugging Face Hub (pinned revisions)
  kgptalkie/cfpb-complaint-routing ──► data/download.py (parquet → rows)
                                          │
                                          ▼
                                   data/prepare.py ──► data/processed/
                    (task.build_request,           train / validation / test.jsonl
                     task.option_subset, seed 7)   train_requests.jsonl  (800, ordered)
                                                   manifest.json
                                          │
            ┌─────────────────────────────┴──────────────────────────────┐
            ▼                                                            ▼
 training/trainer.py (plan → execute)                        evaluation/evaluator.py
   training/loop.py                                            decision: runtime + head,
     mlx_lm.load + LoRA (linear_to_lora_layers)                  one forward pass per batch
     runtime.encode_request → hidden states → JointSchemaHead   chat: base model generate
     AdamW ×2 (LoRA / head), schedule, clipping                  + task.parse_chat_reply
            ▼                                                            ▼
 artifacts/runs/<profile>/                                   reports/evaluation/<label>/
   adapters.safetensors, adapter_config.json (MLX-LM format)
   head.safetensors, head_config.json, run_manifest.json
            │                                                            │
            └──────────────► publishing/model_card.py ◄──────────────────┘
                                   │
                                   ▼
                       publishing/huggingface.py: plan → stage (+ runtime.py as
                       decision_model_runtime.py) → upload only with --execute
```

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | Argument parsing and one handler per command; errors become exit codes |
| `config.py` | The four YAML files as validated frozen dataclasses; paths against the repo root |
| `task.py` | The 9 routes, request building, training augmentation, chat baseline prompt and parser |
| `runtime.py` | **Self-contained MLX module** (shipped with releases): prompt encoding with spans, collation, the joint schema head, `DecisionModel`, typed answers, `decide`, `load_release` |
| `inference.py` | Loading the decision model or the base model; batched decisions; chat replies |
| `data/download.py` | Pinned parquet download and reading |
| `data/prepare.py` | Writes splits, the ordered training stream and the manifest; smoke subset |
| `data/validate.py` | Row, stream-line and file validation; split overlap |
| `training/loop.py` | The MLX training loop, schedule, saving adapter + head |
| `training/trainer.py` | Pre-flight plan (stream, run dir, disk, memory) and guarded execution with manifest |
| `training/artifacts.py` | Run ids, environment/package versions, git commit, run-dir checks |
| `evaluation/evaluator.py` | Decision and chat evaluation runners; JSON + Markdown reports |
| `evaluation/metrics.py` | Accuracy, per-route accuracy, confusion matrix, invalid answers, ECE, latency |
| `reporting.py` | `reports/experiments/summary.md` |
| `publishing/model_card.py` | Hugging Face model card |
| `publishing/huggingface.py` | Upload planning, sanitised staging, leak scan, opt-in upload |
| `utils/paths.py`, `utils/memory.py` | Root discovery (`DECISION_MODEL_ROOT`), memory/disk readings |

`scripts/smoke_test.py` runs the bounded checks and prints a PASS/FAIL table.

## Design rules

* **Side effects need a flag.** `train` and `publish` are dry runs unless `--execute` is
  passed; the Makefile only passes it in `train`, `train-smoke` and `publish`.
* **MLX stays at the edge.** Only `runtime.py`, `inference.py` and the functions of
  `training/loop.py` import MLX, so everything else (and most tests) runs on any OS.
* **One definition of the prompt.** Training, evaluation, the CLI and the published loader
  all use `runtime.encode_request`.
* **Nothing is inferred into reports.** The card and report print "not measured" /
  "pending" for anything not recorded.
* **No overwrite of results.** A run directory that already holds a run blocks training;
  evaluation records are timestamped with a `latest.json` copy.
