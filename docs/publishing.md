# Publishing to Hugging Face

Publishing is opt-in. Nothing is uploaded by installation, tests, smoke checks, training,
evaluation or `make card`.

## What a release contains

| File | Content |
|---|---|
| `README.md` | model card |
| `adapters.safetensors`, `adapter_config.json` | LoRA in MLX-LM format (~41 MB) |
| `head.safetensors`, `head_config.json` | the joint schema head (~63 MB) |
| `decision_model_runtime.py` | a copy of `src/decision_model/runtime.py`: prompt encoding, head, `decide`, `load_release`; imports only MLX, MLX-LM and the standard library |
| `run_manifest.json` | provenance: configuration, versions, loss per update, timings |

The release is not a standalone model: the base model `Qwen/Qwen3.5-0.8B` at the pinned
revision is downloaded by `mlx_lm.load` and the adapter and head are applied on top.
Checkpoints, training logs, data, reports, caches and `.env` stay local.

## 1. Log in (once)

```bash
uv run hf auth login
```

Use a token with **write** access. It is stored by `huggingface_hub`; the project never
reads it from YAML, never prints it and needs it only for the final upload.

## 2. Choose the destination and the run

Set `repo_id` in `configs/publishing.yaml` or pass `REPO_ID=…`. `private: true` (default)
creates a private repository. `run_dir` selects the trained run (default
`artifacts/runs/m1_air_8gb`; override with `RUN_DIR=…`). `adapter_license` sets the card's
license field — confirm it yourself.

## 3. Model card

```bash
make card
```

Writes `artifacts/exports/model_card/README.md` from the configs, the data manifest, the
run manifest and the latest `base-chat` / `decision` evaluation records. A decision
evaluation is only used if it was produced with this exact run.

## 4. Dry run

```bash
make publish-dry-run REPO_ID=<namespace>/<name>
```

Checks without network access: valid repository id; the run directory holds all four
weight/config files and a `run_manifest.json` with status `completed` from a non-smoke
profile; the model card exists; only the listed files (plus the runtime and the card) are
selected. It then stages everything in `artifacts/exports/hf_upload/` with local paths
removed from JSON/YAML files, and scans every staged text file (including the runtime) for
token-like strings and home-directory paths.

## 5. Upload

```bash
make publish REPO_ID=<namespace>/<name>
```

Same checks and staging, verifies the login, creates the repository if needed and uploads
the staged folder in one commit.

## Using a release

```python
import sys
from huggingface_hub import snapshot_download

folder = snapshot_download("<namespace>/<name>")
sys.path.insert(0, folder)
from decision_model_runtime import decide, load_release

model, tokenizer = load_release("Qwen/Qwen3.5-0.8B",
                                "2fc06364715b967f1860aea9cf38778875588b17", folder)
```

Verified locally: the exact usage code of the generated model card ran against a folder
holding only the release files of the smoke run and returned a valid typed answer. Loading
from an actual Hub repository has not been tested, because nothing has been uploaded.
