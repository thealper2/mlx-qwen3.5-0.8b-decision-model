# System 1 decision model (Qwen3.5-0.8B + joint schema head) for complaint routing, MLX.
# Run `make help` for the list of targets. Nothing here trains or uploads unless you
# call `make train`, `make train-smoke` or `make publish` yourself.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV      ?= uv
RUN     := $(UV) run --frozen
CLI     := $(RUN) decision-model
PROFILE ?= m1_air_8gb
RUN_DIR ?=
REPO_ID ?=

RUN_ARG     := $(if $(RUN_DIR),--run-dir $(RUN_DIR),)
REPO_ARG    := $(if $(REPO_ID),--repo-id $(REPO_ID),)

.PHONY: help install check-env lint format typecheck test \
        data-inspect data-download data-prepare data-validate data-smoke \
        train-config-check train-dry-run smoke-test train-smoke train \
        decide-smoke chat-smoke evaluate evaluate-chat evaluate-smoke report \
        card publish-dry-run publish clean-cache clean-artifacts

## General
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [PROFILE=...] [RUN_DIR=...] [REPO_ID=...]\n"} \
	     /^## / {printf "\n%s\n", substr($$0, 4)} \
	     /^[a-zA-Z_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Defaults: PROFILE=$(PROFILE) (training.yaml profile), RUN_DIR=publishing.run_dir"

## Environment
install: ## Create .venv and install the package + dev tools from uv.lock (no model download)
	$(UV) sync --frozen

check-env: ## Check Python, macOS, MLX, packages, memory, disk and configs
	$(CLI) check-env

lint: ## Ruff lint + format check
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

format: ## Apply Ruff formatting and safe fixes
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts

typecheck: ## Run mypy on the package
	$(RUN) mypy

test: ## Offline unit tests (no network, model, dataset or training)
	$(RUN) pytest

## Data
data-inspect: ## Summary statistics of the three splits
	$(CLI) data inspect

data-download: ## Download (or reuse the cached) pinned parquet files (~1 MB)
	$(CLI) data download

data-prepare: ## Write data/processed: splits + the ordered training-request stream
	$(CLI) data prepare

data-validate: ## Validate data/processed (formats, counts, overlap, stream)
	$(CLI) data validate

data-smoke: ## Write the tiny data/smoke subset (canonical data untouched)
	$(CLI) data smoke
	$(CLI) data validate --smoke

## Training
train-config-check: ## Validate all configs and plan the reference profile
	$(CLI) config-check
	$(CLI) train --profile reference

train-dry-run: ## Show the full training plan for PROFILE without training
	$(CLI) train --profile $(PROFILE)

smoke-test: ## Bounded pipeline checks; never trains, never uploads
	$(RUN) python scripts/smoke_test.py

train-smoke: data-smoke ## Tiny REAL training: 2 updates of 1 request (~3.5 GB, ~3 min)
	$(CLI) train --profile smoke --execute

train: ## FULL training with PROFILE (default m1_air_8gb). Many hours on an M1.
	$(CLI) train --profile $(PROFILE) --execute

## Inference and evaluation
decide-smoke: ## Route one example complaint with a trained run (RUN_DIR, e.g. artifacts/runs/smoke)
	$(CLI) decide $(RUN_ARG) \
	  --complaint "$$(cat examples/complaint.txt)"

chat-smoke: ## Ask the base model the same question as a chat (System 2)
	$(CLI) chat --complaint "$$(cat examples/complaint.txt)"

evaluate: ## Evaluate the decision model (RUN_DIR or publishing.run_dir) on 826 test complaints
	$(CLI) evaluate --mode decision $(RUN_ARG)

evaluate-chat: ## Evaluate the base model as a chat on the 826 test complaints (baseline)
	$(CLI) evaluate --mode chat

evaluate-smoke: data-smoke ## Chat baseline on the 8-complaint smoke subset
	$(CLI) evaluate --mode chat --smoke

report: ## Summarise recorded runs and evaluations in reports/experiments/summary.md
	$(CLI) report

## Publication
card: ## Generate the Hugging Face model card locally (no upload)
	$(CLI) card $(RUN_ARG) $(REPO_ARG)

publish-dry-run: card ## Check and stage the upload; uploads nothing
	$(CLI) publish $(RUN_ARG) $(REPO_ARG)

publish: card ## UPLOAD adapter + head + loader + card to REPO_ID (or publishing.repo_id)
	$(CLI) publish --execute $(RUN_ARG) $(REPO_ARG)

## Maintenance
clean-cache: ## Remove Python/tool caches inside this repository only
	find src tests scripts -type d -name __pycache__ -prune -exec rm -r {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

clean-artifacts: ## Remove disposable outputs: smoke data/run, upload staging (keeps real runs + reports)
	rm -rf data/smoke artifacts/runs/smoke artifacts/exports/hf_upload
