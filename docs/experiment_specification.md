# Experiment specification

This document is the source of truth for **what the experiment is**. The code, configs and
tests implement it; if they ever disagree with this page, that is a bug.

## 1. Reference definition

| Aspect | Reference |
|---|---|
| Task | Route a consumer complaint to one of 9 financial products (a typed `choice` question) |
| Routes | `credit_reporting`, `debt_collection`, `credit_card`, `bank_account`, `mortgage`, `money_transfer`, `vehicle_loan`, `student_loan`, `personal_loan`, each with a short description (`task.py`) |
| Base model | `Qwen/Qwen3.5-0.8B` (Apache-2.0), bf16, text model |
| Data | `kgptalkie/cfpb-complaint-routing` (CC0-1.0): train 2,700 (300 per route), validation 270, test 826 |
| Request | `{state: {complaint}, questions: {product: {type: choice, instructions, criteria}}}` |
| Prompt | system line, `STATE:` + state as compact sorted JSON, `SCHEMA FIELDS:` with each field's id, type, instruction and every option as `{"option_id", "description"}` (options sorted by id), an empty think block, `JOINT SCHEMA DECISIONS:`; every piece tokenized separately; state truncated first beyond 4,096 tokens |
| Model | the language model's final hidden states → joint schema head (hidden 1024, width 512, 1 evidence-routing layer, 2 field-decoder layers, 8 heads, feed-forward 2048) → one logit per offered option |
| Head computation | span means of question and option tokens; option = context projection + lexical projection (mean of the options' token embeddings) + question projection; evidence routing (options attend to the prompt); option summary by attention of the field over its options; field vector = question + summary + last-token (global) projection + type embedding; 2 pre-norm decoder layers (fields attend to each other, then to the prompt); logit = scaled cosine prior of option words vs. question+global + sigmoid(gate) × (scaled cosine + residual MLP on [f, o, f·o, abs(f−o)]) |
| Training stream | `random.Random(7)`; one augmentation draw for the first training row (discarded); `rng.sample(train, 800)`; per sampled row, in order: with probability 0.5 offer only the correct route plus `randint(1, 8)` others sampled from the remaining 8 |
| LoRA | rank 16, alpha 32, dropout 0.05 on `q/k/v/o_proj`, `gate/up/down_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj` of all 24 blocks |
| Loss | cross-entropy of the correct option among the offered ones, mean over the batch |
| Optimisation | AdamW, weight decay 0.01, lr 2e-4 (LoRA) and 1e-3 (head), schedule `lr × min(1, (step+1)/20) × ½(1 + cos(π·step/100))`, 100 updates of 8 requests (800 in total), global gradient-norm clipping at 1.0, gradient checkpointing |
| Evaluation | the 826 test complaints with all 9 routes offered; accuracy of the argmax option |
| Baseline | the base model asked the same request as a chat (system instruction + request JSON), greedy, thinking off, 16 new tokens; answer = `product` of the last flat JSON object after the last `</think>`, counted wrong when missing or not a route |
| Output | LoRA adapter + head weights |

### Licenses

| Component | License | Notes |
|---|---|---|
| `Qwen/Qwen3.5-0.8B` | Apache-2.0 | |
| `kgptalkie/cfpb-complaint-routing` | CC0-1.0 | Derived from the US CFPB Consumer Complaint Database (public data) |
| Joint schema head design | Apache-2.0 (Cloudflare Clef) | Independent implementation; not affiliated with Cloudflare |
| This repository's code | MIT | See `LICENSE` |

## 2. Verified properties

Checked on the target machine, not assumed:

* **Training stream.** The 800 requests (complaint, offered routes, label, order) are
  identical to an independent re-derivation of the definition above from the pinned data.
* **Prompt encoding.** For 80 requests (40 from the training stream with option subsets,
  40 test requests with all routes) the token ids and every question/option span are
  identical to an independent encoding with the Hugging Face tokenizer of the pinned model.
* **Head.** With identical weights and inputs (two prompts, one padded, two questions with
  3 and 2 options), the MLX head reproduces an independent float32 reference
  implementation of the head computation to a maximum absolute logit difference of
  1.6 × 10⁻⁷.
* **Baseline.** Chat messages and reply parsing match the definition on test cases
  covering think blocks, invalid routes, several JSON objects and broken JSON.
* **Lengths.** Test prompts are 327–514 tokens (median 418), training prompts at most 522;
  the 4,096-token budget never truncates.
* **Data.** 0 shared ids and 0 identical complaint texts between train, validation and
  test; the stream holds 800 distinct training rows, 46.9% of them with an option subset.

## 3. Preserved, reorganised and changed

| Area | Status | Detail |
|---|---|---|
| Model, data, routes, request and prompt layout, head architecture, training stream, LoRA targets and hyper-parameters, loss, schedule, clipping, evaluation and baseline | Preserved | `configs/`, `task.py`, `runtime.py`, `data/prepare.py`; tests pin the values |
| Model revision, data revision | **Pinned (new)** | `2fc06364…` and `3ec70ba3…` (latest on 2026-10-28) |
| **Framework** | MLX | Base model through MLX-LM; LoRA via MLX-LM's `linear_to_lora_layers` (scale = alpha / rank = 2.0); head and training loop in MLX (`runtime.py`, `training/loop.py`); MLX-LM's own trainer only does next-token loss, so the loop is custom |
| Head attention | Implemented explicitly | Softmax attention written out, because MLX's fused attention kernel has no gradient for very short query sequences; numerically equivalent (§2) |
| Optimizer | Made explicit | MLX AdamW with `bias_correction=True`, betas 0.9/0.999, eps 1e-8, weight decay 0.01 on both parameter groups (MLX's defaults differ) |
| Initial weights | Framework-specific | LoRA and head start from MLX's initialisers with seed 7; runs are reproducible within MLX, not bit-identical across frameworks |
| Batching | Simplified | Right padding to the longest prompt of the batch (no fixed 128-token buckets); the head masks padding, and causal attention makes padding invisible to real tokens |
| Validation split | Prepared, unused | As in the reference recipe |
| Calibration (ECE), per-route accuracy, confusion matrix, latency | Added, secondary | Accuracy stays the headline metric |
| Hardware profile | Added | `m1_air_8gb`: the same 100 updates of the same 8 requests as 8 accumulated micro-batches of 1 |
| Release | Adapter + head + self-contained `decision_model_runtime.py` | The LoRA is not merged into the base weights; `mlx_lm.load(..., adapter_path=...)` applies it |
| Workflow | Implemented | Python package + CLI + Makefile; every step runs from a terminal |

## 4. Known differences of the `m1_air_8gb` profile

Gradients of 8 single-request micro-batches are averaged; with batch 8 the loss is the
mean over 8 requests. Both give the mean per-request loss gradient, so the update is the
same up to floating-point order — except for LoRA dropout masks, which are drawn per
micro-batch. Padding never matters (batch 1 has none).

## 5. Assumptions and limitations

* 800 training requests (30% of the training split) and 100 updates: a short run.
* Labels are the products consumers selected when filing; some are debatable.
* The test set is 826 complaints from the same source and period as training.
* Full training on the M1 MacBook Air 8 GB has not been run yet; see `docs/training.md`.
