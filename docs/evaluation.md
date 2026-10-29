# Evaluation

## What is evaluated

The 826 test complaints in `data/processed/test.jsonl`, never used for training. Every
request offers all 9 routes. Two systems are evaluated and recorded separately:

| Label | Command | System |
|---|---|---|
| `base-chat` | `make evaluate-chat` | **System 2 baseline**: the base model reads the request as a chat and writes JSON |
| `decision` | `make evaluate [RUN_DIR=…]` | **System 1**: base model + LoRA + head, one forward pass, no generation |
| `*-smoke` | `make evaluate-smoke`, `decision-model evaluate --mode decision --smoke --run-dir …` | first 8 test complaints |

A label that has not been run has no record; the report and model card show
"not evaluated" / "pending".

### Decision model

Requests are encoded with `runtime.encode_request`, batched (8 per forward pass, right
padded) and passed through the model once. The answer is the option with the highest
logit; its softmax probability is the confidence. Latency is the batch's wall time divided
by its size.

### Chat baseline

System instruction plus the request (without the model name) as JSON, the model's chat
template with thinking disabled, greedy decoding, at most 16 new tokens. The answer is the
`product` of the last flat JSON object after the last `</think>`; a reply without a valid
route counts as wrong (and as an *invalid answer*).

## Metrics

| Metric | Definition | Role |
|---|---|---|
| **Accuracy** | correct route / complaints | primary |
| Invalid answers | replies without a valid route (chat baseline only; the head always answers) | diagnostic |
| Per-route accuracy | recall per correct route | diagnostic |
| Confusion matrix | correct route × answered route (+ invalid) | diagnostic |
| Mean confidence, ECE | expected calibration error over 10 confidence bins (decision model only) | diagnostic |
| Mean latency | per complaint on the evaluating machine | diagnostic |

## Outputs

`reports/evaluation/<label>/<timestamp>.json` and `.md`, plus `latest.json` / `latest.md`:
model and revision, training run id, test file and SHA-256, mode, peak MLX memory,
environment, all metrics, and per complaint the label, prediction, confidence and
probabilities (decision) or the raw reply (chat). `make report` combines the latest records
with the run manifests into `reports/experiments/summary.md`.

## Limitations

* 826 complaints: a 95% interval on accuracy is up to about ±3.4 points.
* Labels are what consumers selected; some complaints fit several products.
* The chat baseline is sensitive to its 16-token budget and parsing rule by design; it
  measures the base model as a generic chat model, not its best possible prompt.
* Smoke evaluations only show that the pipeline runs.
