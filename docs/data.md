# Data

## Source

[`kgptalkie/cfpb-complaint-routing`](https://huggingface.co/datasets/kgptalkie/cfpb-complaint-routing)
(CC0-1.0), pinned to revision `3ec70ba3690c26b8d9b117629fa3803ce4e4b84f`: short consumer
complaints from the US CFPB Consumer Complaint Database (personal details redacted as `XXXX`
by the CFPB), each labelled with one of 9 products. Three parquet files (~1 MB in total)
are downloaded into the Hugging Face cache; no token is needed.

| Split | Rows | Labels |
|---|---|---|
| train | 2,700 | 300 per route |
| validation | 270 | 30 per route (prepared, not used by the recipe) |
| test | 826 | 59–120 per route |

Columns used: `id` (`cfpb-<complaint id>`), `complaint`, `product`. `make data-inspect`
prints counts, label distributions, duplicates and length statistics per split.

## Preparation (`make data-prepare`)

1. Read the three splits in file order and validate every row (non-empty id and
   complaint, product is one of the 9 routes) and the row counts.
2. Build the **training stream** — the exact, ordered list of requests the trainer reads:
   * `rng = random.Random(7)`;
   * one augmentation draw for the first training row, discarded;
   * `rows = rng.sample(train, 800)` (without replacement);
   * for each row in order: with probability 0.5 the request offers only the correct route
     plus `rng.randint(1, 8)` of the other routes (`rng.sample`), otherwise all 9.
3. Write the files below and `manifest.json` (provenance, counts, label distributions,
   overlap between splits, stream statistics, SHA-256 of every file).

| File | Format |
|---|---|
| `data/processed/{train,validation,test}.jsonl` | `{"id", "complaint", "product"}` |
| `data/processed/train_requests.jsonl` | `{"id", "product", "request"}`, 800 lines in training order |

Because every profile reads the same stream in the same order, the reference profile
(batch 8) and the 8 GB profile (8 × batch 1) see the same 8 requests in every update.

`uv run decision-model data prepare --dry-run` computes everything without writing.

## Validation (`make data-validate`)

Formats of every line, row counts, stream size, every stream request offering the correct
route with the original descriptions, no shared ids or identical texts between splits, and
no test complaint in the stream.

## Smoke subset (`make data-smoke`)

The first 4 stream requests and the first 8 test rows, written to `data/smoke/` from the
canonical files (which are never modified).
