"""Fetch the dataset's parquet files from the Hugging Face Hub (cached, revision-pinned)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from decision_model.config import SPLITS, DatasetSource

log = logging.getLogger(__name__)
COLUMNS = ("id", "complaint", "product")


class DataAccessError(RuntimeError):
    """The dataset could not be downloaded or read."""


def download_split(source: DatasetSource, split: str, local_files_only: bool = False) -> Path:
    """Local path of one split's parquet file, downloading it only if not cached."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=source.repo_id,
            filename=source.files[split],
            repo_type="dataset",
            revision=source.revision,
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as exc:
        raise DataAccessError(
            "The dataset is not in the local Hugging Face cache and downloading is disabled "
            "or the Hub is unreachable. Run `make data-download` with network access."
        ) from exc
    except EntryNotFoundError as exc:
        raise DataAccessError(f"{source.files[split]} not found in {source.repo_id}") from exc
    except Exception as exc:  # network / HTTP errors from the Hub client
        raise DataAccessError(f"Could not download {source.repo_id}: {exc}") from exc
    return Path(path)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a parquet file as dicts, in file order, restricted to the task columns."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    missing = [c for c in COLUMNS if c not in table.column_names]
    if missing:
        raise DataAccessError(f"{path.name} lacks column(s) {missing}")
    return table.select(list(COLUMNS)).to_pylist()


def load_splits(
    source: DatasetSource, local_files_only: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """All three splits, in file order."""
    log.info("Dataset %s @ %s", source.repo_id, (source.revision or "main")[:12])
    return {s: read_parquet_rows(download_split(source, s, local_files_only)) for s in SPLITS}
