"""Process-wide logging setup for the CLI."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once; ``verbose`` switches to DEBUG."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", force=True)
    # Third-party libraries are noisy at INFO; keep their warnings only.
    for name in ("httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
