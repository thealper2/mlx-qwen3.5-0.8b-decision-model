"""Lightweight memory and disk diagnostics for Apple Silicon machines.

All numbers are approximate. Unified memory is shared by macOS, other apps, Python and
MLX, and what is "available" changes from second to second; these readings are meant
for warnings, not guarantees.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

GB = 1024**3
_TIMEOUT_S = 5


@dataclass(frozen=True)
class MemorySnapshot:
    total_gb: float | None
    available_gb: float | None  # free + inactive + purgeable + speculative pages


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def memory_snapshot() -> MemorySnapshot:
    """Total and roughly-available memory (macOS via sysctl/vm_stat; None elsewhere)."""
    if platform.system() != "Darwin":
        return MemorySnapshot(None, None)
    total = _run(["sysctl", "-n", "hw.memsize"])
    total_gb = int(total.strip()) / GB if total and total.strip().isdigit() else None
    vm = _run(["vm_stat"])
    available_gb = None
    if vm:
        page = re.search(r"page size of (\d+) bytes", vm)
        size = int(page.group(1)) if page else 16384
        pages = 0
        for key in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"):
            m = re.search(rf"^{key}:\s+(\d+)\.", vm, re.M)
            pages += int(m.group(1)) if m else 0
        available_gb = pages * size / GB
    return MemorySnapshot(total_gb, available_gb)


def free_disk_gb(path: Path) -> float:
    """Free space on the volume holding ``path`` (or its nearest existing parent)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / GB


def estimate_training_memory_gb(
    weights_gb: float, batch_size: int, seq_length: int, vocab_size: int = 100_352
) -> float:
    """Very rough peak-memory estimate for LoRA training with gradient checkpointing.

    ``seq_length`` is the padded length of the longest batch (MLX-LM pads per batch, not
    to max_seq_length). weights + logits/softmax buffers for the batch (fp32, forward +
    backward) + a fixed allowance for activations, LoRA/optimizer state and allocator
    slack. An order of magnitude only; it has not been calibrated against measured runs.
    """
    logits_gb = batch_size * seq_length * vocab_size * 4 * 3 / GB
    return weights_gb + logits_gb + 1.0


def training_memory_warnings(
    total_gb: float | None,
    estimate_gb: float,
    max_seq_length: int,
    warn_seq_length: int,
) -> list[str]:
    """Human-readable warnings for a planned training run."""
    warnings: list[str] = []
    if total_gb is None:
        warnings.append("System memory could not be read; memory checks are skipped.")
        return warnings
    budget = total_gb * 0.6  # leave room for macOS and everything else
    if estimate_gb > budget:
        warnings.append(
            f"Estimated peak ~{estimate_gb:.1f} GB exceeds ~60% of {total_gb:.0f} GB unified "
            "memory. Expect swapping or an out-of-memory abort: close other apps, and keep "
            "batch_size at 1 (the m1_air_8gb profile)."
        )
    if total_gb <= 8.5 and max_seq_length > warn_seq_length:
        warnings.append(
            f"max_seq_length={max_seq_length} is large for an 8 GB machine. MLX-LM pads each "
            "batch only to its longest example, so short data stays cheap, but any long "
            "example would be allowed to grow memory up to this cap."
        )
    return warnings
