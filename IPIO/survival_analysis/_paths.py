"""Local import-path helpers for IPIO scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_survival_common_on_path() -> Path:
    """Make repo-local shared packages importable."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "survival_common"
        if candidate.exists():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return parent
    raise RuntimeError(
        "Could not locate survival_common relative to "
        f"{current}. Run IPIO survival scripts from this repository checkout."
    )
