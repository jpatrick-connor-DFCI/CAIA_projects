"""Thin re-export of `helpers.loaders.load_caia_parquet` for `CAIA/` callers.

The actual loader lives in `helpers/loaders.py` so both the summary-stats
script and the survival pipeline share one implementation. This file exists
so notebooks under CAIA can `from load_caia_parquet import load_caia_parquet`
without having to know the helper package layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

SURVIVAL_PARENT = Path(__file__).resolve().parents[1]   # .../survival_analysis
if str(SURVIVAL_PARENT) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_PARENT))

from helpers.loaders import load_caia_parquet  # noqa: E402,F401
