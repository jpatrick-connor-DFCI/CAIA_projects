"""COMPASS PROFILE preprocessing defaults."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data")
DATA_PATH = DATA_ROOT / "CAIA" / "COMPASS"
UNIQUE_LABS_CSV = DATA_PATH / "unique_lab_ids_w_units.csv"

# COMPASS-specific exclusions. Shared lab standardization remains unchanged for
# IPIO and other projects.
EXCLUDED_MEASUREMENTS = frozenset(
    {
        "Body weight",
        "Body height",
        "Body temperature",
        "Heart rate",
        "Respiratory rate",
        "Systolic blood pressure",
        "Diastolic blood pressure",
    }
)
