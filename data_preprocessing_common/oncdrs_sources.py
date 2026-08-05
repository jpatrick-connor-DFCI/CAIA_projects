"""Source-agnostic scanning for raw OncDRS tables (CSV or merged Parquet).

The pipeline historically read one raw OncDRS release straight from CSV
(``/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/*.csv``). The
``PROFILE_data_processing`` repo now also publishes a single deduplicated
Parquet per table, merged across every release pull, at::

    /data/gusev/USERS/jpconnor/data/PROFILE_DATA/FINAL/<TABLE>.parquet

Both are valid inputs, so every raw read goes through :func:`scan_source`,
which dispatches on the file suffix. Callers pass whichever path they want;
nothing else in the pipeline needs to know which source it got.

Why the Utf8 cast matters
-------------------------
Every raw CSV read in this pipeline uses ``pl.scan_csv(..., infer_schema_length=0)``,
which makes **every column Utf8** -- no per-block dtype sniffing, no raising on
a later ``'05055/D'``-style dirty value. Downstream code is built on that
assumption: it calls ``.str.to_datetime()`` on date columns, ``.str.to_uppercase()``
on names/codes, runs :func:`data_preprocessing_common.fast_io.recover_numeric` to
pull genuinely-numeric columns back to Float64, and casts ``DFCI_MRN`` back to
Int64 by hand.

Parquet columns are typed, so a bare ``pl.scan_parquet`` would break every
``.str.*`` call. :func:`scan_source` therefore casts the whole Parquet frame to
Utf8, reproducing ``infer_schema_length=0`` exactly.

The only genuinely numeric column on the COMPASS path is ``LABS.NUMERIC_RESULT``
(Float64 upstream). It stringifies to ``"1.0"``-style text and ``recover_numeric``
parses it straight back to Float64, so the round trip is lossless. Every column
COMPASS parses as a string -- ``DFCI_MRN``, all ``*_DT`` dates,
``NCI_PREFERRED_MED_NM``, ``TEST_TYPE_CD``, the ICD-10 codes -- is already
``pl.String`` in the upstream ``COLUMN_MAP``, so it passes through untouched.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Merged, deduplicated Parquet published by PROFILE_data_processing's
# compile_OncDRS_data.ipynb (its OUTPUT_PATH / "FINAL").
DEFAULT_PROFILE_DATA_FINAL = Path(
    "/data/gusev/USERS/jpconnor/data/PROFILE_DATA/FINAL"
)

# Single raw OncDRS release, the pipeline's original source.
DEFAULT_ONCDRS_RELEASE = Path(
    "/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03"
)

# Table key -> (merged Parquet basename, raw-release CSV basename). Keys match
# FILE_TYPES in compile_OncDRS_data.ipynb; the CSV names match the ALL_2025_03
# pull. Only the tables COMPASS reads are listed -- add others as needed.
TABLE_FILES = {
    "EHR_DIAGNOSES": ("EHR_DIAGNOSES.parquet", "EHR_DIAGNOSIS.csv"),
    "MEDICATIONS": ("MEDICATIONS.parquet", "MEDICATIONS.csv"),
    "LABS": ("LABS.parquet", "OUTPT_LAB_RESULTS_LABS.csv"),
    "HEALTH_HISTORY": ("HEALTH_HISTORY.parquet", "HEALTH_HISTORY.csv"),
    "PT_INFO_STATUS_REGISTRATION": (
        "PT_INFO_STATUS_REGISTRATION.parquet",
        "PT_INFO_STATUS_REGISTRATION.csv",
    ),
}

PARQUET_SUFFIXES = (".parquet", ".pq")


def is_parquet(path) -> bool:
    return Path(path).suffix.lower() in PARQUET_SUFFIXES


def scan_source(path) -> pl.LazyFrame:
    """Lazily scan `path` with an all-Utf8 schema, whatever the format.

    Parquet is cast column-wise to Utf8 so it behaves identically to the
    ``pl.scan_csv(..., infer_schema_length=0)`` the rest of the pipeline was
    written against (see the module docstring). CSV -- plain or ``.gz``, which
    polars decompresses transparently -- takes the original path unchanged.
    """
    if is_parquet(path):
        return pl.scan_parquet(path).select(pl.all().cast(pl.Utf8))
    return pl.scan_csv(path, infer_schema_length=0)


def resolve(table: str, root=None) -> Path:
    """Path to `table` under `root`, picking the Parquet or CSV name to match.

    `root` defaults to the merged-Parquet directory. Pass a raw OncDRS release
    directory to get that release's CSV instead.
    """
    if table not in TABLE_FILES:
        raise KeyError(
            f"Unknown OncDRS table {table!r}. "
            f"Known tables: {sorted(TABLE_FILES)}"
        )

    root = Path(root) if root is not None else DEFAULT_PROFILE_DATA_FINAL
    parquet_name, csv_name = TABLE_FILES[table]

    # A directory named FINAL (or any directory already holding the Parquet)
    # gets the Parquet name; anything else is treated as a raw release dir.
    if (root / parquet_name).exists():
        return root / parquet_name
    return root / csv_name
