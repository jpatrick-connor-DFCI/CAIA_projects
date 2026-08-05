"""Source-agnostic scanning for raw OncDRS tables (CSV or merged Parquet).

The pipeline historically read one raw OncDRS release straight from CSV
(``/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/*.csv``). The
``PROFILE_data_processing`` repo now also publishes a single deduplicated
Parquet per table, merged across every release pull, at::

    /data/gusev/USERS/jpconnor/data/PROFILE_DATA/<TABLE>.parquet

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

The genuinely typed upstream columns on the COMPASS path include
``LABS.NUMERIC_RESULT`` (Float64), ``DFCI_MRN`` (Int64), and normalized calendar
dates (Date). They stringify losslessly; ``recover_numeric`` and the downstream
date parsers restore the types expected by the existing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Merged, deduplicated Parquets published directly in PROFILE_data_processing's
# compile_OncDRS_data.ipynb OUTPUT_PATH.
DEFAULT_PROFILE_DATA_ROOT = Path(
    "/data/gusev/USERS/jpconnor/data/PROFILE_DATA"
)
# Compatibility alias for callers written before final Parquets moved out of
# the now-retired PROFILE_DATA/FINAL directory.
DEFAULT_PROFILE_DATA_FINAL = DEFAULT_PROFILE_DATA_ROOT

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

    root = Path(root) if root is not None else DEFAULT_PROFILE_DATA_ROOT
    parquet_name, csv_name = TABLE_FILES[table]

    # A directory already holding the compiled Parquet gets the Parquet name;
    # anything else is treated as a raw release directory.
    if (root / parquet_name).exists():
        return root / parquet_name
    return root / csv_name
