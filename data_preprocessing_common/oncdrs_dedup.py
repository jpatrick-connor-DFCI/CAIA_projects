"""Intra-release deduplication for raw OncDRS tables, ported from the
`PROFILE_data_processing` sibling repo's `compile_OncDRS_data.ipynb`.

That notebook merges seven OncDRS releases into one deduplicated Parquet per
table, folding releases newest-first with an anti-join keyed on
`DEDUP_COL_MAP`. Standalone callers may still read a single raw release
straight off CSV, which does not receive that upstream dedup treatment;
duplicate rows (the same lab result re-pulled across an OncDRS extract, for
instance) can inflate downstream counts (`n_observations`, the PSA-count
cohort gate). The active COMPASS notebook workflow uses only the already
deduplicated merged `profile_data` parquets.

With a single release, the notebook's cross-release fold degenerates to just
its intra-file step: the notebook's own `if not kept_files:` branch
(`.unique(subset=key_cols, keep="first")` for the exact-row-dedup tables) and,
for `PT_INFO_STATUS_REGISTRATION`, one `group_by` coalesce over one frame
instead of many. No release-ordering machinery needs porting.

This module is intentionally polars-only (no pandas import), matching
`fast_io.py`.
"""

from __future__ import annotations

import os

import polars as pl

# ---------------------------------------------------------------------------
# DATA_COLUMNS: ported from the notebook's COLUMN_MAP, restricted to the keys
# of the tables this repo reads (see oncdrs_sources.TABLE_FILES), and with the
# Stage-1-derived `SOURCE_RELEASE` / `<FIELD>_SOURCE_RELEASE` columns dropped
# -- those are added by the notebook's own upstream merge step and do not
# exist in a raw OncDRS release CSV.
#
# IMPORTANT: this is the column set dedup runs on. It is NOT the raw CSV's
# full width (extra columns like RESULT_NBR would break ties the notebook's
# own COLUMN_MAP projection had already dropped, so deduping on raw width
# removes too few rows) and it is NOT whatever narrower `cols=` a caller
# happens to project to (deduping on, say, [DFCI_MRN, TEST_TYPE_CD] alone
# would collapse LABS to ~one row per test code and wreck the PSA gate). See
# `dedup_key_columns` and `apply_dedup` below.
# ---------------------------------------------------------------------------
DATA_COLUMNS: dict[str, list[str]] = {
    "EHR_DIAGNOSES": [
        "DFCI_MRN",
        "START_DT",
        "END_DT",
        "DIAGNOSIS_ICD10_CD",
        "DIAGNOSIS_ICD10_NM",
        "DIAGNOSIS_ICD10_CD2",
        "DIAGNOSIS_ICD10_NM2",
        "DIAGNOSIS_ICD10_CD3",
        "DIAGNOSIS_ICD10_NM3",
    ],
    "MEDICATIONS": [
        "DFCI_MRN",
        "NCI_PREFERRED_MED_NM",
        "MED_START_DT",
    ],
    "LABS": [
        "DFCI_MRN",
        "SPECIMEN_COLLECT_DT",
        "TEST_TYPE_CD",
        "TEST_TYPE_DESCR",
        "NUMERIC_RESULT",
        "TEXT_RESULT",
        "RESULT_UOM_NM",
    ],
    "HEALTH_HISTORY": [
        "DFCI_MRN",
        "START_DT",
        "CODE",
        "HEALTH_HISTORY_TYPE",
        "CODE_TYPE",
        "RESULTS",
        "UNITS_CD",
    ],
    "PT_INFO_STATUS_REGISTRATION": [
        "DFCI_MRN",
        "BIRTH_DT",
        "CLIN_DEATH_DT",
        "HYBRID_DEATH_DT",
        "NDI_DEATH_DT",
        "DERIVED_LAST_ALIVE_DATE",
        "GENDER_NM",
        "GENDER_CD",
        "PT_ONCOPANEL_PROFILED_IND",
    ],
}

# ---------------------------------------------------------------------------
# DEDUP_COL_MAP: ported from the notebook's DEDUP_COL_MAP, restricted to the
# same table subset.
#
# None means: exact full-row dedup over DATA_COLUMNS[table] (the notebook's
# `if not kept_files: release_lf.unique(subset=key_cols, keep="first")`
# branch, where key_cols == data_columns when dedup_cols is None).
#
# A list means: treat those columns as the logical-record identifier. On a
# single release "keep the newest release" is moot -- there is only one
# release -- so this also reduces to `.unique(subset=key_cols, keep="first")`.
#
# PT_INFO_STATUS_REGISTRATION is a coalesce table (see COALESCE_TABLES below);
# its entry here is only the notebook's own "existence check" key, i.e. the
# grouping key for the per-column coalesce in apply_dedup.
# ---------------------------------------------------------------------------
DEDUP_COL_MAP: dict[str, list[str] | None] = {
    "EHR_DIAGNOSES": [
        "DFCI_MRN",
        "START_DT",
        "END_DT",
        "DIAGNOSIS_ICD10_CD",
        "DIAGNOSIS_ICD10_CD2",
        "DIAGNOSIS_ICD10_CD3",
    ],
    "MEDICATIONS": None,
    "LABS": None,
    "HEALTH_HISTORY": None,
    "PT_INFO_STATUS_REGISTRATION": ["DFCI_MRN"],
}

# Tables handled by a per-column coalesce (newest non-null value per column)
# instead of a row-key `.unique()`. Matches the notebook's COALESCE_FILE_TYPES
# minus GENOMIC_SPECIMEN, which has no TABLE_FILES entry or call site here
# (see oncdrs_sources.TABLE_FILES) -- skipped rather than porting dead code.
COALESCE_TABLES = frozenset({"PT_INFO_STATUS_REGISTRATION"})

# Kill switch to reproduce pre-dedup row counts without editing call sites
# (see the plan's verification step (a)/(d): before/after comparisons). Read
# once at import time, matching every other module-level constant here.
DISABLE_DEDUP = os.environ.get("PROFILE_DISABLE_ONCDRS_DEDUP") == "1"


def dedup_key_columns(table: str, available: list[str]) -> list[str]:
    """Resolve the dedup key columns for `table`, intersected with the
    columns actually `available` in the frame being deduped.

    For a `None`-mapped table (exact full-row dedup, e.g. LABS) the "key" is
    `DATA_COLUMNS[table]` itself -- every data column participates. Losing
    most of those columns to a missing-column intersection is the dangerous
    case: it silently turns an "exact row" dedup into a dedup on whatever
    handful of columns happen to survive (e.g. a LABS frame deduped only on
    DFCI_MRN would collapse to ~1 row per patient). That failure mode is
    quiet and catastrophic, so a `None`-mapped table raises loudly if fewer
    than two of its key columns survive. Mirrors the notebook's own
    `missing_dedup_cols` RuntimeError, which fires eagerly on any missing key
    column rather than waiting to see how bad the collapse is.

    A table with an explicit (non-`None`) key list -- EHR_DIAGNOSES's 6-column
    logical-record key, or PT_INFO_STATUS_REGISTRATION's single-column
    `["DFCI_MRN"]` coalesce group key -- is deliberately narrow by design, so
    the same "fewer than two survive" threshold doesn't apply; only an
    entirely-missing key column is an error there.
    """
    key_cols = DEDUP_COL_MAP.get(table)
    is_exact_row_dedup = key_cols is None
    if is_exact_row_dedup:
        key_cols = DATA_COLUMNS[table]

    resolved = [c for c in key_cols if c in available]

    if not resolved or (is_exact_row_dedup and len(resolved) < 2):
        raise RuntimeError(
            f"{table}: too few dedup key columns survive intersection with "
            f"the available columns to dedup safely "
            f"(wanted {key_cols}, available {available}, got {resolved}). "
            "Deduping on this key would collapse the table to near-nothing."
        )
    return resolved


def apply_dedup(lf: pl.LazyFrame, table: str) -> pl.LazyFrame:
    """Apply intra-release dedup for `table` to `lf`.

    `lf` should be the raw scan (all-Utf8, via `oncdrs_sources.scan_source`)
    BEFORE any caller `.select()` projection -- dedup must run on the
    canonical `DATA_COLUMNS[table]` key set, never on a narrower caller
    projection (see the module docstring). Unknown tables pass through
    unchanged, so non-OncDRS sources can share this code path harmlessly.

    Determinism note: `.unique(..., maintain_order=True)` is used throughout.
    `keep="first"` is only meaningful with a defined row order, and
    determinism is worth far more than the saved microseconds at these row
    counts -- it's also what makes the unit tests stable.
    """
    if DISABLE_DEDUP or table not in DEDUP_COL_MAP:
        return lf

    available = lf.collect_schema().names()

    if table in COALESCE_TABLES:
        return _apply_coalesce(lf, table, available)

    key_cols = dedup_key_columns(table, available)
    # `.unique()` treats null == null (the semantics we want here -- e.g.
    # EHR_DIAGNOSES rows with a null END_DT/_CD2/_CD3 should still collapse
    # against another row that is null in exactly the same places).
    return lf.unique(subset=key_cols, keep="first", maintain_order=True)


def _apply_coalesce(lf: pl.LazyFrame, table: str, available: list[str]) -> pl.LazyFrame:
    """Per-column coalesce for PT_INFO_STATUS_REGISTRATION: for each
    DFCI_MRN, each output column independently takes its first non-null
    value (`drop_nulls().first()`).

    On a single release this is a simplification of the notebook's cross-
    release coalesce, which sorts by `_release_rank` first so "first non-null"
    means "newest release with data for this column." With one release there
    is no rank to sort by, so plain `drop_nulls().first()` is exactly
    equivalent -- there's nothing to prefer between releases.

    One further simplification, called out explicitly rather than silently
    assumed: upstream special-cases the five STATUS_PROVENANCE_FIELDS
    (BIRTH_DT, CLIN_DEATH_DT, HYBRID_DEATH_DT, NDI_DEATH_DT,
    DERIVED_LAST_ALIVE_DATE) with
    `pl.col(c).filter(pl.col(f"{c}_SOURCE_RELEASE").is_not_null()).first()`
    instead of `drop_nulls().first()`, to distinguish "this release didn't
    supply the field at all" from "this release supplied it as an explicit
    null." Those `*_SOURCE_RELEASE` companion columns are themselves
    Stage-1-derived and don't exist on a raw release CSV, so the distinction
    is vacuous here -- there is exactly one release, so there is no
    cross-release null-vs-absent ambiguity to resolve. Using the simpler form
    is therefore equivalent in this single-release case, not a shortcut that
    silently drops behavior.
    """
    key_cols = dedup_key_columns(table, available)
    non_key_cols = [c for c in DATA_COLUMNS[table] if c in available and c not in key_cols]

    coalesce_aggs = [pl.col(c).drop_nulls().first().alias(c) for c in non_key_cols]
    return lf.group_by(key_cols, maintain_order=True).agg(coalesce_aggs).select(
        [*key_cols, *non_key_cols]
    )
