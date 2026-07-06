"""Per-patient irAE outcome construction for the IPIO landmark cohort.

Mirrors the landmark-rebasing pattern of COMPASS's helpers.cohort.make_outcome_df,
simplified to a single right-censored endpoint (death and censor are both treated
as censoring; only `event == "irAE"` is the event of interest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "DFCI_MRN"

# Baseline covariates that must survive the per-patient dedup and be carried
# through into the aggregated_landmark{N}.csv / genomic_aggregated.csv outputs
# so the baseline-covariate mechanism in cox_aggregated.py can find them.
# CANCER_TYPE_* is a dynamic (data-dependent) one-hot set, discovered at call
# time via startswith("CANCER_TYPE_"); GENDER_MALE / pd1pdl1 / ctla4 are fixed
# binary 0/1 columns. `combination` is deliberately excluded everywhere.
BASELINE_COVARIATE_COLS = ("AGE_AT_TREATMENTSTART", "GENDER_MALE", "pd1pdl1", "ctla4")


def make_irae_outcome_df(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int = 0,
    anchor_col: str = "t_first_treatment",
    extra_anchor_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build the per-patient irAE outcome table rebased to a landmark.

    `df` may be long-format (one row per patient x lab observation) or already
    patient-level; must carry DFCI_MRN, FIRST_RECORD_DATE, LAST_CONTACT_DATE, IRAE,
    `anchor_col` (days from FIRST_RECORD_DATE -- 0 for the main cohort's
    t_first_treatment, or a per-patient t_sample for the genomic arm), plus the
    baseline covariate columns (AGE_AT_TREATMENTSTART, GENDER_MALE, pd1pdl1, ctla4,
    every CANCER_TYPE_* dummy) and any `extra_anchor_cols`, all passed through
    unchanged.

    Returns a frame indexed by DFCI_MRN, keeping all baseline covariate columns +
    AGE_AT_TREATMENTSTART + IRAE + t_irae + t_irae_from_first_record +
    extra_anchor_cols + the anchor column + FIRST_RECORD_DATE / LAST_CONTACT_DATE
    (the latter two are debug-only duration duplicates; downstream callers may
    drop them before persisting the final aggregated table).
    """
    cancer_type_cols = tuple(c for c in df.columns if c.startswith("CANCER_TYPE_"))

    patient_level_cols = [
        ID_COL,
        *BASELINE_COVARIATE_COLS,
        *cancer_type_cols,
        "FIRST_RECORD_DATE",
        "LAST_CONTACT_DATE",
        "IRAE",
        anchor_col,
        *extra_anchor_cols,
    ]
    # De-dup while preserving first-seen order (anchor_col may already be one of
    # extra_anchor_cols, or equal to a baseline covariate name in edge cases).
    seen: set[str] = set()
    ordered_cols: list[str] = []
    for col in patient_level_cols:
        if col not in seen:
            seen.add(col)
            ordered_cols.append(col)

    available_cols = [col for col in ordered_cols if col in df.columns]
    if ID_COL not in available_cols:
        raise ValueError(f"Input data must contain the id column {ID_COL!r}.")

    required = {"FIRST_RECORD_DATE", "LAST_CONTACT_DATE", "IRAE"}
    missing_required = required - set(available_cols)
    if missing_required:
        raise ValueError(
            f"make_irae_outcome_df: input data is missing required columns: {sorted(missing_required)}"
        )
    if anchor_col not in available_cols:
        raise ValueError(f"make_irae_outcome_df: anchor_col {anchor_col!r} missing from input.")

    pat = df[available_cols].drop_duplicates(ID_COL).set_index(ID_COL)

    pat["FIRST_RECORD_DATE"] = pd.to_datetime(pat["FIRST_RECORD_DATE"], errors="coerce")
    pat["LAST_CONTACT_DATE"] = pd.to_datetime(pat["LAST_CONTACT_DATE"], errors="coerce")
    pat["IRAE"] = pd.to_numeric(pat["IRAE"], errors="coerce").fillna(0).astype(int)
    if "AGE_AT_TREATMENTSTART" in pat.columns:
        pat["AGE_AT_TREATMENTSTART"] = pd.to_numeric(pat["AGE_AT_TREATMENTSTART"], errors="coerce")
    pat[anchor_col] = pd.to_numeric(pat[anchor_col], errors="coerce").astype(float)

    landmark_time = pat[anchor_col].astype(float) + float(landmark_offset_days)
    t_irae_from_first_record = (
        pat["LAST_CONTACT_DATE"] - pat["FIRST_RECORD_DATE"]
    ).dt.days.astype(float)
    pat["t_irae_from_first_record"] = t_irae_from_first_record
    pat["t_irae"] = t_irae_from_first_record - landmark_time

    valid = (
        pat["FIRST_RECORD_DATE"].notna()
        & pat[anchor_col].notna()
        & pat[anchor_col].ge(0)
        & pat["t_irae"].notna()
        & pat["t_irae"].gt(0)
    )
    return pat.loc[valid].copy()
