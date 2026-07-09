"""Per-patient irAE outcome construction for the IPIO landmark cohort.

Mirrors the landmark-rebasing pattern of survival_common.cohort.make_outcome_df.
The cause-specific duration/event pair (t_irae/IRAE) treats death and censor
both as right-censoring, unchanged from the original design. Additionally
builds a 3-level `event_type` (0=censored, 1=irAE, 2=death) consumed by
survival_common.finegray for the Fine-Gray competing-risks univariate arm,
where death competes with irAE instead of being treated as plain censoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from survival_common import cohort

# Kept for backward compatibility (callers/tests that import ipio_cohort.ID_COL).
# The runtime source of truth is survival_common.cohort.ID_COL, set by
# configure_id_columns(); make_irae_outcome_df reads that dynamically so a
# non-default --id-col is honored instead of this import-time constant.
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
    AGE_AT_TREATMENTSTART + IRAE + DEATH + t_irae + t_irae_from_first_record +
    event_type + extra_anchor_cols + the anchor column + FIRST_RECORD_DATE /
    LAST_CONTACT_DATE (the latter two are debug-only duration duplicates;
    downstream callers may drop them before persisting the final aggregated
    table).

    `event_type` is a 3-level column (0=censored, 1=irAE, 2=death) for the
    Fine-Gray competing-risks univariate arm (survival_common.finegray). irAE
    takes precedence when both are flagged, matching survival_common.cohort.
    make_outcome_df's platinum-over-death precedence. t_irae is reused as the
    shared subdistribution duration for death subjects too: IPIO has no
    separate death-date column, and LAST_CONTACT_DATE already equals the death
    date when a patient's raw `event` was 'death' (see
    longitudinal_data_processing.py's DEATH derivation), so t_irae (time to
    LAST_CONTACT_DATE, landmark-rebased) is already the correct competing-event
    time.
    """
    # Read the id column at call time from the shared cohort config so a runtime
    # configure_id_columns() (non-default --id-col) is honored; falls back to the
    # default "DFCI_MRN" when never configured.
    id_col = cohort.ID_COL

    cancer_type_cols = tuple(c for c in df.columns if c.startswith("CANCER_TYPE_"))

    patient_level_cols = [
        id_col,
        *BASELINE_COVARIATE_COLS,
        *cancer_type_cols,
        "FIRST_RECORD_DATE",
        "LAST_CONTACT_DATE",
        "IRAE",
        "DEATH",
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
    if id_col not in available_cols:
        raise ValueError(f"Input data must contain the id column {id_col!r}.")

    required = {"FIRST_RECORD_DATE", "LAST_CONTACT_DATE", "IRAE"}
    missing_required = required - set(available_cols)
    if missing_required:
        raise ValueError(
            f"make_irae_outcome_df: input data is missing required columns: {sorted(missing_required)}"
        )
    if anchor_col not in available_cols:
        raise ValueError(f"make_irae_outcome_df: anchor_col {anchor_col!r} missing from input.")

    pat = df[available_cols].drop_duplicates(id_col).set_index(id_col)

    pat["FIRST_RECORD_DATE"] = pd.to_datetime(pat["FIRST_RECORD_DATE"], errors="coerce")
    pat["LAST_CONTACT_DATE"] = pd.to_datetime(pat["LAST_CONTACT_DATE"], errors="coerce")
    pat["IRAE"] = pd.to_numeric(pat["IRAE"], errors="coerce").fillna(0).astype(int)
    if "DEATH" in pat.columns:
        pat["DEATH"] = pd.to_numeric(pat["DEATH"], errors="coerce").fillna(0).astype(int)
    else:
        pat["DEATH"] = 0
    if "AGE_AT_TREATMENTSTART" in pat.columns:
        pat["AGE_AT_TREATMENTSTART"] = pd.to_numeric(pat["AGE_AT_TREATMENTSTART"], errors="coerce")
    pat[anchor_col] = pd.to_numeric(pat[anchor_col], errors="coerce").astype(float)

    landmark_time = pat[anchor_col].astype(float) + float(landmark_offset_days)
    t_irae_from_first_record = (
        pat["LAST_CONTACT_DATE"] - pat["FIRST_RECORD_DATE"]
    ).dt.days.astype(float)
    pat["t_irae_from_first_record"] = t_irae_from_first_record
    pat["t_irae"] = t_irae_from_first_record - landmark_time

    pat["event_type"] = np.where(
        pat["IRAE"].eq(1), 1, np.where(pat["DEATH"].eq(1), 2, 0)
    ).astype(int)

    valid = (
        pat["FIRST_RECORD_DATE"].notna()
        & pat[anchor_col].notna()
        & pat[anchor_col].ge(0)
        & pat["t_irae"].notna()
        & pat["t_irae"].gt(0)
    )
    return pat.loc[valid].copy()
