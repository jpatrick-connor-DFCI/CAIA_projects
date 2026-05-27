"""Cohort loaders shared by PROFILE and CAIA pipelines.

  * `load_profile_longitudinal(path)` reads the existing PROFILE
    `longitudinal_prediction_data.csv` and splits it into patient-level and
    lab-level frames.
  * `load_caia_parquet(path)` reads the new CAIA cohort parquet, renames
    columns to the PROFILE-pipeline names, recomputes timing fields from the
    raw datetime objects (the parquet's pre-computed `days_relative_to_*`
    columns are sanity-checked but not trusted), and splits into patient/lab
    frames.

Both loaders return `(patient_df, labs_df)` where:
  * `patient_df` is one row per patient ID with outcomes + demographics.
  * `labs_df` is one row per (patient, lab measurement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# CAIA parquet column -> PROFILE pipeline column name
CAIA_COLUMN_RENAME = {
    "age_at_diagnosis": "AGE_AT_DIAGNOSIS",
    "diagnosis_date": "DIAGNOSIS_DATE",
    "first_treatment_start_date_post_diagnosis": "FIRST_TREATMENT_DATE",
    "platinum_start_date": "PLATINUM_DATE",
    "last_followup_date": "LAST_CONTACT_DATE",
    "is_deceased": "DEATH",
    "event_platinum": "PLATINUM",
    "lab_name": "LAB_NAME",
    "lab_value": "LAB_VALUE",
    "lab_unit": "LAB_UNIT",
    "measurement_date": "LAB_DATE",
}

CAIA_PATIENT_COLS = [
    "person_id",
    "race",
    "ethnicity",
    "platinum_type",
    "AGE_AT_DIAGNOSIS",
    "DIAGNOSIS_DATE",
    "FIRST_TREATMENT_DATE",
    "PLATINUM_DATE",
    "LAST_CONTACT_DATE",
    "DEATH",
    "PLATINUM",
    "t_diagnosis",
    "t_first_treatment",
    "t_platinum",
    "t_death",
    "t_last_contact",
    "t_dx_to_tx",
]

CAIA_LAB_COLS = [
    "person_id",
    "LAB_NAME",
    "LAB_VALUE",
    "LAB_UNIT",
    "LAB_DATE",
    "t_lab",
]


def _parse_dates(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")


def load_caia_parquet(
    path: str | Path,
    *,
    id_col: str = "person_id",
    admin_censor_days: int | None = None,
    timing_tolerance_days: int = 1,
    write_csvs: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the CAIA cohort parquet and return (patient_df, labs_df).

    Recomputes `t_lab`, `t_platinum`, `t_death`, `t_dx_to_tx`,
    `t_first_treatment` (== 0 by definition), `t_last_contact`, and
    `t_diagnosis` from the raw datetime columns rather than trusting the
    parquet's pre-computed `days_relative_to_*` fields.  The pre-computed
    values are still loaded and compared; rows whose recomputed value differs
    from the parquet's by more than `timing_tolerance_days` are reported but
    not dropped.

    By default this loader does not administrative-censor outcomes. Downstream
    model runners apply a landmark-relative finite prediction window; load-time
    censoring would shorten nonzero landmark windows. Pass `admin_censor_days`
    only for standalone summaries that explicitly need treatment-start-relative
    censoring.

    CAIA parquet has no explicit death date; for deceased patients we use
    `LAST_CONTACT_DATE` as the proxy death date (which is what the parquet's
    `time_to_death_or_censor` was presumably built from).
    """
    df = pd.read_parquet(path)
    df = df.rename(columns=CAIA_COLUMN_RENAME)
    _parse_dates(df, ["DIAGNOSIS_DATE", "FIRST_TREATMENT_DATE",
                      "PLATINUM_DATE", "LAST_CONTACT_DATE", "LAB_DATE"])

    # Death-date proxy: deceased -> LAST_CONTACT_DATE, alive -> NaT.
    df["DEATH_DATE"] = df["LAST_CONTACT_DATE"].where(df["DEATH"].astype(bool), pd.NaT)

    # --- timing recomputation from datetime objects -----------------------
    ft = df["FIRST_TREATMENT_DATE"]

    df["t_lab"] = (df["LAB_DATE"] - ft).dt.days.astype("Float64")
    df["t_first_treatment"] = 0
    df["t_diagnosis"] = (df["DIAGNOSIS_DATE"] - ft).dt.days.astype("Float64")
    df["t_last_contact"] = (df["LAST_CONTACT_DATE"] - ft).dt.days.astype("Float64")
    df["t_dx_to_tx"] = (ft - df["DIAGNOSIS_DATE"]).dt.days.astype("Float64")

    platinum_anchor = df["PLATINUM_DATE"].fillna(df["LAST_CONTACT_DATE"])
    df["t_platinum"] = (platinum_anchor - ft).dt.days.astype("Float64")

    death_anchor = df["DEATH_DATE"].fillna(df["LAST_CONTACT_DATE"])
    df["t_death"] = (death_anchor - ft).dt.days.astype("Float64")

    # Sanity check against parquet's pre-computed columns (do not fail).
    def _mismatch(recomputed: pd.Series, original_name: str) -> int:
        if original_name not in df.columns:
            return 0
        orig = pd.to_numeric(df[original_name], errors="coerce")
        diff = (recomputed - orig).abs()
        return int(((diff > timing_tolerance_days) & diff.notna()).sum())

    mismatches = {
        "t_platinum vs time_to_platinum_or_censor":
            _mismatch(df["t_platinum"], "time_to_platinum_or_censor"),
        "t_death vs time_to_death_or_censor":
            _mismatch(df["t_death"], "time_to_death_or_censor"),
        "t_lab vs days_relative_to_first_treatment_start_post_diagnosis":
            _mismatch(df["t_lab"], "days_relative_to_first_treatment_start_post_diagnosis"),
        "t_diagnosis vs days_relative_to_diagnosis":
            _mismatch(df["t_diagnosis"], "days_relative_to_diagnosis"),
        "t_last_contact vs days_relative_to_last_followup":
            _mismatch(df["t_last_contact"], "days_relative_to_last_followup"),
    }
    if verbose:
        n_rows = len(df)
        for label, n_bad in mismatches.items():
            pct = 100.0 * n_bad / max(n_rows, 1)
            print(f"  [load_caia_parquet] {label}: {n_bad:,} rows mismatched "
                  f"by >{timing_tolerance_days}d ({pct:.2f}%)")

    if admin_censor_days is not None:
        cap = float(admin_censor_days)
        for evt_col, dur_col in [("PLATINUM", "t_platinum"), ("DEATH", "t_death")]:
            late = df[dur_col] > cap
            df.loc[late, evt_col] = 0
            df.loc[late, dur_col] = cap

    # --- split into patient_df / labs_df ---------------------------------
    pat_cols = [c for c in CAIA_PATIENT_COLS if c in df.columns]
    patient_df = (
        df[pat_cols]
        .drop_duplicates(subset=[id_col])
        .reset_index(drop=True)
    )

    lab_cols = [c for c in CAIA_LAB_COLS if c in df.columns]
    labs_df = (
        df.loc[df["LAB_NAME"].notna(), lab_cols]
        .reset_index(drop=True)
    )

    if write_csvs is not None:
        write_csvs = Path(write_csvs)
        write_csvs.mkdir(parents=True, exist_ok=True)
        patient_df.to_csv(write_csvs / "caia_patient_df.csv", index=False)
        labs_df.to_csv(write_csvs / "caia_labs_long.csv", index=False)

    return patient_df, labs_df


def load_profile_longitudinal(
    path: str | Path,
    *,
    id_col: str = "DFCI_MRN",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the PROFILE-cohort `longitudinal_prediction_data.csv` and split
    into (patient_df, labs_df).

    The CSV is long-format with patient-level columns repeating per lab row.
    Patient-level fields are deduplicated by `id_col`; lab-level fields stay
    long.
    """
    df = pd.read_csv(path, low_memory=False)
    _parse_dates(df, ["DIAGNOSIS_DATE", "FIRST_TREATMENT_DATE",
                      "PLATINUM_DATE", "LAST_CONTACT_DATE", "LAB_DATE",
                      "FIRST_RECORD_DATE"])

    if "t_dx_to_tx" not in df.columns and {"DIAGNOSIS_DATE", "FIRST_TREATMENT_DATE"} <= set(df.columns):
        df["t_dx_to_tx"] = (df["FIRST_TREATMENT_DATE"] - df["DIAGNOSIS_DATE"]).dt.days.astype("Float64")

    patient_level = [
        id_col, "AGE_AT_TREATMENTSTART", "FIRST_RECORD_DATE", "DIAGNOSIS_DATE",
        "FIRST_TREATMENT_DATE", "FIRST_TREATMENT", "LAST_CONTACT_DATE",
        "DEATH", "PLATINUM_MEDICATION", "PLATINUM_DATE", "PLATINUM",
        "t_diagnosis", "t_first_treatment", "t_platinum", "t_last_contact",
        "t_death", "t_dx_to_tx",
    ]
    pat_cols = [c for c in patient_level if c in df.columns]
    patient_df = (
        df[pat_cols]
        .drop_duplicates(subset=[id_col])
        .reset_index(drop=True)
    )

    lab_cols_candidates = [id_col, "LAB_NAME", "LAB_VALUE", "LAB_UNIT",
                           "LAB_DATE", "t_lab"]
    lab_cols = [c for c in lab_cols_candidates if c in df.columns]
    labs_df = (
        df.loc[df["LAB_NAME"].notna(), lab_cols]
        .reset_index(drop=True)
    )
    return patient_df, labs_df


def record_span_days(labs_df: pd.DataFrame, *, id_col: str, date_col: str = "LAB_DATE") -> pd.Series:
    """Compute per-patient record span (max - min of `date_col`) in days."""
    g = labs_df.groupby(id_col)[date_col]
    span = (g.max() - g.min()).dt.days.astype("Float64")
    span.name = "record_span_days"
    return span
