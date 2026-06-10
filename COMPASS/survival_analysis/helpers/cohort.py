"""Shared cohort construction and pre-landmark lab feature engineering.

This module is the single home for the landmarked-cohort builders that several
survival-analysis entry points consume: ``build_prediction_inputs`` (the source
of truth for splits / aggregated + longitudinal inputs), ``build_genomic_inputs``
(sample-anchored variant + lab features), and the analysis scripts that re-export
these via ``cox_aggregated``.

Patient identifiers are configurable at runtime. ``ID_COL`` / ``AGE_COL`` default
to the PROFILE schema; entry points that operate on a different schema (e.g. CAIA
``person_id`` / ``AGE_AT_DIAGNOSIS``) call :func:`configure_id_columns` once in
``main`` so the builders below pick up the right columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Runtime-configurable patient-identifier / age columns. Default to the PROFILE
# schema; reconfigured via configure_id_columns() by entry points on other
# schemas (CAIA passes person_id / AGE_AT_DIAGNOSIS).
ID_COL = "DFCI_MRN"
AGE_COL = "AGE_AT_TREATMENTSTART"

PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}
MIN_SLOPE_OBS = 4
MIN_SLOPE_UNIQUE_TIMES = 3
MIN_SLOPE_SPAN_DAYS = 14.0
MIN_DELTA_OBS = 2


def configure_id_columns(id_col: str, age_col: str) -> None:
    """Set the patient-id / age columns used by every builder in this module.

    Entry points call this once in ``main`` (alongside setting their own module
    globals) so the cohort builders below operate on the requested schema.
    """
    global ID_COL, AGE_COL
    ID_COL = id_col
    AGE_COL = age_col


def normalize_landmark_days(raw_landmark_days: list[int]) -> list[int]:
    landmark_days: list[int] = []
    for raw_day in raw_landmark_days:
        day = int(raw_day)
        if day < 0:
            raise ValueError(f"Landmark days must be non-negative, got {day}.")
        if day not in landmark_days:
            landmark_days.append(day)
    return sorted(landmark_days)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _coerce_duration(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce").astype(float)


def _derive_duration(
    patient_df: pd.DataFrame,
    *,
    duration_col: str,
    event_date_col: str,
    fallback_duration_col: str | None = None,
) -> pd.Series:
    if duration_col in patient_df.columns:
        existing = _coerce_duration(patient_df[duration_col])
        if existing is not None:
            return existing

    derived = pd.Series(np.nan, index=patient_df.index, dtype=float)
    if event_date_col in patient_df.columns and "FIRST_RECORD_DATE" in patient_df.columns:
        event_date = _coerce_datetime(patient_df[event_date_col])
        first_record = _coerce_datetime(patient_df["FIRST_RECORD_DATE"])
        derived = (event_date - first_record).dt.days.astype(float)

    if fallback_duration_col and fallback_duration_col in patient_df.columns:
        fallback = _coerce_duration(patient_df[fallback_duration_col])
        if fallback is not None:
            derived = derived.fillna(fallback)

    return derived


def _coerce_platinum(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    platinum = series.astype(str).str.upper().isin(PLATINUM_MEDS)
    return numeric.fillna(platinum.astype(int)).fillna(0).astype(int)


def make_outcome_df(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int = 0,
    anchor_col: str = "t_first_treatment",
    extra_anchor_cols: tuple[str, ...] = (),
    require_first_treatment: bool = True,
) -> pd.DataFrame:
    """Build the per-patient outcome table rebased to a landmark.

    Args:
        landmark_offset_days: days added to ``anchor_col`` to define the landmark.
        anchor_col: column whose value (in days from first record) anchors the
            landmark. Defaults to ``t_first_treatment``; the genomic arm passes
            ``t_sample``.
        extra_anchor_cols: additional patient-level columns to preserve through
            the dedup (e.g. ``("t_sample", "SAMPLE_COLLECTION_DT")`` for the
            genomic arm). These are kept in the returned frame.
        require_first_treatment: whether the cohort filter requires
            ``FIRST_TREATMENT == 1``. Off for the genomic arm where treatment
            timing is irrelevant to the outcome window.
    """
    patient_level_cols = [
        ID_COL,
        AGE_COL,
        "FIRST_RECORD_DATE",
        "DIAGNOSIS_DATE",
        "FIRST_TREATMENT_DATE",
        "FIRST_TREATMENT",
        "LAST_CONTACT_DATE",
        "PLATINUM_DATE",
        "PLATINUM",
        "DEATH",
        "t_diagnosis",
        "t_first_treatment",
        "t_platinum",
        "t_last_contact",
        "t_death",
        *extra_anchor_cols,
    ]
    available_cols = [col for col in patient_level_cols if col in df.columns]
    if ID_COL not in available_cols:
        raise ValueError(f"Input data must contain the id column {ID_COL!r}.")

    pat = df[available_cols].drop_duplicates(ID_COL).set_index(ID_COL)

    if "FIRST_RECORD_DATE" not in pat.columns:
        if "LAB_DATE" not in df.columns:
            raise ValueError("Input data must contain FIRST_RECORD_DATE or LAB_DATE.")
        first_record = _coerce_datetime(df["LAB_DATE"]).groupby(df[ID_COL]).min()
        pat["FIRST_RECORD_DATE"] = first_record

    for date_col in [
        "FIRST_RECORD_DATE",
        "DIAGNOSIS_DATE",
        "FIRST_TREATMENT_DATE",
        "LAST_CONTACT_DATE",
        "PLATINUM_DATE",
    ]:
        if date_col in pat.columns:
            pat[date_col] = _coerce_datetime(pat[date_col])

    if AGE_COL in pat.columns:
        pat[AGE_COL] = pd.to_numeric(pat[AGE_COL], errors="coerce")
    else:
        pat[AGE_COL] = np.nan
    pat["DEATH"] = pd.to_numeric(pat.get("DEATH"), errors="coerce").fillna(0).astype(int)
    pat["PLATINUM"] = _coerce_platinum(pat.get("PLATINUM", pd.Series(0, index=pat.index)))
    pat["FIRST_TREATMENT"] = pd.to_numeric(
        pat.get(
            "FIRST_TREATMENT",
            pat.get("FIRST_TREATMENT_DATE", pd.Series(index=pat.index)).notna(),
        ),
        errors="coerce",
    ).fillna(0).astype(int)

    pat["t_last_contact"] = _derive_duration(
        pat,
        duration_col="t_last_contact",
        event_date_col="LAST_CONTACT_DATE",
    )
    pat["t_death"] = _derive_duration(
        pat,
        duration_col="t_death",
        event_date_col="LAST_CONTACT_DATE",
        fallback_duration_col="t_last_contact",
    )
    pat["t_diagnosis"] = _derive_duration(
        pat,
        duration_col="t_diagnosis",
        event_date_col="DIAGNOSIS_DATE",
        fallback_duration_col="t_last_contact",
    )
    pat["t_first_treatment"] = _derive_duration(
        pat,
        duration_col="t_first_treatment",
        event_date_col="FIRST_TREATMENT_DATE",
        fallback_duration_col="t_last_contact",
    )
    pat["t_platinum"] = _derive_duration(
        pat,
        duration_col="t_platinum",
        event_date_col="PLATINUM_DATE",
    )
    pat["t_platinum"] = pat["t_platinum"].where(
        pat["PLATINUM"].eq(1),
        pat["t_platinum"].fillna(pat["t_last_contact"]),
    )

    if anchor_col not in pat.columns:
        raise ValueError(f"make_outcome_df: anchor_col {anchor_col!r} missing from input.")
    landmark_time = pat[anchor_col].astype(float) + float(landmark_offset_days)
    for duration_col in ["t_last_contact", "t_death", "t_platinum"]:
        pat[f"{duration_col}_from_first_record"] = pat[duration_col]
        pat[duration_col] = pat[duration_col].astype(float) - landmark_time

    platinum_event_time = np.where(pat["PLATINUM"].eq(1), pat["t_platinum"], np.inf)
    death_event_time = np.where(pat["DEATH"].eq(1), pat["t_death"], np.inf)
    first_event_time = np.minimum(platinum_event_time, death_event_time)

    pat["EITHER"] = np.isfinite(first_event_time).astype(int)
    pat["t_either"] = np.where(pat["EITHER"].eq(1), first_event_time, pat["t_death"])

    valid = (
        pat["FIRST_RECORD_DATE"].notna()
        & pat[anchor_col].notna()
        & pat[anchor_col].ge(0)
        & pat["t_platinum"].notna()
        & pat["t_death"].notna()
        & pat["t_last_contact"].notna()
        & pat["t_either"].notna()
        & pat["t_platinum"].gt(0)
        & pat["t_death"].gt(0)
        & pat["t_last_contact"].gt(0)
        & pat["t_either"].gt(0)
    )
    if require_first_treatment:
        valid = valid & pat["FIRST_TREATMENT"].eq(1)
    return pat.loc[valid].copy()


def build_pre_treatment_lab_long(
    df: pd.DataFrame,
    *,
    cohort_index: pd.Index | None = None,
    landmark_offset_days: int = 0,
    anchor_col: str = "t_first_treatment",
    anchor_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Long-format pre-landmark lab observations used for canonical-lab selection.

    Returns columns DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab, <anchor_col>.
    Restricts to observations with t_lab < <anchor_col> + landmark_offset_days
    so the lab presence used for coverage matches the aggregated feature
    engineering and Dynamic DeepHit person-period builder windows.

    ``anchor_series`` (MRN-indexed) overrides the column lookup, supporting the
    genomic arm where ``t_sample`` is per-patient and not carried on every row.
    """
    base_required = {ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"}
    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_pre_treatment_lab_long missing columns: {sorted(missing)}"
        )

    cols = list(base_required)
    if anchor_series is None:
        if anchor_col not in df.columns:
            raise ValueError(
                f"build_pre_treatment_lab_long needs anchor column {anchor_col!r} on df or via anchor_series."
            )
        cols.append(anchor_col)
    out = df[cols].copy()
    out["LAB_NAME"] = out["LAB_NAME"].astype(str).str.strip()
    out["LAB_VALUE"] = pd.to_numeric(out["LAB_VALUE"], errors="coerce")
    out["t_lab"] = pd.to_numeric(out["t_lab"], errors="coerce")
    if anchor_series is not None:
        out[anchor_col] = out[ID_COL].map(anchor_series.astype(float)).astype(float)
    else:
        out[anchor_col] = pd.to_numeric(out[anchor_col], errors="coerce")
    out = out.dropna(subset=[ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab", anchor_col])
    landmark_t = out[anchor_col] + float(landmark_offset_days)
    out = out.loc[out["t_lab"] < landmark_t].copy()
    if cohort_index is not None:
        out = out.loc[out[ID_COL].isin(cohort_index)].copy()
    return out


def _compute_patient_lab_slopes(pre_treatment: pd.DataFrame) -> pd.DataFrame:
    """OLS slope of LAB_VALUE vs t_lab (per day) per (DFCI_MRN, LAB_NAME).

    Returns NaN unless a patient has enough observations, enough unique
    timepoints, and a sufficient time span; these stricter requirements keep
    the original OLS-style slope definition while reducing unstable estimates
    from sparse, short-span trajectories.
    """
    def _slope(group: pd.DataFrame) -> float:
        if len(group) < MIN_SLOPE_OBS:
            return np.nan
        if group["t_lab"].nunique(dropna=True) < MIN_SLOPE_UNIQUE_TIMES:
            return np.nan

        x = group["t_lab"].to_numpy(dtype=float)
        y = group["LAB_VALUE"].to_numpy(dtype=float)
        if (x.max() - x.min()) < MIN_SLOPE_SPAN_DAYS:
            return np.nan
        cov = np.cov(x, y, ddof=0)
        var_x = cov[0, 0]
        if not np.isfinite(var_x) or var_x <= 0:
            return np.nan
        slope = float(cov[0, 1] / var_x)
        return slope if np.isfinite(slope) else np.nan

    slopes = (
        pre_treatment.groupby([ID_COL, "LAB_NAME"])[["t_lab", "LAB_VALUE"]]
        .apply(_slope)
        .rename("slope")
        .reset_index()
    )
    return slopes


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int = 0,
    anchor_col: str = "t_first_treatment",
    anchor_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-patient lab summary features for the pre-landmark window.

    Args:
        landmark_offset_days: days added to ``anchor_col`` to define the landmark.
        anchor_col: column on ``df`` whose value (in days from first record) is
            the anchor. Default ``t_first_treatment``; the genomic arm uses
            ``t_sample``.
        anchor_series: optional MRN-indexed Series providing per-patient anchor
            values when the column isn't carried on every lab row (e.g.
            ``t_sample`` joined externally). Takes precedence over ``anchor_col``.
    """
    working = df.copy()
    required_cols = {ID_COL, "LAB_NAME", "LAB_VALUE"}
    missing_required = required_cols - set(working.columns)
    if missing_required:
        missing_str = ", ".join(sorted(missing_required))
        raise ValueError(f"Input data is missing required columns for feature engineering: {missing_str}")

    working["LAB_NAME"] = working["LAB_NAME"].astype(str).str.strip()
    working["LAB_VALUE"] = pd.to_numeric(working["LAB_VALUE"], errors="coerce")

    if "t_lab" not in working.columns:
        if "LAB_DATE" not in working.columns:
            raise ValueError("Input data must contain t_lab or LAB_DATE.")
        if "FIRST_RECORD_DATE" not in working.columns:
            working["FIRST_RECORD_DATE"] = _coerce_datetime(working["LAB_DATE"]).groupby(working[ID_COL]).transform("min")
        working["t_lab"] = (
            _coerce_datetime(working["LAB_DATE"]) - _coerce_datetime(working["FIRST_RECORD_DATE"])
        ).dt.days.astype(float)
    else:
        working["t_lab"] = _coerce_duration(working["t_lab"])

    if anchor_series is not None:
        anchor_map = anchor_series.astype(float)
        working[anchor_col] = working[ID_COL].map(anchor_map).astype(float)
    elif anchor_col not in working.columns:
        if anchor_col == "t_first_treatment" and {"FIRST_TREATMENT_DATE", "FIRST_RECORD_DATE"}.issubset(working.columns):
            working["t_first_treatment"] = (
                _coerce_datetime(working["FIRST_TREATMENT_DATE"])
                - _coerce_datetime(working["FIRST_RECORD_DATE"])
            ).dt.days.astype(float)
        else:
            raise ValueError(
                f"build_feature_matrix needs anchor column {anchor_col!r} on the df or via anchor_series."
            )
    else:
        working[anchor_col] = _coerce_duration(working[anchor_col])

    working = working.dropna(
        subset=[ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab", anchor_col]
    )

    landmark_time = working[anchor_col].astype(float) + float(landmark_offset_days)
    pre_treatment = working.loc[working["t_lab"].lt(landmark_time)].copy()
    if pre_treatment.empty:
        raise ValueError("No pre-landmark lab rows were available to build lab summary features.")

    sort_cols = [ID_COL, "LAB_NAME", "t_lab"]
    if "LAB_DATE" in pre_treatment.columns:
        pre_treatment["LAB_DATE"] = _coerce_datetime(pre_treatment["LAB_DATE"])
        sort_cols.append("LAB_DATE")
    pre_treatment = pre_treatment.sort_values(sort_cols)

    feature_long = (
        pre_treatment.groupby([ID_COL, "LAB_NAME"])["LAB_VALUE"]
        .agg(
            mean="mean",
            min="min",
            max="max",
            first="first",
            last="last",
            n_observations="count",
        )
        .reset_index()
    )
    feature_long["delta"] = np.where(
        feature_long["n_observations"] >= MIN_DELTA_OBS,
        feature_long["last"] - feature_long["first"],
        np.nan,
    )
    feature_long = feature_long.drop(columns=["first"])
    slope_long = _compute_patient_lab_slopes(pre_treatment)
    feature_long = feature_long.merge(slope_long, on=[ID_COL, "LAB_NAME"], how="left")
    feature_df = (
        feature_long.set_index([ID_COL, "LAB_NAME"])
        .stack()
        .rename("value")
        .reset_index()
        .rename(columns={"level_2": "feature_stat"})
    )
    feature_df["feature_name"] = feature_df["LAB_NAME"] + "__" + feature_df["feature_stat"]
    feature_df = feature_df.pivot(index=ID_COL, columns="feature_name", values="value")
    feature_df = feature_df.sort_index(axis=1)

    print(f"Raw feature matrix: {feature_df.shape[0]} patients x {feature_df.shape[1]} summary-lab features")
    return feature_df


def build_landmark_merged(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome_df = make_outcome_df(df, landmark_offset_days=landmark_offset_days)
    print(f"Outcome table @ landmark +{landmark_offset_days}d: {len(outcome_df)} patients")

    print(f"Building raw aggregated lab summary feature matrix through landmark +{landmark_offset_days}d...")
    feature_df = build_feature_matrix(df, landmark_offset_days=landmark_offset_days)

    merged = feature_df.join(outcome_df, how="inner")
    merged = merged.loc[merged[AGE_COL].notna()].copy()
    if merged.empty:
        raise ValueError("No patients have both engineered features and valid outcomes.")
    return outcome_df, feature_df, merged


def build_landmark_availability_table(
    merged_by_landmark: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Index]:
    if not merged_by_landmark:
        raise ValueError("No landmark cohorts were provided.")

    all_mrns = pd.Index([])
    common_mrns: pd.Index | None = None
    for merged in merged_by_landmark.values():
        all_mrns = all_mrns.union(merged.index)
        common_mrns = merged.index if common_mrns is None else common_mrns.intersection(merged.index)

    availability = pd.DataFrame(index=all_mrns)
    landmark_cols: list[str] = []
    for landmark_day in sorted(merged_by_landmark):
        col = f"eligible_landmark_{landmark_day}"
        landmark_cols.append(col)
        availability[col] = availability.index.isin(merged_by_landmark[landmark_day].index)
    availability["eligible_all_landmarks"] = availability[landmark_cols].all(axis=1)
    availability = availability.rename_axis(ID_COL).reset_index()
    return availability, (common_mrns if common_mrns is not None else pd.Index([]))
