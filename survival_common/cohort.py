"""Shared cohort construction and pre-landmark lab feature engineering.

This module is the single home for the landmarked-cohort builders that several
survival-analysis entry points consume: ``build_prediction_inputs`` (the source
of truth for splits / aggregated + longitudinal inputs), ``build_genomic_inputs``
(sample-anchored variant + lab features), and the analysis scripts that re-export
these via ``cox_aggregated``.

Patient identifiers are configurable at runtime. ``ID_COL`` / ``AGE_COL`` default
to the PROFILE schema; entry points can call :func:`configure_id_columns` once in
``main`` if they need alternate column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Runtime-configurable patient-identifier / age columns. Default to the PROFILE
# schema; reconfigured via configure_id_columns() by entry points that need
# alternate names.
ID_COL = "DFCI_MRN"
AGE_COL = "AGE_AT_TREATMENTSTART"

PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}
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
    anchor_col: str | None = "t_first_treatment",
    extra_anchor_cols: tuple[str, ...] = (),
    require_first_treatment: bool = True,
    max_followup_days: float | None = 3650.0,
) -> pd.DataFrame:
    """Build the per-patient outcome table rebased to a landmark.

    Args:
        landmark_offset_days: days added to ``anchor_col`` to define the landmark.
        anchor_col: column whose value (in days from first record) anchors the
            landmark. Defaults to ``t_first_treatment``; the genomic arm passes
            ``t_sample``. Pass ``None`` when the durations are already measured
            from the index date (e.g. COMPASS's treatment-anchor-relative clock):
            the landmark is then the pure offset ``landmark_offset_days`` from time
            0, with no anchor term and no anchor-based patient filtering.
        extra_anchor_cols: additional patient-level columns to preserve through
            the dedup (e.g. ``("t_sample", "SAMPLE_COLLECTION_DT")`` for the
            genomic arm). These are kept in the returned frame.
        require_first_treatment: whether the cohort filter requires
            ``FIRST_TREATMENT == 1``. Off for the genomic arm where treatment
            timing is irrelevant to the outcome window.
        max_followup_days: administrative-censoring horizon on the
            landmark-relative clock (days from time 0). Patients whose platinum
            or death event lands beyond this horizon are censored at the horizon
            (event flag -> 0, duration -> horizon); censored follow-up
            (``t_last_contact``) is clipped to the horizon too. Defaults to
            3650 (10 years): platinum >10y after the treatment anchor is very
            unlikely to reflect the modeled aggressive-transformation process,
            and the sparse tail destabilizes the Cox/Fine-Gray fits. Pass
            ``None`` to disable and use full follow-up.
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

    if anchor_col is None:
        # Durations are already measured from the index date; the landmark is a
        # pure offset from time 0 (no anchor term, no anchor-based filtering).
        landmark_time = float(landmark_offset_days)
    else:
        if anchor_col not in pat.columns:
            raise ValueError(f"make_outcome_df: anchor_col {anchor_col!r} missing from input.")
        landmark_time = pat[anchor_col].astype(float) + float(landmark_offset_days)
    for duration_col in ["t_last_contact", "t_death", "t_platinum"]:
        pat[f"{duration_col}_from_first_record"] = pat[duration_col]
        pat[duration_col] = pat[duration_col].astype(float) - landmark_time

    # Administrative censoring at max_followup_days (on the landmark-relative
    # clock). An event whose time exceeds the horizon becomes a censored
    # observation AT the horizon; censored follow-up is clipped to the horizon
    # too. Done before the event_type / first_event_time derivations below so
    # PLATINUM/DEATH/EITHER/event_type all stay consistent with the clipped
    # durations. The *_from_first_record columns are left uncapped (they record
    # the raw timing for diagnostics, not the modeled outcome).
    if max_followup_days is not None:
        horizon = float(max_followup_days)
        platinum_past = pat["PLATINUM"].eq(1) & pat["t_platinum"].gt(horizon)
        death_past = pat["DEATH"].eq(1) & pat["t_death"].gt(horizon)
        pat.loc[platinum_past, "PLATINUM"] = 0
        pat.loc[death_past, "DEATH"] = 0
        for duration_col in ["t_last_contact", "t_death", "t_platinum"]:
            pat[duration_col] = pat[duration_col].clip(upper=horizon)

    platinum_event_time = np.where(pat["PLATINUM"].eq(1), pat["t_platinum"], np.inf)
    death_event_time = np.where(pat["DEATH"].eq(1), pat["t_death"], np.inf)
    first_event_time = np.minimum(platinum_event_time, death_event_time)

    pat["EITHER"] = np.isfinite(first_event_time).astype(int)
    pat["t_either"] = np.where(pat["EITHER"].eq(1), first_event_time, pat["t_death"])

    # 3-level event type for Fine-Gray competing-risks fitting (survival_common.finegray):
    # 1 = event of interest (platinum), 2 = competing event (death), 0 = censored.
    # Platinum takes precedence when both are flagged -- t_platinum is already the
    # correct subdistribution time (time to platinum, regardless of later death).
    pat["event_type"] = np.where(
        pat["PLATINUM"].eq(1), 1, np.where(pat["DEATH"].eq(1), 2, 0)
    ).astype(int)

    # Individual validity conditions, kept separate so the attrition each one
    # causes can be reported below. Durations here are already landmark-rebased
    # (shifted by landmark_time above), so a non-positive value means the event /
    # last contact falls at or before the landmark.
    conditions: dict[str, pd.Series] = {
        "FIRST_RECORD_DATE notna": pat["FIRST_RECORD_DATE"].notna(),
        "t_platinum notna": pat["t_platinum"].notna(),
        "t_death notna": pat["t_death"].notna(),
        "t_last_contact notna": pat["t_last_contact"].notna(),
        "t_either notna": pat["t_either"].notna(),
        "t_platinum > 0": pat["t_platinum"].gt(0),
        "t_death > 0": pat["t_death"].gt(0),
        "t_last_contact > 0": pat["t_last_contact"].gt(0),
        "t_either > 0": pat["t_either"].gt(0),
    }
    if anchor_col is not None:
        # A real anchor column must be present and on-or-after first record; with
        # anchor_col=None the durations are already index-relative and there is no
        # anchor to gate on.
        conditions[f"{anchor_col} notna"] = pat[anchor_col].notna()
        conditions[f"{anchor_col} >= 0"] = pat[anchor_col].ge(0)
    if require_first_treatment:
        conditions["FIRST_TREATMENT == 1"] = pat["FIRST_TREATMENT"].eq(1)

    valid = pd.Series(True, index=pat.index)
    for cond in conditions.values():
        valid = valid & cond

    # Per-condition attrition report: for each condition, how many patients it
    # fails on its own ("failed alone") and how many are lost cumulatively as the
    # conditions are ANDed in order ("dropped this step"). "failed alone" columns
    # sum to more than the total dropped because a patient can fail several.
    n_in = len(pat)
    print(f"[make_outcome_df @ landmark +{landmark_offset_days}d] "
          f"validity attrition from {n_in} patients:")
    running = pd.Series(True, index=pat.index)
    for name, cond in conditions.items():
        failed_alone = int((~cond).sum())
        before = int(running.sum())
        running = running & cond
        dropped_step = before - int(running.sum())
        print(f"    {name:<24} failed alone={failed_alone:<5} "
              f"dropped this step={dropped_step:<5} remaining={int(running.sum())}")
    print(f"    => {int(valid.sum())}/{n_in} patients pass all conditions.")

    return pat.loc[valid].copy()


def build_pre_treatment_lab_long(
    df: pd.DataFrame,
    *,
    cohort_index: pd.Index | None = None,
    landmark_offset_days: int = 0,
    anchor_col: str | None = "t_first_treatment",
    anchor_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Long-format pre-landmark lab observations used for canonical-lab selection.

    Returns columns DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab[, <anchor_col>].
    Restricts to observations with t_lab < <anchor_col> + landmark_offset_days
    so the lab presence used for coverage matches the aggregated feature
    engineering and Dynamic DeepHit person-period builder windows.

    ``anchor_series`` (MRN-indexed) overrides the column lookup, supporting the
    genomic arm where ``t_sample`` is per-patient and not carried on every row.

    Pass ``anchor_col=None`` when ``t_lab`` is already measured from the index
    date (COMPASS's treatment-anchor clock): the window is then the pure offset
    ``t_lab < landmark_offset_days`` with no anchor column required.
    """
    base_required = {ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"}
    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_pre_treatment_lab_long missing columns: {sorted(missing)}"
        )

    cols = list(base_required)
    if anchor_col is not None and anchor_series is None:
        if anchor_col not in df.columns:
            raise ValueError(
                f"build_pre_treatment_lab_long needs anchor column {anchor_col!r} on df or via anchor_series."
            )
        cols.append(anchor_col)
    out = df[cols].copy()
    out["LAB_NAME"] = out["LAB_NAME"].astype(str).str.strip()
    out["LAB_VALUE"] = pd.to_numeric(out["LAB_VALUE"], errors="coerce")
    out["t_lab"] = pd.to_numeric(out["t_lab"], errors="coerce")
    dropna_cols = [ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"]
    if anchor_col is not None:
        if anchor_series is not None:
            out[anchor_col] = out[ID_COL].map(anchor_series.astype(float)).astype(float)
        else:
            out[anchor_col] = pd.to_numeric(out[anchor_col], errors="coerce")
        dropna_cols.append(anchor_col)
    out = out.dropna(subset=dropna_cols)
    if anchor_col is None:
        # t_lab is already index-relative; window is the pure offset from 0.
        landmark_t = float(landmark_offset_days)
    else:
        landmark_t = out[anchor_col] + float(landmark_offset_days)
    out = out.loc[out["t_lab"] < landmark_t].copy()
    if cohort_index is not None:
        out = out.loc[out[ID_COL].isin(cohort_index)].copy()
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int = 0,
    anchor_col: str | None = "t_first_treatment",
    anchor_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-patient lab summary features for the pre-landmark window.

    Args:
        landmark_offset_days: days added to ``anchor_col`` to define the landmark.
        anchor_col: column on ``df`` whose value (in days from first record) is
            the anchor. Default ``t_first_treatment``; the genomic arm uses
            ``t_sample``. Pass ``None`` when ``t_lab`` is already index-relative
            (COMPASS's treatment-anchor clock): the window is the pure offset
            ``t_lab < landmark_offset_days`` with no anchor column required.
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

    dropna_cols = [ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"]
    if anchor_col is None:
        # t_lab is already index-relative; the pre-landmark window is the pure
        # offset from time 0, with no anchor column required.
        working = working.dropna(subset=dropna_cols)
        landmark_time = float(landmark_offset_days)
    else:
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
        working = working.dropna(subset=[*dropna_cols, anchor_col])
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
    anchor_col: str | None = "t_first_treatment",
    require_first_treatment: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the landmarked outcome + feature merge for a given anchor.

    ``anchor_col`` defaults to ``t_first_treatment`` (the standard pipeline). A
    non-standard anchor (e.g. ``t_sample``) is not among the patient-level columns
    make_outcome_df keeps by default, so it is passed through ``extra_anchor_cols``
    to survive the per-patient dedup — mirroring how the genomic arm carries
    ``t_sample``. Pass ``anchor_col=None`` when durations are already index-relative
    (COMPASS): the landmark is then a pure offset and no anchor column is needed.
    """
    extra_anchor_cols = (
        () if anchor_col in (None, "t_first_treatment") else (anchor_col,)
    )
    outcome_df = make_outcome_df(
        df,
        landmark_offset_days=landmark_offset_days,
        anchor_col=anchor_col,
        extra_anchor_cols=extra_anchor_cols,
        require_first_treatment=require_first_treatment,
    )
    print(f"Outcome table @ landmark +{landmark_offset_days}d: {len(outcome_df)} patients")

    print(f"Building raw aggregated lab summary feature matrix through landmark +{landmark_offset_days}d...")
    feature_df = build_feature_matrix(
        df, landmark_offset_days=landmark_offset_days, anchor_col=anchor_col
    )

    n_outcome_only = len(outcome_df.index.difference(feature_df.index))
    n_feature_only = len(feature_df.index.difference(outcome_df.index))
    print(
        f"Feature/outcome index overlap @ landmark +{landmark_offset_days}d: "
        f"outcome_df n={len(outcome_df)}, feature_df n={len(feature_df)}; "
        f"{n_outcome_only} outcome-valid patients have no pre-landmark lab feature "
        f"(kept, features left all-NaN for downstream imputation); "
        f"{n_feature_only} feature-only patients dropped (no valid outcome)."
    )
    # Left join on outcome_df: every patient with a valid landmark outcome is kept
    # even if they have zero pre-landmark labs (build_feature_matrix's pivot only
    # emits rows for patients with >=1 qualifying lab, so an inner join here would
    # silently drop otherwise-eligible, just lab-sparse patients). Missing feature
    # columns are handled downstream via per-feature mean imputation (see
    # SimpleImputer usage in cox_models.py), so an all-NaN feature row is valid
    # input, not a defect.
    merged = outcome_df.join(feature_df, how="left")
    n_before_age_filter = len(merged)
    n_missing_age = int(merged[AGE_COL].isna().sum())
    merged = merged.loc[merged[AGE_COL].notna()].copy()
    print(
        f"Feature+outcome merge @ landmark +{landmark_offset_days}d: {n_before_age_filter} patients; "
        f"{AGE_COL} missing for {n_missing_age}; {len(merged)} remaining."
    )
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
