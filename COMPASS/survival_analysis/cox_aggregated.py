"""
Two-arm survival analysis on landmarked lab summary features.

Features per lab (all pre-landmark): mean, min, max, last, slope, delta
(last - first), n_observations.

Arm 1 (univariate, full dataset):
  - For each feature, fit Cox on [AGE + feature] using all patients.
  - Extract coefficient, HR per SD, 95% CI, and p-value.

Arm 1b (univariate, full dataset, n_obs-adjusted):
  - For each non-count feature, fit Cox on
    [AGE + matching LAB__n_observations + feature] using all patients.
  - Extract coefficients, HRs per SD, 95% CIs, and p-values.

Arm 2 (multivariable elastic-net Cox):
  - 80% train/val + 20% held-out test.
  - 5-fold CV over (penalizer x l1_ratio) grid on the 80%; AGE is unpenalized.
  - Refit on full 80% with chosen (penalizer, l1_ratio) and evaluate on 20% test:
    C-index and IPCW cumulative/dynamic AUC(t).
  - AUC(t) horizons are read from build_manifest.json so Cox, XGBoost, and
    Dynamic DeepHit use the same per-landmark quantile grid.

Endpoints: platinum, death.

Supports landmark offsets relative to first treatment via --landmark-days. When
multiple landmark offsets are requested, analyses are restricted to the
intersection MRN set eligible at every requested landmark so comparisons are
made on a fixed cohort.

Expected input:
  Row-level longitudinal data with at least
    DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab, t_first_treatment, t_platinum,
    t_death (or t_last_contact), PLATINUM, DEATH, AGE_AT_TREATMENTSTART

Outputs:
  results/cox_agg_landmark_mrn_availability.csv
  results/cox_agg_feature_selection.csv   (selected lab features + coverage)
  results/cox_agg_univariate_nobs_adjusted.csv  (n_obs-adjusted log HRs, p, q)
  results/cox_agg_multivariable.csv       (coefs + C-indices + AUC(t))
  results/cox_agg_multivariable_metrics.csv
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from helper import (
    assert_disjoint_folds,
    assert_no_test_leakage,
    breslow_survival_at_horizons,
    compute_brier,
    compute_horizon_grid,
    horizon_grid_frame,
    select_canonical_labs,
)

try:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError
    from lifelines.utils import concordance_index

    LIFELINES_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    CoxPHFitter = None
    ConvergenceError = RuntimeError
    concordance_index = None
    LIFELINES_IMPORT_ERROR = exc

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    SKSURV_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    CoxnetSurvivalAnalysis = None
    cumulative_dynamic_auc = None
    Surv = None
    SKSURV_IMPORT_ERROR = exc

DEFAULT_COXNET_MAX_ITER = 20000
DEFAULT_MAX_ABS_COXNET_COEF = 25.0

BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")
DEFAULT_V3_LABELS_PATH = DATA_PATH / "v3_outputs" / "LLM_v3_labels.tsv"

AGE_COL = "AGE_AT_TREATMENTSTART"
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_N_FOLDS = 5
DEFAULT_LANDMARK_DAYS = [0, 90]  # treatment start and 90 days post first treatment
DEFAULT_MIN_PATIENT_COVERAGE = 0.20
DEFAULT_MIN_EVENTS_PER_FEATURE = 10
DEFAULT_CV_PENALIZERS = [
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.40,
    0.80,
    1.60,
    3.20,
]
DEFAULT_CV_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
DEFAULT_AUC_QUANTILES = (0.25, 0.375, 0.50, 0.625, 0.75)
DEFAULT_AUC_TIME_UNIT_DAYS = 7
PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}
MIN_SLOPE_OBS = 4
MIN_SLOPE_UNIQUE_TIMES = 3
MIN_SLOPE_SPAN_DAYS = 14.0
MIN_DELTA_OBS = 2
SPLIT_ASSIGNMENTS_FILENAME = "cox_agg_split_assignments.csv"
LANDMARK_AVAILABILITY_FILENAME = "cox_agg_landmark_mrn_availability.csv"
HORIZON_GRID_FILENAME = "cox_agg_horizon_grid.csv"
CANONICAL_LABS_TRAIN_VAL_FILENAME = "cox_agg_canonical_labs_train_val.csv"
CANONICAL_LABS_FOLDS_FILENAME = "cox_agg_canonical_labs_folds.csv"

ENDPOINTS = {
    "platinum": {
        "duration_col": "t_platinum",
        "event_col": "PLATINUM",
        "description": "Time from first treatment start to first platinum exposure",
    },
    "death": {
        "duration_col": "t_death",
        "event_col": "DEATH",
        "description": "Time from first treatment start to death / last contact",
    },
}
OUTCOME_COLUMNS = {
    AGE_COL,
    "FIRST_RECORD_DATE",
    "DIAGNOSIS_DATE",
    "FIRST_TREATMENT_DATE",
    "FIRST_TREATMENT",
    "LAST_CONTACT_DATE",
    "PLATINUM_DATE",
    "PLATINUM",
    "DEATH",
    "EITHER",
    "t_diagnosis",
    "t_first_treatment",
    "t_platinum",
    "t_platinum_from_first_record",
    "t_last_contact",
    "t_last_contact_from_first_record",
    "t_death",
    "t_death_from_first_record",
    "t_either",
    "split",
}


def require_lifelines() -> None:
    if CoxPHFitter is None or concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to run the aggregated Cox association pipeline."
        ) from LIFELINES_IMPORT_ERROR


def require_sksurv() -> None:
    if cumulative_dynamic_auc is None:
        raise ModuleNotFoundError(
            "scikit-survival is required for the Cox IPCW AUC(t) evaluation."
        ) from SKSURV_IMPORT_ERROR


def parse_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "value"
    return feature_name.rsplit("__", 1)


def matching_n_obs_feature(feature_name: str) -> str:
    lab_name, _ = parse_feature_name(feature_name)
    return f"{lab_name}__n_observations"


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.notna()
    if not valid.any():
        return q_values

    valid_values = p_values.loc[valid].astype(float)
    order = np.argsort(valid_values.values)
    ordered_values = valid_values.values[order]
    ranks = np.arange(1, len(ordered_values) + 1, dtype=float)

    adjusted = ordered_values * len(ordered_values) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    ordered_index = valid_values.index[order]
    q_values.loc[ordered_index] = adjusted
    return q_values


def normalize_endpoints(raw_endpoints: list[str]) -> list[str]:
    endpoints: list[str] = []
    for endpoint in raw_endpoints:
        normalized = endpoint.lower()
        if normalized not in ENDPOINTS:
            valid = ", ".join(sorted(ENDPOINTS))
            raise ValueError(f"Unsupported endpoint '{endpoint}'. Choose from: {valid}")
        if normalized not in endpoints:
            endpoints.append(normalized)
    return endpoints


def load_v3_label_mrns(path: Path) -> set[int]:
    labels = pd.read_csv(path, sep="\t", usecols=["DFCI_MRN"])
    return set(labels["DFCI_MRN"].dropna().astype(int).unique())


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


def make_outcome_df(df: pd.DataFrame, *, landmark_offset_days: int = 0) -> pd.DataFrame:
    patient_level_cols = [
        "DFCI_MRN",
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
    ]
    available_cols = [col for col in patient_level_cols if col in df.columns]
    if "DFCI_MRN" not in available_cols:
        raise ValueError("Input data must contain DFCI_MRN.")

    pat = df[available_cols].drop_duplicates("DFCI_MRN").set_index("DFCI_MRN")

    if "FIRST_RECORD_DATE" not in pat.columns:
        if "LAB_DATE" not in df.columns:
            raise ValueError("Input data must contain FIRST_RECORD_DATE or LAB_DATE.")
        first_record = _coerce_datetime(df["LAB_DATE"]).groupby(df["DFCI_MRN"]).min()
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

    landmark_time = pat["t_first_treatment"].astype(float) + float(landmark_offset_days)
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
        & pat["FIRST_TREATMENT"].eq(1)
        & pat["t_first_treatment"].notna()
        & pat["t_first_treatment"].ge(0)
        & pat["t_platinum"].notna()
        & pat["t_death"].notna()
        & pat["t_last_contact"].notna()
        & pat["t_either"].notna()
        & pat["t_platinum"].gt(0)
        & pat["t_death"].gt(0)
        & pat["t_last_contact"].gt(0)
        & pat["t_either"].gt(0)
    )
    return pat.loc[valid].copy()


def build_pre_treatment_lab_long(
    df: pd.DataFrame,
    *,
    cohort_index: pd.Index | None = None,
    landmark_offset_days: int = 0,
) -> pd.DataFrame:
    """Long-format pre-landmark lab observations used for canonical-lab selection.

    Returns columns DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab, t_first_treatment.
    Restricts to observations with t_lab < t_first_treatment + landmark_offset_days
    so the lab presence used for coverage is the same window the aggregated
    feature engineering and Dynamic DeepHit person-period builder consume.
    """
    required = {"DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab", "t_first_treatment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_pre_treatment_lab_long missing columns: {sorted(missing)}"
        )
    out = df[list(required)].copy()
    out["LAB_NAME"] = out["LAB_NAME"].astype(str).str.strip()
    out["LAB_VALUE"] = pd.to_numeric(out["LAB_VALUE"], errors="coerce")
    out["t_lab"] = pd.to_numeric(out["t_lab"], errors="coerce")
    out["t_first_treatment"] = pd.to_numeric(out["t_first_treatment"], errors="coerce")
    out = out.dropna(subset=["DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab", "t_first_treatment"])
    landmark_t = out["t_first_treatment"] + float(landmark_offset_days)
    out = out.loc[out["t_lab"] < landmark_t].copy()
    if cohort_index is not None:
        out = out.loc[out["DFCI_MRN"].isin(cohort_index)].copy()
    return out


def _patient_lab_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return np.nan
    return float(np.std(values.to_numpy(dtype=float), ddof=0))


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
        pre_treatment.groupby(["DFCI_MRN", "LAB_NAME"])[["t_lab", "LAB_VALUE"]]
        .apply(_slope)
        .rename("slope")
        .reset_index()
    )
    return slopes


def build_feature_matrix(df: pd.DataFrame, *, landmark_offset_days: int = 0) -> pd.DataFrame:
    working = df.copy()
    required_cols = {"DFCI_MRN", "LAB_NAME", "LAB_VALUE"}
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
            working["FIRST_RECORD_DATE"] = _coerce_datetime(working["LAB_DATE"]).groupby(working["DFCI_MRN"]).transform("min")
        working["t_lab"] = (
            _coerce_datetime(working["LAB_DATE"]) - _coerce_datetime(working["FIRST_RECORD_DATE"])
        ).dt.days.astype(float)
    else:
        working["t_lab"] = _coerce_duration(working["t_lab"])

    if "t_first_treatment" not in working.columns:
        if not {"FIRST_TREATMENT_DATE", "FIRST_RECORD_DATE"}.issubset(working.columns):
            raise ValueError(
                "Input data must contain t_first_treatment or both FIRST_TREATMENT_DATE and FIRST_RECORD_DATE."
            )
        working["t_first_treatment"] = (
            _coerce_datetime(working["FIRST_TREATMENT_DATE"])
            - _coerce_datetime(working["FIRST_RECORD_DATE"])
        ).dt.days.astype(float)
    else:
        working["t_first_treatment"] = _coerce_duration(working["t_first_treatment"])

    working = working.dropna(
        subset=["DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab", "t_first_treatment"]
    )

    landmark_time = working["t_first_treatment"].astype(float) + float(landmark_offset_days)
    pre_treatment = working.loc[working["t_lab"].lt(landmark_time)].copy()
    if pre_treatment.empty:
        raise ValueError("No pre-landmark lab rows were available to build lab summary features.")

    sort_cols = ["DFCI_MRN", "LAB_NAME", "t_lab"]
    if "LAB_DATE" in pre_treatment.columns:
        pre_treatment["LAB_DATE"] = _coerce_datetime(pre_treatment["LAB_DATE"])
        sort_cols.append("LAB_DATE")
    pre_treatment = pre_treatment.sort_values(sort_cols)

    feature_long = (
        pre_treatment.groupby(["DFCI_MRN", "LAB_NAME"])["LAB_VALUE"]
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
    feature_long = feature_long.merge(slope_long, on=["DFCI_MRN", "LAB_NAME"], how="left")
    feature_df = (
        feature_long.set_index(["DFCI_MRN", "LAB_NAME"])
        .stack()
        .rename("value")
        .reset_index()
        .rename(columns={"level_2": "feature_stat"})
    )
    feature_df["feature_name"] = feature_df["LAB_NAME"] + "__" + feature_df["feature_stat"]
    feature_df = feature_df.pivot(index="DFCI_MRN", columns="feature_name", values="value")
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


def apply_split_assignments(
    merged: pd.DataFrame,
    *,
    split_assignments: pd.Series,
    split_stratification: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, str]:
    aligned_splits = split_assignments.reindex(merged.index)
    if aligned_splits.isna().any():
        missing = aligned_splits.index[aligned_splits.isna()].tolist()
        missing_preview = ", ".join(str(mrn) for mrn in missing[:5])
        raise ValueError(
            "Provided split assignments do not cover the merged cohort"
            + (f" (e.g. {missing_preview})" if missing_preview else "")
            + "."
        )

    merged = merged.copy()
    merged["split"] = aligned_splits
    train_val = merged.loc[aligned_splits.eq("train_val")].copy()
    test = merged.loc[aligned_splits.eq("test")].copy()
    train_val["split"] = "train_val"
    test["split"] = "test"
    return merged, train_val, test, aligned_splits, split_stratification


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
    availability = availability.rename_axis("DFCI_MRN").reset_index()
    return availability, (common_mrns if common_mrns is not None else pd.Index([]))


def combined_event_label(df: pd.DataFrame) -> np.ndarray:
    p = df["PLATINUM"].astype(int).to_numpy()
    d = df["DEATH"].astype(int).to_numpy()
    return p + 2 * d


def choose_stratification_labels(df: pd.DataFrame, *, min_count: int) -> tuple[np.ndarray | None, str]:
    candidates = [
        ("combined", combined_event_label(df)),
        ("either", df["EITHER"].astype(int).to_numpy()),
        ("platinum", df["PLATINUM"].astype(int).to_numpy()),
        ("death", df["DEATH"].astype(int).to_numpy()),
    ]
    for label_name, labels in candidates:
        counts = pd.Series(labels).value_counts()
        if len(counts) > 1 and counts.min() >= min_count:
            return labels, label_name
    return None, "unstratified"


def split_train_test(
    merged: pd.DataFrame,
    *,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    labels, label_name = choose_stratification_labels(merged, min_count=2)
    stratify = labels if labels is not None else None

    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(merged)),
            test_size=test_frac,
            stratify=stratify,
            random_state=seed,
        )
    except ValueError:
        label_name = "unstratified"
        train_idx, test_idx = train_test_split(
            np.arange(len(merged)),
            test_size=test_frac,
            random_state=seed,
        )

    split_assignments = pd.Series("train_val", index=merged.index, name="split")
    split_assignments.iloc[test_idx] = "test"
    train_val = merged.iloc[train_idx].copy()
    test = merged.iloc[test_idx].copy()
    return train_val, test, split_assignments, label_name


def build_aligned_cohort(
    df: pd.DataFrame,
    *,
    test_frac: float,
    seed: int,
    landmark_offset_days: int = 0,
    required_mrns: pd.Index | None = None,
    split_assignments: pd.Series | None = None,
    split_stratification: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Build the shared landmarked patient cohort and held-out test split."""
    outcome_df, feature_df, merged = build_landmark_merged(
        df,
        landmark_offset_days=landmark_offset_days,
    )
    if required_mrns is not None:
        merged = merged.loc[merged.index.intersection(required_mrns)].copy()
        if merged.empty:
            raise ValueError(
                f"No patients remained after restricting to the requested landmark cohort at +{landmark_offset_days}d."
            )

    if split_assignments is None:
        train_val, test, split_assignments, split_stratification = split_train_test(
            merged,
            test_frac=test_frac,
            seed=seed,
        )
        merged = merged.copy()
        merged["split"] = split_assignments
        train_val = train_val.copy()
        train_val["split"] = "train_val"
        test = test.copy()
        test["split"] = "test"
    else:
        if split_stratification is None:
            raise ValueError("split_stratification must be provided when reusing split_assignments.")
        merged, train_val, test, split_assignments, split_stratification = apply_split_assignments(
            merged,
            split_assignments=split_assignments,
            split_stratification=split_stratification,
        )

    outcome_df = outcome_df.loc[outcome_df.index.intersection(merged.index)].copy()
    feature_df = feature_df.loc[feature_df.index.intersection(merged.index)].copy()
    return outcome_df, feature_df, merged, train_val, test, split_assignments, split_stratification


def select_feature_columns(
    data: pd.DataFrame,
    raw_feature_cols: list[str],
    *,
    min_patient_coverage: float,
    restrict_to_labs: list[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select features on the training/validation block to avoid test leakage.

    `restrict_to_labs`, when provided, intersects the per-stat survivors with
    the canonical lab list (computed upstream on the same MRN block). This is
    the gate that enforces an identical lab set across pipelines.
    """
    coverage = data[raw_feature_cols].notna().mean()
    unique_non_missing = data[raw_feature_cols].nunique(dropna=True)

    feature_meta = pd.DataFrame(
        {
            "feature": raw_feature_cols,
            "coverage": coverage.reindex(raw_feature_cols).values,
            "unique_non_missing": unique_non_missing.reindex(raw_feature_cols).values,
        }
    )
    parsed = feature_meta["feature"].map(parse_feature_name)
    feature_meta["lab_name"] = parsed.str[0]
    feature_meta["feature_stat"] = parsed.str[1]
    feature_meta["selected"] = (
        feature_meta["coverage"].ge(min_patient_coverage)
        & feature_meta["unique_non_missing"].gt(1)
    )
    if restrict_to_labs is not None:
        canonical = set(str(lab) for lab in restrict_to_labs)
        feature_meta["in_canonical_labs"] = feature_meta["lab_name"].astype(str).isin(canonical)
        feature_meta["selected"] = feature_meta["selected"] & feature_meta["in_canonical_labs"]
    feature_meta = feature_meta.sort_values(
        ["selected", "coverage", "feature"],
        ascending=[False, False, True],
    )

    selected = feature_meta.loc[feature_meta["selected"], "feature"].tolist()
    if not selected:
        raise ValueError("No features passed coverage and variability filters.")
    return selected, feature_meta.reset_index(drop=True)


def fit_cox_with_fallback(
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    penalizers: list[float],
    l1_ratio: float,
    unpenalized_cols: list[str] | None = None,
    covariate_cols: list[str] | None = None,
) -> tuple[CoxPHFitter | None, float, str]:
    require_lifelines()

    if covariate_cols is None:
        covariate_cols = [c for c in model_df.columns if c not in {duration_col, event_col}]
    unpenalized = set(unpenalized_cols or [])

    last_error = ""
    for penalizer in penalizers:
        try:
            if unpenalized:
                penalty_vec = np.array(
                    [0.0 if c in unpenalized else float(penalizer) for c in covariate_cols],
                    dtype=float,
                )
                pen_arg: float | np.ndarray = penalty_vec
            else:
                pen_arg = float(penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CoxPHFitter(penalizer=pen_arg, l1_ratio=l1_ratio)
                model.fit(model_df, duration_col=duration_col, event_col=event_col)
            note = "fit_ok" if penalizer == 0 else f"fit_ok_penalizer_{penalizer:g}"
            return model, penalizer, note
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)

    return None, float("nan"), f"fit_failed: {last_error}"


def build_model_matrices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    duration_col: str,
    event_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Always includes age as the last covariate; age is the unpenalized column."""
    base_feature_cols = [
        col
        for col in feature_cols
        if train_df[col].notna().any() and train_df[col].nunique(dropna=True) > 1
    ]
    missing_indicator_cols = [
        f"{col}__missing"
        for col in base_feature_cols
        if train_df[col].isna().nunique(dropna=False) > 1
    ]
    covariate_cols = list(base_feature_cols) + missing_indicator_cols

    train_model = pd.DataFrame(index=train_df.index)
    eval_model = pd.DataFrame(index=eval_df.index)

    if base_feature_cols:
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        x_train_values = imputer.fit_transform(train_df[base_feature_cols].values)
        x_eval_values = imputer.transform(eval_df[base_feature_cols].values)

        if missing_indicator_cols:
            indicator_source_cols = [
                col.removesuffix("__missing") for col in missing_indicator_cols
            ]
            x_train_missing = train_df[indicator_source_cols].isna().astype(float).to_numpy()
            x_eval_missing = eval_df[indicator_source_cols].isna().astype(float).to_numpy()
            x_train = np.hstack([x_train_values, x_train_missing])
            x_eval = np.hstack([x_eval_values, x_eval_missing])
            scaled_cols = base_feature_cols + missing_indicator_cols
        else:
            x_train = x_train_values
            x_eval = x_eval_values
            scaled_cols = base_feature_cols

        x_train = scaler.fit_transform(x_train)
        x_eval = scaler.transform(x_eval)
        train_model = pd.DataFrame(x_train, columns=scaled_cols, index=train_df.index)
        eval_model = pd.DataFrame(x_eval, columns=scaled_cols, index=eval_df.index)

    age_scaler = StandardScaler()
    train_age = age_scaler.fit_transform(
        train_df[[AGE_COL]].to_numpy(dtype=float)
    ).reshape(-1)
    eval_age = age_scaler.transform(
        eval_df[[AGE_COL]].to_numpy(dtype=float)
    ).reshape(-1)
    train_model["age"] = train_age
    eval_model["age"] = eval_age
    covariate_cols.append("age")

    if not covariate_cols:
        raise ValueError("No usable covariates remained after train-fold filtering.")

    train_model[duration_col] = train_df[duration_col].to_numpy(dtype=float)
    train_model[event_col] = train_df[event_col].to_numpy(dtype=int)
    eval_model[duration_col] = eval_df[duration_col].to_numpy(dtype=float)
    eval_model[event_col] = eval_df[event_col].to_numpy(dtype=int)
    return train_model, eval_model, covariate_cols


def _duration_to_auc_units(duration: pd.Series, *, time_unit_days: int) -> np.ndarray:
    if time_unit_days <= 0:
        raise ValueError("time_unit_days must be positive.")
    duration_days = pd.to_numeric(duration, errors="coerce").to_numpy(dtype=float)
    return np.ceil(duration_days / float(time_unit_days))


def _make_survival_array(event: np.ndarray, duration: np.ndarray) -> np.ndarray:
    survival = np.empty(
        dtype=[("event", bool), ("time", np.float64)],
        shape=len(duration),
    )
    survival["event"] = event.astype(bool)
    survival["time"] = duration.astype(float)
    return survival


def _apply_auc_admin_censoring(
    event: np.ndarray,
    duration: np.ndarray,
    *,
    max_time_unit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Censor durations at a fixed AUC horizon for comparable finite-horizon metrics."""
    event = np.asarray(event, dtype=int).copy()
    duration = np.asarray(duration, dtype=float).copy()
    if max_time_unit is None:
        return event, duration
    if max_time_unit <= 0:
        raise ValueError("max_time_unit must be positive when provided.")

    within_horizon = event.astype(bool) & (duration > 0) & (duration < float(max_time_unit))
    duration = np.where(duration < float(max_time_unit), duration, float(max_time_unit))
    return within_horizon.astype(int), duration


def compute_ipcw_auc_t(
    eval_df: pd.DataFrame,
    risk_score: np.ndarray,
    *,
    duration_col: str,
    event_col: str,
    reference_df: pd.DataFrame,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
    quantiles: tuple[float, ...] = DEFAULT_AUC_QUANTILES,
    max_time_unit: int | None = None,
    fixed_horizons: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Compute IPCW cumulative/dynamic AUC(t).

    When `fixed_horizons` is provided, the horizon grid is taken from those
    values (in time-units) and `quantiles` is ignored. This lets every CV fold
    and the held-out test eval use one shared grid derived once from train_val
    event times — no test-driven horizon selection.
    """
    require_sksurv()

    empty_cols = [
        "horizon_quantile",
        "horizon_time_unit",
        "horizon_days",
        "auc_t",
        "n_eval",
        "n_eval_events",
        "admin_censor_time_unit",
        "note",
    ]

    train_duration = _duration_to_auc_units(
        reference_df[duration_col],
        time_unit_days=time_unit_days,
    )
    train_event = reference_df[event_col].to_numpy(dtype=int)
    eval_duration = _duration_to_auc_units(
        eval_df[duration_col],
        time_unit_days=time_unit_days,
    )
    eval_event = eval_df[event_col].to_numpy(dtype=int)
    risk_score = np.asarray(risk_score, dtype=float).reshape(-1)

    train_event, train_duration = _apply_auc_admin_censoring(
        train_event,
        train_duration,
        max_time_unit=max_time_unit,
    )
    eval_event, eval_duration = _apply_auc_admin_censoring(
        eval_event,
        eval_duration,
        max_time_unit=max_time_unit,
    )

    train_valid = np.isfinite(train_duration) & (train_duration > 0)
    eval_valid = np.isfinite(eval_duration) & (eval_duration > 0) & np.isfinite(risk_score)
    if train_valid.sum() == 0 or eval_valid.sum() == 0:
        return np.nan, pd.DataFrame(columns=empty_cols)

    train_surv = _make_survival_array(train_event[train_valid], train_duration[train_valid])
    eval_surv = _make_survival_array(eval_event[eval_valid], eval_duration[eval_valid])
    eval_risk = risk_score[eval_valid]

    if fixed_horizons is not None:
        horizon_times = np.asarray(fixed_horizons, dtype=float).reshape(-1)
        horizon_times = np.unique(horizon_times[horizon_times > 0])
        if max_time_unit is not None:
            horizon_times = horizon_times[horizon_times <= float(max_time_unit)]
        if len(horizon_times) == 0:
            return np.nan, pd.DataFrame(columns=empty_cols)
        # Pad quantile column with NaN since we no longer derive horizons from quantiles.
        horizon_quantiles: tuple[float, ...] = tuple([np.nan] * len(horizon_times))
    else:
        event_times = eval_duration[eval_valid & (eval_event == 1)]
        event_times = event_times[np.isfinite(event_times) & (event_times > 0)]
        if len(event_times) == 0:
            return np.nan, pd.DataFrame(columns=empty_cols)
        horizon_times = np.asarray(
            [int(val) for val in np.quantile(event_times, quantiles)],
            dtype=float,
        )
        horizon_quantiles = tuple(quantiles)

    rows = []
    for quantile, horizon in zip(horizon_quantiles, horizon_times):
        auc_t = np.nan
        note = ""
        if horizon <= 0:
            note = "non_positive_horizon"
        else:
            try:
                auc_values, _ = cumulative_dynamic_auc(
                    train_surv,
                    eval_surv,
                    eval_risk,
                    np.asarray([horizon], dtype=float),
                )
                auc_t = float(auc_values[0])
            except ValueError as exc:
                note = f"auc_failed: {exc}"
        rows.append(
            {
                "horizon_quantile": quantile,
                "horizon_time_unit": horizon,
                "horizon_days": horizon * float(time_unit_days),
                "auc_t": auc_t,
                "n_eval": int(eval_valid.sum()),
                "n_eval_events": int((eval_event[eval_valid] == 1).sum()),
                "admin_censor_time_unit": max_time_unit,
                "note": note,
            }
        )

    auc_df = pd.DataFrame(rows)
    if len(horizon_times) < 2 or horizon_times[-1] <= horizon_times[0]:
        return np.nan, auc_df

    mean_auc_times = np.arange(horizon_times[0], horizon_times[-1], dtype=float)
    mean_auc_times = mean_auc_times[mean_auc_times > 0]
    if len(mean_auc_times) == 0:
        return np.nan, auc_df

    try:
        _, mean_auc = cumulative_dynamic_auc(
            train_surv,
            eval_surv,
            eval_risk,
            mean_auc_times,
        )
    except ValueError:
        mean_auc = np.nan
    return float(mean_auc) if np.isfinite(mean_auc) else np.nan, auc_df


def score_cox_model(
    model: CoxPHFitter,
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
) -> tuple[float, np.ndarray]:
    # Use the linear predictor (log partial hazard) rather than exp(linear predictor):
    # ranking is identical, and it avoids exp() overflow when the linear predictor
    # is large for a handful of outlier rows — which would otherwise produce inf and
    # crash downstream AUC routines.
    log_pred = np.asarray(model.predict_log_partial_hazard(model_df)).reshape(-1)
    c_index = float(concordance_index(model_df[duration_col], -log_pred, model_df[event_col]))
    return c_index, log_pred


def _build_coxnet_xy(
    model_df: pd.DataFrame,
    *,
    covariate_cols: list[str],
    duration_col: str,
    event_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = model_df[covariate_cols].to_numpy(dtype=float)
    y = Surv.from_arrays(
        event=model_df[event_col].astype(bool).to_numpy(),
        time=model_df[duration_col].astype(float).to_numpy(),
    )
    return X, y


def fit_coxnet_with_fallback(
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    penalizers: list[float],
    l1_ratio: float,
    covariate_cols: list[str],
    unpenalized_cols: list[str] | None = None,
    max_iter: int = DEFAULT_COXNET_MAX_ITER,
) -> tuple[CoxnetSurvivalAnalysis | None, float, str]:
    """Elastic-net Cox via sksurv's coordinate-descent CoxnetSurvivalAnalysis.

    Stability-first wrapper around sksurv's coordinate-descent Coxnet.

    Fits that emit warnings, produce non-finite coefficients, or blow up to
    implausibly large absolute coefficients on z-scored inputs are rejected and
    treated as unusable, triggering the penalizer fallback.
    """
    require_sksurv()
    X, y = _build_coxnet_xy(
        model_df,
        covariate_cols=covariate_cols,
        duration_col=duration_col,
        event_col=event_col,
    )
    unpenalized = set(unpenalized_cols or [])
    penalty_factor = np.array(
        [0.0 if c in unpenalized else 1.0 for c in covariate_cols],
        dtype=float,
    )

    last_error = ""
    for penalizer in penalizers:
        try:
            model = CoxnetSurvivalAnalysis(
                alphas=[float(penalizer)],
                l1_ratio=float(l1_ratio),
                penalty_factor=penalty_factor,
                max_iter=int(max_iter),
                fit_baseline_model=False,
            )
            # Warnings emitted by sksurv (incl. ConvergenceWarning) are NOT
            # treated as failures — the finite + bounded coefficient checks
            # below catch genuinely unusable fits.
            model.fit(X, y)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)
            continue

        coefs = np.asarray(model.coef_, dtype=float)
        if coefs.ndim == 2:
            coefs = coefs[:, -1]
        coefs = coefs.reshape(-1)
        if coefs.size != len(covariate_cols):
            last_error = (
                f"coef_size_mismatch: expected {len(covariate_cols)} got {coefs.size}"
            )
            continue
        if not np.all(np.isfinite(coefs)):
            last_error = "non_finite_coefficients"
            continue
        max_abs_coef = float(np.max(np.abs(coefs))) if coefs.size else 0.0
        if max_abs_coef > DEFAULT_MAX_ABS_COXNET_COEF:
            last_error = (
                f"coef_blowup_max_abs_{max_abs_coef:.3f}_gt_{DEFAULT_MAX_ABS_COXNET_COEF:g}"
            )
            continue

        note = f"fit_ok_penalizer_{penalizer:g}"
        return model, float(penalizer), note

    return None, float("nan"), f"fit_failed: {last_error}"


def score_coxnet_model(
    model: CoxnetSurvivalAnalysis,
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
) -> tuple[float, np.ndarray]:
    X = model_df[covariate_cols].to_numpy(dtype=float)
    log_pred = np.asarray(model.predict(X)).reshape(-1)
    c_index = float(concordance_index(model_df[duration_col], -log_pred, model_df[event_col]))
    return c_index, log_pred


def coxnet_survival_at_horizons(
    model: CoxnetSurvivalAnalysis,
    train_mdf: pd.DataFrame,
    eval_mdf: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    horizons: np.ndarray,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
) -> np.ndarray:
    """Survival probabilities for each eval row at each horizon (in time-units).

    Coxnet is fit with fit_baseline_model=False, so we recover S(t|x) via a
    Breslow estimator on the training linear predictor. Train durations are
    converted into the same time unit as `horizons` so the baseline cumulative
    hazard and the horizons share a clock.
    """
    X_train = train_mdf[covariate_cols].to_numpy(dtype=float)
    X_eval = eval_mdf[covariate_cols].to_numpy(dtype=float)
    train_lp = np.asarray(model.predict(X_train)).reshape(-1)
    eval_lp = np.asarray(model.predict(X_eval)).reshape(-1)
    train_event = train_mdf[event_col].astype(int).to_numpy()
    train_duration = pd.to_numeric(train_mdf[duration_col], errors="coerce").to_numpy(dtype=float)
    train_duration_units = np.ceil(train_duration / float(time_unit_days))
    return breslow_survival_at_horizons(
        train_event=train_event,
        train_duration=train_duration_units,
        train_lp=train_lp,
        eval_lp=eval_lp,
        horizons=horizons,
    )


def coxnet_coefficients(
    model: CoxnetSurvivalAnalysis, covariate_cols: list[str]
) -> pd.Series:
    coefs = np.asarray(model.coef_)
    if coefs.ndim == 2:
        coefs = coefs[:, -1]
    return pd.Series(coefs.reshape(-1), index=covariate_cols, name="coef")


def run_univariate_associations(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> pd.DataFrame:
    """Arm 1: for each feature, fit Cox on [AGE + feature] using the full dataset."""
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
    total_patients = len(data)
    rows = []

    for feature in feature_cols:
        lab_name, feature_stat = parse_feature_name(feature)
        feature_df = data[[feature, duration_col, event_col, AGE_COL]].copy()
        coverage = float(feature_df[feature].notna().mean())

        feature_df = feature_df.dropna(subset=[duration_col, event_col, AGE_COL])

        observed_non_missing = int(feature_df[feature].notna().sum())
        imputed_count = int(len(feature_df) - observed_non_missing)
        result = {
            "endpoint": endpoint,
            "feature": feature,
            "lab_name": lab_name,
            "feature_stat": feature_stat,
            "coverage": coverage,
            "n_patients_total": total_patients,
            "n_patients_used": len(feature_df),
            "n_patients_observed": observed_non_missing,
            "n_patients_imputed": imputed_count,
            "n_events_used": int(feature_df[event_col].sum()),
            "coef_feature": np.nan,
            "hazard_ratio_per_sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "coef_missing": np.nan,
            "p_value_missing": np.nan,
            "coef_age": np.nan,
            "p_value_age": np.nan,
            "fit_penalizer": np.nan,
            "note": "",
        }

        if len(feature_df) == 0:
            result["note"] = "no_rows_with_outcomes"
            rows.append(result)
            continue

        if observed_non_missing == 0:
            result["note"] = "no_non_missing_rows"
            rows.append(result)
            continue

        if result["n_events_used"] < min_events_per_feature:
            result["note"] = f"too_few_events_lt_{min_events_per_feature}"
            rows.append(result)
            continue

        missing_indicator = feature_df[feature].isna().astype(float).to_numpy()
        include_missing_indicator = bool(np.unique(missing_indicator).size > 1)

        imputer = SimpleImputer(strategy="mean")
        feature_values = imputer.fit_transform(feature_df[[feature]]).reshape(-1)
        feature_sd = float(np.std(feature_values, ddof=0))
        if not np.isfinite(feature_sd) or feature_sd <= 0:
            result["note"] = "feature_has_no_variation"
            rows.append(result)
            continue

        feature_mean = float(np.mean(feature_values))
        feature_df = feature_df.copy()
        feature_df["feature_z"] = (feature_values - feature_mean) / feature_sd
        age_values = feature_df[AGE_COL].to_numpy(dtype=float)
        age_sd = float(np.std(age_values, ddof=0))
        if np.isfinite(age_sd) and age_sd > 0:
            feature_df["age"] = (age_values - float(np.mean(age_values))) / age_sd
        else:
            feature_df["age"] = age_values - float(np.mean(age_values))
        model_cols = ["feature_z", "age"]
        if include_missing_indicator:
            feature_df["feature_missing"] = missing_indicator
            model_cols.insert(1, "feature_missing")

        model, used_penalizer, note = fit_cox_with_fallback(
            feature_df[model_cols + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col,
            penalizers=[0.0, fallback_penalizer],
            l1_ratio=0.0,
        )
        result["fit_penalizer"] = used_penalizer
        result["note"] = note

        if model is None:
            rows.append(result)
            continue

        summary_row = model.summary.loc["feature_z"]
        result["coef_feature"] = float(summary_row["coef"])
        result["hazard_ratio_per_sd"] = float(summary_row["exp(coef)"])
        result["ci_lower"] = float(summary_row["exp(coef) lower 95%"])
        result["ci_upper"] = float(summary_row["exp(coef) upper 95%"])
        result["p_value"] = float(summary_row["p"])
        if include_missing_indicator and "feature_missing" in model.summary.index:
            missing_row = model.summary.loc["feature_missing"]
            result["coef_missing"] = float(missing_row["coef"])
            result["p_value_missing"] = float(missing_row["p"])
        age_row = model.summary.loc["age"]
        result["coef_age"] = float(age_row["coef"])
        result["p_value_age"] = float(age_row["p"])
        rows.append(result)

    associations = pd.DataFrame(rows)
    associations["q_value"] = benjamini_hochberg(associations["p_value"])
    return associations.sort_values(["p_value", "q_value", "feature"], na_position="last").reset_index(drop=True)


def run_univariate_nobs_adjusted_associations(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> pd.DataFrame:
    """Arm 1b: fit Cox on [AGE + matching LAB__n_observations + feature]."""
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
    total_patients = len(data)
    rows = []

    for feature in feature_cols:
        lab_name, feature_stat = parse_feature_name(feature)
        n_obs_feature = matching_n_obs_feature(feature)
        result = {
            "endpoint": endpoint,
            "feature": feature,
            "lab_name": lab_name,
            "feature_stat": feature_stat,
            "n_obs_feature": n_obs_feature,
            "coverage": np.nan,
            "n_obs_coverage": np.nan,
            "n_patients_total": total_patients,
            "n_patients_used": 0,
            "n_patients_observed": 0,
            "n_patients_imputed": 0,
            "n_patients_n_obs_observed": 0,
            "n_patients_n_obs_imputed": 0,
            "n_events_used": 0,
            "coef_feature": np.nan,
            "hazard_ratio_per_sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "coef_n_obs": np.nan,
            "hazard_ratio_n_obs_per_sd": np.nan,
            "ci_lower_n_obs": np.nan,
            "ci_upper_n_obs": np.nan,
            "p_value_n_obs": np.nan,
            "coef_missing": np.nan,
            "p_value_missing": np.nan,
            "coef_age": np.nan,
            "p_value_age": np.nan,
            "fit_penalizer": np.nan,
            "note": "",
        }

        if feature_stat == "n_observations":
            result["note"] = "target_is_n_observations"
            rows.append(result)
            continue

        if n_obs_feature not in data.columns:
            result["note"] = "missing_matching_n_obs_feature"
            rows.append(result)
            continue

        feature_df = data[[feature, n_obs_feature, duration_col, event_col, AGE_COL]].copy()
        result["coverage"] = float(feature_df[feature].notna().mean())
        result["n_obs_coverage"] = float(feature_df[n_obs_feature].notna().mean())

        feature_df = feature_df.dropna(subset=[duration_col, event_col, AGE_COL])
        observed_non_missing = int(feature_df[feature].notna().sum())
        observed_n_obs = int(feature_df[n_obs_feature].notna().sum())
        result["n_patients_used"] = len(feature_df)
        result["n_patients_observed"] = observed_non_missing
        result["n_patients_imputed"] = int(len(feature_df) - observed_non_missing)
        result["n_patients_n_obs_observed"] = observed_n_obs
        result["n_patients_n_obs_imputed"] = int(len(feature_df) - observed_n_obs)
        result["n_events_used"] = int(feature_df[event_col].sum())

        if len(feature_df) == 0:
            result["note"] = "no_rows_with_outcomes"
            rows.append(result)
            continue

        if observed_non_missing == 0:
            result["note"] = "no_non_missing_rows"
            rows.append(result)
            continue

        if observed_n_obs == 0:
            result["note"] = "no_non_missing_n_obs_rows"
            rows.append(result)
            continue

        if result["n_events_used"] < min_events_per_feature:
            result["note"] = f"too_few_events_lt_{min_events_per_feature}"
            rows.append(result)
            continue

        missing_indicator = feature_df[feature].isna().astype(float).to_numpy()
        include_missing_indicator = bool(np.unique(missing_indicator).size > 1)

        feature_values = SimpleImputer(strategy="mean").fit_transform(feature_df[[feature]]).reshape(-1)
        feature_sd = float(np.std(feature_values, ddof=0))
        if not np.isfinite(feature_sd) or feature_sd <= 0:
            result["note"] = "feature_has_no_variation"
            rows.append(result)
            continue

        n_obs_values = (
            SimpleImputer(strategy="mean")
            .fit_transform(feature_df[[n_obs_feature]])
            .reshape(-1)
        )
        n_obs_sd = float(np.std(n_obs_values, ddof=0))
        if not np.isfinite(n_obs_sd) or n_obs_sd <= 0:
            result["note"] = "n_obs_has_no_variation"
            rows.append(result)
            continue

        feature_df = feature_df.copy()
        feature_df["feature_z"] = (feature_values - float(np.mean(feature_values))) / feature_sd
        feature_df["n_obs_z"] = (n_obs_values - float(np.mean(n_obs_values))) / n_obs_sd

        age_values = feature_df[AGE_COL].to_numpy(dtype=float)
        age_sd = float(np.std(age_values, ddof=0))
        if np.isfinite(age_sd) and age_sd > 0:
            feature_df["age"] = (age_values - float(np.mean(age_values))) / age_sd
        else:
            feature_df["age"] = age_values - float(np.mean(age_values))

        model_cols = ["feature_z", "n_obs_z", "age"]
        if include_missing_indicator:
            feature_df["feature_missing"] = missing_indicator
            model_cols.insert(1, "feature_missing")

        model, used_penalizer, note = fit_cox_with_fallback(
            feature_df[model_cols + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col,
            penalizers=[0.0, fallback_penalizer],
            l1_ratio=0.0,
        )
        result["fit_penalizer"] = used_penalizer
        result["note"] = note

        if model is None:
            rows.append(result)
            continue

        summary_row = model.summary.loc["feature_z"]
        result["coef_feature"] = float(summary_row["coef"])
        result["hazard_ratio_per_sd"] = float(summary_row["exp(coef)"])
        result["ci_lower"] = float(summary_row["exp(coef) lower 95%"])
        result["ci_upper"] = float(summary_row["exp(coef) upper 95%"])
        result["p_value"] = float(summary_row["p"])

        n_obs_row = model.summary.loc["n_obs_z"]
        result["coef_n_obs"] = float(n_obs_row["coef"])
        result["hazard_ratio_n_obs_per_sd"] = float(n_obs_row["exp(coef)"])
        result["ci_lower_n_obs"] = float(n_obs_row["exp(coef) lower 95%"])
        result["ci_upper_n_obs"] = float(n_obs_row["exp(coef) upper 95%"])
        result["p_value_n_obs"] = float(n_obs_row["p"])

        if include_missing_indicator and "feature_missing" in model.summary.index:
            missing_row = model.summary.loc["feature_missing"]
            result["coef_missing"] = float(missing_row["coef"])
            result["p_value_missing"] = float(missing_row["p"])

        age_row = model.summary.loc["age"]
        result["coef_age"] = float(age_row["coef"])
        result["p_value_age"] = float(age_row["p"])
        rows.append(result)

    associations = pd.DataFrame(rows)
    associations["q_value"] = benjamini_hochberg(associations["p_value"])
    return associations.sort_values(["p_value", "q_value", "feature"], na_position="last").reset_index(drop=True)


def make_cv_splitter(
    train_val: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
) -> tuple[StratifiedKFold | KFold, np.ndarray | None, str]:
    labels, label_name = choose_stratification_labels(train_val, min_count=n_folds)
    if labels is not None:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed), labels, label_name
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed), None, "unstratified"


def tune_multivariable_model(
    train_val: pd.DataFrame,
    *,
    raw_feature_cols: list[str],
    endpoint: str,
    penalizers: list[float],
    l1_ratios: list[float],
    n_folds: int,
    seed: int,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    pre_treatment_lab_df: pd.DataFrame,
    horizon_grid: np.ndarray,
    min_patient_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Arm 2: 5-fold CV over (penalizer x l1_ratio) grid (elastic-net), AGE unpenalized.

    Backed by sksurv's CoxnetSurvivalAnalysis for speed on high-dimensional
    feature sets. Hyperparameter selection is restricted to combinations with
    valid fits in every CV fold.

    Strict no-leakage: canonical labs are recomputed inside each fold from
    fold_train MRNs only, and AUC(t)/Brier use the fixed horizon_grid passed
    in (derived once on full train_val). Test never participates.

    Returns (fold_df, cv_df, best_row, fold_canonical_labs_df).
    """
    require_sksurv()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    splitter, strat_labels, cv_stratification = make_cv_splitter(train_val, n_folds=n_folds, seed=seed)
    fold_rows = []
    fold_canonical_labs_rows: list[dict] = []

    split_args = (np.arange(len(train_val)), strat_labels) if strat_labels is not None else (np.arange(len(train_val)),)

    # Materialize splits once so canonical-lab selection and the (penalizer, l1_ratio)
    # grid traverse the same fold partition deterministically.
    fold_partitions = list(enumerate(splitter.split(*split_args), 1))
    fold_canonical_labs: dict[int, list[str]] = {}
    fold_selected_features: dict[int, list[str]] = {}
    for fold, (tr_idx, val_idx) in fold_partitions:
        fold_train_idx = train_val.index[tr_idx]
        fold_val_idx = train_val.index[val_idx]
        assert_disjoint_folds(
            fold_train_mrns=fold_train_idx,
            fold_val_mrns=fold_val_idx,
            fold=fold,
        )
        canonical = select_canonical_labs(
            pre_treatment_lab_df,
            mrns=fold_train_idx,
            min_coverage=min_patient_coverage,
        )
        fold_canonical_labs[fold] = canonical
        fold_train = train_val.iloc[tr_idx]
        selected, _ = select_feature_columns(
            fold_train,
            raw_feature_cols,
            min_patient_coverage=min_patient_coverage,
            restrict_to_labs=canonical,
        )
        fold_selected_features[fold] = selected
        for lab in canonical:
            fold_canonical_labs_rows.append(
                {"endpoint": endpoint, "fold": fold, "lab_name": lab}
            )

    for penalizer, l1_ratio in product(penalizers, l1_ratios):
        for fold, (tr_idx, val_idx) in fold_partitions:
            fold_train = train_val.iloc[tr_idx]
            fold_val = train_val.iloc[val_idx]
            fold_features = fold_selected_features[fold]
            row = {
                "endpoint": endpoint,
                "fold": fold,
                "penalizer": float(penalizer),
                "l1_ratio": float(l1_ratio),
                "n_train": len(fold_train),
                "n_val": len(fold_val),
                "n_events_train": int(fold_train[event_col].sum()),
                "n_events_val": int(fold_val[event_col].sum()),
                "n_canonical_labs": len(fold_canonical_labs[fold]),
                "n_selected_features": len(fold_features),
                "cv_stratification": cv_stratification,
                "c_index_val": np.nan,
                "mean_auc_t_val": np.nan,
                "n_valid_auc_horizons_val": 0,
                "integrated_brier_val": np.nan,
                "n_valid_brier_horizons_val": 0,
                "n_covariates": np.nan,
                "note": "",
            }

            try:
                train_mdf, val_mdf, covariate_cols = build_model_matrices(
                    fold_train,
                    fold_val,
                    feature_cols=fold_features,
                    duration_col=duration_col,
                    event_col=event_col,
                )
                model, _, note = fit_coxnet_with_fallback(
                    train_mdf,
                    duration_col=duration_col,
                    event_col=event_col,
                    penalizers=[float(penalizer)],
                    l1_ratio=float(l1_ratio),
                    covariate_cols=covariate_cols,
                    unpenalized_cols=["age"],
                )
                row["note"] = note
                row["n_covariates"] = len(covariate_cols)
                if model is not None:
                    c_index, val_pred = score_coxnet_model(
                        model,
                        val_mdf,
                        duration_col=duration_col,
                        event_col=event_col,
                        covariate_cols=covariate_cols,
                    )
                    row["c_index_val"] = c_index
                    mean_auc_val, auc_df_val = compute_ipcw_auc_t(
                        val_mdf,
                        val_pred,
                        duration_col=duration_col,
                        event_col=event_col,
                        reference_df=train_mdf,
                        time_unit_days=auc_time_unit_days,
                        max_time_unit=auc_max_time_units,
                        fixed_horizons=horizon_grid,
                    )
                    row["mean_auc_t_val"] = mean_auc_val
                    row["n_valid_auc_horizons_val"] = (
                        int(auc_df_val["auc_t"].notna().sum()) if not auc_df_val.empty else 0
                    )

                    val_surv = coxnet_survival_at_horizons(
                        model,
                        train_mdf,
                        val_mdf,
                        duration_col=duration_col,
                        event_col=event_col,
                        covariate_cols=covariate_cols,
                        horizons=horizon_grid,
                        time_unit_days=auc_time_unit_days,
                    )
                    train_dur_units = np.ceil(
                        pd.to_numeric(train_mdf[duration_col], errors="coerce").to_numpy(dtype=float)
                        / float(auc_time_unit_days)
                    )
                    val_dur_units = np.ceil(
                        pd.to_numeric(val_mdf[duration_col], errors="coerce").to_numpy(dtype=float)
                        / float(auc_time_unit_days)
                    )
                    brier_df, ibs = compute_brier(
                        train_event=train_mdf[event_col].to_numpy(dtype=int),
                        train_duration=train_dur_units,
                        eval_event=val_mdf[event_col].to_numpy(dtype=int),
                        eval_duration=val_dur_units,
                        surv_at_horizons=val_surv,
                        horizons=horizon_grid,
                        time_unit_days=auc_time_unit_days,
                    )
                    row["integrated_brier_val"] = ibs
                    row["n_valid_brier_horizons_val"] = (
                        int(brier_df["brier"].notna().sum()) if not brier_df.empty else 0
                    )
            except Exception as exc:  # pragma: no cover - defensive for unstable folds
                row["note"] = f"fold_failed: {exc}"

            fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    cv_df = (
        fold_df.groupby(["endpoint", "penalizer", "l1_ratio"], dropna=False)
        .agg(
            cv_mean=("c_index_val", "mean"),
            cv_std=("c_index_val", "std"),
            n_valid_folds=("c_index_val", lambda s: int(s.notna().sum())),
            mean_auc_t_cv_mean=("mean_auc_t_val", "mean"),
            mean_auc_t_cv_std=("mean_auc_t_val", "std"),
            n_valid_auc_t_folds=("mean_auc_t_val", lambda s: int(s.notna().sum())),
            integrated_brier_cv_mean=("integrated_brier_val", "mean"),
            integrated_brier_cv_std=("integrated_brier_val", "std"),
            n_valid_brier_folds=("integrated_brier_val", lambda s: int(s.notna().sum())),
            n_covariates_mean=("n_covariates", "mean"),
            cv_stratification=("cv_stratification", "first"),
        )
        .reset_index()
    )
    cv_df["all_folds_valid"] = cv_df["n_valid_folds"].eq(int(n_folds))

    if cv_df["n_valid_folds"].eq(0).all():
        raise RuntimeError(f"All CV fits failed for endpoint '{endpoint}'.")
    if not cv_df["all_folds_valid"].any():
        best_valid = int(cv_df["n_valid_folds"].max())
        raise RuntimeError(
            f"No hyperparameter setting produced valid fits in all {n_folds} folds for endpoint "
            f"'{endpoint}'. Best observed validity was {best_valid}/{n_folds} folds."
        )

    best_row = (
        cv_df.loc[cv_df["all_folds_valid"]]
        .sort_values(
            ["cv_mean", "n_valid_folds", "penalizer", "l1_ratio"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        .iloc[0]
        .to_dict()
    )
    fold_canonical_labs_df = pd.DataFrame(fold_canonical_labs_rows)
    return fold_df, cv_df, best_row, fold_canonical_labs_df


def fit_final_multivariable_model(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    penalizer: float,
    l1_ratio: float,
    penalizer_grid: list[float],
    split_stratification: str,
    cv_stratification: str,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    horizon_grid: np.ndarray,
    canonical_labs: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit on full train_val using the train_val canonical labs and evaluate on test.

    No-leak: canonical_labs and horizon_grid must be derived upstream from
    train_val only. The returned tuple is
        (metrics_row, summary_df, predictions, test_auc_df, test_brier_df)
    where test_brier_df contains per-horizon Brier on the test fold.
    """
    require_sksurv()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    assert_no_test_leakage(
        test_mrns=test.index,
        train_mrns=train_val.index,
        context=f"fit_final_multivariable_model[{endpoint}]",
    )
    train_mdf, test_mdf, covariate_cols = build_model_matrices(
        train_val,
        test,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
    )
    fallback_penalizers = [float(penalizer)] + sorted(
        {float(p) for p in penalizer_grid if float(p) > float(penalizer)}
    )
    model, used_penalizer, note = fit_coxnet_with_fallback(
        train_mdf,
        duration_col=duration_col,
        event_col=event_col,
        penalizers=fallback_penalizers,
        l1_ratio=float(l1_ratio),
        covariate_cols=covariate_cols,
        unpenalized_cols=["age"],
    )
    if model is None:
        raise RuntimeError(f"Final multivariable model failed for endpoint '{endpoint}': {note}")
    if used_penalizer != penalizer:
        print(
            f"  [fallback] CV-chosen penalizer={penalizer:g} did not produce a usable fit on full 80% "
            f"for '{endpoint}'; used penalizer={used_penalizer:g} instead."
        )

    train_c, train_pred = score_coxnet_model(
        model,
        train_mdf,
        duration_col=duration_col,
        event_col=event_col,
        covariate_cols=covariate_cols,
    )
    test_c, test_pred = score_coxnet_model(
        model,
        test_mdf,
        duration_col=duration_col,
        event_col=event_col,
        covariate_cols=covariate_cols,
    )
    train_mean_auc, train_auc_df = compute_ipcw_auc_t(
        train_mdf,
        train_pred,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_mdf,
        time_unit_days=auc_time_unit_days,
        max_time_unit=auc_max_time_units,
        fixed_horizons=horizon_grid,
    )
    test_mean_auc, test_auc_df = compute_ipcw_auc_t(
        test_mdf,
        test_pred,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_mdf,
        time_unit_days=auc_time_unit_days,
        max_time_unit=auc_max_time_units,
        fixed_horizons=horizon_grid,
    )

    test_surv = coxnet_survival_at_horizons(
        model,
        train_mdf,
        test_mdf,
        duration_col=duration_col,
        event_col=event_col,
        covariate_cols=covariate_cols,
        horizons=horizon_grid,
        time_unit_days=auc_time_unit_days,
    )
    train_dur_units = np.ceil(
        pd.to_numeric(train_mdf[duration_col], errors="coerce").to_numpy(dtype=float)
        / float(auc_time_unit_days)
    )
    test_dur_units = np.ceil(
        pd.to_numeric(test_mdf[duration_col], errors="coerce").to_numpy(dtype=float)
        / float(auc_time_unit_days)
    )
    test_brier_df, test_ibs = compute_brier(
        train_event=train_mdf[event_col].to_numpy(dtype=int),
        train_duration=train_dur_units,
        eval_event=test_mdf[event_col].to_numpy(dtype=int),
        eval_duration=test_dur_units,
        surv_at_horizons=test_surv,
        horizons=horizon_grid,
        time_unit_days=auc_time_unit_days,
    )
    test_brier_df = test_brier_df.copy()
    if not test_brier_df.empty:
        test_brier_df.insert(0, "endpoint", endpoint)

    metrics_row = {
        "endpoint": endpoint,
        "description": ENDPOINTS[endpoint]["description"],
        "n_train_val": len(train_val),
        "n_test": len(test),
        "n_events_train_val": int(train_val[event_col].sum()),
        "n_events_test": int(test[event_col].sum()),
        "selected_penalizer": used_penalizer,
        "selected_l1_ratio": float(l1_ratio),
        "n_covariates": len(covariate_cols),
        "n_canonical_labs": len(canonical_labs),
        "train_val_c_index": train_c,
        "test_c_index": test_c,
        "train_val_mean_auc_t": train_mean_auc,
        "test_mean_auc_t": test_mean_auc,
        "test_integrated_brier": test_ibs,
        "auc_quantiles": ",".join(f"{q:g}" for q in DEFAULT_AUC_QUANTILES),
        "auc_time_unit_days": auc_time_unit_days,
        "auc_admin_censor_time_unit": auc_max_time_units,
        "horizon_grid": ",".join(f"{float(h):g}" for h in np.asarray(horizon_grid, dtype=float).reshape(-1)),
        "train_val_n_valid_auc_horizons": int(train_auc_df["auc_t"].notna().sum()) if not train_auc_df.empty else 0,
        "test_n_valid_auc_horizons": int(test_auc_df["auc_t"].notna().sum()) if not test_auc_df.empty else 0,
        "test_n_valid_brier_horizons": int(test_brier_df["brier"].notna().sum()) if not test_brier_df.empty else 0,
        "split_stratification": split_stratification,
        "cv_stratification": cv_stratification,
        "note": note,
    }
    for label, auc_df in [("train_val", train_auc_df), ("test", test_auc_df)]:
        if auc_df.empty:
            continue
        for _, auc_row in auc_df.iterrows():
            horizon = float(auc_row["horizon_time_unit"])
            horizon_label = f"h{int(horizon)}"
            metrics_row[f"{label}_auc_{horizon_label}"] = float(auc_row["auc_t"])
    if not test_brier_df.empty:
        for _, brier_row in test_brier_df.iterrows():
            horizon = float(brier_row["horizon_time_unit"])
            horizon_label = f"h{int(horizon)}"
            metrics_row[f"test_brier_{horizon_label}"] = float(brier_row["brier"])

    # CoxnetSurvivalAnalysis does not produce SE / p-values / CIs — penalized
    # MLE estimates do not have meaningful analytic standard errors, so we only
    # emit coef and exp(coef) here.
    coefs = coxnet_coefficients(model, covariate_cols)
    summary = coefs.reset_index().rename(columns={"index": "feature", "coef": "coef"})
    summary["exp(coef)"] = np.exp(summary["coef"].to_numpy(dtype=float))
    parsed = summary["feature"].map(parse_feature_name)
    summary["endpoint"] = endpoint
    summary["lab_name"] = parsed.str[0]
    summary["feature_stat"] = parsed.str[1]
    summary["is_age_covariate"] = summary["feature"].eq("age")
    summary["selected_penalizer"] = used_penalizer
    summary["selected_l1_ratio"] = float(l1_ratio)
    summary["n_covariates"] = len(covariate_cols)
    summary["train_val_c_index"] = train_c
    summary["test_c_index"] = test_c
    summary["train_val_mean_auc_t"] = train_mean_auc
    summary["test_mean_auc_t"] = test_mean_auc
    summary["auc_admin_censor_time_unit"] = auc_max_time_units
    summary["note"] = note
    keep_cols = [
        "endpoint",
        "feature",
        "lab_name",
        "feature_stat",
        "is_age_covariate",
        "coef",
        "exp(coef)",
        "selected_penalizer",
        "selected_l1_ratio",
        "n_covariates",
        "train_val_c_index",
        "test_c_index",
        "train_val_mean_auc_t",
        "test_mean_auc_t",
        "auc_admin_censor_time_unit",
        "note",
    ]
    summary = summary[[c for c in keep_cols if c in summary.columns]]
    summary = summary.sort_values("coef", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    predictions = pd.DataFrame(
        {
            "DFCI_MRN": test.index,
            "endpoint": endpoint,
            "dataset": "test",
            "duration_days": test[duration_col].to_numpy(dtype=float),
            "event": test[event_col].to_numpy(dtype=int),
            "risk_score": np.asarray(test_pred).reshape(-1),
        }
    )
    test_auc_df = test_auc_df.copy()
    if not test_auc_df.empty:
        test_auc_df.insert(0, "endpoint", endpoint)
    return metrics_row, summary, predictions, test_auc_df, test_brier_df


def print_top_hits(df: pd.DataFrame, *, endpoint: str, label: str = "univariate") -> None:
    estimable = df.loc[df["p_value"].notna()]
    n_tested = len(estimable)
    n_sig_p = int((estimable["p_value"] < 0.05).sum()) if n_tested else 0
    n_sig_q = (
        int((estimable["q_value"] < 0.05).sum())
        if n_tested and "q_value" in estimable.columns
        else 0
    )
    print(f"\nTop {label} associations for {endpoint}:")
    print(
        f"  Significant hits: {n_sig_p}/{n_tested} at p<0.05, "
        f"{n_sig_q}/{n_tested} at q<0.05 (BH)"
    )
    hits = estimable[["feature", "hazard_ratio_per_sd", "p_value", "q_value"]].head(10)
    if hits.empty:
        print("  No estimable feature associations.")
        return
    print(hits.to_string(index=False))


def _load_build_manifest(inputs_dir: Path) -> dict:
    from build_prediction_inputs import BUILD_MANIFEST_FILENAME

    manifest_path = inputs_dir / BUILD_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing build manifest at {manifest_path}. Run build_prediction_inputs.py first."
        )
    import json as _json

    return _json.loads(manifest_path.read_text())


def _load_prebuilt_landmark(
    inputs_dir: Path,
    landmark_day: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load aggregated + pre-treatment lab data written by build_prediction_inputs.py.

    Returns (merged, train_val, test, pre_treatment_lab_df).
    `merged` carries the original 3-way split column; `train_val` is the row
    subset where split is in {train, valid} and `test` is split == "test".
    """
    from build_prediction_inputs import (
        aggregated_filename,
        pre_treatment_lab_filename,
    )

    agg_path = inputs_dir / aggregated_filename(landmark_day)
    if not agg_path.exists():
        raise FileNotFoundError(
            f"Missing aggregated input for landmark +{landmark_day}d at {agg_path}. "
            "Run build_prediction_inputs.py first."
        )
    aggregated = pd.read_csv(agg_path).set_index("DFCI_MRN")
    if "split" not in aggregated.columns:
        raise ValueError(f"{agg_path} is missing the 'split' column.")
    train_val = aggregated.loc[aggregated["split"].isin(["train", "valid"])].copy()
    test = aggregated.loc[aggregated["split"].eq("test")].copy()
    if train_val.empty or test.empty:
        raise ValueError(
            f"Landmark +{landmark_day}d aggregated input has empty train_val or test "
            f"({len(train_val)} / {len(test)})."
        )

    pre_path = inputs_dir / pre_treatment_lab_filename(landmark_day)
    if not pre_path.exists():
        raise FileNotFoundError(
            f"Missing pre-landmark lab data for landmark +{landmark_day}d at {pre_path}. "
            "Run build_prediction_inputs.py first."
        )
    pre_treatment_lab_df = pd.read_csv(pre_path)
    return aggregated, train_val, test, pre_treatment_lab_df


def main(args: argparse.Namespace) -> None:
    global RESULTS
    RESULTS = Path(args.output_dir)
    RESULTS.mkdir(parents=True, exist_ok=True)
    endpoints = normalize_endpoints(args.endpoints)
    landmark_days = normalize_landmark_days(args.landmark_days)
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(
            f"Inputs dir {inputs_dir} not found. Run build_prediction_inputs.py first."
        )
    build_manifest = _load_build_manifest(inputs_dir)
    min_patient_coverage = float(build_manifest["min_patient_coverage"])
    args.auc_time_unit_days = int(build_manifest["auc_time_unit_days"])
    args.auc_quantiles = tuple(build_manifest["auc_quantiles"])
    args.auc_max_time_units = None
    auc_horizons_by_landmark = build_manifest["auc_horizons_by_landmark"]
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={min_patient_coverage}, "
        f"auc_time_unit_days={args.auc_time_unit_days} per build manifest)"
    )

    feature_selection_frames: list[pd.DataFrame] = []
    univariate_nobs_adjusted_frames: list[pd.DataFrame] = []
    multivariable_frames: list[pd.DataFrame] = []
    multivariable_metric_rows: list[dict] = []
    multivariable_test_auc_frames: list[pd.DataFrame] = []
    multivariable_test_brier_frames: list[pd.DataFrame] = []
    horizon_grid_rows: list[pd.DataFrame] = []
    canonical_labs_train_val_rows: list[dict] = []
    canonical_labs_fold_rows: list[pd.DataFrame] = []

    univariate_nobs_adjusted_keep_cols = [
        "landmark_days",
        "endpoint",
        "feature",
        "lab_name",
        "feature_stat",
        "n_obs_feature",
        "coverage",
        "n_obs_coverage",
        "n_patients_used",
        "n_patients_observed",
        "n_patients_imputed",
        "n_patients_n_obs_observed",
        "n_patients_n_obs_imputed",
        "n_events_used",
        "coef_feature",
        "hazard_ratio_per_sd",
        "ci_lower",
        "ci_upper",
        "p_value",
        "q_value",
        "coef_n_obs",
        "hazard_ratio_n_obs_per_sd",
        "ci_lower_n_obs",
        "ci_upper_n_obs",
        "p_value_n_obs",
        "coef_missing",
        "p_value_missing",
        "note",
    ]

    for landmark_day in landmark_days:
        print(f"\n##### LANDMARK ANALYSES: +{landmark_day} DAYS #####")
        merged, train_val, test, pre_treatment_lab_df = _load_prebuilt_landmark(
            inputs_dir, landmark_day
        )
        raw_feature_cols = [c for c in merged.columns if c not in OUTCOME_COLUMNS]
        univariate_data = merged.copy()

        split_stratification = "prebuilt"

        assert_no_test_leakage(
            test_mrns=test.index,
            train_mrns=train_val.index,
            context=f"main[landmark+{landmark_day}d]",
        )

        # Canonical lab list: derived from pre-treatment lab observations of
        # train_val MRNs only, mirroring build_prediction_inputs.py so the
        # per-stat feature filter agrees with the upstream canonical set.
        canonical_labs = select_canonical_labs(
            pre_treatment_lab_df,
            mrns=train_val.index,
            min_coverage=min_patient_coverage,
        )
        for lab in canonical_labs:
            canonical_labs_train_val_rows.append(
                {"landmark_days": landmark_day, "lab_name": lab}
            )

        # Feature selection on train_val only to avoid leaking test-set coverage,
        # restricted to the canonical lab set so Cox / XGBoost / DeepHit operate
        # on the same labs (different feature representations only).
        selected_feature_cols, feature_meta = select_feature_columns(
            train_val,
            raw_feature_cols,
            min_patient_coverage=min_patient_coverage,
            restrict_to_labs=canonical_labs,
        )
        feature_meta_selected = feature_meta.loc[
            feature_meta["selected"],
            ["feature", "lab_name", "feature_stat", "coverage", "unique_non_missing"],
        ].copy()
        feature_meta_selected.insert(0, "landmark_days", landmark_day)
        feature_selection_frames.append(feature_meta_selected)

        # NOTE: do NOT subset train_val / merged / test to selected_feature_cols.
        # The multivariable arm's per-fold selection rebuilds the canonical lab
        # set and per-stat filter against the *raw* feature universe inside each
        # CV fold (see tune_multivariable_model), so it needs access to all
        # raw_feature_cols on train_val. Downstream call sites already specify
        # the column subset they consume via select_feature_columns and
        # build_model_matrices, so leaving the full feature universe in place
        # is harmless for univariate / final-fit paths.

        print(f"Full cohort: {len(merged)} patients")
        print(f"Train/val (Arm 2): {len(train_val)} patients")
        print(f"Test (Arm 2):      {len(test)} patients")
        print(f"Canonical labs (train_val): {len(canonical_labs)}")
        print(f"Selected summary-lab features (train_val pre-filter): {len(selected_feature_cols)}")

        # AUC(t) horizon grid is loaded from build_manifest.json so Cox / XGBoost /
        # DeepHit all evaluate on the identical horizon set per (landmark, endpoint).
        landmark_horizons = auc_horizons_by_landmark.get(str(int(landmark_day)))
        if landmark_horizons is None:
            raise KeyError(
                f"build_manifest.json has no auc_horizons_by_landmark entry for landmark +{landmark_day}d. "
                "Re-run build_prediction_inputs.py for this landmark."
            )
        endpoint_horizon_grids: dict[str, np.ndarray] = {}
        for endpoint in endpoints:
            if endpoint not in landmark_horizons:
                raise KeyError(
                    f"build_manifest.json missing horizons for endpoint {endpoint!r} at landmark +{landmark_day}d."
                )
            grid = np.asarray(landmark_horizons[endpoint], dtype=float)
            endpoint_horizon_grids[endpoint] = grid
            grid_df = horizon_grid_frame(
                grid,
                quantiles=args.auc_quantiles,
                time_unit_days=args.auc_time_unit_days,
                endpoint=endpoint,
            )
            grid_df.insert(0, "landmark_days", landmark_day)
            horizon_grid_rows.append(grid_df)
            print(
                f"Horizon grid ({endpoint}, from manifest): "
                + ", ".join(f"{int(h)}" for h in grid)
                + f" {args.auc_time_unit_days}-day units"
            )

        if args.analysis in {"univariate", "both"}:
            print("\n##### ARM 1: UNIVARIATE (n_obs-adjusted, all endpoints) #####")
            for endpoint in endpoints:
                print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
                print(ENDPOINTS[endpoint]["description"])
                adjusted_df = run_univariate_nobs_adjusted_associations(
                    univariate_data,
                    feature_cols=selected_feature_cols,
                    endpoint=endpoint,
                    min_events_per_feature=args.min_events_per_feature,
                    fallback_penalizer=args.univariate_penalizer,
                )
                adjusted_df.insert(0, "landmark_days", landmark_day)
                univariate_nobs_adjusted_frames.append(
                    adjusted_df[univariate_nobs_adjusted_keep_cols].copy()
                )
                print_top_hits(
                    adjusted_df,
                    endpoint=endpoint,
                    label="n_obs-adjusted univariate",
                )

        if args.analysis in {"multivariable", "both"}:
            print("\n##### ARM 2: MULTIVARIABLE ELASTIC-NET (all endpoints) #####")
            for endpoint in endpoints:
                print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
                print(ENDPOINTS[endpoint]["description"])
                horizon_grid = endpoint_horizon_grids[endpoint]
                _, _, best_row, fold_canonical_labs_df = tune_multivariable_model(
                    train_val,
                    raw_feature_cols=raw_feature_cols,
                    endpoint=endpoint,
                    penalizers=args.cv_penalizers,
                    l1_ratios=args.cv_l1_ratios,
                    n_folds=args.n_folds,
                    seed=args.seed,
                    auc_time_unit_days=args.auc_time_unit_days,
                    auc_max_time_units=args.auc_max_time_units,
                    pre_treatment_lab_df=pre_treatment_lab_df,
                    horizon_grid=horizon_grid,
                    min_patient_coverage=min_patient_coverage,
                )
                if not fold_canonical_labs_df.empty:
                    fold_canonical_labs_df.insert(0, "landmark_days", landmark_day)
                    canonical_labs_fold_rows.append(fold_canonical_labs_df)

                (
                    metrics_row,
                    summary_df,
                    _,
                    test_auc_df,
                    test_brier_df,
                ) = fit_final_multivariable_model(
                    train_val,
                    test,
                    feature_cols=selected_feature_cols,
                    endpoint=endpoint,
                    penalizer=float(best_row["penalizer"]),
                    l1_ratio=float(best_row["l1_ratio"]),
                    penalizer_grid=args.cv_penalizers,
                    split_stratification=split_stratification,
                    cv_stratification=str(best_row["cv_stratification"]),
                    auc_time_unit_days=args.auc_time_unit_days,
                    auc_max_time_units=args.auc_max_time_units,
                    horizon_grid=horizon_grid,
                    canonical_labs=canonical_labs,
                )
                metrics_row["landmark_days"] = landmark_day
                summary_df.insert(0, "landmark_days", landmark_day)
                multivariable_metric_rows.append(metrics_row)
                multivariable_frames.append(summary_df)
                if not test_auc_df.empty:
                    test_auc_df = test_auc_df.copy()
                    test_auc_df.insert(0, "landmark_days", landmark_day)
                    multivariable_test_auc_frames.append(test_auc_df)
                if not test_brier_df.empty:
                    test_brier_df = test_brier_df.copy()
                    test_brier_df.insert(0, "landmark_days", landmark_day)
                    multivariable_test_brier_frames.append(test_brier_df)

                top_cols = [c for c in ["feature", "coef", "exp(coef)"] if c in summary_df.columns]
                top = summary_df.loc[~summary_df["is_age_covariate"], top_cols].head(10)
                print("\nChosen hyperparameters (elastic-net, age unpenalized):")
                print(
                    f"  penalizer={best_row['penalizer']}  l1_ratio={best_row['l1_ratio']}  "
                    f"cv_mean C-index={best_row['cv_mean']:.4f}"
                )
                print(f"  CV mean AUC(t)={best_row['mean_auc_t_cv_mean']:.4f}")
                print(f"  CV mean integrated Brier={best_row['integrated_brier_cv_mean']:.4f}")
                print(
                    f"  train/val C-index={metrics_row['train_val_c_index']:.4f}  "
                    f"mean AUC(t)={metrics_row['train_val_mean_auc_t']:.4f}"
                )
                print(f"  held-out test C-index={metrics_row['test_c_index']:.4f}")
                print(f"  held-out test mean AUC(t)={metrics_row['test_mean_auc_t']:.4f}")
                print(f"  held-out test integrated Brier={metrics_row['test_integrated_brier']:.4f}")
                print("Top multivariable coefficients:")
                print(top.to_string(index=False))

    if feature_selection_frames:
        pd.concat(feature_selection_frames, ignore_index=True).to_csv(
            RESULTS / "cox_agg_feature_selection.csv", index=False
        )
    if horizon_grid_rows:
        pd.concat(horizon_grid_rows, ignore_index=True).to_csv(
            RESULTS / HORIZON_GRID_FILENAME, index=False
        )
    # canonical_labs_train_val is owned by build_prediction_inputs.py; we keep
    # canonical_labs_train_val_rows in memory only for cross-checking.
    _ = canonical_labs_train_val_rows
    if canonical_labs_fold_rows:
        pd.concat(canonical_labs_fold_rows, ignore_index=True).to_csv(
            RESULTS / CANONICAL_LABS_FOLDS_FILENAME, index=False
        )
    if univariate_nobs_adjusted_frames:
        pd.concat(univariate_nobs_adjusted_frames, ignore_index=True).to_csv(
            RESULTS / "cox_agg_univariate_nobs_adjusted.csv", index=False
        )
    if multivariable_frames:
        pd.concat(multivariable_frames, ignore_index=True).to_csv(
            RESULTS / "cox_agg_multivariable.csv", index=False
        )
    if multivariable_test_auc_frames:
        pd.concat(multivariable_test_auc_frames, ignore_index=True).to_csv(
            RESULTS / "cox_agg_multivariable_test_auc_t.csv", index=False
        )
    if multivariable_test_brier_frames:
        pd.concat(multivariable_test_brier_frames, ignore_index=True).to_csv(
            RESULTS / "cox_agg_multivariable_test_brier.csv", index=False
        )
    if multivariable_metric_rows:
        pd.DataFrame(multivariable_metric_rows).to_csv(
            RESULTS / "cox_agg_multivariable_metrics.csv", index=False
        )

    print("\nSaved:")
    print("  results/cox_agg_feature_selection.csv")
    if horizon_grid_rows:
        print(f"  results/{HORIZON_GRID_FILENAME}")
    if canonical_labs_fold_rows:
        print(f"  results/{CANONICAL_LABS_FOLDS_FILENAME}")
    if univariate_nobs_adjusted_frames:
        print("  results/cox_agg_univariate_nobs_adjusted.csv")
    if multivariable_frames:
        print("  results/cox_agg_multivariable.csv")
    if multivariable_test_auc_frames:
        print("  results/cox_agg_multivariable_test_auc_t.csv")
    if multivariable_test_brier_frames:
        print("  results/cox_agg_multivariable_test_brier.csv")
    if multivariable_metric_rows:
        print("  results/cox_agg_multivariable_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs-dir",
        default=str(RESULTS / "prediction_inputs"),
        help="Directory containing prebuilt inputs from build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS),
        help="Directory for Cox result CSVs.",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["platinum", "death"],
        choices=list(ENDPOINTS),
        help="Endpoints to analyze.",
    )
    parser.add_argument(
        "--landmark-days",
        nargs="+",
        type=int,
        default=DEFAULT_LANDMARK_DAYS,
        help="Landmark offsets to analyze. Each must have prebuilt inputs in --inputs-dir.",
    )
    parser.add_argument(
        "--analysis",
        choices=["univariate", "multivariable", "both"],
        default="both",
        help="Association analyses to run on the aggregated feature set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for cross-validation. The patient split is fixed by build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of cross-validation folds within the train/validation cohort.",
    )
    parser.add_argument(
        "--min-events-per-feature",
        type=int,
        default=DEFAULT_MIN_EVENTS_PER_FEATURE,
        help="Skip univariate associations when too few endpoint events remain after outcome filtering.",
    )
    parser.add_argument(
        "--univariate-penalizer",
        type=float,
        default=0.05,
        help="Fallback penalizer used only when an univariate Cox model does not converge without regularization.",
    )
    parser.add_argument(
        "--cv-penalizers",
        nargs="+",
        type=float,
        default=DEFAULT_CV_PENALIZERS,
        help="Penalizer values searched during 5-fold CV on the 80%% train/val block.",
    )
    parser.add_argument(
        "--cv-l1-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_CV_L1_RATIOS,
        help="Elastic-net L1 mixing values (0=ridge, 1=lasso) searched during 5-fold CV.",
    )
    # AUC(t) time unit, quantiles, horizons all come from build_manifest.json
    # so Cox/XGB/DeepHit evaluate on the identical horizon set.
    main(parser.parse_args())
