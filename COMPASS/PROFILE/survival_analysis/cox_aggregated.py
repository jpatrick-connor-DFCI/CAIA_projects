"""
Two-arm survival analysis on pre-treatment lab summary features.

Features per lab (all pre-first-treatment): mean, min, max, last, slope,
delta (last - first), n_observations.

Arm 1 (univariate, full dataset):
  - For each feature, fit Cox on [AGE + feature] using all patients.
  - Extract coefficient, HR per SD, 95% CI, and p-value.

Arm 2 (multivariable elastic-net Cox):
  - 80% train/val + 20% held-out test.
  - 5-fold CV over (penalizer x l1_ratio) grid on the 80%; AGE is unpenalized.
  - Refit on full 80% with chosen (penalizer, l1_ratio) and evaluate on 20% test:
    C-index and SurvLatent-style IPCW cumulative/dynamic AUC(t).

Endpoints: platinum, death.

Expected input:
  Row-level longitudinal data with at least
    DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab, t_first_treatment, t_platinum,
    t_death (or t_last_contact), PLATINUM, DEATH, AGE_AT_TREATMENTSTART

Outputs:
  results/cox_agg_feature_selection.csv   (selected lab features + coverage)
  results/cox_agg_univariate.csv          (log HRs, p-values, FDR per endpoint)
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
    from sksurv.metrics import cumulative_dynamic_auc

    SKSURV_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    cumulative_dynamic_auc = None
    SKSURV_IMPORT_ERROR = exc

BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")

AGE_COL = "AGE_AT_TREATMENTSTART"
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_N_FOLDS = 5
DEFAULT_MIN_PATIENT_COVERAGE = 0.20
DEFAULT_MIN_EVENTS_PER_FEATURE = 10
DEFAULT_CV_PENALIZERS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
DEFAULT_CV_L1_RATIOS = [0.5, 1.0]
DEFAULT_AUC_QUANTILES = (0.25, 0.375, 0.50, 0.625, 0.75)
DEFAULT_AUC_TIME_UNIT_DAYS = 7
PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}
MIN_SLOPE_OBS = 3
MIN_SLOPE_SPAN_DAYS = 7.0
MIN_DELTA_OBS = 2
SPLIT_ASSIGNMENTS_FILENAME = "cox_agg_split_assignments.csv"

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
            "scikit-survival is required for the SurvLatent-style Cox AUC(t) evaluation."
        ) from SKSURV_IMPORT_ERROR


def parse_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "value"
    return feature_name.rsplit("__", 1)


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


def make_outcome_df(df: pd.DataFrame) -> pd.DataFrame:
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

    if "PLATINUM_DATE" in pat.columns and "LAST_CONTACT_DATE" in pat.columns:
        pat["PLATINUM_DATE"] = pat["PLATINUM_DATE"].fillna(pat["LAST_CONTACT_DATE"])

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
        fallback_duration_col="t_last_contact",
    )

    landmark_time = pat["t_first_treatment"].astype(float)
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


def _patient_lab_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return np.nan
    return float(np.std(values.to_numpy(dtype=float), ddof=0))


def _compute_patient_lab_slopes(pre_treatment: pd.DataFrame) -> pd.DataFrame:
    """Slope of LAB_VALUE vs t_lab (per day) per (DFCI_MRN, LAB_NAME).

    Returns NaN unless a patient has >= MIN_SLOPE_OBS observations spanning at
    least MIN_SLOPE_SPAN_DAYS; short-span fits produce unstable, extreme slopes.
    """
    def _slope(group: pd.DataFrame) -> float:
        if len(group) < MIN_SLOPE_OBS:
            return np.nan
        x = group["t_lab"].to_numpy(dtype=float)
        y = group["LAB_VALUE"].to_numpy(dtype=float)
        if (x.max() - x.min()) < MIN_SLOPE_SPAN_DAYS:
            return np.nan
        cov = np.cov(x, y, ddof=0)
        var_x = cov[0, 0]
        if var_x == 0:
            return np.nan
        return float(cov[0, 1] / var_x)

    slopes = (
        pre_treatment.groupby(["DFCI_MRN", "LAB_NAME"])[["t_lab", "LAB_VALUE"]]
        .apply(_slope)
        .rename("slope")
        .reset_index()
    )
    return slopes


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
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

    pre_treatment = working.loc[working["t_lab"].lt(working["t_first_treatment"])].copy()
    if pre_treatment.empty:
        raise ValueError("No pre-treatment lab rows were available to build lab summary features.")

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Build the shared landmarked patient cohort and held-out test split."""
    outcome_df = make_outcome_df(df)
    print(f"Outcome table: {len(outcome_df)} patients")

    print("Building raw aggregated pre-treatment lab summary feature matrix...")
    feature_df = build_feature_matrix(df)

    merged = feature_df.join(outcome_df, how="inner")
    merged = merged.loc[merged[AGE_COL].notna()].copy()
    if merged.empty:
        raise ValueError("No patients have both engineered features and valid outcomes.")

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
    return outcome_df, feature_df, merged, train_val, test, split_assignments, split_stratification


def select_feature_columns(
    data: pd.DataFrame,
    raw_feature_cols: list[str],
    *,
    min_patient_coverage: float,
) -> tuple[list[str], pd.DataFrame]:
    """Select features on the training/validation block to avoid test leakage."""
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

    train_model["age"] = train_df[AGE_COL].to_numpy(dtype=float)
    eval_model["age"] = eval_df[AGE_COL].to_numpy(dtype=float)
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


def compute_survlatent_auc_t(
    eval_df: pd.DataFrame,
    risk_score: np.ndarray,
    *,
    duration_col: str,
    event_col: str,
    reference_df: pd.DataFrame,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
    quantiles: tuple[float, ...] = DEFAULT_AUC_QUANTILES,
) -> tuple[float, pd.DataFrame]:
    """Match SurvLatent ODE's IPCW cumulative/dynamic AUC(t) evaluation."""
    require_sksurv()

    empty_cols = [
        "horizon_quantile",
        "horizon_time_unit",
        "horizon_days",
        "auc_t",
        "n_eval",
        "n_eval_events",
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

    train_valid = np.isfinite(train_duration) & (train_duration > 0)
    eval_valid = np.isfinite(eval_duration) & (eval_duration > 0) & np.isfinite(risk_score)
    if train_valid.sum() == 0 or eval_valid.sum() == 0:
        return np.nan, pd.DataFrame(columns=empty_cols)

    train_surv = _make_survival_array(train_event[train_valid], train_duration[train_valid])
    eval_surv = _make_survival_array(eval_event[eval_valid], eval_duration[eval_valid])
    eval_risk = risk_score[eval_valid]

    event_times = eval_duration[eval_valid & (eval_event == 1)]
    event_times = event_times[np.isfinite(event_times) & (event_times > 0)]
    if len(event_times) == 0:
        return np.nan, pd.DataFrame(columns=empty_cols)

    horizon_times = np.asarray(
        [int(val) for val in np.quantile(event_times, quantiles)],
        dtype=float,
    )
    rows = []
    for quantile, horizon in zip(quantiles, horizon_times):
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
        feature_df["age"] = feature_df[AGE_COL]
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
    feature_cols: list[str],
    endpoint: str,
    penalizers: list[float],
    l1_ratios: list[float],
    n_folds: int,
    seed: int,
    auc_time_unit_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Arm 2: 5-fold CV over (penalizer x l1_ratio) grid (elastic-net), AGE unpenalized."""
    require_lifelines()
    require_sksurv()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    splitter, strat_labels, cv_stratification = make_cv_splitter(train_val, n_folds=n_folds, seed=seed)
    fold_rows = []

    split_args = (np.arange(len(train_val)), strat_labels) if strat_labels is not None else (np.arange(len(train_val)),)

    for penalizer, l1_ratio in product(penalizers, l1_ratios):
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(*split_args), 1):
            fold_train = train_val.iloc[tr_idx]
            fold_val = train_val.iloc[val_idx]
            row = {
                "endpoint": endpoint,
                "fold": fold,
                "penalizer": float(penalizer),
                "l1_ratio": float(l1_ratio),
                "n_train": len(fold_train),
                "n_val": len(fold_val),
                "n_events_train": int(fold_train[event_col].sum()),
                "n_events_val": int(fold_val[event_col].sum()),
                "cv_stratification": cv_stratification,
                "c_index_val": np.nan,
                "mean_auc_t_val": np.nan,
                "n_valid_auc_horizons_val": 0,
                "n_covariates": np.nan,
                "note": "",
            }

            try:
                train_mdf, val_mdf, covariate_cols = build_model_matrices(
                    fold_train,
                    fold_val,
                    feature_cols=feature_cols,
                    duration_col=duration_col,
                    event_col=event_col,
                )
                model, _, note = fit_cox_with_fallback(
                    train_mdf,
                    duration_col=duration_col,
                    event_col=event_col,
                    penalizers=[float(penalizer)],
                    l1_ratio=float(l1_ratio),
                    unpenalized_cols=["age"],
                    covariate_cols=covariate_cols,
                )
                row["note"] = note
                row["n_covariates"] = len(covariate_cols)
                if model is not None:
                    c_index, val_pred = score_cox_model(
                        model,
                        val_mdf,
                        duration_col=duration_col,
                        event_col=event_col,
                    )
                    row["c_index_val"] = c_index
                    mean_auc_val, auc_df_val = compute_survlatent_auc_t(
                        val_mdf,
                        val_pred,
                        duration_col=duration_col,
                        event_col=event_col,
                        reference_df=train_mdf,
                        time_unit_days=auc_time_unit_days,
                    )
                    row["mean_auc_t_val"] = mean_auc_val
                    row["n_valid_auc_horizons_val"] = (
                        int(auc_df_val["auc_t"].notna().sum()) if not auc_df_val.empty else 0
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
            n_covariates_mean=("n_covariates", "mean"),
            cv_stratification=("cv_stratification", "first"),
        )
        .reset_index()
    )

    if cv_df["n_valid_folds"].eq(0).all():
        raise RuntimeError(f"All CV fits failed for endpoint '{endpoint}'.")

    best_row = (
        cv_df.sort_values(
            ["cv_mean", "n_valid_folds", "penalizer", "l1_ratio"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        .iloc[0]
        .to_dict()
    )
    return fold_df, cv_df, best_row


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
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    require_lifelines()
    require_sksurv()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

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
    model, used_penalizer, note = fit_cox_with_fallback(
        train_mdf,
        duration_col=duration_col,
        event_col=event_col,
        penalizers=fallback_penalizers,
        l1_ratio=float(l1_ratio),
        unpenalized_cols=["age"],
        covariate_cols=covariate_cols,
    )
    if model is None:
        raise RuntimeError(f"Final multivariable model failed for endpoint '{endpoint}': {note}")
    if used_penalizer != penalizer:
        print(
            f"  [fallback] CV-chosen penalizer={penalizer:g} failed to converge on full 80% "
            f"for '{endpoint}'; used penalizer={used_penalizer:g} instead."
        )

    train_c, train_pred = score_cox_model(
        model,
        train_mdf,
        duration_col=duration_col,
        event_col=event_col,
    )
    test_c, test_pred = score_cox_model(
        model,
        test_mdf,
        duration_col=duration_col,
        event_col=event_col,
    )
    train_mean_auc, train_auc_df = compute_survlatent_auc_t(
        train_mdf,
        train_pred,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_mdf,
        time_unit_days=auc_time_unit_days,
    )
    test_mean_auc, test_auc_df = compute_survlatent_auc_t(
        test_mdf,
        test_pred,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_mdf,
        time_unit_days=auc_time_unit_days,
    )

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
        "train_val_c_index": train_c,
        "test_c_index": test_c,
        "train_val_mean_auc_t": train_mean_auc,
        "test_mean_auc_t": test_mean_auc,
        "auc_quantiles": ",".join(f"{q:g}" for q in DEFAULT_AUC_QUANTILES),
        "auc_time_unit_days": auc_time_unit_days,
        "train_val_n_valid_auc_horizons": int(train_auc_df["auc_t"].notna().sum()) if not train_auc_df.empty else 0,
        "test_n_valid_auc_horizons": int(test_auc_df["auc_t"].notna().sum()) if not test_auc_df.empty else 0,
        "split_stratification": split_stratification,
        "cv_stratification": cv_stratification,
        "note": note,
    }
    for label, auc_df in [("train_val", train_auc_df), ("test", test_auc_df)]:
        if auc_df.empty:
            continue
        for _, auc_row in auc_df.iterrows():
            quantile = float(auc_row["horizon_quantile"])
            quantile_label = f"{quantile * 100:g}".replace(".", "_")
            metrics_row[f"{label}_auc_{quantile_label}_percentile"] = float(auc_row["auc_t"])
            metrics_row[f"{label}_auc_{quantile_label}_time_unit"] = float(
                auc_row["horizon_time_unit"]
            )

    summary = model.summary.reset_index().rename(columns={"covariate": "feature"})
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
    return metrics_row, summary, predictions


def print_top_hits(df: pd.DataFrame, *, endpoint: str) -> None:
    estimable = df.loc[df["p_value"].notna()]
    n_tested = len(estimable)
    n_sig_p = int((estimable["p_value"] < 0.05).sum()) if n_tested else 0
    n_sig_q = (
        int((estimable["q_value"] < 0.05).sum())
        if n_tested and "q_value" in estimable.columns
        else 0
    )
    print(f"\nTop univariate associations for {endpoint}:")
    print(
        f"  Significant hits: {n_sig_p}/{n_tested} at p<0.05, "
        f"{n_sig_q}/{n_tested} at q<0.05 (BH)"
    )
    hits = estimable[["feature", "hazard_ratio_per_sd", "p_value", "q_value"]].head(10)
    if hits.empty:
        print("  No estimable feature associations.")
        return
    print(hits.to_string(index=False))


def main(args: argparse.Namespace) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    endpoints = normalize_endpoints(args.endpoints)

    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)

    print("Building landmarked outcome table and aligned patient split...")
    _, feature_df, merged, train_val, test, split_assignments, split_stratification = build_aligned_cohort(
        df,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    raw_feature_cols = feature_df.columns.tolist()

    # Feature selection on train_val only to avoid leaking test-set coverage.
    selected_feature_cols, feature_meta = select_feature_columns(
        train_val,
        raw_feature_cols,
        min_patient_coverage=args.min_patient_coverage,
    )

    keep_cols = selected_feature_cols + [c for c in merged.columns if c in OUTCOME_COLUMNS]
    merged = merged[keep_cols].copy()
    train_val = train_val[keep_cols].copy()
    test = test[keep_cols].copy()

    print(f"Full cohort: {len(merged)} patients")
    print(f"Train/val (Arm 2): {len(train_val)} patients")
    print(f"Test (Arm 2):      {len(test)} patients")
    print(f"Selected summary-lab features: {len(selected_feature_cols)}")
    print(f"Split stratification: {split_stratification}")

    split_assignments.rename_axis("DFCI_MRN").reset_index().to_csv(
        RESULTS / SPLIT_ASSIGNMENTS_FILENAME,
        index=False,
    )

    feature_meta.loc[
        feature_meta["selected"],
        ["feature", "lab_name", "feature_stat", "coverage", "unique_non_missing"],
    ].to_csv(RESULTS / "cox_agg_feature_selection.csv", index=False)

    univariate_frames: list[pd.DataFrame] = []
    multivariable_frames: list[pd.DataFrame] = []
    multivariable_metric_rows: list[dict] = []

    univariate_keep_cols = [
        "endpoint",
        "feature",
        "lab_name",
        "feature_stat",
        "coverage",
        "n_patients_used",
        "n_patients_observed",
        "n_patients_imputed",
        "n_events_used",
        "coef_feature",
        "hazard_ratio_per_sd",
        "ci_lower",
        "ci_upper",
        "p_value",
        "q_value",
        "coef_missing",
        "p_value_missing",
    ]

    if args.analysis in {"univariate", "both"}:
        print("\n##### ARM 1: UNIVARIATE (all endpoints) #####")
        for endpoint in endpoints:
            print(f"\n=== {endpoint.upper()} ===")
            print(ENDPOINTS[endpoint]["description"])
            univariate_df = run_univariate_associations(
                merged,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                min_events_per_feature=args.min_events_per_feature,
                fallback_penalizer=args.univariate_penalizer,
            )
            univariate_frames.append(univariate_df[univariate_keep_cols].copy())
            print_top_hits(univariate_df, endpoint=endpoint)

    if args.analysis in {"multivariable", "both"}:
        print("\n##### ARM 2: MULTIVARIABLE ELASTIC-NET (all endpoints) #####")
        for endpoint in endpoints:
            print(f"\n=== {endpoint.upper()} ===")
            print(ENDPOINTS[endpoint]["description"])
            _, _, best_row = tune_multivariable_model(
                train_val,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                penalizers=args.cv_penalizers,
                l1_ratios=args.cv_l1_ratios,
                n_folds=args.n_folds,
                seed=args.seed,
                auc_time_unit_days=args.auc_time_unit_days,
            )

            metrics_row, summary_df, _ = fit_final_multivariable_model(
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
            )
            multivariable_metric_rows.append(metrics_row)
            multivariable_frames.append(summary_df)

            top_cols = [c for c in ["feature", "coef", "exp(coef)"] if c in summary_df.columns]
            top = summary_df.loc[~summary_df["is_age_covariate"], top_cols].head(10)
            print("\nChosen hyperparameters (elastic-net, age unpenalized):")
            print(
                f"  penalizer={best_row['penalizer']}  l1_ratio={best_row['l1_ratio']}  "
                f"cv_mean C-index={best_row['cv_mean']:.4f}"
            )
            print(f"  CV mean AUC(t)={best_row['mean_auc_t_cv_mean']:.4f}")
            print(
                f"  train/val C-index={metrics_row['train_val_c_index']:.4f}  "
                f"mean AUC(t)={metrics_row['train_val_mean_auc_t']:.4f}"
            )
            print(f"  held-out test C-index={metrics_row['test_c_index']:.4f}")
            print(f"  held-out test mean AUC(t)={metrics_row['test_mean_auc_t']:.4f}")
            print("Top multivariable coefficients:")
            print(top.to_string(index=False))

    if univariate_frames:
        pd.concat(univariate_frames, ignore_index=True).to_csv(
            RESULTS / "cox_agg_univariate.csv", index=False
        )
    if multivariable_frames:
        pd.concat(multivariable_frames, ignore_index=True).to_csv(
            RESULTS / "cox_agg_multivariable.csv", index=False
        )
    if multivariable_metric_rows:
        pd.DataFrame(multivariable_metric_rows).to_csv(
            RESULTS / "cox_agg_multivariable_metrics.csv", index=False
        )

    print("\nSaved:")
    print(f"  results/{SPLIT_ASSIGNMENTS_FILENAME}")
    print("  results/cox_agg_feature_selection.csv")
    if univariate_frames:
        print("  results/cox_agg_univariate.csv")
    if multivariable_frames:
        print("  results/cox_agg_multivariable.csv")
    if multivariable_metric_rows:
        print("  results/cox_agg_multivariable_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["platinum", "death"],
        choices=list(ENDPOINTS),
        help="Endpoints to analyze.",
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
        help="Random seed for the patient split and cross-validation.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Fraction of patients reserved for the held-out test set.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of cross-validation folds within the train/validation cohort.",
    )
    parser.add_argument(
        "--min-patient-coverage",
        "--min-lab-availability",
        dest="min_patient_coverage",
        type=float,
        default=DEFAULT_MIN_PATIENT_COVERAGE,
        help="Minimum train/validation lab availability required for a feature to be selected.",
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
    parser.add_argument(
        "--auc-time-unit-days",
        type=int,
        default=DEFAULT_AUC_TIME_UNIT_DAYS,
        help="Time unit used for Cox AUC(t), matching SurvLatent ODE input bins by default.",
    )
    main(parser.parse_args())
