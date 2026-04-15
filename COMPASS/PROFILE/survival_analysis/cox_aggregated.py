"""
Two-arm survival analysis on pre-treatment lab summary features.

Features per lab (all pre-first-treatment): mean, min, max, last.

Arm 1 (univariate, full dataset):
  - For each feature, fit Cox on [AGE + feature] using all patients.
  - Extract coefficient, HR per SD, 95% CI, and p-value.

Arm 2 (multivariable elastic-net Cox, sksurv CoxnetSurvivalAnalysis):
  - 80% train/val + 20% held-out test.
  - 5-fold CV over (alpha x l1_ratio) grid on the 80%; AGE is unpenalized via
    sksurv's penalty_factor.
  - Refit on full 80% with chosen (alpha, l1_ratio) and evaluate on 20% test:
    C-index and integrated AUC(t) over the 5-95 percentile of event times.

Endpoints: platinum, death.

Expected input:
  Row-level longitudinal data with at least
    DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab, t_first_treatment, t_platinum,
    t_death (or t_last_contact), PLATINUM, DEATH, AGE_AT_TREATMENTSTART

Outputs:
  results/cox_agg_feature_matrix_raw.csv
  results/cox_agg_outcome_table.csv
  results/cox_agg_split_assignments.csv
  results/cox_agg_feature_selection.csv
  results/cox_agg_endpoint_summary.csv
  results/cox_agg_univariate_<endpoint>.csv
  results/cox_agg_univariate_overview.csv
  results/cox_agg_cv_metrics.csv
  results/cox_agg_cv_fold_metrics.csv
  results/cox_agg_selected_models.csv
  results/cox_agg_test_metrics.csv
  results/cox_agg_multivariable_<endpoint>.csv
  results/cox_agg_test_predictions_<endpoint>.csv
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
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
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    SKSURV_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    CoxnetSurvivalAnalysis = None
    concordance_index_censored = None
    Surv = None
    SKSURV_IMPORT_ERROR = exc


BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")
RESULTS.mkdir(parents=True, exist_ok=True)

AGE_COL = "AGE_AT_TREATMENTSTART"
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_N_FOLDS = 5
DEFAULT_MIN_PATIENT_COVERAGE = 0.20
DEFAULT_MIN_EVENTS_PER_FEATURE = 10
DEFAULT_CV_PENALIZERS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
DEFAULT_CV_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_AUC_PERCENTILE_RANGE = (0.05, 0.95)
DEFAULT_AUC_N_POINTS = 50
COXNET_MAX_ITER = 10_000
PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}

ENDPOINTS = {
    "platinum": {
        "duration_col": "t_platinum",
        "event_col": "PLATINUM",
        "description": "Time to first platinum exposure",
    },
    "death": {
        "duration_col": "t_death",
        "event_col": "DEATH",
        "description": "Time to death / last contact",
    },
    "nepc": {
        "duration_col": "t_nepc",
        "event_col": "NEPC",
        "description": "Time to NEPC transformation (LLM-labeled)",
    },
}
OUTCOME_COLUMNS = {
    AGE_COL,
    "FIRST_RECORD_DATE",
    "DIAGNOSIS_DATE",
    "DIAGNOSIS",
    "FIRST_TREATMENT_DATE",
    "FIRST_TREATMENT",
    "LAST_CONTACT_DATE",
    "PLATINUM_DATE",
    "PLATINUM",
    "DEATH",
    "EITHER",
    "NEPC_DATE",
    "NEPC",
    "t_diagnosis",
    "t_first_treatment",
    "t_platinum",
    "t_last_contact",
    "t_death",
    "t_either",
    "t_nepc",
    "split",
}


def require_lifelines() -> None:
    if CoxPHFitter is None or concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to run the aggregated Cox association pipeline."
        ) from LIFELINES_IMPORT_ERROR


def require_sksurv() -> None:
    if CoxnetSurvivalAnalysis is None or Surv is None or concordance_index_censored is None:
        raise ModuleNotFoundError(
            "scikit-survival (sksurv) is required for the multivariable elastic-net Cox fits."
        ) from SKSURV_IMPORT_ERROR


def _to_surv_y(durations: np.ndarray, events: np.ndarray):
    return Surv.from_arrays(
        event=np.asarray(events).astype(bool),
        time=np.asarray(durations).astype(float),
    )


def _penalty_factor(covariate_cols: list[str]) -> np.ndarray:
    """Return per-feature penalty multipliers (0.0 for unpenalized age, 1.0 elsewhere)."""
    return np.array([0.0 if c == "age" else 1.0 for c in covariate_cols], dtype=float)


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
        "DIAGNOSIS",
        "FIRST_TREATMENT_DATE",
        "FIRST_TREATMENT",
        "LAST_CONTACT_DATE",
        "PLATINUM_DATE",
        "PLATINUM",
        "DEATH",
        "NEPC_DATE",
        "NEPC",
        "t_diagnosis",
        "t_first_treatment",
        "t_platinum",
        "t_last_contact",
        "t_death",
        "t_nepc",
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
        "NEPC_DATE",
    ]:
        if date_col in pat.columns:
            pat[date_col] = _coerce_datetime(pat[date_col])

    if AGE_COL in pat.columns:
        pat[AGE_COL] = pd.to_numeric(pat[AGE_COL], errors="coerce")
    else:
        pat[AGE_COL] = np.nan
    pat["DEATH"] = pd.to_numeric(pat.get("DEATH"), errors="coerce").fillna(0).astype(int)
    pat["PLATINUM"] = _coerce_platinum(pat.get("PLATINUM", pd.Series(0, index=pat.index)))
    pat["NEPC"] = pd.to_numeric(pat.get("NEPC", pd.Series(0, index=pat.index)), errors="coerce").fillna(0).astype(int)
    pat["DIAGNOSIS"] = pd.to_numeric(
        pat.get("DIAGNOSIS", pat.get("DIAGNOSIS_DATE", pd.Series(index=pat.index)).notna()),
        errors="coerce",
    ).fillna(0).astype(int)
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
    pat["t_nepc"] = _derive_duration(
        pat,
        duration_col="t_nepc",
        event_date_col="NEPC_DATE",
        fallback_duration_col="t_last_contact",
    )

    platinum_event_time = np.where(pat["PLATINUM"].eq(1), pat["t_platinum"], np.inf)
    death_event_time = np.where(pat["DEATH"].eq(1), pat["t_death"], np.inf)
    first_event_time = np.minimum(platinum_event_time, death_event_time)

    pat["EITHER"] = np.isfinite(first_event_time).astype(int)
    pat["t_either"] = np.where(pat["EITHER"].eq(1), first_event_time, pat["t_death"])

    valid = (
        pat["FIRST_RECORD_DATE"].notna()
        & pat["t_platinum"].notna()
        & pat["t_death"].notna()
        & pat["t_either"].notna()
        & pat["t_nepc"].notna()
        & pat["t_first_treatment"].notna()
        & pat["t_platinum"].ge(0)
        & pat["t_death"].ge(0)
        & pat["t_either"].ge(0)
        & pat["t_nepc"].ge(0)
    )
    return pat.loc[valid].copy()


def _patient_lab_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return np.nan
    return float(np.std(values.to_numpy(dtype=float), ddof=0))


def _compute_patient_lab_slopes(pre_treatment: pd.DataFrame) -> pd.DataFrame:
    """Slope of LAB_VALUE vs t_lab (per day) per (DFCI_MRN, LAB_NAME).

    Returns NaN when fewer than 2 observations exist or t_lab has no variation.
    """
    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        x = group["t_lab"].to_numpy(dtype=float)
        y = group["LAB_VALUE"].to_numpy(dtype=float)
        if np.std(x) == 0:
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
            last="last",
        )
        .reset_index()
    )
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
    if "NEPC" in df.columns:
        candidates.append(("nepc", df["NEPC"].astype(int).to_numpy()))
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


def select_feature_columns(
    data: pd.DataFrame,
    raw_feature_cols: list[str],
    *,
    min_patient_coverage: float,
) -> tuple[list[str], pd.DataFrame]:
    """Select features on full dataset (used by both Arm 1 and Arm 2)."""
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


def summarize_endpoints(
    merged: pd.DataFrame,
    endpoints: list[str],
) -> pd.DataFrame:
    rows = []
    for split_name, split_df in [
        ("train_val", merged.loc[merged["split"].eq("train_val")]),
        ("test", merged.loc[merged["split"].eq("test")]),
    ]:
        for endpoint in endpoints:
            duration_col = ENDPOINTS[endpoint]["duration_col"]
            event_col = ENDPOINTS[endpoint]["event_col"]
            event_times = split_df.loc[split_df[event_col].eq(1), duration_col]
            rows.append(
                {
                    "split": split_name,
                    "endpoint": endpoint,
                    "description": ENDPOINTS[endpoint]["description"],
                    "n_patients": len(split_df),
                    "n_events": int(split_df[event_col].sum()),
                    "event_rate": float(split_df[event_col].mean()) if len(split_df) else np.nan,
                    "median_time_days_all": float(split_df[duration_col].median()) if len(split_df) else np.nan,
                    "median_time_days_events": float(event_times.median()) if not event_times.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


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
    covariate_cols = list(base_feature_cols)

    train_model = pd.DataFrame(index=train_df.index)
    eval_model = pd.DataFrame(index=eval_df.index)

    if base_feature_cols:
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        x_train = imputer.fit_transform(train_df[base_feature_cols].values)
        x_eval = imputer.transform(eval_df[base_feature_cols].values)
        x_train = scaler.fit_transform(x_train)
        x_eval = scaler.transform(x_eval)
        train_model = pd.DataFrame(x_train, columns=base_feature_cols, index=train_df.index)
        eval_model = pd.DataFrame(x_eval, columns=base_feature_cols, index=eval_df.index)

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


def select_auc_horizons(
    reference_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    percentile_range: tuple[float, float],
    n_points: int,
) -> list[float]:
    event_times = pd.to_numeric(
        reference_df.loc[reference_df[event_col].eq(1), duration_col],
        errors="coerce",
    ).dropna()
    event_times = event_times.loc[np.isfinite(event_times) & event_times.gt(0)]
    if event_times.empty:
        return []

    t_lo, t_hi = np.quantile(event_times.to_numpy(dtype=float), percentile_range)
    if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_hi <= t_lo:
        return []
    horizons = np.linspace(float(t_lo), float(t_hi), int(n_points))
    return [float(h) for h in horizons if np.isfinite(h) and h > 0]


def auc_at_horizon(
    *,
    duration: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    horizon: float,
) -> tuple[float, int, int, int]:
    positive = (event == 1) & (duration <= horizon)
    negative = duration > horizon
    usable = positive | negative

    n_usable = int(usable.sum())
    n_positive = int(positive[usable].sum())
    n_negative = int(negative[usable].sum())

    if n_usable == 0 or n_positive == 0 or n_negative == 0:
        return np.nan, n_usable, n_positive, n_negative

    auc = float(roc_auc_score(positive[usable].astype(int), risk_score[usable]))
    return auc, n_usable, n_positive, n_negative


def compute_integrated_auc_t(
    eval_df: pd.DataFrame,
    risk_score: np.ndarray,
    *,
    duration_col: str,
    event_col: str,
    reference_df: pd.DataFrame,
    percentile_range: tuple[float, float] = DEFAULT_AUC_PERCENTILE_RANGE,
    n_points: int = DEFAULT_AUC_N_POINTS,
) -> tuple[float, pd.DataFrame]:
    """Time-averaged AUC(t) integrated over the 5-95 percentile of reference event times."""
    horizons = select_auc_horizons(
        reference_df,
        duration_col=duration_col,
        event_col=event_col,
        percentile_range=percentile_range,
        n_points=n_points,
    )
    empty_cols = ["horizon_days", "auc_t", "n_usable", "n_positive", "n_negative"]
    if not horizons:
        return np.nan, pd.DataFrame(columns=empty_cols)

    duration = eval_df[duration_col].to_numpy(dtype=float)
    event = eval_df[event_col].to_numpy(dtype=int)
    risk_score = np.asarray(risk_score, dtype=float).reshape(-1)

    rows = []
    for horizon in horizons:
        auc_t, n_usable, n_positive, n_negative = auc_at_horizon(
            duration=duration,
            event=event,
            risk_score=risk_score,
            horizon=horizon,
        )
        rows.append(
            {
                "horizon_days": horizon,
                "auc_t": auc_t,
                "n_usable": n_usable,
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
        )

    auc_df = pd.DataFrame(rows)
    valid = auc_df["auc_t"].notna()
    if valid.sum() < 2:
        return np.nan, auc_df
    x = auc_df.loc[valid, "horizon_days"].to_numpy(dtype=float)
    y = auc_df.loc[valid, "auc_t"].to_numpy(dtype=float)
    span = x[-1] - x[0]
    if not np.isfinite(span) or span <= 0:
        return np.nan, auc_df
    trapezoid = getattr(np, "trapezoid", np.trapz)
    integrated = float(trapezoid(y, x) / span)
    return integrated, auc_df


def score_cox_model(
    model: CoxPHFitter,
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
) -> tuple[float, np.ndarray]:
    pred = np.asarray(model.predict_partial_hazard(model_df)).reshape(-1)
    c_index = float(concordance_index(model_df[duration_col], -pred, model_df[event_col]))
    return c_index, pred


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Arm 2: 5-fold CV over (alpha x l1_ratio) grid via sksurv CoxnetSurvivalAnalysis.

    For each fold and l1_ratio, fits the full alpha path in one call, then scores
    every alpha on the held-out fold. AGE is unpenalized via penalty_factor=0.
    """
    require_sksurv()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    splitter, strat_labels, cv_stratification = make_cv_splitter(train_val, n_folds=n_folds, seed=seed)
    fold_rows = []

    split_args = (np.arange(len(train_val)), strat_labels) if strat_labels is not None else (np.arange(len(train_val)),)

    alphas_desc = sorted({float(p) for p in penalizers if float(p) > 0}, reverse=True)
    if not alphas_desc:
        raise ValueError("cv_penalizers must contain at least one positive value.")

    for l1_ratio in l1_ratios:
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(*split_args), 1):
            fold_train = train_val.iloc[tr_idx]
            fold_val = train_val.iloc[val_idx]

            base_row = {
                "endpoint": endpoint,
                "fold": fold,
                "l1_ratio": float(l1_ratio),
                "n_train": len(fold_train),
                "n_val": len(fold_val),
                "n_events_train": int(fold_train[event_col].sum()),
                "n_events_val": int(fold_val[event_col].sum()),
                "cv_stratification": cv_stratification,
                "n_covariates": np.nan,
            }

            try:
                train_mdf, val_mdf, covariate_cols = build_model_matrices(
                    fold_train,
                    fold_val,
                    feature_cols=feature_cols,
                    duration_col=duration_col,
                    event_col=event_col,
                )
            except Exception as exc:  # pragma: no cover
                for alpha in alphas_desc:
                    fold_rows.append({
                        **base_row,
                        "penalizer": alpha,
                        "c_index_val": np.nan,
                        "integrated_auc_t_val": np.nan,
                        "n_valid_auc_horizons_val": 0,
                        "note": f"matrix_failed: {exc}",
                    })
                continue

            base_row["n_covariates"] = len(covariate_cols)
            X_tr = train_mdf[covariate_cols].to_numpy(dtype=float)
            X_val = val_mdf[covariate_cols].to_numpy(dtype=float)
            y_tr = _to_surv_y(
                train_mdf[duration_col].to_numpy(),
                train_mdf[event_col].to_numpy(),
            )
            y_val_event = val_mdf[event_col].to_numpy().astype(bool)
            y_val_time = val_mdf[duration_col].to_numpy().astype(float)
            pf = _penalty_factor(covariate_cols)

            fitted_alphas: set[float] = set()
            fit_note = ""
            cox: CoxnetSurvivalAnalysis | None = None
            try:
                cox = CoxnetSurvivalAnalysis(
                    l1_ratio=float(l1_ratio),
                    alphas=alphas_desc,
                    penalty_factor=pf,
                    fit_baseline_model=False,
                    max_iter=COXNET_MAX_ITER,
                )
                cox.fit(X_tr, y_tr)
                fitted_alphas = {float(a) for a in cox.alphas_}
                fit_note = "fit_ok"
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"Coxnet path fit failed for endpoint='{endpoint}' fold={fold} "
                    f"l1_ratio={l1_ratio}: {exc}"
                )
                fit_note = f"path_failed: {exc}"
                if cox is not None:
                    fitted_alphas = {float(a) for a in getattr(cox, "alphas_", [])}

            for alpha in alphas_desc:
                row = {
                    **base_row,
                    "penalizer": alpha,
                    "c_index_val": np.nan,
                    "integrated_auc_t_val": np.nan,
                    "n_valid_auc_horizons_val": 0,
                    "note": fit_note,
                }
                if cox is None or alpha not in fitted_alphas:
                    if cox is not None and alpha not in fitted_alphas:
                        row["note"] = "alpha_not_in_path"
                    fold_rows.append(row)
                    continue
                try:
                    risk = np.asarray(cox.predict(X_val, alpha=alpha), dtype=float).reshape(-1)
                    row["c_index_val"] = float(
                        concordance_index_censored(y_val_event, y_val_time, risk)[0]
                    )
                    integrated_auc_val, auc_df_val = compute_integrated_auc_t(
                        val_mdf,
                        risk,
                        duration_col=duration_col,
                        event_col=event_col,
                        reference_df=train_mdf,
                    )
                    row["integrated_auc_t_val"] = integrated_auc_val
                    row["n_valid_auc_horizons_val"] = (
                        int(auc_df_val["auc_t"].notna().sum()) if not auc_df_val.empty else 0
                    )
                except Exception as exc:  # pragma: no cover
                    row["note"] = f"score_failed: {exc}"
                fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    cv_df = (
        fold_df.groupby(["endpoint", "penalizer", "l1_ratio"], dropna=False)
        .agg(
            cv_mean=("c_index_val", "mean"),
            cv_std=("c_index_val", "std"),
            n_valid_folds=("c_index_val", lambda s: int(s.notna().sum())),
            integrated_auc_t_cv_mean=("integrated_auc_t_val", "mean"),
            integrated_auc_t_cv_std=("integrated_auc_t_val", "std"),
            n_valid_auc_t_folds=("integrated_auc_t_val", lambda s: int(s.notna().sum())),
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
    split_stratification: str,
    cv_stratification: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
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
    X_tr = train_mdf[covariate_cols].to_numpy(dtype=float)
    X_te = test_mdf[covariate_cols].to_numpy(dtype=float)
    y_tr = _to_surv_y(
        train_mdf[duration_col].to_numpy(),
        train_mdf[event_col].to_numpy(),
    )
    y_tr_event = train_mdf[event_col].to_numpy().astype(bool)
    y_tr_time = train_mdf[duration_col].to_numpy().astype(float)
    y_te_event = test_mdf[event_col].to_numpy().astype(bool)
    y_te_time = test_mdf[duration_col].to_numpy().astype(float)
    pf = _penalty_factor(covariate_cols)

    model = CoxnetSurvivalAnalysis(
        l1_ratio=float(l1_ratio),
        alphas=[float(penalizer)],
        penalty_factor=pf,
        fit_baseline_model=False,
        max_iter=COXNET_MAX_ITER,
    )
    try:
        model.fit(X_tr, y_tr)
        note = "fit_ok"
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            f"Coxnet final fit did not converge for endpoint='{endpoint}' "
            f"alpha={penalizer} l1_ratio={l1_ratio}: {exc}. "
            "Proceeding with the current (possibly partial) estimate."
        )
        note = f"fit_not_converged: {exc}"
    used_alpha = float(penalizer)

    if hasattr(model, "coef_"):
        train_pred = np.asarray(model.predict(X_tr), dtype=float).reshape(-1)
        test_pred = np.asarray(model.predict(X_te), dtype=float).reshape(-1)
        try:
            train_c = float(concordance_index_censored(y_tr_event, y_tr_time, train_pred)[0])
        except (ValueError, ZeroDivisionError):
            train_c = float("nan")
        try:
            test_c = float(concordance_index_censored(y_te_event, y_te_time, test_pred)[0])
        except (ValueError, ZeroDivisionError):
            test_c = float("nan")
    else:
        train_pred = np.zeros(len(X_tr), dtype=float)
        test_pred = np.zeros(len(X_te), dtype=float)
        train_c = float("nan")
        test_c = float("nan")

    train_iauc, train_auc_df = compute_integrated_auc_t(
        train_mdf,
        train_pred,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_mdf,
    )
    test_iauc, test_auc_df = compute_integrated_auc_t(
        test_mdf,
        test_pred,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_mdf,
    )

    metrics_row = {
        "endpoint": endpoint,
        "description": ENDPOINTS[endpoint]["description"],
        "n_train_val": len(train_val),
        "n_test": len(test),
        "n_events_train_val": int(train_val[event_col].sum()),
        "n_events_test": int(test[event_col].sum()),
        "selected_penalizer": used_alpha,
        "selected_l1_ratio": float(l1_ratio),
        "n_covariates": len(covariate_cols),
        "train_val_c_index": train_c,
        "test_c_index": test_c,
        "train_val_integrated_auc_t": train_iauc,
        "test_integrated_auc_t": test_iauc,
        "auc_percentile_range": f"{DEFAULT_AUC_PERCENTILE_RANGE[0]:g}-{DEFAULT_AUC_PERCENTILE_RANGE[1]:g}",
        "auc_n_points": DEFAULT_AUC_N_POINTS,
        "train_val_n_valid_auc_horizons": int(train_auc_df["auc_t"].notna().sum()) if not train_auc_df.empty else 0,
        "test_n_valid_auc_horizons": int(test_auc_df["auc_t"].notna().sum()) if not test_auc_df.empty else 0,
        "split_stratification": split_stratification,
        "cv_stratification": cv_stratification,
        "backend": "sksurv.CoxnetSurvivalAnalysis",
        "note": note,
    }

    if hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_, dtype=float).reshape(-1)
    else:
        coefs = np.zeros(len(covariate_cols), dtype=float)
    summary_rows = []
    for col, coef in zip(covariate_cols, coefs):
        lab_name, feature_stat = parse_feature_name(col)
        summary_rows.append(
            {
                "endpoint": endpoint,
                "feature": col,
                "lab_name": lab_name,
                "feature_stat": feature_stat,
                "is_age_covariate": col == "age",
                "coef": float(coef),
                "exp(coef)": float(np.exp(coef)),
                "selected_penalizer": used_alpha,
                "selected_l1_ratio": float(l1_ratio),
                "n_covariates": len(covariate_cols),
                "train_val_c_index": train_c,
                "test_c_index": test_c,
                "train_val_integrated_auc_t": train_iauc,
                "test_integrated_auc_t": test_iauc,
                "note": note,
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("coef", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    predictions = pd.DataFrame(
        {
            "DFCI_MRN": test.index,
            "endpoint": endpoint,
            "dataset": "test",
            "duration_days": test[duration_col].to_numpy(dtype=float),
            "event": test[event_col].to_numpy(dtype=int),
            "risk_score": test_pred,
        }
    )
    return metrics_row, summary, predictions


def build_univariate_overview(univariate_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    overview: pd.DataFrame | None = None
    key_cols = ["feature", "lab_name", "feature_stat"]

    for endpoint, endpoint_df in univariate_results.items():
        keep = endpoint_df[
            [
                *key_cols,
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
                "coef_age",
                "p_value_age",
                "note",
            ]
        ].copy()
        rename_map = {col: f"{endpoint}_{col}" for col in keep.columns if col not in key_cols}
        keep = keep.rename(columns=rename_map)

        if overview is None:
            overview = keep
        else:
            overview = overview.merge(keep, on=key_cols, how="outer")

    if overview is None:
        return pd.DataFrame(columns=key_cols)

    sort_cols = [
        col
        for col in ["platinum_p_value", "death_p_value"]
        if col in overview.columns
    ]
    if sort_cols:
        overview = overview.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return overview


def print_top_hits(df: pd.DataFrame, *, endpoint: str) -> None:
    hits = df.loc[
        df["p_value"].notna(),
        ["feature", "hazard_ratio_per_sd", "p_value", "q_value"],
    ].head(10)
    print(f"\nTop univariate associations for {endpoint}:")
    if hits.empty:
        print("  No estimable feature associations.")
        return
    print(hits.to_string(index=False))


def main(args: argparse.Namespace) -> None:
    endpoints = normalize_endpoints(args.endpoints)

    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)

    print("Building outcome table...")
    outcome_df = make_outcome_df(df)
    print(f"Outcome table: {len(outcome_df)} patients")

    print("Building raw aggregated pre-treatment lab summary feature matrix...")
    feature_df = build_feature_matrix(df)

    merged = feature_df.join(outcome_df, how="inner")
    merged = merged.loc[merged[AGE_COL].notna()].copy()
    if merged.empty:
        raise ValueError("No patients have both engineered features and valid outcomes.")

    raw_feature_cols = feature_df.columns.tolist()
    selected_feature_cols, feature_meta = select_feature_columns(
        merged,
        raw_feature_cols,
        min_patient_coverage=args.min_patient_coverage,
    )

    merged = merged[
        selected_feature_cols + [col for col in merged.columns if col in OUTCOME_COLUMNS]
    ].copy()

    train_val, test, split_assignments, split_stratification = split_train_test(
        merged,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    merged["split"] = split_assignments

    print(f"Full cohort: {len(merged)} patients")
    print(f"Train/val (Arm 2): {len(train_val)} patients")
    print(f"Test (Arm 2):      {len(test)} patients")
    print(f"Selected summary-lab features: {len(selected_feature_cols)}")
    print(f"Split stratification: {split_stratification}")

    feature_df.to_csv(RESULTS / "cox_agg_feature_matrix_raw.csv")
    outcome_df.to_csv(RESULTS / "cox_agg_outcome_table.csv")
    split_assignments.to_frame().to_csv(RESULTS / "cox_agg_split_assignments.csv")
    feature_meta.to_csv(RESULTS / "cox_agg_feature_selection.csv", index=False)

    endpoint_summary = summarize_endpoints(merged, endpoints)
    endpoint_summary.to_csv(RESULTS / "cox_agg_endpoint_summary.csv", index=False)

    univariate_results: dict[str, pd.DataFrame] = {}
    cv_metric_rows = []
    cv_fold_rows = []
    selected_model_rows = []
    test_metric_rows = []

    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} ===")
        print(ENDPOINTS[endpoint]["description"])

        if args.analysis in {"univariate", "both"}:
            univariate_df = run_univariate_associations(
                merged,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                min_events_per_feature=args.min_events_per_feature,
                fallback_penalizer=args.univariate_penalizer,
            )
            univariate_results[endpoint] = univariate_df
            univariate_df.to_csv(RESULTS / f"cox_agg_univariate_{endpoint}.csv", index=False)
            print_top_hits(univariate_df, endpoint=endpoint)

        if args.analysis in {"multivariable", "both"}:
            fold_df, cv_df, best_row = tune_multivariable_model(
                train_val,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                penalizers=args.cv_penalizers,
                l1_ratios=args.cv_l1_ratios,
                n_folds=args.n_folds,
                seed=args.seed,
            )
            cv_fold_rows.append(fold_df)
            cv_metric_rows.append(cv_df)
            selected_model_rows.append(
                {
                    "endpoint": endpoint,
                    "description": ENDPOINTS[endpoint]["description"],
                    "selected_penalizer": best_row["penalizer"],
                    "selected_l1_ratio": best_row["l1_ratio"],
                    "cv_mean_c_index": best_row["cv_mean"],
                    "cv_std_c_index": best_row["cv_std"],
                    "cv_mean_integrated_auc_t": best_row["integrated_auc_t_cv_mean"],
                    "cv_std_integrated_auc_t": best_row["integrated_auc_t_cv_std"],
                    "n_valid_folds": best_row["n_valid_folds"],
                    "n_valid_auc_t_folds": best_row["n_valid_auc_t_folds"],
                    "cv_stratification": best_row["cv_stratification"],
                }
            )

            metrics_row, summary_df, pred_df = fit_final_multivariable_model(
                train_val,
                test,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                penalizer=float(best_row["penalizer"]),
                l1_ratio=float(best_row["l1_ratio"]),
                split_stratification=split_stratification,
                cv_stratification=str(best_row["cv_stratification"]),
            )
            test_metric_rows.append(metrics_row)
            summary_df.to_csv(RESULTS / f"cox_agg_multivariable_{endpoint}.csv", index=False)
            pred_df.to_csv(RESULTS / f"cox_agg_test_predictions_{endpoint}.csv", index=False)

            top_cols = [c for c in ["feature", "coef", "exp(coef)"] if c in summary_df.columns]
            top = summary_df.loc[~summary_df["is_age_covariate"], top_cols].head(10)
            print("\nChosen hyperparameters (elastic-net, age unpenalized):")
            print(
                f"  penalizer={best_row['penalizer']}  l1_ratio={best_row['l1_ratio']}  "
                f"cv_mean C-index={best_row['cv_mean']:.4f}"
            )
            print(f"  CV integrated AUC(t)={best_row['integrated_auc_t_cv_mean']:.4f}")
            print(
                f"  train/val C-index={metrics_row['train_val_c_index']:.4f}  "
                f"integrated AUC(t)={metrics_row['train_val_integrated_auc_t']:.4f}"
            )
            print(f"  held-out test C-index={metrics_row['test_c_index']:.4f}")
            print(f"  held-out test integrated AUC(t)={metrics_row['test_integrated_auc_t']:.4f}")
            print("Top multivariable coefficients:")
            print(top.to_string(index=False))

    if univariate_results:
        overview = build_univariate_overview(univariate_results)
        overview.to_csv(RESULTS / "cox_agg_univariate_overview.csv", index=False)
        print("\nSaved: results/cox_agg_univariate_overview.csv")

    if cv_metric_rows:
        pd.concat(cv_metric_rows, ignore_index=True).to_csv(RESULTS / "cox_agg_cv_metrics.csv", index=False)
        pd.concat(cv_fold_rows, ignore_index=True).to_csv(RESULTS / "cox_agg_cv_fold_metrics.csv", index=False)
        pd.DataFrame(selected_model_rows).to_csv(RESULTS / "cox_agg_selected_models.csv", index=False)
        pd.DataFrame(test_metric_rows).to_csv(RESULTS / "cox_agg_test_metrics.csv", index=False)
        print("Saved:")
        print("  results/cox_agg_cv_metrics.csv")
        print("  results/cox_agg_cv_fold_metrics.csv")
        print("  results/cox_agg_selected_models.csv")
        print("  results/cox_agg_test_metrics.csv")

    print("Saved:")
    print("  results/cox_agg_feature_matrix_raw.csv")
    print("  results/cox_agg_outcome_table.csv")
    print("  results/cox_agg_split_assignments.csv")
    print("  results/cox_agg_feature_selection.csv")
    print("  results/cox_agg_endpoint_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["platinum", "death", "nepc"],
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
    main(parser.parse_args())
