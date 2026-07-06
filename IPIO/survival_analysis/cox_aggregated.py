"""
Shared library for the landmarked Cox survival analyses.

This module holds the model-fitting primitives (univariate n_obs-adjusted Cox,
elastic-net Coxnet with CV, IPCW AUC(t)/Brier evaluation), the prebuilt-input
loaders, and the shared per-landmark setup (prepare_landmark_context /
build_endpoint_horizon_grids). The runnable analyses live in two thin scripts
that import from here:
  * cox_univariate.py    — Arm 1 univariate (n_obs-adjusted) associations.
  * cox_multivariable.py — Arm 2 multivariable elastic-net Cox + age(+baseline
    covariates) model.

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
  - 5-fold CV over (penalizer x l1_ratio) grid on the 80%; AGE (and the
    baseline covariates below) are unpenalized.
  - Refit on full 80% with chosen (penalizer, l1_ratio) and evaluate on 20% test:
    C-index and IPCW cumulative/dynamic AUC(t).
  - AUC(t) horizons are read from build_manifest.json so Cox, XGBoost, and
    Dynamic DeepHit use the same per-landmark quantile grid.

Endpoint: irae (time to first immune-related adverse event).

Supports landmark offsets relative to first treatment via --landmark-days. When
multiple landmark offsets are requested, analyses are restricted to the
intersection MRN set eligible at every requested landmark so comparisons are
made on a fixed cohort.

Expected input:
  Row-level longitudinal data with at least
    DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab, t_first_treatment, t_irae,
    t_last_contact, IRAE, AGE_AT_TREATMENTSTART, plus the baseline covariates
    GENDER_MALE, pd1pdl1, ctla4, and one-hot CANCER_TYPE_<type> dummies.

Outputs:
  results/cox_agg_landmark_mrn_availability.csv
  results/cox_agg_feature_selection.csv   (selected lab features + coverage)
  results/cox_agg_univariate_nobs_adjusted.csv  (n_obs-adjusted log HRs, p, q)
  results/cox_agg_multivariable.csv       (coefs + C-indices + AUC(t))
  results/cox_agg_multivariable_metrics.csv
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# IPIO does not duplicate COMPASS's helpers/ package; resolve `helpers.*`
# imports against the COMPASS survival_analysis directory where it actually
# lives. This must run before any `from helpers...` import below.
sys.path.insert(
    0,
    "/data/gusev/USERS/jpconnor/code/CAIA/COMPASS/survival_analysis",
)

from helpers.helper import (  # noqa: E402
    _make_survival_array,
    assert_disjoint_folds,
    assert_no_test_leakage,
    breslow_survival_at_horizons,
    choose_stratification_labels,
    compute_brier,
    compute_horizon_grid,
    horizon_grid_frame,
    select_canonical_labs,
)
# Cohort construction + pre-landmark feature engineering live in helpers.cohort
# (the shared input-building layer used by build_prediction_inputs /
# build_genomic_inputs). Imported here so the patient-id/age config and the
# builders stay importable as `cox_aggregated.<name>` for the analysis scripts.
from helpers.cohort import (  # noqa: E402,F401
    AGE_COL,
    ID_COL,
    build_feature_matrix,
    build_landmark_availability_table,
    build_landmark_merged,
    build_pre_treatment_lab_long,
    make_outcome_df,
    normalize_landmark_days,
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
# Effectively-unpenalized alpha used when every covariate is unpenalized (the
# age-only baseline). sksurv's Coxnet rejects alpha=0 outright, but a negligible
# alpha with a uniform penalty_factor recovers the unpenalized Cox MLE.
DEFAULT_UNPENALIZED_ALPHA = 1e-6

BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/IPIO/")
RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/IPIO/survival_analysis")

DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_N_FOLDS = 5
DEFAULT_LANDMARK_DAYS = [0, 90]  # treatment start and 90 days post first treatment
DEFAULT_MIN_PATIENT_COVERAGE = 0.20
DEFAULT_MIN_EVENTS_PER_FEATURE = 10
# Upper bound (in time-units) for the IPCW AUC(t)/Brier evaluation horizons. This
# administratively censors only the *evaluation* grid (not model fitting), so AUC/Brier
# stay comparable across Cox / XGBoost / DeepHit. Overridable via --auc-max-time-units.
DEFAULT_AUC_MAX_TIME_UNITS = 260
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
HORIZON_GRID_FILENAME = "cox_agg_horizon_grid.csv"
CANONICAL_LABS_FOLDS_FILENAME = "cox_agg_canonical_labs_folds.csv"

ENDPOINTS = {
    "irae": {
        "duration_col": "t_irae",
        "event_col": "IRAE",
        "description": "Time from IO start to first immune-related adverse event",
    },
}

# Administrative censoring of *model fitting* was removed: Cox + XGBoost now train on
# full follow-up (previously capped at 1820 days = DeepHit's DEFAULT_MAX_PRED_WINDOW * 7).
# NOTE: the IPCW AUC(t)/Brier *evaluation* grid is still capped at DEFAULT_AUC_MAX_TIME_UNITS
# (see below) so the reported discrimination/calibration metrics stay comparable across models.
OUTCOME_COLUMNS = {
    AGE_COL,
    "FIRST_RECORD_DATE",
    "FIRST_TREATMENT_DATE",
    "FIRST_TREATMENT",
    "LAST_CONTACT_DATE",
    "IO_START",
    "LAST_DATE",
    "IRAE",
    "t_irae",
    "t_irae_from_first_record",
    "split",
}

# Fixed baseline covariates that are always included (unpenalized) in every Cox
# model, plus the dynamic set of one-hot CANCER_TYPE_<type> dummy columns. This
# is the IPIO replacement for COMPASS's Gleason/cancer-stage baseline covariate
# mechanism (which doesn't apply here).
BASELINE_STATIC_FIXED = ("GENDER_MALE", "pd1pdl1", "ctla4")
BASELINE_STATIC_PREFIX = "CANCER_TYPE_"


def baseline_covariate_columns(df: pd.DataFrame) -> list[str]:
    """Fixed baseline covariates always included (unpenalized) in every Cox model:
    GENDER_MALE, pd1pdl1, ctla4, plus every one-hot CANCER_TYPE_<type> dummy column
    present in df (the set of dummy columns depends on which cancer types are in the
    data, so it's discovered dynamically rather than hardcoded).
    """
    cols = [c for c in BASELINE_STATIC_FIXED if c in df.columns]
    cols += sorted(c for c in df.columns if c.startswith(BASELINE_STATIC_PREFIX))
    return cols


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
    static_covariate_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Always includes age as a covariate; age is unpenalized.

    ``static_covariate_cols`` (e.g. the baseline covariates returned by
    ``baseline_covariate_columns`` — GENDER_MALE, pd1pdl1, ctla4, CANCER_TYPE_*)
    are scaled with their own scaler and appended like age, so callers can mark
    them unpenalized too. Mean-impute + StandardScaler on a 0/1 indicator is
    harmless for CoxnetSurvivalAnalysis, so binary baseline covariates need no
    special-casing here.
    """
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
                col[: -len("__missing")] if col.endswith("__missing") else col
                for col in missing_indicator_cols
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

    # Always-included static covariates (e.g. GENDER_MALE, pd1pdl1, ctla4,
    # CANCER_TYPE_*): mean-imputed + scaled with their own scaler, appended like
    # age. Callers add them to unpenalized_cols.
    for cov in static_covariate_cols:
        if cov not in train_df.columns:
            raise ValueError(f"static covariate {cov!r} not present in the model frame.")
        cov_imputer = SimpleImputer(strategy="mean")
        cov_scaler = StandardScaler()
        train_cov = cov_scaler.fit_transform(
            cov_imputer.fit_transform(train_df[[cov]].to_numpy(dtype=float))
        ).reshape(-1)
        eval_cov = cov_scaler.transform(
            cov_imputer.transform(eval_df[[cov]].to_numpy(dtype=float))
        ).reshape(-1)
        train_model[cov] = train_cov
        eval_model[cov] = eval_cov
        covariate_cols.append(cov)

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

    within_horizon = event.astype(bool) & (duration > 0) & (duration <= float(max_time_unit))
    duration = np.where(duration <= float(max_time_unit), duration, float(max_time_unit))
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

    Warnings are allowed: convergence warnings, overflow, and any other
    non-error warning do NOT invalidate a fit (they are suppressed during fit).
    A penalizer is only rejected when ``model.fit`` raises a genuine exception
    (or, defensively, when the model returns the wrong number of coefficients).
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

    # When every covariate is unpenalized (the age-only baseline), sksurv
    # rescales penalty_factor to sum to n_features via pf * n_features / pf.sum(),
    # so an all-zero vector divides by zero and silently zeroes every coefficient
    # -> constant risk score, C-index collapses to exactly 0.5. Fall back to the
    # unpenalized MLE: a uniform penalty_factor with a negligible alpha.
    if float(penalty_factor.sum()) == 0.0 and len(covariate_cols) > 0:
        penalty_factor = np.ones(len(covariate_cols), dtype=float)
        penalizers = [DEFAULT_UNPENALIZED_ALPHA]

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
            # Suppress (allow) all warnings during fit — convergence warnings,
            # overflow, etc. do not invalidate the fit. Only a raised exception does.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)
            continue

        coefs = np.asarray(model.coef_, dtype=float)
        if coefs.ndim == 2:
            coefs = coefs[:, -1]
        coefs = coefs.reshape(-1)
        # Defensive structural guard only (not a warning/overflow rejection): the
        # model must return one coefficient per covariate so downstream extraction
        # aligns. Non-finite / large coefficients are NOT rejected here.
        if coefs.size != len(covariate_cols):
            last_error = (
                f"coef_size_mismatch: expected {len(covariate_cols)} got {coefs.size}"
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


def _static_covariate_association_row(
    data: pd.DataFrame,
    *,
    covariate_col: str,
    endpoint: str,
    duration_col: str,
    event_col: str,
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> dict:
    """Univariate Cox [AGE + covariate] for one static baseline covariate (e.g.
    GENDER_MALE, pd1pdl1, ctla4, or a CANCER_TYPE_<type> dummy).

    Returns a row in the same schema as the lab-feature associations (n_obs / missing
    columns left NaN) so it slots straight into the univariate table and the shared
    BH q-value pool.
    """
    total_patients = len(data)
    result = {
        "endpoint": endpoint,
        "feature": covariate_col,
        "lab_name": covariate_col,
        "feature_stat": "",
        "n_obs_feature": "",
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
        "note": "static_covariate",
    }
    if covariate_col not in data.columns:
        result["note"] = "static_covariate_missing_column"
        return result

    cov_df = data[[covariate_col, duration_col, event_col, AGE_COL]].copy()
    result["coverage"] = float(cov_df[covariate_col].notna().mean())
    cov_df = cov_df.dropna(subset=[covariate_col, duration_col, event_col, AGE_COL])
    result["n_patients_used"] = len(cov_df)
    result["n_patients_observed"] = len(cov_df)
    result["n_events_used"] = int(cov_df[event_col].sum()) if len(cov_df) else 0
    if len(cov_df) == 0:
        result["note"] = "no_rows_with_outcomes"
        return result
    if result["n_events_used"] < min_events_per_feature:
        result["note"] = f"too_few_events_lt_{min_events_per_feature}"
        return result

    cov_values = cov_df[covariate_col].to_numpy(dtype=float)
    cov_sd = float(np.std(cov_values, ddof=0))
    if not np.isfinite(cov_sd) or cov_sd <= 0:
        result["note"] = "static_covariate_no_variation"
        return result
    cov_df["feature_z"] = (cov_values - float(np.mean(cov_values))) / cov_sd

    age_values = cov_df[AGE_COL].to_numpy(dtype=float)
    age_sd = float(np.std(age_values, ddof=0))
    if np.isfinite(age_sd) and age_sd > 0:
        cov_df["age"] = (age_values - float(np.mean(age_values))) / age_sd
    else:
        cov_df["age"] = age_values - float(np.mean(age_values))

    model, used_penalizer, note = fit_cox_with_fallback(
        cov_df[["feature_z", "age", duration_col, event_col]],
        duration_col=duration_col,
        event_col=event_col,
        penalizers=[0.0, fallback_penalizer],
        l1_ratio=0.0,
    )
    result["fit_penalizer"] = used_penalizer
    result["note"] = f"static_covariate;{note}" if note else "static_covariate"
    if model is None:
        return result

    summary_row = model.summary.loc["feature_z"]
    result["coef_feature"] = float(summary_row["coef"])
    result["hazard_ratio_per_sd"] = float(summary_row["exp(coef)"])
    result["ci_lower"] = float(summary_row["exp(coef) lower 95%"])
    result["ci_upper"] = float(summary_row["exp(coef) upper 95%"])
    result["p_value"] = float(summary_row["p"])
    age_row = model.summary.loc["age"]
    result["coef_age"] = float(age_row["coef"])
    result["p_value_age"] = float(age_row["p"])
    return result


def run_univariate_nobs_adjusted_associations(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    min_events_per_feature: int,
    fallback_penalizer: float,
    static_covariate_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Arm 1b: fit Cox on [AGE + matching LAB__n_observations + feature].

    ``static_covariate_cols`` (the baseline covariates from
    ``baseline_covariate_columns`` — e.g. ``GENDER_MALE``) are additionally fit as
    [AGE + covariate] and appended as rows in the same schema, so they share the
    BH q-value pool with the lab associations.
    """
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

    for covariate_col in static_covariate_cols:
        rows.append(
            _static_covariate_association_row(
                data,
                covariate_col=covariate_col,
                endpoint=endpoint,
                duration_col=duration_col,
                event_col=event_col,
                min_events_per_feature=min_events_per_feature,
                fallback_penalizer=fallback_penalizer,
            )
        )

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


def _summarize_fold_failures(fold_df: pd.DataFrame) -> str:
    """Render a per-(penalizer, l1_ratio, fold) note table for CV diagnostics."""
    if fold_df is None or fold_df.empty:
        return "  (no fold rows recorded)"
    cols = [
        "fold",
        "penalizer",
        "l1_ratio",
        "n_events_train",
        "n_events_val",
        "n_canonical_labs",
        "n_selected_features",
        "c_index_val",
        "note",
    ]
    available = [c for c in cols if c in fold_df.columns]
    notes = (
        fold_df["note"].fillna("").astype(str).value_counts().head(5).to_dict()
        if "note" in fold_df.columns
        else {}
    )
    note_summary = "\n".join(f"    {n:>4}x  {note!r}" for note, n in notes.items())
    table = fold_df[available].head(20).to_string(index=False)
    return (
        "  Most common per-fold notes:\n"
        f"{note_summary}\n"
        "  First 20 fold rows:\n"
        f"{table}"
    )


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
    static_covariate_cols: tuple[str, ...] = (),
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
            id_col=ID_COL,
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
                    static_covariate_cols=static_covariate_cols,
                )
                model, _, note = fit_coxnet_with_fallback(
                    train_mdf,
                    duration_col=duration_col,
                    event_col=event_col,
                    penalizers=[float(penalizer)],
                    l1_ratio=float(l1_ratio),
                    covariate_cols=covariate_cols,
                    unpenalized_cols=["age", *static_covariate_cols],
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
        diagnostic = _summarize_fold_failures(fold_df)
        raise RuntimeError(
            f"All CV fits failed for endpoint '{endpoint}'.\n{diagnostic}"
        )
    if not cv_df["all_folds_valid"].any():
        best_valid = int(cv_df["n_valid_folds"].max())
        diagnostic = _summarize_fold_failures(fold_df)
        raise RuntimeError(
            f"No hyperparameter setting produced valid fits in all {n_folds} folds for endpoint "
            f"'{endpoint}'. Best observed validity was {best_valid}/{n_folds} folds.\n{diagnostic}"
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
    split_stratification: str,
    cv_stratification: str,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    horizon_grid: np.ndarray,
    canonical_labs: list[str],
    static_covariate_cols: tuple[str, ...] = (),
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
        static_covariate_cols=static_covariate_cols,
    )
    # Fit with the CV-selected penalizer only — no fallback escalation.
    # tune_multivariable_model already restricts selection to (penalizer, l1_ratio)
    # combinations that produce a valid fit in every CV fold, so the chosen value is
    # the automatically selected best. If it cannot produce a usable fit on the full
    # 80% train_val, fail loudly rather than silently swapping in a different penalizer.
    model, used_penalizer, note = fit_coxnet_with_fallback(
        train_mdf,
        duration_col=duration_col,
        event_col=event_col,
        penalizers=[float(penalizer)],
        l1_ratio=float(l1_ratio),
        covariate_cols=covariate_cols,
        unpenalized_cols=["age", *static_covariate_cols],
    )
    if model is None:
        raise RuntimeError(
            f"Final multivariable model failed for endpoint '{endpoint}' with the "
            f"CV-selected penalizer={penalizer:g}, l1_ratio={l1_ratio:g}: {note}"
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
            ID_COL: test.index,
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
    aggregated = pd.read_csv(agg_path).set_index(ID_COL)
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


@dataclass
class LandmarkContext:
    """Shared per-landmark setup consumed by the univariate / multivariable
    arms (cox_univariate.py and cox_multivariable.py).

    Produced once per landmark by :func:`prepare_landmark_context` so both arms
    operate on an identical cohort, canonical lab set, and selected-feature set.
    """

    landmark_day: int
    merged: pd.DataFrame
    train_val: pd.DataFrame
    test: pd.DataFrame
    pre_treatment_lab_df: pd.DataFrame
    raw_feature_cols: list[str]
    univariate_data: pd.DataFrame
    split_stratification: str
    canonical_labs: list[str]
    selected_feature_cols: list[str]
    feature_meta_selected: pd.DataFrame


def prepare_landmark_context(
    inputs_dir: Path,
    landmark_day: int,
    *,
    min_patient_coverage: float,
) -> LandmarkContext:
    """Load prebuilt inputs and run the shared cohort + feature setup for a landmark.

    Loads the aggregated table written by build_prediction_inputs.py, derives
    the canonical lab set from train_val pre-treatment labs, and runs the
    train_val feature selection. The returned context is everything the
    univariate and multivariable arms consume, so the two analysis scripts
    share an identical cohort, canonical lab set, and selected-feature set.

    Baseline covariates (GENDER_MALE, pd1pdl1, ctla4, CANCER_TYPE_*) are always
    included (see baseline_covariate_columns) and are excluded here from
    raw_feature_cols so they are never swept as candidate lab features.
    """
    print(f"\n##### LANDMARK ANALYSES: +{landmark_day} DAYS #####")
    merged, train_val, test, pre_treatment_lab_df = _load_prebuilt_landmark(
        inputs_dir, landmark_day
    )
    excluded = OUTCOME_COLUMNS | set(baseline_covariate_columns(merged))
    raw_feature_cols = [c for c in merged.columns if c not in excluded]
    univariate_data = merged.copy()

    split_stratification = "prebuilt"

    assert_no_test_leakage(
        test_mrns=test.index,
        train_mrns=train_val.index,
        context=f"prepare_landmark_context[landmark+{landmark_day}d]",
    )

    # Canonical lab list: derived from pre-treatment lab observations of
    # train_val MRNs only, mirroring build_prediction_inputs.py so the
    # per-stat feature filter agrees with the upstream canonical set.
    canonical_labs = select_canonical_labs(
        pre_treatment_lab_df,
        mrns=train_val.index,
        min_coverage=min_patient_coverage,
        id_col=ID_COL,
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

    return LandmarkContext(
        landmark_day=landmark_day,
        merged=merged,
        train_val=train_val,
        test=test,
        pre_treatment_lab_df=pre_treatment_lab_df,
        raw_feature_cols=raw_feature_cols,
        univariate_data=univariate_data,
        split_stratification=split_stratification,
        canonical_labs=canonical_labs,
        selected_feature_cols=selected_feature_cols,
        feature_meta_selected=feature_meta_selected,
    )


def build_endpoint_horizon_grids(
    landmark_day: int,
    *,
    endpoints: list[str],
    auc_horizons_by_landmark: dict,
    auc_quantiles: tuple[float, ...],
    auc_time_unit_days: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load the per-endpoint AUC(t) horizon grid for a landmark from build_manifest.

    Returns the endpoint -> grid mapping plus a tidy frame (one row per horizon)
    for cox_agg_horizon_grid.csv. Grids come from build_manifest.json so Cox /
    XGBoost / DeepHit evaluate on the identical horizon set per (landmark, endpoint).
    """
    landmark_horizons = auc_horizons_by_landmark.get(str(int(landmark_day)))
    if landmark_horizons is None:
        raise KeyError(
            f"build_manifest.json has no auc_horizons_by_landmark entry for landmark +{landmark_day}d. "
            "Re-run build_prediction_inputs.py for this landmark."
        )
    endpoint_horizon_grids: dict[str, np.ndarray] = {}
    grid_frames: list[pd.DataFrame] = []
    for endpoint in endpoints:
        if endpoint not in landmark_horizons:
            raise KeyError(
                f"build_manifest.json missing horizons for endpoint {endpoint!r} at landmark +{landmark_day}d."
            )
        grid = np.asarray(landmark_horizons[endpoint], dtype=float)
        endpoint_horizon_grids[endpoint] = grid
        grid_df = horizon_grid_frame(
            grid,
            quantiles=auc_quantiles,
            time_unit_days=auc_time_unit_days,
            endpoint=endpoint,
        )
        grid_df.insert(0, "landmark_days", landmark_day)
        grid_frames.append(grid_df)
        print(
            f"Horizon grid ({endpoint}, from manifest): "
            + ", ".join(f"{int(h)}" for h in grid)
            + f" {auc_time_unit_days}-day units"
        )
    horizon_grid_df = (
        pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()
    )
    return endpoint_horizon_grids, horizon_grid_df
