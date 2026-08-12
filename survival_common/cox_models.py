"""Shared Cox model orchestration for landmarked survival analyses.

Project-specific modules supply endpoint definitions, cohort restrictions, and
baseline-covariate discovery. This module owns the duplicated model mechanics:
feature selection, model matrices, univariate Cox associations, Coxnet CV/final
fits, manifest loading, and endpoint horizon-grid handling.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from survival_common.cohort import AGE_COL as DEFAULT_AGE_COL
from survival_common.cohort import ID_COL as DEFAULT_ID_COL
from survival_common.cox_engine import (
    benjamini_hochberg,
    compute_ipcw_auc_t,
    coxnet_coefficients,
    coxnet_survival_at_horizons,
    fit_cox_with_fallback,
    fit_coxnet_with_fallback,
    make_cv_splitter,
    matching_n_obs_feature,
    parse_feature_name,
    require_sksurv,
    score_coxnet_model,
    summarize_auc_timeline,
    summarize_fold_failures as _summarize_fold_failures,
)
from survival_common.helper import (
    AUC_TIMELINE_SCHEMA_VERSION,
    assert_disjoint_folds,
    assert_no_test_leakage,
    compute_brier,
    horizon_grid_frame,
    select_canonical_labs,
    summarize_brier_timeline,
)

try:  # pragma: no cover - depends on local environment
    from lifelines.exceptions import ConvergenceError
except ModuleNotFoundError:  # pragma: no cover - mirror survival_common.cox_engine fallback
    ConvergenceError = RuntimeError

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - tqdm is optional
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        if iterable is None:
            class _Null:
                def update(self, *_a, **_kw): pass
                def set_postfix(self, *_a, **_kw): pass
                def set_description(self, *_a, **_kw): pass
                def close(self): pass
                def __enter__(self): return self
                def __exit__(self, *_): return False
            return _Null()
        return iterable

# Expected numerical / convergence failures for a single CV fold. Matches the
# exception sets caught by fit_cox_with_fallback (ConvergenceError, ValueError,
# LinAlgError) and fit_coxnet_with_fallback (ArithmeticError, ValueError,
# LinAlgError) in survival_common.cox_engine, so genuine bugs propagate instead
# of being silently converted into "fold_failed" notes. FloatingPointError is a
# subclass of ArithmeticError and is therefore already covered.
_FOLD_FIT_ERRORS = (
    ConvergenceError,
    ArithmeticError,
    ValueError,
    np.linalg.LinAlgError,
)

EndpointMap = Mapping[str, Mapping[str, str]]


def normalize_endpoints(raw_endpoints: list[str], endpoint_map: EndpointMap) -> list[str]:
    endpoints: list[str] = []
    for endpoint in raw_endpoints:
        normalized = endpoint.lower()
        if normalized not in endpoint_map:
            valid = ", ".join(sorted(endpoint_map))
            raise ValueError(f"Unsupported endpoint '{endpoint}'. Choose from: {valid}")
        if normalized not in endpoints:
            endpoints.append(normalized)
    return endpoints


def _endpoint_columns(endpoint_map: EndpointMap, endpoint: str) -> tuple[str, str]:
    spec = endpoint_map[endpoint]
    return spec["duration_col"], spec["event_col"]


def select_feature_columns(
    data: pd.DataFrame,
    raw_feature_cols: list[str],
    *,
    min_patient_coverage: float,
    restrict_to_labs: list[str] | None = None,
    always_include: list[str] | None = None,
    genomic_feature_cols: list[str] | None = None,
    min_genomic_prevalence: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select candidate features on a train/validation block to avoid leakage.

    `restrict_to_labs`, when provided, gates lab-summary features to the
    canonical lab set. `always_include` exempts explicit non-lab features, such
    as genomic indicators, from that lab gate while still applying coverage and
    variability filters. `genomic_feature_cols`, when provided together with
    `min_genomic_prevalence`, additionally requires those binary indicators to
    have value 1 (mutated) in at least that fraction of patients.
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
        if always_include:
            feature_meta["in_canonical_labs"] = (
                feature_meta["in_canonical_labs"]
                | feature_meta["feature"].isin(set(always_include))
            )
        feature_meta["selected"] = feature_meta["selected"] & feature_meta["in_canonical_labs"]
    if genomic_feature_cols and min_genomic_prevalence is not None:
        genomic_set = set(genomic_feature_cols)
        is_genomic = feature_meta["feature"].isin(genomic_set)
        mutation_prevalence = data[raw_feature_cols].eq(1).mean()
        feature_meta["mutation_prevalence"] = np.where(
            is_genomic,
            mutation_prevalence.reindex(raw_feature_cols).values,
            np.nan,
        )
        meets_prevalence = feature_meta["mutation_prevalence"].ge(min_genomic_prevalence)
        feature_meta["selected"] = feature_meta["selected"] & (~is_genomic | meets_prevalence)
    feature_meta = feature_meta.sort_values(
        ["selected", "coverage", "feature"],
        ascending=[False, False, True],
    )

    selected = feature_meta.loc[feature_meta["selected"], "feature"].tolist()
    if not selected:
        raise ValueError("No features passed coverage and variability filters.")
    return selected, feature_meta.reset_index(drop=True)


def build_model_matrices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    duration_col: str,
    event_col: str,
    static_covariate_cols: tuple[str, ...] = (),
    age_col: str = DEFAULT_AGE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build scaled Coxnet matrices with age and static covariates unpenalized."""
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
        train_df[[age_col]].to_numpy(dtype=float)
    ).reshape(-1)
    eval_age = age_scaler.transform(
        eval_df[[age_col]].to_numpy(dtype=float)
    ).reshape(-1)
    train_model["age"] = train_age
    eval_model["age"] = eval_age
    covariate_cols.append("age")

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


def _static_feature_association_row(
    data: pd.DataFrame,
    *,
    feature: str,
    endpoint: str,
    duration_col: str,
    event_col: str,
    min_events_per_feature: int,
    fallback_penalizer: float,
    age_col: str,
    baseline_covariate_cols: tuple[str, ...] = (),
    model_type: str = "cox",
) -> dict:
    """Test a static feature (feature_z + age + baseline covariates, no n_obs)."""
    total_patients = len(data)
    result = {
        "endpoint": endpoint,
        "feature": feature,
        "lab_name": feature,
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
        "note": "static_feature",
        "model_type": model_type,
    }
    if feature not in data.columns:
        result["note"] = "static_missing_column"
        return result

    baseline_cols = [c for c in baseline_covariate_cols if c in data.columns]

    feature_df = data[[feature, duration_col, event_col, age_col, *baseline_cols]].copy()
    result["coverage"] = float(feature_df[feature].notna().mean())
    feature_df = feature_df.dropna(
        subset=[feature, duration_col, event_col, age_col, *baseline_cols]
    )
    result["n_patients_used"] = len(feature_df)
    result["n_patients_observed"] = len(feature_df)
    result["n_events_used"] = int(feature_df[event_col].sum()) if len(feature_df) else 0
    if len(feature_df) == 0:
        result["note"] = "no_rows_with_outcomes"
        return result
    if result["n_events_used"] < min_events_per_feature:
        result["note"] = f"too_few_events_lt_{min_events_per_feature}"
        return result

    feature_values = feature_df[feature].to_numpy(dtype=float)
    finite_unique = set(np.unique(feature_values[np.isfinite(feature_values)]).tolist())
    feature_kind = (
        "genomic_binary"
        if finite_unique.issubset({0.0, 1.0})
        else "static_continuous"
    )
    feature_sd = float(np.std(feature_values, ddof=0))
    if not np.isfinite(feature_sd) or feature_sd <= 0:
        result["note"] = "feature_has_no_variation"
        return result
    feature_df["feature_z"] = (feature_values - float(np.mean(feature_values))) / feature_sd

    age_values = feature_df[age_col].to_numpy(dtype=float)
    age_sd = float(np.std(age_values, ddof=0))
    if np.isfinite(age_sd) and age_sd > 0:
        feature_df["age"] = (age_values - float(np.mean(age_values))) / age_sd
    else:
        feature_df["age"] = age_values - float(np.mean(age_values))

    model_cols = ["feature_z", "age", *baseline_cols]

    model, used_penalizer, note = fit_cox_with_fallback(
        feature_df[model_cols + [duration_col, event_col]],
        duration_col=duration_col,
        event_col=event_col,
        penalizers=[0.0, fallback_penalizer],
        l1_ratio=0.0,
    )
    result["fit_penalizer"] = used_penalizer
    result["note"] = f"{feature_kind};{note}" if note else feature_kind
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
    endpoint_map: EndpointMap,
    baseline_covariate_cols: tuple[str, ...] = (),
    genomic_feature_cols: list[str] | tuple[str, ...] | None = None,
    age_col: str = DEFAULT_AGE_COL,
    model_type: str = "cox",
) -> pd.DataFrame:
    """Fit Cox models on age + baseline covariates + n-observation count + one feature at a time.

    `baseline_covariate_cols` (e.g. gender, cancer type, treatment) are
    always-included adjustment terms in every per-feature fit, alongside age.

    `genomic_feature_cols` identifies static features rather than repeated-
    measurement labs (usually binary genomic indicators, but continuous static
    annotations are supported): they have no matching `_n_observations` column
    and are tested as the raw standardized feature + age + baseline covariates,
    instead of the lab n_obs-adjusted model.
    """
    genomic_feature_set = set(genomic_feature_cols) if genomic_feature_cols else set()
    duration_col, event_col = _endpoint_columns(endpoint_map, endpoint)
    total_patients = len(data)
    rows = []

    for feature in tqdm(feature_cols, desc=f"univariate[{endpoint}]", dynamic_ncols=True):
        if feature in genomic_feature_set:
            rows.append(
                _static_feature_association_row(
                    data,
                    feature=feature,
                    endpoint=endpoint,
                    duration_col=duration_col,
                    event_col=event_col,
                    min_events_per_feature=min_events_per_feature,
                    fallback_penalizer=fallback_penalizer,
                    age_col=age_col,
                    baseline_covariate_cols=baseline_covariate_cols,
                    model_type=model_type,
                )
            )
            continue

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
            "model_type": model_type,
        }

        if feature_stat == "n_observations":
            result["note"] = "target_is_n_observations"
            rows.append(result)
            continue

        if n_obs_feature not in data.columns:
            result["note"] = "missing_matching_n_obs_feature"
            rows.append(result)
            continue

        baseline_cols = [c for c in baseline_covariate_cols if c in data.columns]
        feature_df = data[
            [feature, n_obs_feature, duration_col, event_col, age_col, *baseline_cols]
        ].copy()
        result["coverage"] = float(feature_df[feature].notna().mean())
        result["n_obs_coverage"] = float(feature_df[n_obs_feature].notna().mean())

        feature_df = feature_df.dropna(subset=[duration_col, event_col, age_col, *baseline_cols])
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

        age_values = feature_df[age_col].to_numpy(dtype=float)
        age_sd = float(np.std(age_values, ddof=0))
        if np.isfinite(age_sd) and age_sd > 0:
            feature_df["age"] = (age_values - float(np.mean(age_values))) / age_sd
        else:
            feature_df["age"] = age_values - float(np.mean(age_values))

        model_cols = ["feature_z", "n_obs_z", "age", *baseline_cols]
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
    endpoint_map: EndpointMap,
    static_covariate_cols: tuple[str, ...] = (),
    always_include_feature_cols: tuple[str, ...] = (),
    genomic_feature_cols: tuple[str, ...] | None = None,
    min_genomic_prevalence: float | None = None,
    id_col: str = DEFAULT_ID_COL,
    age_col: str = DEFAULT_AGE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Run per-fold canonical-lab selection and elastic-net Cox CV.

    `genomic_feature_cols`, when given, is the subset of
    `always_include_feature_cols` that are genuine binary mutation
    indicators eligible for the `min_genomic_prevalence` floor (Finding 10).
    Defaults to `always_include_feature_cols` itself for callers (e.g. IPIO's
    genomics/all feature subsets) where every always-include feature is
    genomic; callers whose always-include set also contains non-genomic
    features (e.g. COMPASS's Gleason score) must pass this explicitly so the
    prevalence floor isn't applied to a continuous clinical score.
    """
    require_sksurv()
    duration_col, event_col = _endpoint_columns(endpoint_map, endpoint)
    if genomic_feature_cols is None:
        genomic_feature_cols = always_include_feature_cols

    splitter, strat_labels, cv_stratification = make_cv_splitter(
        train_val,
        n_folds=n_folds,
        seed=seed,
        event_col=event_col,
    )
    fold_rows = []
    fold_canonical_labs_rows: list[dict] = []

    split_args = (np.arange(len(train_val)), strat_labels) if strat_labels is not None else (np.arange(len(train_val)),)

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
            id_col=id_col,
        )
        fold_canonical_labs[fold] = canonical
        fold_train = train_val.iloc[tr_idx]
        selected, _ = select_feature_columns(
            fold_train,
            raw_feature_cols,
            min_patient_coverage=min_patient_coverage,
            restrict_to_labs=canonical,
            always_include=list(always_include_feature_cols),
            genomic_feature_cols=list(genomic_feature_cols),
            min_genomic_prevalence=min_genomic_prevalence,
        )
        fold_selected_features[fold] = selected
        for lab in canonical:
            fold_canonical_labs_rows.append(
                {"endpoint": endpoint, "fold": fold, "lab_name": lab}
            )

    grid = list(product(penalizers, l1_ratios))
    total_runs = len(grid) * len(fold_partitions)
    cv_bar = tqdm(total=total_runs, desc=f"coxnet CV[{endpoint}]", dynamic_ncols=True)
    for penalizer, l1_ratio in grid:
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
                "auc_timeline_coverage_val": np.nan,
                "auc_timeline_status_val": "not_evaluated",
                "integrated_brier_val": np.nan,
                "n_valid_brier_horizons_val": 0,
                "brier_timeline_coverage_val": np.nan,
                "brier_timeline_status_val": "not_evaluated",
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
                    age_col=age_col,
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
                    auc_diag = summarize_auc_timeline(auc_df_val)
                    row["n_valid_auc_horizons_val"] = auc_diag["n_valid_auc_horizons"]
                    row["auc_timeline_coverage_val"] = auc_diag["auc_timeline_coverage"]
                    row["auc_timeline_status_val"] = auc_diag["auc_timeline_status"]

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
                        max_time_unit=auc_max_time_units,
                    )
                    row["integrated_brier_val"] = ibs
                    brier_diag = summarize_brier_timeline(brier_df)
                    row["n_valid_brier_horizons_val"] = brier_diag["n_valid_brier_horizons"]
                    row["brier_timeline_coverage_val"] = brier_diag["brier_timeline_coverage"]
                    row["brier_timeline_status_val"] = brier_diag["brier_timeline_status"]
            except _FOLD_FIT_ERRORS as exc:  # expected numerical/convergence failures only
                row["note"] = f"fold_failed: {exc}"

            fold_rows.append(row)
            cv_bar.update(1)
    cv_bar.close()

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
            auc_timeline_coverage_cv_mean=("auc_timeline_coverage_val", "mean"),
            n_adequate_auc_timeline_folds=(
                "auc_timeline_status_val", lambda s: int(s.eq("ok").sum())
            ),
            integrated_brier_cv_mean=("integrated_brier_val", "mean"),
            integrated_brier_cv_std=("integrated_brier_val", "std"),
            n_valid_brier_folds=("integrated_brier_val", lambda s: int(s.notna().sum())),
            brier_timeline_coverage_cv_mean=("brier_timeline_coverage_val", "mean"),
            n_adequate_brier_timeline_folds=(
                "brier_timeline_status_val", lambda s: int(s.eq("ok").sum())
            ),
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

    valid_candidates = cv_df.loc[cv_df["all_folds_valid"]]
    # Select on the same capped, IPCW-weighted mean AUC(t) that is ultimately
    # reported, not the uncapped/unweighted Harrell C (cv_mean) — otherwise the
    # "best" hyperparameter setting can be tuned to a metric different from the
    # one published for it (Finding 4). Fall back to cv_mean only when no
    # candidate has an estimable IPCW mean for every fold (n_valid_auc_t_folds
    # short of n_folds), since a NaN-sorted candidate can't be meaningfully
    # compared on that key at all.
    has_full_auc_t_coverage = valid_candidates["n_valid_auc_t_folds"].eq(int(n_folds))
    selection_key = "mean_auc_t_cv_mean" if has_full_auc_t_coverage.any() else "cv_mean"
    best_row = (
        valid_candidates.sort_values(
            # On an exact selection-key/cv_mean/n_valid_folds tie, prefer the
            # MORE regularized setting (larger penalizer/l1_ratio) rather than
            # the least, since an unmotivated preference for the more complex
            # model is not justified by the CV score alone.
            [selection_key, "cv_mean", "n_valid_folds", "penalizer", "l1_ratio"],
            ascending=[False, False, False, False, False],
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
    endpoint_map: EndpointMap,
    static_covariate_cols: tuple[str, ...] = (),
    auc_quantiles: tuple[float, ...] = (),
    id_col: str = DEFAULT_ID_COL,
    age_col: str = DEFAULT_AGE_COL,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit on full train/validation data and evaluate on the held-out test set."""
    require_sksurv()
    duration_col, event_col = _endpoint_columns(endpoint_map, endpoint)

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
        age_col=age_col,
    )
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
    train_auc_diag = summarize_auc_timeline(train_auc_df)
    test_auc_diag = summarize_auc_timeline(test_auc_df)

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
        max_time_unit=auc_max_time_units,
    )
    test_brier_diag = summarize_brier_timeline(test_brier_df)
    test_brier_df = test_brier_df.copy()
    if not test_brier_df.empty:
        test_brier_df.insert(0, "endpoint", endpoint)

    metrics_row = {
        "endpoint": endpoint,
        "description": endpoint_map[endpoint]["description"],
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
        "auc_quantiles": ",".join(f"{q:g}" for q in auc_quantiles),
        "auc_time_unit_days": auc_time_unit_days,
        "auc_admin_censor_time_unit": auc_max_time_units,
        "horizon_grid": ",".join(f"{float(h):g}" for h in np.asarray(horizon_grid, dtype=float).reshape(-1)),
        "split_stratification": split_stratification,
        "cv_stratification": cv_stratification,
        "note": note,
    }
    metrics_row.update({f"train_val_{key}": value for key, value in train_auc_diag.items()})
    metrics_row.update({f"test_{key}": value for key, value in test_auc_diag.items()})
    metrics_row.update({f"test_{key}": value for key, value in test_brier_diag.items()})
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
            id_col: test.index,
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


def load_build_manifest(inputs_dir: Path, *, manifest_filename: str) -> dict:
    manifest_path = inputs_dir / manifest_filename
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing build manifest at {manifest_path}. Run build_prediction_inputs.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    version = manifest.get("auc_timeline_schema_version")
    if version != AUC_TIMELINE_SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path} has AUC timeline schema version {version!r}; expected "
            f"{AUC_TIMELINE_SCHEMA_VERSION}. Re-run the applicable prediction/genomic "
            "input builder to construct the robust evaluation timeline."
        )
    return manifest


def load_prebuilt_landmark(
    inputs_dir: Path,
    landmark_day: int,
    *,
    aggregated_filename: Callable[[int], str],
    pre_treatment_lab_filename: Callable[[int], str],
    id_col: str = DEFAULT_ID_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load aggregate-statistic and pre-treatment lab data for one landmark."""
    agg_path = inputs_dir / aggregated_filename(landmark_day)
    if not agg_path.exists():
        raise FileNotFoundError(
            f"Missing aggregated input for landmark +{landmark_day}d at {agg_path}. "
            "Run build_prediction_inputs.py first."
        )
    aggregated = pd.read_csv(agg_path).set_index(id_col)

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


def build_endpoint_horizon_grids(
    landmark_day: int,
    *,
    endpoints: list[str],
    auc_horizons_by_landmark: dict,
    auc_quantiles: tuple[float, ...],
    auc_time_unit_days: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Build endpoint horizon grids from the prebuilt input manifest."""
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
