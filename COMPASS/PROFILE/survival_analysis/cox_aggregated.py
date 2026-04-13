"""
Feature-based association and evaluation pipeline using pre-treatment mean labs.

Workflow:
  1. Build patient-level mean lab features from longitudinal rows observed before first treatment.
  2. Split patients into 80% train/validation and 20% held-out test.
  3. Select labs using only the train/validation cohort and a minimum availability threshold.
  4. Run train/validation univariate Cox models with mean-imputed lab values.
  5. Tune penalized multivariable Cox models with 5-fold CV on train/validation.
  6. Refit the chosen model on the full 80% block and evaluate on held-out test.

Endpoints:
  - platinum: time to first platinum exposure or censoring
  - death:    time to death / last contact
  - either:   composite first of platinum or death

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
  results/cox_agg_univariate_trainval_<endpoint>.csv
  results/cox_agg_univariate_trainval_overview.csv
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


BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

AGE_COL = "AGE_AT_TREATMENTSTART"
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_N_FOLDS = 5
DEFAULT_MIN_PATIENT_COVERAGE = 0.20
DEFAULT_MIN_EVENTS_PER_FEATURE = 10
DEFAULT_CV_PENALIZERS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
DEFAULT_CV_L1_RATIOS = [0.0, 0.5]
PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}

JOINT_ENDPOINT_ALIASES = {
    "joint": "either",
    "combined": "either",
    "composite": "either",
}
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
    "either": {
        "duration_col": "t_either",
        "event_col": "EITHER",
        "description": "Composite: first platinum or death",
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
    "t_diagnosis",
    "t_first_treatment",
    "t_platinum",
    "t_last_contact",
    "t_death",
    "t_either",
    "split",
}


def require_lifelines() -> None:
    if CoxPHFitter is None or concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to run the aggregated Cox association pipeline."
        ) from LIFELINES_IMPORT_ERROR


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
        normalized = JOINT_ENDPOINT_ALIASES.get(endpoint.lower(), endpoint.lower())
        if normalized not in ENDPOINTS:
            valid = ", ".join(sorted(set(ENDPOINTS) | set(JOINT_ENDPOINT_ALIASES)))
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
        & pat["t_first_treatment"].notna()
        & pat["t_platinum"].ge(0)
        & pat["t_death"].ge(0)
        & pat["t_either"].ge(0)
    )
    return pat.loc[valid].copy()


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
        raise ValueError("No pre-treatment lab rows were available to build mean lab features.")

    feature_long = (
        pre_treatment.groupby(["DFCI_MRN", "LAB_NAME"])["LAB_VALUE"]
        .mean()
        .rename("lab_mean")
        .reset_index()
    )
    feature_df = feature_long.pivot(index="DFCI_MRN", columns="LAB_NAME", values="lab_mean")
    feature_df.columns = [f"{col}__mean" for col in feature_df.columns]
    feature_df = feature_df.sort_index(axis=1)

    print(f"Raw feature matrix: {feature_df.shape[0]} patients x {feature_df.shape[1]} mean-lab features")
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


def select_feature_columns(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    raw_feature_cols: list[str],
    *,
    min_patient_coverage: float,
) -> tuple[list[str], pd.DataFrame]:
    train_cov = train_val[raw_feature_cols].notna().mean()
    test_cov = test[raw_feature_cols].notna().mean()
    train_unique = train_val[raw_feature_cols].nunique(dropna=True)

    feature_meta = pd.DataFrame(
        {
            "feature": raw_feature_cols,
            "train_val_coverage": train_cov.reindex(raw_feature_cols).values,
            "test_coverage": test_cov.reindex(raw_feature_cols).values,
            "train_val_unique_non_missing": train_unique.reindex(raw_feature_cols).values,
        }
    )
    parsed = feature_meta["feature"].map(parse_feature_name)
    feature_meta["lab_name"] = parsed.str[0]
    feature_meta["feature_stat"] = parsed.str[1]
    feature_meta["selected"] = (
        feature_meta["train_val_coverage"].ge(min_patient_coverage)
        & feature_meta["train_val_unique_non_missing"].gt(1)
    )
    feature_meta = feature_meta.sort_values(
        ["selected", "train_val_coverage", "feature"],
        ascending=[False, False, True],
    )

    selected = feature_meta.loc[feature_meta["selected"], "feature"].tolist()
    if not selected:
        raise ValueError("No features passed train/validation coverage and variability filters.")
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
) -> tuple[CoxPHFitter | None, float, str]:
    require_lifelines()

    last_error = ""
    for penalizer in penalizers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
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
    adjust_age: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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

    if adjust_age:
        train_model["age_10y"] = train_df[AGE_COL].to_numpy(dtype=float) / 10.0
        eval_model["age_10y"] = eval_df[AGE_COL].to_numpy(dtype=float) / 10.0
        covariate_cols.append("age_10y")

    if not covariate_cols:
        raise ValueError("No usable covariates remained after train-fold filtering.")

    train_model[duration_col] = train_df[duration_col].to_numpy(dtype=float)
    train_model[event_col] = train_df[event_col].to_numpy(dtype=int)
    eval_model[duration_col] = eval_df[duration_col].to_numpy(dtype=float)
    eval_model[event_col] = eval_df[event_col].to_numpy(dtype=int)
    return train_model, eval_model, covariate_cols


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
    train_val: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    adjust_age: bool,
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> pd.DataFrame:
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
    total_patients = len(train_val)
    rows = []

    for feature in feature_cols:
        lab_name, feature_stat = parse_feature_name(feature)
        feature_df = train_val[[feature, duration_col, event_col, AGE_COL]].copy()
        coverage = float(feature_df[feature].notna().mean())

        required_cols = [duration_col, event_col]
        if adjust_age:
            required_cols.append(AGE_COL)
        feature_df = feature_df.dropna(subset=required_cols)

        observed_non_missing = int(feature_df[feature].notna().sum())
        imputed_count = int(len(feature_df) - observed_non_missing)
        result = {
            "endpoint": endpoint,
            "feature": feature,
            "lab_name": lab_name,
            "feature_stat": feature_stat,
            "coverage_train_val": coverage,
            "n_patients_train_val": total_patients,
            "n_patients_used": len(feature_df),
            "n_patients_observed": observed_non_missing,
            "n_patients_imputed": imputed_count,
            "n_events_used": int(feature_df[event_col].sum()),
            "adjusted_for_age": adjust_age,
            "coef": np.nan,
            "hazard_ratio_per_sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "concordance_index": np.nan,
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
        model_cols = ["feature_z"]

        if adjust_age:
            feature_df["age_10y"] = feature_df[AGE_COL] / 10.0
            model_cols.append("age_10y")

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
        result["coef"] = float(summary_row["coef"])
        result["hazard_ratio_per_sd"] = float(summary_row["exp(coef)"])
        result["ci_lower"] = float(summary_row["exp(coef) lower 95%"])
        result["ci_upper"] = float(summary_row["exp(coef) upper 95%"])
        result["p_value"] = float(summary_row["p"])
        result["concordance_index"] = float(model.concordance_index_)
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
    adjust_age: bool,
    penalizers: list[float],
    l1_ratios: list[float],
    n_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
                "penalizer": penalizer,
                "l1_ratio": l1_ratio,
                "n_train": len(fold_train),
                "n_val": len(fold_val),
                "n_events_train": int(fold_train[event_col].sum()),
                "n_events_val": int(fold_val[event_col].sum()),
                "cv_stratification": cv_stratification,
                "c_index_val": np.nan,
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
                    adjust_age=adjust_age,
                )
                model, _, note = fit_cox_with_fallback(
                    train_mdf,
                    duration_col=duration_col,
                    event_col=event_col,
                    penalizers=[penalizer],
                    l1_ratio=l1_ratio,
                )
                row["note"] = note
                row["n_covariates"] = len(covariate_cols)
                if model is not None:
                    c_index, _ = score_cox_model(
                        model,
                        val_mdf,
                        duration_col=duration_col,
                        event_col=event_col,
                    )
                    row["c_index_val"] = c_index
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
    adjust_age: bool,
    penalizer: float,
    l1_ratio: float,
    split_stratification: str,
    cv_stratification: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    train_mdf, test_mdf, covariate_cols = build_model_matrices(
        train_val,
        test,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        adjust_age=adjust_age,
    )
    model, used_penalizer, note = fit_cox_with_fallback(
        train_mdf,
        duration_col=duration_col,
        event_col=event_col,
        penalizers=[penalizer],
        l1_ratio=l1_ratio,
    )
    if model is None:
        raise RuntimeError(f"Final multivariable model failed for endpoint '{endpoint}': {note}")

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

    metrics_row = {
        "endpoint": endpoint,
        "description": ENDPOINTS[endpoint]["description"],
        "n_train_val": len(train_val),
        "n_test": len(test),
        "n_events_train_val": int(train_val[event_col].sum()),
        "n_events_test": int(test[event_col].sum()),
        "selected_penalizer": used_penalizer,
        "selected_l1_ratio": l1_ratio,
        "n_covariates": len(covariate_cols),
        "train_val_c_index": train_c,
        "test_c_index": test_c,
        "adjusted_for_age": adjust_age,
        "split_stratification": split_stratification,
        "cv_stratification": cv_stratification,
        "note": note,
    }

    summary = model.summary.reset_index().rename(columns={"covariate": "feature"})
    parsed = summary["feature"].map(parse_feature_name)
    summary["endpoint"] = endpoint
    summary["lab_name"] = parsed.str[0]
    summary["feature_stat"] = parsed.str[1]
    summary["is_age_covariate"] = summary["feature"].eq("age_10y")
    summary["selected_penalizer"] = used_penalizer
    summary["selected_l1_ratio"] = l1_ratio
    summary["n_covariates"] = len(covariate_cols)
    summary["train_val_c_index"] = train_c
    summary["test_c_index"] = test_c
    summary["adjusted_for_age"] = adjust_age
    summary["note"] = note
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


def build_univariate_overview(univariate_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    overview: pd.DataFrame | None = None
    key_cols = ["feature", "lab_name", "feature_stat"]

    for endpoint, endpoint_df in univariate_results.items():
        keep = endpoint_df[
            [
                *key_cols,
                "coverage_train_val",
                "n_patients_used",
                "n_patients_observed",
                "n_patients_imputed",
                "n_events_used",
                "coef",
                "hazard_ratio_per_sd",
                "ci_lower",
                "ci_upper",
                "p_value",
                "q_value",
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
        for col in ["either_p_value", "platinum_p_value", "death_p_value"]
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
    print(f"\nTop univariate associations on train/val for {endpoint}:")
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

    print("Building raw aggregated pre-treatment mean lab feature matrix...")
    feature_df = build_feature_matrix(df)

    merged = feature_df.join(outcome_df, how="inner")
    if args.adjust_age:
        merged = merged.loc[merged[AGE_COL].notna()].copy()
    if merged.empty:
        raise ValueError("No patients have both engineered features and valid outcomes.")

    train_val, test, split_assignments, split_stratification = split_train_test(
        merged,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    merged = merged.copy()
    merged["split"] = split_assignments

    raw_feature_cols = feature_df.columns.tolist()
    selected_feature_cols, feature_meta = select_feature_columns(
        train_val,
        test,
        raw_feature_cols,
        min_patient_coverage=args.min_patient_coverage,
    )

    keep_cols = selected_feature_cols + [col for col in merged.columns if col in OUTCOME_COLUMNS]
    train_val = train_val[
        selected_feature_cols + [col for col in train_val.columns if col in OUTCOME_COLUMNS]
    ].copy()
    test = test[selected_feature_cols + [col for col in test.columns if col in OUTCOME_COLUMNS]].copy()
    merged = merged[keep_cols].copy()
    merged["split"] = split_assignments

    print(f"Train/val: {len(train_val)} patients")
    print(f"Test:      {len(test)} patients")
    print(f"Selected mean-lab features from train/val only: {len(selected_feature_cols)}")
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
                train_val,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                adjust_age=args.adjust_age,
                min_events_per_feature=args.min_events_per_feature,
                fallback_penalizer=args.univariate_penalizer,
            )
            univariate_results[endpoint] = univariate_df
            univariate_df.to_csv(RESULTS / f"cox_agg_univariate_trainval_{endpoint}.csv", index=False)
            print_top_hits(univariate_df, endpoint=endpoint)

        if args.analysis in {"multivariable", "both"}:
            fold_df, cv_df, best_row = tune_multivariable_model(
                train_val,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                adjust_age=args.adjust_age,
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
                    "cv_mean": best_row["cv_mean"],
                    "cv_std": best_row["cv_std"],
                    "n_valid_folds": best_row["n_valid_folds"],
                    "cv_stratification": best_row["cv_stratification"],
                }
            )

            metrics_row, summary_df, pred_df = fit_final_multivariable_model(
                train_val,
                test,
                feature_cols=selected_feature_cols,
                endpoint=endpoint,
                adjust_age=args.adjust_age,
                penalizer=float(best_row["penalizer"]),
                l1_ratio=float(best_row["l1_ratio"]),
                split_stratification=split_stratification,
                cv_stratification=str(best_row["cv_stratification"]),
            )
            test_metric_rows.append(metrics_row)
            summary_df.to_csv(RESULTS / f"cox_agg_multivariable_{endpoint}.csv", index=False)
            pred_df.to_csv(RESULTS / f"cox_agg_test_predictions_{endpoint}.csv", index=False)

            top = summary_df.loc[
                ~summary_df["is_age_covariate"],
                ["feature", "coef", "exp(coef)", "p"],
            ].head(10)
            print("\nChosen hyperparameters:")
            print(
                f"  penalizer={best_row['penalizer']}  l1_ratio={best_row['l1_ratio']}  cv_mean={best_row['cv_mean']:.4f}"
            )
            print(f"  held-out test C-index={metrics_row['test_c_index']:.4f}")
            print("Top multivariable coefficients:")
            print(top.to_string(index=False))

    if univariate_results:
        overview = build_univariate_overview(univariate_results)
        overview.to_csv(RESULTS / "cox_agg_univariate_trainval_overview.csv", index=False)
        print("\nSaved: results/cox_agg_univariate_trainval_overview.csv")

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
        default=["platinum", "death", "either"],
        help="Endpoints to analyze. 'joint' is accepted as an alias for 'either'.",
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
        help="Penalizer values searched during train/validation cross-validation.",
    )
    parser.add_argument(
        "--cv-l1-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_CV_L1_RATIOS,
        help="Elastic-net mixing values searched during train/validation cross-validation.",
    )
    parser.add_argument(
        "--adjust-age",
        dest="adjust_age",
        action="store_true",
        help="Adjust association and multivariable models for age at treatment start.",
    )
    parser.add_argument(
        "--no-adjust-age",
        dest="adjust_age",
        action="store_false",
        help="Run models without the age covariate.",
    )
    parser.set_defaults(adjust_age=True)
    main(parser.parse_args())
