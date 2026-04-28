"""
Landmarking + XGBoost survival baseline for longitudinal lab summaries.

For each requested landmark, this script reuses cox_aggregated.py's landmarked
feature engineering and held-out split, then fits endpoint-specific XGBoost Cox
models on train_val and evaluates on the fixed test set.

Outputs:
  landmark_xgboost_metrics.csv
  landmark_xgboost_auc_t.csv
  landmark_xgboost_feature_importance.csv
  landmark_xgboost_patient_risks.csv
  landmark_xgboost_feature_selection.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    XGBOOST_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    xgb = None
    XGBOOST_IMPORT_ERROR = exc

try:
    from lifelines.utils import concordance_index

    LIFELINES_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    concordance_index = None
    LIFELINES_IMPORT_ERROR = exc

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from cox_aggregated import (  # noqa: E402
    AGE_COL,
    DATA_PATH,
    DEFAULT_AUC_QUANTILES,
    DEFAULT_AUC_TIME_UNIT_DAYS,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_MIN_PATIENT_COVERAGE,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    ENDPOINTS,
    OUTCOME_COLUMNS,
    RESULTS,
    build_aligned_cohort,
    build_landmark_availability_table,
    compute_survlatent_auc_t,
    normalize_endpoints,
    normalize_landmark_days,
    select_feature_columns,
)


def require_xgboost() -> None:
    if xgb is None:
        raise ModuleNotFoundError(
            "xgboost is required for landmark_xgboost.py."
        ) from XGBOOST_IMPORT_ERROR


def require_lifelines() -> None:
    if concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to compute concordance indices."
        ) from LIFELINES_IMPORT_ERROR


def fit_preprocessor(train_df: pd.DataFrame, *, feature_cols: list[str]) -> dict:
    base_feature_cols = [
        col
        for col in feature_cols
        if train_df[col].notna().any() and train_df[col].nunique(dropna=True) > 1
    ]
    missing_cols = [
        f"{col}__missing"
        for col in base_feature_cols
        if train_df[col].isna().nunique(dropna=False) > 1
    ]
    covariate_cols = list(base_feature_cols) + missing_cols + ["age"]
    if not base_feature_cols:
        raise ValueError("No usable XGBoost covariates remained after train-fold filtering.")

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_train_values = imputer.fit_transform(train_df[base_feature_cols])

    if missing_cols:
        missing_source = [col.removesuffix("__missing") for col in missing_cols]
        x_train_values = np.hstack(
            [x_train_values, train_df[missing_source].isna().astype(float).to_numpy()]
        )

    scaler.fit(x_train_values)

    age_scaler = StandardScaler()
    age_scaler.fit(train_df[[AGE_COL]])
    return {
        "base_feature_cols": base_feature_cols,
        "missing_cols": missing_cols,
        "covariate_cols": covariate_cols,
        "imputer": imputer,
        "scaler": scaler,
        "age_scaler": age_scaler,
    }


def transform_xgb_matrix(df: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    base_feature_cols = preprocessor["base_feature_cols"]
    missing_cols = preprocessor["missing_cols"]
    x_values = preprocessor["imputer"].transform(df[base_feature_cols])
    if missing_cols:
        missing_source = [col.removesuffix("__missing") for col in missing_cols]
        x_values = np.hstack(
            [x_values, df[missing_source].isna().astype(float).to_numpy()]
        )
    x_values = preprocessor["scaler"].transform(x_values)
    age = preprocessor["age_scaler"].transform(df[[AGE_COL]]).reshape(-1, 1)
    return np.hstack([x_values, age]).astype(float)


def signed_cox_label(duration: pd.Series, event: pd.Series) -> np.ndarray:
    y = pd.to_numeric(duration, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(event, errors="coerce").fillna(0).astype(int).to_numpy()
    y = np.maximum(y, 1e-6)
    return np.where(e == 1, y, -y)


def make_train_valid_split(
    train_val: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_frac <= 0:
        return train_val, train_val.iloc[0:0].copy()
    stratify = train_val[event_col].astype(int)
    if stratify.value_counts().min() < 2:
        stratify = None
    try:
        train_idx, valid_idx = train_test_split(
            np.arange(len(train_val)),
            test_size=val_frac,
            stratify=stratify,
            random_state=seed,
        )
    except ValueError:
        train_idx, valid_idx = train_test_split(
            np.arange(len(train_val)),
            test_size=val_frac,
            random_state=seed,
        )
    return train_val.iloc[train_idx].copy(), train_val.iloc[valid_idx].copy()


def fit_xgb_cox(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    duration_col: str,
    event_col: str,
    args: argparse.Namespace,
) -> tuple[object, list[str], dict, dict]:
    require_xgboost()
    preprocessor = fit_preprocessor(train_df, feature_cols=feature_cols)
    covariate_cols = preprocessor["covariate_cols"]
    x_train = transform_xgb_matrix(train_df, preprocessor)
    dtrain = xgb.DMatrix(
        x_train,
        label=signed_cox_label(train_df[duration_col], train_df[event_col]),
        feature_names=covariate_cols,
    )
    evals = [(dtrain, "train")]
    if len(valid_df):
        x_valid = transform_xgb_matrix(valid_df, preprocessor)
        dvalid = xgb.DMatrix(
            x_valid,
            label=signed_cox_label(valid_df[duration_col], valid_df[event_col]),
            feature_names=covariate_cols,
        )
        evals.append((dvalid, "valid"))

    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "seed": args.seed,
        "tree_method": args.tree_method,
    }
    train_kwargs = {
        "params": params,
        "dtrain": dtrain,
        "num_boost_round": args.num_boost_round,
        "evals": evals,
        "verbose_eval": args.verbose_eval,
    }
    if len(valid_df) and args.early_stopping_rounds:
        train_kwargs["early_stopping_rounds"] = args.early_stopping_rounds
    model = xgb.train(**train_kwargs)
    return model, covariate_cols, params, preprocessor


def best_iteration(model) -> int | None:
    try:
        return int(model.best_iteration)
    except (AttributeError, TypeError):
        return None


def predict_risk(
    model,
    eval_df: pd.DataFrame,
    *,
    preprocessor: dict,
) -> tuple[np.ndarray, list[str]]:
    covariate_cols = preprocessor["covariate_cols"]
    x_eval = transform_xgb_matrix(eval_df, preprocessor)
    dtest = xgb.DMatrix(x_eval, feature_names=covariate_cols)
    best_iter = best_iteration(model)
    if best_iter is None:
        risk = model.predict(dtest)
    else:
        risk = model.predict(dtest, iteration_range=(0, best_iter + 1))
    return np.asarray(risk, dtype=float).reshape(-1), covariate_cols


def feature_importance_frame(
    model,
    *,
    covariate_cols: list[str],
    endpoint: str,
    landmark_day: int,
) -> pd.DataFrame:
    frames = []
    for importance_type in ["gain", "weight", "cover"]:
        raw = model.get_score(importance_type=importance_type)
        frame = pd.DataFrame(
            {
                "feature": covariate_cols,
                importance_type: [raw.get(feature, 0.0) for feature in covariate_cols],
            }
        )
        frames.append(frame.set_index("feature"))
    out = pd.concat(frames, axis=1).reset_index()
    out.insert(0, "endpoint", endpoint)
    out.insert(0, "landmark_day", landmark_day)
    return out.sort_values(["gain", "weight"], ascending=False)


def run_one_endpoint(
    *,
    merged: pd.DataFrame,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    raw_feature_cols: list[str],
    endpoint: str,
    landmark_day: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_lifelines()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
    selected_features, feature_meta = select_feature_columns(
        train_val,
        raw_feature_cols,
        min_patient_coverage=args.min_patient_coverage,
    )
    if args.max_features is not None and len(selected_features) > args.max_features:
        feature_meta = feature_meta.copy()
        ranked = (
            feature_meta.loc[feature_meta["selected"]]
            .sort_values(["coverage", "feature"], ascending=[False, True])
            .head(args.max_features)["feature"]
            .tolist()
        )
        selected_features = ranked
        feature_meta["selected"] = feature_meta["feature"].isin(selected_features)

    train_fit, valid_fit = make_train_valid_split(
        train_val,
        duration_col=duration_col,
        event_col=event_col,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    model, _, params, preprocessor = fit_xgb_cox(
        train_fit,
        valid_fit,
        feature_cols=selected_features,
        duration_col=duration_col,
        event_col=event_col,
        args=args,
    )
    risk, covariate_cols = predict_risk(
        model,
        test,
        preprocessor=preprocessor,
    )
    event = test[event_col].astype(int).to_numpy()
    duration = test[duration_col].astype(float).to_numpy()
    valid = np.isfinite(duration) & (duration > 0) & np.isfinite(risk)
    c_index = np.nan
    if valid.sum() and event[valid].sum():
        c_index = float(concordance_index(duration[valid], -risk[valid], event[valid]))

    mean_auc, auc_t = compute_survlatent_auc_t(
        test,
        risk,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_val,
        time_unit_days=args.auc_time_unit_days,
        quantiles=tuple(args.auc_quantiles),
        max_time_unit=args.auc_max_time_units,
    )
    auc_t.insert(0, "endpoint", endpoint)
    auc_t.insert(0, "landmark_day", landmark_day)

    metrics = pd.DataFrame(
        [
            {
                "landmark_day": landmark_day,
                "endpoint": endpoint,
                "n_train_val": len(train_val),
                "n_test": len(test),
                "n_train_val_events": int(train_val[event_col].sum()),
                "n_test_events": int(test[event_col].sum()),
                "n_selected_features": len(selected_features),
                "n_covariates_with_missing_indicators": len(covariate_cols),
                "best_iteration": best_iteration(model),
                "c_index": c_index,
                "mean_auc_t": mean_auc,
                "xgb_params": repr(params),
            }
        ]
    )

    risks = pd.DataFrame(
        {
            "landmark_day": landmark_day,
            "endpoint": endpoint,
            "DFCI_MRN": test.index,
            "duration": duration,
            "event": event,
            "risk_score": risk,
        }
    )
    feature_meta = feature_meta.copy()
    feature_meta.insert(0, "endpoint", endpoint)
    feature_meta.insert(0, "landmark_day", landmark_day)
    importance = feature_importance_frame(
        model,
        covariate_cols=covariate_cols,
        endpoint=endpoint,
        landmark_day=landmark_day,
    )
    return metrics, auc_t, risks, pd.concat([feature_meta, importance], ignore_index=True, sort=False)


def main(args: argparse.Namespace) -> None:
    require_xgboost()
    require_lifelines()
    endpoints = normalize_endpoints(args.endpoints)
    landmark_days = normalize_landmark_days(args.landmark_days)
    df = pd.read_csv(args.data, low_memory=False)

    merged_by_landmark = {}
    initial_split = None
    split_stratification = None
    for landmark_day in landmark_days:
        _, _, merged, _, _, _, _ = build_aligned_cohort(
            df,
            seed=args.seed,
            test_frac=args.test_frac,
            landmark_offset_days=landmark_day,
        )
        merged_by_landmark[landmark_day] = merged

    availability, common_mrns = build_landmark_availability_table(merged_by_landmark)
    all_metrics = []
    all_auc = []
    all_risks = []
    all_features = []

    for landmark_day in landmark_days:
        _, _, merged, train_val, test, split_assignments, split_stratification = build_aligned_cohort(
            df,
            seed=args.seed,
            test_frac=args.test_frac,
            landmark_offset_days=landmark_day,
            required_mrns=common_mrns,
            split_assignments=initial_split,
            split_stratification=split_stratification,
        )
        if initial_split is None:
            initial_split = split_assignments

        raw_feature_cols = [
            col
            for col in merged.columns
            if col not in OUTCOME_COLUMNS and col != "split"
        ]
        print(
            f"\nLandmark +{landmark_day}d: train_val={len(train_val)} test={len(test)} "
            f"raw_features={len(raw_feature_cols)}"
        )
        for endpoint in endpoints:
            print(f"Fitting XGBoost Cox endpoint={endpoint} ...")
            metrics, auc_t, risks, features = run_one_endpoint(
                merged=merged,
                train_val=train_val,
                test=test,
                raw_feature_cols=raw_feature_cols,
                endpoint=endpoint,
                landmark_day=landmark_day,
                args=args,
            )
            all_metrics.append(metrics)
            all_auc.append(auc_t)
            all_risks.append(risks)
            all_features.append(features)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "landmark_xgboost_metrics.csv"
    auc_path = output_dir / "landmark_xgboost_auc_t.csv"
    risks_path = output_dir / "landmark_xgboost_patient_risks.csv"
    features_path = output_dir / "landmark_xgboost_feature_selection.csv"
    availability_path = output_dir / "landmark_xgboost_landmark_mrn_availability.csv"

    pd.concat(all_metrics, ignore_index=True).to_csv(metrics_path, index=False)
    pd.concat(all_auc, ignore_index=True).to_csv(auc_path, index=False)
    pd.concat(all_risks, ignore_index=True).to_csv(risks_path, index=False)
    pd.concat(all_features, ignore_index=True, sort=False).to_csv(features_path, index=False)
    availability.to_csv(availability_path, index=False)
    print(f"\nSaved:\n  {metrics_path}\n  {auc_path}\n  {risks_path}\n  {features_path}\n  {availability_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS))
    parser.add_argument("--landmark-days", nargs="+", type=int, default=DEFAULT_LANDMARK_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--min-patient-coverage", type=float, default=DEFAULT_MIN_PATIENT_COVERAGE)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.80)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--tree-method", default="hist")
    parser.add_argument("--verbose-eval", type=int, default=50)
    parser.add_argument("--auc-time-unit-days", type=int, default=DEFAULT_AUC_TIME_UNIT_DAYS)
    parser.add_argument("--auc-quantiles", nargs="+", type=float, default=list(DEFAULT_AUC_QUANTILES))
    parser.add_argument("--auc-max-time-units", type=int, default=None)
    main(parser.parse_args())
