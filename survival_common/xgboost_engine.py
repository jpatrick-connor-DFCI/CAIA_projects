"""Shared XGBoost survival:cox mechanics for landmarked survival analyses."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from survival_common.helper import breslow_survival_at_horizons

try:
    import xgboost as xgb

    XGBOOST_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    xgb = None
    XGBOOST_IMPORT_ERROR = exc


def require_xgboost() -> None:
    if xgb is None:
        raise ModuleNotFoundError(
            "xgboost is required for XGBoost survival analyses."
        ) from XGBOOST_IMPORT_ERROR


def strip_suffix(value: str, suffix: str) -> str:
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def xgb_safe_name(name: str) -> str:
    """Map feature names to the subset accepted by XGBoost."""
    return name.replace("[", "(").replace("]", ")").replace("<", "_lt_")


def truncate_features_by_rank(selected, feature_meta, max_features):
    """Keep top features by coverage, with deterministic name tie-breaking."""
    if max_features is None or len(selected) <= max_features:
        return list(selected)
    candidates = feature_meta
    if "selected" in feature_meta.columns:
        candidates = feature_meta.loc[feature_meta["selected"]]
    return (
        candidates.sort_values(["coverage", "feature"], ascending=[False, True])
        .head(max_features)["feature"]
        .tolist()
    )


def fit_preprocessor(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    age_col: str,
    static_cols: tuple[str, ...] = (),
) -> dict:
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
    static_cols = tuple(static_cols)
    covariate_cols = list(base_feature_cols) + missing_cols + ["age"] + list(static_cols)

    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    if base_feature_cols:
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        x_train_values = imputer.fit_transform(train_df[base_feature_cols])

        if missing_cols:
            missing_source = [strip_suffix(col, "__missing") for col in missing_cols]
            x_train_values = np.hstack(
                [x_train_values, train_df[missing_source].isna().astype(float).to_numpy()]
            )

        scaler.fit(x_train_values)

    age_scaler = StandardScaler()
    age_scaler.fit(train_df[[age_col]])

    static_imputer: SimpleImputer | None = None
    static_scaler: StandardScaler | None = None
    if static_cols:
        static_imputer = SimpleImputer(strategy="mean")
        static_scaler = StandardScaler()
        static_scaler.fit(static_imputer.fit_transform(train_df[list(static_cols)]))

    xgb_feature_names = [xgb_safe_name(c) for c in covariate_cols]
    if len(set(xgb_feature_names)) != len(xgb_feature_names):
        raise ValueError(
            "XGBoost-sanitized feature names collided; update xgb_safe_name."
        )
    return {
        "age_col": age_col,
        "base_feature_cols": base_feature_cols,
        "missing_cols": missing_cols,
        "static_cols": list(static_cols),
        "covariate_cols": covariate_cols,
        "xgb_feature_names": xgb_feature_names,
        "imputer": imputer,
        "scaler": scaler,
        "age_scaler": age_scaler,
        "static_imputer": static_imputer,
        "static_scaler": static_scaler,
    }


def transform_xgb_matrix(df: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    base_feature_cols = preprocessor["base_feature_cols"]
    missing_cols = preprocessor["missing_cols"]
    static_cols = preprocessor.get("static_cols", [])
    age_col = preprocessor["age_col"]
    age = preprocessor["age_scaler"].transform(df[[age_col]]).reshape(-1, 1)
    blocks: list[np.ndarray] = []
    if base_feature_cols:
        x_values = preprocessor["imputer"].transform(df[base_feature_cols])
        if missing_cols:
            missing_source = [strip_suffix(col, "__missing") for col in missing_cols]
            x_values = np.hstack(
                [x_values, df[missing_source].isna().astype(float).to_numpy()]
            )
        blocks.append(preprocessor["scaler"].transform(x_values))
    blocks.append(age)
    if static_cols:
        static_vals = preprocessor["static_imputer"].transform(df[static_cols])
        blocks.append(preprocessor["static_scaler"].transform(static_vals))
    return np.hstack(blocks).astype(float)


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
    age_col: str,
    static_cols: tuple[str, ...] = (),
    eta: float | None = None,
    max_depth: int | None = None,
    min_child_weight: float | None = None,
    num_boost_round: int | None = None,
) -> tuple[object, list[str], dict, dict]:
    """Fit one XGBoost survival:cox model."""
    require_xgboost()
    rounds = int(num_boost_round if num_boost_round is not None else args.num_boost_round)
    if rounds < 1:
        raise ValueError("num_boost_round must be >= 1.")
    preprocessor = fit_preprocessor(
        train_df,
        feature_cols=feature_cols,
        age_col=age_col,
        static_cols=static_cols,
    )
    covariate_cols = preprocessor["covariate_cols"]
    xgb_names = preprocessor["xgb_feature_names"]
    x_train = transform_xgb_matrix(train_df, preprocessor)
    dtrain = xgb.DMatrix(
        x_train,
        label=signed_cox_label(train_df[duration_col], train_df[event_col]),
        feature_names=xgb_names,
    )
    evals = [(dtrain, "train")]
    if len(valid_df):
        x_valid = transform_xgb_matrix(valid_df, preprocessor)
        dvalid = xgb.DMatrix(
            x_valid,
            label=signed_cox_label(valid_df[duration_col], valid_df[event_col]),
            feature_names=xgb_names,
        )
        evals.append((dvalid, "valid"))

    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "eta": float(eta if eta is not None else args.eta),
        "max_depth": int(max_depth if max_depth is not None else args.max_depth),
        "min_child_weight": float(
            min_child_weight if min_child_weight is not None else args.min_child_weight
        ),
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
        "num_boost_round": rounds,
        "evals": evals,
        "verbose_eval": args.verbose_eval if args.verbose_eval else False,
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
    dtest = xgb.DMatrix(x_eval, feature_names=preprocessor["xgb_feature_names"])
    best_iter = best_iteration(model)
    if best_iter is None:
        risk = model.predict(dtest)
    else:
        risk = model.predict(dtest, iteration_range=(0, best_iter + 1))
    return np.asarray(risk, dtype=float).reshape(-1), covariate_cols


def predict_xgb_margin(
    model,
    eval_df: pd.DataFrame,
    *,
    preprocessor: dict,
) -> np.ndarray:
    """Raw margin (log relative hazard), used by Breslow survival curves."""
    x_eval = transform_xgb_matrix(eval_df, preprocessor)
    dtest = xgb.DMatrix(x_eval, feature_names=preprocessor["xgb_feature_names"])
    best_iter = best_iteration(model)
    if best_iter is None:
        margin = model.predict(dtest, output_margin=True)
    else:
        margin = model.predict(
            dtest, iteration_range=(0, best_iter + 1), output_margin=True
        )
    return np.asarray(margin, dtype=float).reshape(-1)


def xgb_survival_at_horizons(
    model,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    preprocessor: dict,
    duration_col: str,
    event_col: str,
    horizons: np.ndarray,
    time_unit_days: int,
) -> np.ndarray:
    """Survival probabilities at each horizon via Breslow baseline hazards."""
    train_lp = predict_xgb_margin(model, train_df, preprocessor=preprocessor)
    eval_lp = predict_xgb_margin(model, eval_df, preprocessor=preprocessor)
    train_event = train_df[event_col].astype(int).to_numpy()
    train_duration = pd.to_numeric(train_df[duration_col], errors="coerce").to_numpy(
        dtype=float
    )
    train_duration_units = np.ceil(train_duration / float(time_unit_days))
    return breslow_survival_at_horizons(
        train_event=train_event,
        train_duration=train_duration_units,
        train_lp=train_lp,
        eval_lp=eval_lp,
        horizons=horizons,
    )


def feature_importance_frame(
    model,
    *,
    covariate_cols: list[str],
    endpoint: str,
    landmark_day: int | None = None,
    xgb_feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return gain/weight/cover importances keyed to original feature names."""
    lookup_keys = list(xgb_feature_names) if xgb_feature_names is not None else list(covariate_cols)
    if len(lookup_keys) != len(covariate_cols):
        raise ValueError("xgb_feature_names and covariate_cols length mismatch.")
    frames = []
    for importance_type in ["gain", "weight", "cover"]:
        raw = model.get_score(importance_type=importance_type)
        frame = pd.DataFrame(
            {
                "feature": covariate_cols,
                importance_type: [raw.get(k, 0.0) for k in lookup_keys],
            }
        )
        frames.append(frame.set_index("feature"))
    out = pd.concat(frames, axis=1).reset_index()
    out.insert(0, "endpoint", endpoint)
    if landmark_day is not None:
        out.insert(0, "landmark_day", landmark_day)
    return out.sort_values(["gain", "weight"], ascending=False)


def chosen_from_best_row(best_row: dict) -> dict:
    """Build the standard chosen-hyperparameters dict from a CV best row."""
    return {
        "max_depth": int(best_row["max_depth"]),
        "eta": float(best_row["eta"]),
        "min_child_weight": float(best_row["min_child_weight"]),
        "selected_num_boost_round": num_boost_round_from_best_iteration(
            best_row.get("best_iteration_mean", np.nan)
        ),
        "cv_mean_c_index": float(best_row.get("cv_mean", np.nan)),
        "cv_mean_auc_t": float(best_row.get("mean_auc_t_cv_mean", np.nan)),
        "cv_mean_integrated_brier": float(best_row.get("integrated_brier_cv_mean", np.nan)),
        "cv_stratification": str(best_row.get("cv_stratification", "")),
    }


def num_boost_round_from_best_iteration(value: object) -> int | None:
    """Convert XGBoost's zero-based best_iteration to a positive round count."""
    try:
        best_iter = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(best_iter) or best_iter < 0:
        return None
    return max(1, int(round(best_iter)) + 1)
