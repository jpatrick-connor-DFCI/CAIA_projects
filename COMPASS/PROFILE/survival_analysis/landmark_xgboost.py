"""
Landmarking + XGBoost survival baseline for longitudinal lab summaries.

For each requested landmark, this script reuses cox_aggregated.py's landmarked
feature engineering and held-out split, then runs 5-fold stratified CV on
train_val over a (max_depth x eta x min_child_weight) grid to pick
hyperparameters before refitting on full train_val and evaluating on the
held-out test fold. Per-fold canonical labs and the fixed horizon grid keep
test data fully isolated from selection.

Outputs:
  landmark_xgboost_metrics.csv
  landmark_xgboost_auc_t.csv
  landmark_xgboost_brier.csv
  landmark_xgboost_cv_folds.csv
  landmark_xgboost_cv_summary.csv
  landmark_xgboost_canonical_labs_folds.csv
  landmark_xgboost_feature_importance.csv
  landmark_xgboost_patient_risks.csv
  landmark_xgboost_feature_selection.csv
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
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
    build_pre_treatment_lab_long,
    compute_survlatent_auc_t,
    normalize_endpoints,
    normalize_landmark_days,
    select_feature_columns,
)
from helper import (  # noqa: E402
    assert_disjoint_folds,
    assert_no_test_leakage,
    breslow_survival_at_horizons,
    compute_brier,
    compute_horizon_grid,
    iter_stratified_folds,
    select_canonical_labs,
)

DEFAULT_CV_MAX_DEPTHS = [2, 3, 4]
DEFAULT_CV_ETAS = [0.03, 0.05, 0.10]
DEFAULT_CV_MIN_CHILD_WEIGHTS = [3.0, 5.0, 10.0]
DEFAULT_N_FOLDS = 5


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


class TqdmXGBCallback:
    """Per-boosting-round tqdm progress for xgb.train.

    Implemented as a duck-typed `xgb.callback.TrainingCallback` subclass when
    XGBoost is available; this avoids touching xgb at module import time so the
    rest of the file still loads if XGBoost is missing.
    """

    def __init__(self, total_rounds: int, desc: str = "xgb"):
        self.bar = tqdm(
            total=int(total_rounds),
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )

    def before_training(self, model):
        return model

    def after_iteration(self, model, epoch, evals_log):
        latest = {}
        for split, metrics in evals_log.items():
            for metric, vals in metrics.items():
                if vals:
                    latest[f"{split[:5]}_{metric[:6]}"] = f"{float(vals[-1]):.4f}"
        if hasattr(self.bar, "set_postfix"):
            self.bar.set_postfix(latest)
        self.bar.update(1)
        return False  # don't stop training

    def after_training(self, model):
        if hasattr(self.bar, "close"):
            self.bar.close()
        return model


def _make_xgb_callback(total_rounds: int, desc: str):
    """Return a real xgb TrainingCallback subclass instance, or None."""
    if xgb is None or not TQDM_AVAILABLE:
        return None
    base = xgb.callback.TrainingCallback
    cls = type(
        "_TqdmXGBCallbackImpl",
        (base,),
        {
            "__init__": TqdmXGBCallback.__init__,
            "before_training": TqdmXGBCallback.before_training,
            "after_iteration": TqdmXGBCallback.after_iteration,
            "after_training": TqdmXGBCallback.after_training,
        },
    )
    return cls(total_rounds=total_rounds, desc=desc)


def strip_suffix(value: str, suffix: str) -> str:
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


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
        missing_source = [strip_suffix(col, "__missing") for col in missing_cols]
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
        missing_source = [strip_suffix(col, "__missing") for col in missing_cols]
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
    eta: float | None = None,
    max_depth: int | None = None,
    min_child_weight: float | None = None,
) -> tuple[object, list[str], dict, dict]:
    """Fit one XGBoost survival:cox model.

    `eta`, `max_depth`, `min_child_weight` override the corresponding args.*
    defaults so the CV grid can sweep them without touching args. Other
    hyperparameters (subsample / colsample / lambda / alpha) come from args.
    """
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
    bar_desc = getattr(args, "_xgb_progress_desc", "xgb-train")
    cb = _make_xgb_callback(total_rounds=args.num_boost_round, desc=bar_desc)
    train_kwargs = {
        "params": params,
        "dtrain": dtrain,
        "num_boost_round": args.num_boost_round,
        "evals": evals,
        # tqdm replaces per-round prints; only fall back to xgboost's own
        # logger when tqdm isn't installed.
        "verbose_eval": False if cb is not None else args.verbose_eval,
    }
    if cb is not None:
        train_kwargs["callbacks"] = [cb]
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


def predict_xgb_margin(
    model,
    eval_df: pd.DataFrame,
    *,
    preprocessor: dict,
) -> np.ndarray:
    """Raw margin (log relative hazard) — needed by Breslow for survival probs."""
    covariate_cols = preprocessor["covariate_cols"]
    x_eval = transform_xgb_matrix(eval_df, preprocessor)
    dtest = xgb.DMatrix(x_eval, feature_names=covariate_cols)
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
    """Survival probabilities at each horizon (in time-units) via Breslow.

    XGBoost survival:cox emits a margin (log relative hazard); we use it to
    fit Breslow's baseline cumulative hazard on the training set and then
    evaluate S(t|x) for each eval row. Train durations are converted into the
    same time unit as `horizons` so the baseline and the horizons share a clock.
    """
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


def cv_one_endpoint(
    *,
    train_val: pd.DataFrame,
    raw_feature_cols: list[str],
    pre_treatment_lab_df: pd.DataFrame,
    horizon_grid: np.ndarray,
    endpoint: str,
    landmark_day: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """5-fold stratified CV over (max_depth x eta x min_child_weight).

    Strict no-leakage: per fold, canonical labs and per-stat feature selection
    are recomputed from fold_train MRNs only. Early stopping uses the held-out
    fold (fold_val) as the watch — same set used for metric reporting, which
    is the standard CV-with-early-stopping convention.

    Returns (fold_df, cv_df, best_row, fold_canonical_labs_df).
    """
    require_xgboost()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    fold_partitions = list(
        iter_stratified_folds(train_val, n_folds=args.n_folds, seed=args.seed)
    )
    if not fold_partitions:
        raise RuntimeError(f"No CV folds produced for endpoint '{endpoint}'.")
    cv_stratification = fold_partitions[0][3]

    # Materialize fold-level canonical labs and selected features once so the
    # 27-combo grid traverses identical fold partitions.
    fold_canonical_labs: dict[int, list[str]] = {}
    fold_selected_features: dict[int, list[str]] = {}
    fold_canonical_labs_rows: list[dict] = []
    for fold, tr_idx, val_idx, _ in fold_partitions:
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
            min_coverage=args.min_patient_coverage,
        )
        fold_canonical_labs[fold] = canonical
        fold_train = train_val.iloc[tr_idx]
        selected, _ = select_feature_columns(
            fold_train,
            raw_feature_cols,
            min_patient_coverage=args.min_patient_coverage,
            restrict_to_labs=canonical,
        )
        if args.max_features is not None and len(selected) > args.max_features:
            selected = selected[: args.max_features]
        fold_selected_features[fold] = selected
        for lab in canonical:
            fold_canonical_labs_rows.append(
                {
                    "landmark_day": landmark_day,
                    "endpoint": endpoint,
                    "fold": fold,
                    "lab_name": lab,
                }
            )

    grid = list(
        product(args.cv_max_depths, args.cv_etas, args.cv_min_child_weights)
    )
    fold_rows: list[dict] = []
    total_runs = len(grid) * len(fold_partitions)
    cv_bar = tqdm(
        total=total_runs,
        desc=f"xgb CV[{endpoint}@+{landmark_day}d]",
        dynamic_ncols=True,
    )
    for max_depth, eta, min_child_weight in grid:
        for fold, tr_idx, val_idx, _ in fold_partitions:
            fold_train = train_val.iloc[tr_idx]
            fold_val = train_val.iloc[val_idx]
            fold_features = fold_selected_features[fold]
            row = {
                "landmark_day": landmark_day,
                "endpoint": endpoint,
                "fold": fold,
                "max_depth": int(max_depth),
                "eta": float(eta),
                "min_child_weight": float(min_child_weight),
                "n_train": len(fold_train),
                "n_val": len(fold_val),
                "n_events_train": int(fold_train[event_col].sum()),
                "n_events_val": int(fold_val[event_col].sum()),
                "n_canonical_labs": len(fold_canonical_labs[fold]),
                "n_selected_features": len(fold_features),
                "cv_stratification": cv_stratification,
                "best_iteration": np.nan,
                "c_index_val": np.nan,
                "mean_auc_t_val": np.nan,
                "n_valid_auc_horizons_val": 0,
                "integrated_brier_val": np.nan,
                "n_valid_brier_horizons_val": 0,
                "note": "",
            }
            args._xgb_progress_desc = (
                f"d={int(max_depth)} eta={float(eta):g} "
                f"mcw={float(min_child_weight):g} fold={fold}"
            )
            try:
                if not fold_features:
                    raise ValueError("no usable features after fold-level selection")
                model, _, _, preprocessor = fit_xgb_cox(
                    fold_train,
                    fold_val,
                    feature_cols=fold_features,
                    duration_col=duration_col,
                    event_col=event_col,
                    args=args,
                    eta=eta,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                )
                row["best_iteration"] = best_iteration(model)
                risk, _ = predict_risk(model, fold_val, preprocessor=preprocessor)
                event = fold_val[event_col].astype(int).to_numpy()
                duration = fold_val[duration_col].astype(float).to_numpy()
                valid = np.isfinite(duration) & (duration > 0) & np.isfinite(risk)
                if valid.sum() and event[valid].sum():
                    row["c_index_val"] = float(
                        concordance_index(
                            duration[valid], -risk[valid], event[valid]
                        )
                    )
                mean_auc_val, auc_df_val = compute_survlatent_auc_t(
                    fold_val,
                    risk,
                    duration_col=duration_col,
                    event_col=event_col,
                    reference_df=fold_train,
                    time_unit_days=args.auc_time_unit_days,
                    quantiles=tuple(args.auc_quantiles),
                    max_time_unit=args.auc_max_time_units,
                    fixed_horizons=horizon_grid,
                )
                row["mean_auc_t_val"] = mean_auc_val
                row["n_valid_auc_horizons_val"] = (
                    int(auc_df_val["auc_t"].notna().sum())
                    if not auc_df_val.empty
                    else 0
                )

                val_surv = xgb_survival_at_horizons(
                    model,
                    fold_train,
                    fold_val,
                    preprocessor=preprocessor,
                    duration_col=duration_col,
                    event_col=event_col,
                    horizons=horizon_grid,
                    time_unit_days=args.auc_time_unit_days,
                )
                train_dur_units = np.ceil(
                    pd.to_numeric(fold_train[duration_col], errors="coerce").to_numpy(
                        dtype=float
                    )
                    / float(args.auc_time_unit_days)
                )
                val_dur_units = np.ceil(
                    pd.to_numeric(fold_val[duration_col], errors="coerce").to_numpy(
                        dtype=float
                    )
                    / float(args.auc_time_unit_days)
                )
                brier_df, ibs = compute_brier(
                    train_event=fold_train[event_col].to_numpy(dtype=int),
                    train_duration=train_dur_units,
                    eval_event=fold_val[event_col].to_numpy(dtype=int),
                    eval_duration=val_dur_units,
                    surv_at_horizons=val_surv,
                    horizons=horizon_grid,
                    time_unit_days=args.auc_time_unit_days,
                )
                row["integrated_brier_val"] = ibs
                row["n_valid_brier_horizons_val"] = (
                    int(brier_df["brier"].notna().sum()) if not brier_df.empty else 0
                )
            except Exception as exc:  # pragma: no cover - defensive
                row["note"] = f"fold_failed: {exc}"
            fold_rows.append(row)
            if hasattr(cv_bar, "set_postfix"):
                cv_bar.set_postfix(
                    {
                        "C": (
                            f"{row['c_index_val']:.4f}"
                            if np.isfinite(row.get("c_index_val", np.nan))
                            else "nan"
                        ),
                        "iter": int(row["best_iteration"])
                        if np.isfinite(row.get("best_iteration", np.nan))
                        else "-",
                    }
                )
            cv_bar.update(1)
    cv_bar.close()
    if hasattr(args, "_xgb_progress_desc"):
        delattr(args, "_xgb_progress_desc")

    fold_df = pd.DataFrame(fold_rows)
    cv_df = (
        fold_df.groupby(
            ["landmark_day", "endpoint", "max_depth", "eta", "min_child_weight"],
            dropna=False,
        )
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
            best_iteration_mean=("best_iteration", "mean"),
            cv_stratification=("cv_stratification", "first"),
        )
        .reset_index()
    )
    cv_df["all_folds_valid"] = cv_df["n_valid_folds"].eq(int(args.n_folds))

    if cv_df["n_valid_folds"].eq(0).all():
        raise RuntimeError(
            f"All XGBoost CV fits failed for endpoint '{endpoint}'."
        )
    candidate = cv_df.loc[cv_df["all_folds_valid"]]
    if candidate.empty:
        # Fall back to whatever combo had the most valid folds.
        candidate = cv_df.sort_values("n_valid_folds", ascending=False)
    best_row = (
        candidate.sort_values(
            ["cv_mean", "n_valid_folds", "max_depth", "eta", "min_child_weight"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )
        .iloc[0]
        .to_dict()
    )
    fold_canonical_labs_df = pd.DataFrame(fold_canonical_labs_rows)
    return fold_df, cv_df, best_row, fold_canonical_labs_df


def run_one_endpoint(
    *,
    merged: pd.DataFrame,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    raw_feature_cols: list[str],
    canonical_labs: list[str],
    pre_treatment_lab_df: pd.DataFrame,
    horizon_grid: np.ndarray,
    endpoint: str,
    landmark_day: int,
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    require_lifelines()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    assert_no_test_leakage(
        test_mrns=test.index,
        train_mrns=train_val.index,
        context=f"landmark_xgboost.run_one_endpoint[{endpoint}@+{landmark_day}d]",
    )

    # CV-driven hyperparameter selection (skip with --no-cv to keep the legacy
    # single-fit path).
    cv_fold_df = pd.DataFrame()
    cv_summary_df: pd.DataFrame = pd.DataFrame()
    fold_canonical_labs_df = pd.DataFrame()
    chosen: dict | None = None
    if not args.no_cv:
        cv_fold_df, cv_summary_df, best_row, fold_canonical_labs_df = cv_one_endpoint(
            train_val=train_val,
            raw_feature_cols=raw_feature_cols,
            pre_treatment_lab_df=pre_treatment_lab_df,
            horizon_grid=horizon_grid,
            endpoint=endpoint,
            landmark_day=landmark_day,
            args=args,
        )
        chosen = {
            "max_depth": int(best_row["max_depth"]),
            "eta": float(best_row["eta"]),
            "min_child_weight": float(best_row["min_child_weight"]),
            "cv_mean_c_index": float(best_row.get("cv_mean", np.nan)),
            "cv_mean_auc_t": float(best_row.get("mean_auc_t_cv_mean", np.nan)),
            "cv_mean_integrated_brier": float(
                best_row.get("integrated_brier_cv_mean", np.nan)
            ),
            "cv_stratification": str(best_row.get("cv_stratification", "")),
        }
        print(
            f"  CV chose max_depth={chosen['max_depth']} eta={chosen['eta']:g} "
            f"min_child_weight={chosen['min_child_weight']:g}; "
            f"cv C-index={chosen['cv_mean_c_index']:.4f} "
            f"AUC(t)={chosen['cv_mean_auc_t']:.4f} "
            f"IBS={chosen['cv_mean_integrated_brier']:.4f}"
        )

    # Final selection on full train_val with the canonical labs already in scope.
    selected_features, feature_meta = select_feature_columns(
        train_val,
        raw_feature_cols,
        min_patient_coverage=args.min_patient_coverage,
        restrict_to_labs=canonical_labs,
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
    final_eta = chosen["eta"] if chosen is not None else None
    final_max_depth = chosen["max_depth"] if chosen is not None else None
    final_min_child_weight = chosen["min_child_weight"] if chosen is not None else None
    model, _, params, preprocessor = fit_xgb_cox(
        train_fit,
        valid_fit,
        feature_cols=selected_features,
        duration_col=duration_col,
        event_col=event_col,
        args=args,
        eta=final_eta,
        max_depth=final_max_depth,
        min_child_weight=final_min_child_weight,
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
        fixed_horizons=horizon_grid,
    )
    auc_t.insert(0, "endpoint", endpoint)
    auc_t.insert(0, "landmark_day", landmark_day)

    # Test-set Brier via Breslow on the train_val linear predictor.
    test_surv = xgb_survival_at_horizons(
        model,
        train_val,
        test,
        preprocessor=preprocessor,
        duration_col=duration_col,
        event_col=event_col,
        horizons=horizon_grid,
        time_unit_days=args.auc_time_unit_days,
    )
    train_dur_units = np.ceil(
        pd.to_numeric(train_val[duration_col], errors="coerce").to_numpy(dtype=float)
        / float(args.auc_time_unit_days)
    )
    test_dur_units = np.ceil(
        pd.to_numeric(test[duration_col], errors="coerce").to_numpy(dtype=float)
        / float(args.auc_time_unit_days)
    )
    brier_t, integrated_brier = compute_brier(
        train_event=train_val[event_col].to_numpy(dtype=int),
        train_duration=train_dur_units,
        eval_event=test[event_col].to_numpy(dtype=int),
        eval_duration=test_dur_units,
        surv_at_horizons=test_surv,
        horizons=horizon_grid,
        time_unit_days=args.auc_time_unit_days,
    )
    if not brier_t.empty:
        brier_t = brier_t.copy()
        brier_t.insert(0, "endpoint", endpoint)
        brier_t.insert(0, "landmark_day", landmark_day)

    metrics_row = {
        "landmark_day": landmark_day,
        "endpoint": endpoint,
        "n_train_val": len(train_val),
        "n_test": len(test),
        "n_train_val_events": int(train_val[event_col].sum()),
        "n_test_events": int(test[event_col].sum()),
        "n_canonical_labs": len(canonical_labs),
        "n_selected_features": len(selected_features),
        "n_covariates_with_missing_indicators": len(covariate_cols),
        "best_iteration": best_iteration(model),
        "c_index": c_index,
        "mean_auc_t": mean_auc,
        "integrated_brier": integrated_brier,
        "xgb_params": repr(params),
        "horizon_grid": ",".join(
            f"{float(h):g}" for h in np.asarray(horizon_grid, dtype=float).reshape(-1)
        ),
    }
    if chosen is not None:
        metrics_row.update(
            {
                "selected_max_depth": chosen["max_depth"],
                "selected_eta": chosen["eta"],
                "selected_min_child_weight": chosen["min_child_weight"],
                "cv_mean_c_index": chosen["cv_mean_c_index"],
                "cv_mean_auc_t": chosen["cv_mean_auc_t"],
                "cv_mean_integrated_brier": chosen["cv_mean_integrated_brier"],
                "cv_stratification": chosen["cv_stratification"],
            }
        )
    metrics = pd.DataFrame([metrics_row])

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
    return (
        metrics,
        auc_t,
        brier_t,
        risks,
        feature_meta,
        importance,
        cv_fold_df,
        cv_summary_df,
        fold_canonical_labs_df,
    )


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
    all_brier = []
    all_risks = []
    all_features = []
    all_importance = []
    all_cv_folds = []
    all_cv_summaries = []
    all_fold_canonical_labs = []

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

        assert_no_test_leakage(
            test_mrns=test.index,
            train_mrns=train_val.index,
            context=f"landmark_xgboost.main[+{landmark_day}d]",
        )

        raw_feature_cols = [
            col
            for col in merged.columns
            if col not in OUTCOME_COLUMNS and col != "split"
        ]
        print(
            f"\nLandmark +{landmark_day}d: train_val={len(train_val)} test={len(test)} "
            f"raw_features={len(raw_feature_cols)}"
        )
        pre_treatment_lab_df = build_pre_treatment_lab_long(
            df,
            cohort_index=merged.index,
            landmark_offset_days=landmark_day,
        )
        canonical_labs = select_canonical_labs(
            pre_treatment_lab_df,
            mrns=train_val.index,
            min_coverage=args.min_patient_coverage,
        )
        print(f"Canonical labs (train_val): {len(canonical_labs)}")
        for endpoint in endpoints:
            print(f"Fitting XGBoost Cox endpoint={endpoint} ...")
            horizon_grid = compute_horizon_grid(
                train_val,
                duration_col=ENDPOINTS[endpoint]["duration_col"],
                event_col=ENDPOINTS[endpoint]["event_col"],
                quantiles=tuple(args.auc_quantiles),
                time_unit_days=args.auc_time_unit_days,
            )
            (
                metrics,
                auc_t,
                brier_t,
                risks,
                features,
                importance,
                cv_folds,
                cv_summary,
                fold_canonical_labs,
            ) = run_one_endpoint(
                merged=merged,
                train_val=train_val,
                test=test,
                raw_feature_cols=raw_feature_cols,
                canonical_labs=canonical_labs,
                pre_treatment_lab_df=pre_treatment_lab_df,
                horizon_grid=horizon_grid,
                endpoint=endpoint,
                landmark_day=landmark_day,
                args=args,
            )
            all_metrics.append(metrics)
            all_auc.append(auc_t)
            if not brier_t.empty:
                all_brier.append(brier_t)
            all_risks.append(risks)
            all_features.append(features)
            all_importance.append(importance)
            if not cv_folds.empty:
                all_cv_folds.append(cv_folds)
            if not cv_summary.empty:
                all_cv_summaries.append(cv_summary)
            if not fold_canonical_labs.empty:
                all_fold_canonical_labs.append(fold_canonical_labs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "landmark_xgboost_metrics.csv"
    auc_path = output_dir / "landmark_xgboost_auc_t.csv"
    brier_path = output_dir / "landmark_xgboost_brier.csv"
    risks_path = output_dir / "landmark_xgboost_patient_risks.csv"
    features_path = output_dir / "landmark_xgboost_feature_selection.csv"
    importance_path = output_dir / "landmark_xgboost_feature_importance.csv"
    cv_folds_path = output_dir / "landmark_xgboost_cv_folds.csv"
    cv_summary_path = output_dir / "landmark_xgboost_cv_summary.csv"
    fold_labs_path = output_dir / "landmark_xgboost_canonical_labs_folds.csv"
    availability_path = output_dir / "landmark_xgboost_landmark_mrn_availability.csv"

    pd.concat(all_metrics, ignore_index=True).to_csv(metrics_path, index=False)
    pd.concat(all_auc, ignore_index=True).to_csv(auc_path, index=False)
    pd.concat(all_risks, ignore_index=True).to_csv(risks_path, index=False)
    pd.concat(all_features, ignore_index=True, sort=False).to_csv(features_path, index=False)
    pd.concat(all_importance, ignore_index=True, sort=False).to_csv(importance_path, index=False)
    saved = [metrics_path, auc_path, risks_path, features_path, importance_path]
    if all_brier:
        pd.concat(all_brier, ignore_index=True).to_csv(brier_path, index=False)
        saved.append(brier_path)
    if all_cv_folds:
        pd.concat(all_cv_folds, ignore_index=True).to_csv(cv_folds_path, index=False)
        saved.append(cv_folds_path)
    if all_cv_summaries:
        pd.concat(all_cv_summaries, ignore_index=True).to_csv(cv_summary_path, index=False)
        saved.append(cv_summary_path)
    if all_fold_canonical_labs:
        pd.concat(all_fold_canonical_labs, ignore_index=True).to_csv(
            fold_labs_path, index=False
        )
        saved.append(fold_labs_path)
    availability.to_csv(availability_path, index=False)
    saved.append(availability_path)
    print("\nSaved:")
    for path in saved:
        print(f"  {path}")


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
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument(
        "--cv-max-depths",
        nargs="+",
        type=int,
        default=list(DEFAULT_CV_MAX_DEPTHS),
    )
    parser.add_argument(
        "--cv-etas",
        nargs="+",
        type=float,
        default=list(DEFAULT_CV_ETAS),
    )
    parser.add_argument(
        "--cv-min-child-weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_CV_MIN_CHILD_WEIGHTS),
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip 5-fold CV; fit a single model with the args.* hyperparameters.",
    )
    main(parser.parse_args())
