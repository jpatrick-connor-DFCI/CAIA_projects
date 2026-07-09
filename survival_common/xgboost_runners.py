"""Shared CLI and orchestration for XGBoost survival:cox analyses.

Mirrors ``survival_common.cox_runners`` for the tree arm: this module owns the
per-landmark loop, CV tuning, final fit, and output writing, parameterized by
``(config, cox, args)`` where ``cox`` is the project's ``cox_aggregated`` module
and ``config`` is its :class:`~survival_common.config.CoxProjectConfig`.

The low-level XGBoost mechanics (preprocessing, DMatrix construction, Breslow
survival, importances) live in ``survival_common.xgboost_engine`` and are reused
here rather than reimplemented. Project-specific pieces (static covariates, the
baseline covariate wiring, and the feature-selection / patient-risk output
schemas) are supplied through the config hooks.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from survival_common.config import CoxProjectConfig
from survival_common.helper import (
    assert_disjoint_folds,
    assert_no_test_leakage,
    compute_brier,
    iter_stratified_folds,
    select_canonical_labs,
)
from survival_common.xgboost_engine import (
    best_iteration,
    chosen_from_best_row,
    feature_importance_frame,
    fit_xgb_cox,
    predict_risk,
    require_xgboost,
    truncate_features_by_rank,
    xgb_survival_at_horizons,
)

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


def require_lifelines() -> None:
    if concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to compute concordance indices."
        ) from LIFELINES_IMPORT_ERROR


# --------------------------------------------------------------------------
# Default output-schema builders (IPIO conventions). COMPASS overrides these
# via config to keep its historically-wider feature-selection / risk schemas.
# --------------------------------------------------------------------------

def _default_feature_meta_frame(
    ctx: Any,
    endpoint: str,
    landmark_day: int,
    selected_features: list[str],
    feature_meta: pd.DataFrame,
) -> pd.DataFrame:
    """IPIO default: trim ctx.feature_meta_selected and re-key by landmark_day."""
    frame = feature_meta.drop(columns=["landmark_days"], errors="ignore").copy()
    frame.insert(0, "endpoint", endpoint)
    frame.insert(0, "landmark_day", landmark_day)
    return frame


def _default_risk_frame(
    test: pd.DataFrame,
    endpoint: str,
    landmark_day: int,
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    id_col: str,
) -> pd.DataFrame:
    """IPIO default: dataset-tagged risk frame (landmark_day inserted later)."""
    return pd.DataFrame(
        {
            id_col: test.index,
            "endpoint": endpoint,
            "dataset": "test",
            "duration": duration,
            "event": event,
            "risk_score": risk,
        }
    )


# --------------------------------------------------------------------------
# CV tuning + final fit (shared; formerly COMPASS cv_one_endpoint /
# run_one_endpoint and IPIO tune_xgboost_model / fit_final_xgboost_model).
# --------------------------------------------------------------------------

def tune_xgboost_model(
    cox: Any,
    train_val: pd.DataFrame,
    *,
    raw_feature_cols: list[str],
    endpoint: str,
    pre_treatment_lab_df: pd.DataFrame,
    horizon_grid: np.ndarray,
    min_patient_coverage: float,
    static_covariate_cols: tuple[str, ...] = (),
    always_include_feature_cols: tuple[str, ...] = (),
    args: Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """5-fold stratified CV over (max_depth x eta x min_child_weight).

    Strict no-leakage: per fold, canonical labs and per-stat feature selection
    are recomputed from fold_train MRNs only. Early stopping uses the held-out
    fold (fold_val) as the watch — the same set used for metric reporting, the
    standard CV-with-early-stopping convention.

    Returns (fold_df, cv_df, best_row, fold_canonical_labs_df). Rows carry
    'endpoint' but no landmark-day/feature-set label; the caller inserts that
    after the fact (mirrors tune_multivariable_model's contract).
    """
    require_xgboost()
    id_col = cox.ID_COL
    age_col = cox.AGE_COL
    endpoint_map = cox.ENDPOINTS
    duration_col = endpoint_map[endpoint]["duration_col"]
    event_col = endpoint_map[endpoint]["event_col"]
    max_features = getattr(args, "max_features", None)

    fold_partitions = list(
        iter_stratified_folds(
            train_val,
            n_folds=args.n_folds,
            seed=args.seed,
            event_col=event_col,
        )
    )
    if not fold_partitions:
        raise RuntimeError(f"No CV folds produced for endpoint '{endpoint}'.")
    cv_stratification = fold_partitions[0][3]

    # Materialize fold-level canonical labs and selected features once so the
    # full grid traverses identical fold partitions.
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
            min_coverage=min_patient_coverage,
            id_col=id_col,
        )
        fold_canonical_labs[fold] = canonical
        fold_train = train_val.iloc[tr_idx]
        selected, fold_feature_meta = cox.select_feature_columns(
            fold_train,
            raw_feature_cols,
            min_patient_coverage=min_patient_coverage,
            restrict_to_labs=canonical,
            always_include=list(always_include_feature_cols),
        )
        selected = truncate_features_by_rank(selected, fold_feature_meta, max_features)
        fold_selected_features[fold] = selected
        for lab in canonical:
            fold_canonical_labs_rows.append(
                {"endpoint": endpoint, "fold": fold, "lab_name": lab}
            )

    grid = list(product(args.cv_max_depths, args.cv_etas, args.cv_min_child_weights))
    fold_rows: list[dict] = []
    total_runs = len(grid) * len(fold_partitions)
    cv_bar = tqdm(total=total_runs, desc=f"xgb CV[{endpoint}]", dynamic_ncols=True)
    for max_depth, eta, min_child_weight in grid:
        for fold, tr_idx, val_idx, _ in fold_partitions:
            fold_train = train_val.iloc[tr_idx]
            fold_val = train_val.iloc[val_idx]
            fold_features = fold_selected_features[fold]
            row = {
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
            try:
                if not fold_features and not static_covariate_cols:
                    raise ValueError("no usable features after fold-level selection")
                model, _, _, preprocessor = fit_xgb_cox(
                    fold_train,
                    fold_val,
                    feature_cols=fold_features,
                    duration_col=duration_col,
                    event_col=event_col,
                    args=args,
                    age_col=age_col,
                    static_cols=static_covariate_cols,
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
                        concordance_index(duration[valid], -risk[valid], event[valid])
                    )
                mean_auc_val, auc_df_val = cox.compute_ipcw_auc_t(
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
                        "d": int(max_depth),
                        "eta": f"{float(eta):g}",
                        "mcw": f"{float(min_child_weight):g}",
                        "fold": fold,
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

    fold_df = pd.DataFrame(fold_rows)
    cv_df = (
        fold_df.groupby(
            ["endpoint", "max_depth", "eta", "min_child_weight"],
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
        notes = (
            fold_df.loc[fold_df["note"].astype(str).str.len() > 0, "note"]
            .astype(str)
            .value_counts()
            .head(5)
        )
        notes_str = (
            "\n  ".join(f"({n}x) {msg}" for msg, n in notes.items())
            if not notes.empty
            else "(no per-fold notes captured; predict_risk likely returned NaN)"
        )
        raise RuntimeError(
            f"All XGBoost CV fits failed for endpoint '{endpoint}'.\n"
            f"Top fold-failure notes:\n  {notes_str}"
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


def fit_final_xgboost_model(
    cox: Any,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    chosen: dict | None,
    static_covariate_cols: tuple[str, ...] = (),
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    horizon_grid: np.ndarray,
    canonical_labs: list[str],
    args: Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Refit on full train_val and evaluate on the held-out test fold.

    ``chosen`` is the dict from :func:`chosen_from_best_row` (CV-selected
    max_depth/eta/min_child_weight and round count), or ``None`` to fit with
    the ``args.*`` defaults directly (the --baseline / --no-cv path).

    Returns (metrics_df, importance_df, duration, event, risk, auc_t, brier_t).
    The raw (duration, event, risk) arrays are returned so the caller can build
    the project-specific patient-risk frame.
    """
    require_lifelines()
    id_col = cox.ID_COL
    age_col = cox.AGE_COL
    endpoint_map = cox.ENDPOINTS
    duration_col = endpoint_map[endpoint]["duration_col"]
    event_col = endpoint_map[endpoint]["event_col"]

    assert_no_test_leakage(
        test_mrns=test.index,
        train_mrns=train_val.index,
        context=f"landmark_xgboost.fit_final_xgboost_model[{endpoint}]",
    )

    final_eta = chosen["eta"] if chosen is not None else None
    final_max_depth = chosen["max_depth"] if chosen is not None else None
    final_min_child_weight = chosen["min_child_weight"] if chosen is not None else None
    final_num_boost_round = int(
        (chosen.get("selected_num_boost_round") if chosen is not None else None)
        or args.num_boost_round
    )
    model, _, params, preprocessor = fit_xgb_cox(
        train_val,
        train_val.iloc[0:0].copy(),
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        args=args,
        age_col=age_col,
        static_cols=static_covariate_cols,
        eta=final_eta,
        max_depth=final_max_depth,
        min_child_weight=final_min_child_weight,
        num_boost_round=final_num_boost_round,
    )
    risk, covariate_cols = predict_risk(model, test, preprocessor=preprocessor)
    event = test[event_col].astype(int).to_numpy()
    duration = test[duration_col].astype(float).to_numpy()
    valid = np.isfinite(duration) & (duration > 0) & np.isfinite(risk)
    c_index = np.nan
    if valid.sum() and event[valid].sum():
        c_index = float(concordance_index(duration[valid], -risk[valid], event[valid]))

    mean_auc, auc_t = cox.compute_ipcw_auc_t(
        test,
        risk,
        duration_col=duration_col,
        event_col=event_col,
        reference_df=train_val,
        time_unit_days=auc_time_unit_days,
        quantiles=tuple(args.auc_quantiles),
        max_time_unit=auc_max_time_units,
        fixed_horizons=horizon_grid,
    )
    auc_t = auc_t.copy()
    if not auc_t.empty:
        auc_t.insert(0, "endpoint", endpoint)

    test_surv = xgb_survival_at_horizons(
        model,
        train_val,
        test,
        preprocessor=preprocessor,
        duration_col=duration_col,
        event_col=event_col,
        horizons=horizon_grid,
        time_unit_days=auc_time_unit_days,
    )
    train_dur_units = np.ceil(
        pd.to_numeric(train_val[duration_col], errors="coerce").to_numpy(dtype=float)
        / float(auc_time_unit_days)
    )
    test_dur_units = np.ceil(
        pd.to_numeric(test[duration_col], errors="coerce").to_numpy(dtype=float)
        / float(auc_time_unit_days)
    )
    brier_t, integrated_brier = compute_brier(
        train_event=train_val[event_col].to_numpy(dtype=int),
        train_duration=train_dur_units,
        eval_event=test[event_col].to_numpy(dtype=int),
        eval_duration=test_dur_units,
        surv_at_horizons=test_surv,
        horizons=horizon_grid,
        time_unit_days=auc_time_unit_days,
    )
    brier_t = brier_t.copy()
    if not brier_t.empty:
        brier_t.insert(0, "endpoint", endpoint)

    metrics_row = {
        "endpoint": endpoint,
        "n_train_val": len(train_val),
        "n_test": len(test),
        "n_train_val_events": int(train_val[event_col].sum()),
        "n_test_events": int(test[event_col].sum()),
        "n_canonical_labs": len(canonical_labs),
        "n_selected_features": len(feature_cols),
        "n_covariates_with_missing_indicators": len(covariate_cols),
        "final_num_boost_round": final_num_boost_round,
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
                "selected_num_boost_round": final_num_boost_round,
                "cv_mean_c_index": chosen["cv_mean_c_index"],
                "cv_mean_auc_t": chosen["cv_mean_auc_t"],
                "cv_mean_integrated_brier": chosen["cv_mean_integrated_brier"],
                "cv_stratification": chosen["cv_stratification"],
            }
        )
    metrics = pd.DataFrame([metrics_row])

    importance = feature_importance_frame(
        model,
        covariate_cols=covariate_cols,
        xgb_feature_names=preprocessor["xgb_feature_names"],
        endpoint=endpoint,
    )
    return metrics, importance, duration, event, risk, auc_t, brier_t


# --------------------------------------------------------------------------
# Per-landmark arms.
# --------------------------------------------------------------------------

def _run_multivariable_landmark(
    config: CoxProjectConfig,
    cox: Any,
    ctx: Any,
    endpoint_horizon_grids: dict[str, Any],
    *,
    landmark_day: int,
    endpoints: list[str],
    args: Namespace,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    min_patient_coverage: float,
    out: dict[str, list],
) -> None:
    """Baseline+labs XGBoost arm for one landmark; appends to ``out`` lists."""
    id_col = cox.ID_COL
    static_covariate_cols = config.static_covariates(ctx, args, cox)
    feature_meta_frame = config.xgb_feature_meta_frame or _default_feature_meta_frame
    risk_frame = config.xgb_risk_frame or _default_risk_frame
    print("\n##### XGBOOST ARM: BASELINE+LABS (survival:cox, all endpoints) #####")
    if static_covariate_cols:
        print(f"  always-included covariates: age + {', '.join(static_covariate_cols)}")
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(cox.ENDPOINTS[endpoint]["description"])
        horizon_grid = endpoint_horizon_grids[endpoint]

        chosen = None
        fold_df = pd.DataFrame()
        cv_df = pd.DataFrame()
        fold_canonical_labs_df = pd.DataFrame()
        if not args.no_cv:
            fold_df, cv_df, best_row, fold_canonical_labs_df = tune_xgboost_model(
                cox,
                ctx.train_val,
                raw_feature_cols=ctx.raw_feature_cols,
                endpoint=endpoint,
                pre_treatment_lab_df=ctx.pre_treatment_lab_df,
                horizon_grid=horizon_grid,
                min_patient_coverage=min_patient_coverage,
                static_covariate_cols=static_covariate_cols,
                always_include_feature_cols=tuple(
                    getattr(ctx, "always_include_feature_cols", ())
                ),
                args=args,
            )
            chosen = chosen_from_best_row(best_row)
            print(
                f"  CV chose max_depth={chosen['max_depth']} eta={chosen['eta']:g} "
                f"min_child_weight={chosen['min_child_weight']:g}; "
                f"cv C-index={chosen['cv_mean_c_index']:.4f} "
                f"AUC(t)={chosen['cv_mean_auc_t']:.4f} "
                f"IBS={chosen['cv_mean_integrated_brier']:.4f}"
            )

        # Final feature set: reuse ctx.selected_feature_cols (computed on
        # train_val with the canonical labs in scope by prepare_landmark_context),
        # optionally truncated by rank.
        selected_features = list(ctx.selected_feature_cols)
        feature_meta = ctx.feature_meta_selected
        if args.max_features is not None and len(selected_features) > args.max_features:
            selected_features = truncate_features_by_rank(
                selected_features, feature_meta, args.max_features
            )
            feature_meta = feature_meta.loc[feature_meta["feature"].isin(selected_features)]

        metrics, importance, duration, event, risk, auc_t, brier_t = fit_final_xgboost_model(
            cox,
            ctx.train_val,
            ctx.test,
            feature_cols=selected_features,
            endpoint=endpoint,
            chosen=chosen,
            static_covariate_cols=static_covariate_cols,
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grid=horizon_grid,
            canonical_labs=ctx.canonical_labs,
            args=args,
        )
        predictions = risk_frame(
            ctx.test, endpoint, landmark_day, duration, event, risk, id_col
        )

        for df in (metrics, importance, predictions, auc_t, brier_t, fold_df, cv_df, fold_canonical_labs_df):
            if not df.empty:
                df.insert(0, "landmark_day", landmark_day)

        feature_meta_out = feature_meta_frame(
            ctx, endpoint, landmark_day, selected_features, feature_meta
        )

        out["metrics"].append(metrics)
        out["auc_t"].append(auc_t)
        if not brier_t.empty:
            out["brier"].append(brier_t)
        out["risks"].append(predictions)
        out["features"].append(feature_meta_out)
        out["importance"].append(importance)
        if not fold_df.empty:
            out["cv_folds"].append(fold_df)
        if not cv_df.empty:
            out["cv_summary"].append(cv_df)
        if not fold_canonical_labs_df.empty:
            out["canonical_labs_folds"].append(fold_canonical_labs_df)

        metrics_row = metrics.iloc[0]
        print(f"  held-out test C-index={metrics_row['c_index']:.4f}")
        print(f"  held-out test mean AUC(t)={metrics_row['mean_auc_t']:.4f}")
        print(f"  held-out test integrated Brier={metrics_row['integrated_brier']:.4f}")


def _run_baseline_landmark(
    config: CoxProjectConfig,
    cox: Any,
    ctx: Any,
    endpoint_horizon_grids: dict[str, Any],
    *,
    landmark_day: int,
    endpoints: list[str],
    args: Namespace,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    out: dict[str, list],
) -> None:
    """Age(+baseline-covariate)-only XGBoost arm for one landmark.

    No lab features, no CV / feature selection. Reuses fit_final_xgboost_model
    with chosen=None so it fits with the args.* hyperparameter defaults, on the
    identical horizon grid as the labs model.

    Two project wirings, selected by ``config.xgb_baseline_as_features``:
      * feature path (COMPASS): baseline_feature_cols go through the tree
        feature path (feature_cols=..., static_cols=()); missing indicators and
        scaling apply.
      * static path (IPIO): feature_cols=[], static_cols=static_covariates.
    """
    id_col = cox.ID_COL
    risk_frame = config.xgb_risk_frame or _default_risk_frame
    if config.xgb_baseline_as_features:
        feature_cols = list(config.baseline_feature_cols(ctx, args, cox))
        static_covariate_cols: tuple[str, ...] = ()
        baseline_cols = feature_cols
    else:
        feature_cols = []
        static_covariate_cols = config.static_covariates(ctx, args, cox)
        baseline_cols = list(static_covariate_cols)

    print("\n##### XGBOOST ARM: AGE(+BASELINE COVARIATES)-ONLY (all endpoints) #####")
    print(
        "  covariates: age"
        + (f" + {', '.join(baseline_cols)}" if baseline_cols else " (no baseline covariates found)")
    )
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(cox.ENDPOINTS[endpoint]["description"])
        horizon_grid = endpoint_horizon_grids[endpoint]

        metrics, importance, duration, event, risk, auc_t, brier_t = fit_final_xgboost_model(
            cox,
            ctx.train_val,
            ctx.test,
            feature_cols=feature_cols,
            endpoint=endpoint,
            chosen=None,
            static_covariate_cols=static_covariate_cols,
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grid=horizon_grid,
            canonical_labs=[],
            args=args,
        )
        predictions = risk_frame(
            ctx.test, endpoint, landmark_day, duration, event, risk, id_col
        )
        for df in (metrics, importance, predictions, auc_t, brier_t):
            if not df.empty:
                df.insert(0, "landmark_day", landmark_day)

        feature_meta = pd.DataFrame(
            {
                "landmark_day": landmark_day,
                "endpoint": endpoint,
                "feature": list(baseline_cols),
                "lab_name": list(baseline_cols),
                "feature_stat": config.xgb_baseline_feature_stat,
            }
        )

        out["metrics"].append(metrics)
        out["auc_t"].append(auc_t)
        if not brier_t.empty:
            out["brier"].append(brier_t)
        out["risks"].append(predictions)
        out["features"].append(feature_meta)
        out["importance"].append(importance)

        metrics_row = metrics.iloc[0]
        print(f"  held-out test C-index={metrics_row['c_index']:.4f}")
        print(f"  held-out test mean AUC(t)={metrics_row['mean_auc_t']:.4f}")
        print(f"  held-out test integrated Brier={metrics_row['integrated_brier']:.4f}")


# --------------------------------------------------------------------------
# CLI orchestration.
# --------------------------------------------------------------------------

def run_xgboost(config: CoxProjectConfig, cox: Any, args: Namespace) -> None:
    cox.ID_COL = args.id_col
    cox.AGE_COL = args.age_col
    require_xgboost()
    require_lifelines()
    endpoints = cox.normalize_endpoints(args.endpoints)
    landmark_days = cox.normalize_landmark_days(args.landmark_days)
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(
            f"Inputs dir {inputs_dir} not found. Run build_prediction_inputs.py first."
        )
    build_manifest = cox._load_build_manifest(inputs_dir)
    min_patient_coverage = float(build_manifest["min_patient_coverage"])
    args.auc_time_unit_days = int(build_manifest["auc_time_unit_days"])
    args.auc_quantiles = tuple(build_manifest["auc_quantiles"])
    auc_max_time_units = (
        args.auc_max_time_units
        if args.auc_max_time_units is not None
        else cox.DEFAULT_AUC_MAX_TIME_UNITS
    )
    args.auc_max_time_units = auc_max_time_units
    auc_horizons_by_landmark = build_manifest["auc_horizons_by_landmark"]
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={min_patient_coverage}, "
        f"auc_time_unit_days={args.auc_time_unit_days} per build manifest)"
    )

    out = {
        "metrics": [],
        "auc_t": [],
        "brier": [],
        "risks": [],
        "features": [],
        "importance": [],
        "cv_folds": [],
        "cv_summary": [],
        "canonical_labs_folds": [],
    }

    for landmark_day in landmark_days:
        ctx = cox.prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
            **config.prepare_context_kwargs(args),
        )
        endpoint_horizon_grids, _horizon_grid_df = cox.build_endpoint_horizon_grids(
            landmark_day,
            endpoints=endpoints,
            auc_horizons_by_landmark=auc_horizons_by_landmark,
            auc_quantiles=args.auc_quantiles,
            auc_time_unit_days=args.auc_time_unit_days,
        )

        if args.baseline:
            _run_baseline_landmark(
                config,
                cox,
                ctx,
                endpoint_horizon_grids,
                landmark_day=landmark_day,
                endpoints=endpoints,
                args=args,
                auc_time_unit_days=args.auc_time_unit_days,
                auc_max_time_units=auc_max_time_units,
                out=out,
            )
        else:
            _run_multivariable_landmark(
                config,
                cox,
                ctx,
                endpoint_horizon_grids,
                landmark_day=landmark_day,
                endpoints=endpoints,
                args=args,
                auc_time_unit_days=args.auc_time_unit_days,
                auc_max_time_units=auc_max_time_units,
                min_patient_coverage=min_patient_coverage,
                out=out,
            )

    _write_xgboost_outputs(args, out)


def _write_xgboost_outputs(args: Namespace, out: dict[str, list]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_baseline" if args.baseline else ""
    metrics_path = output_dir / f"landmark_xgboost{suffix}_metrics.csv"
    auc_path = output_dir / f"landmark_xgboost{suffix}_auc_t.csv"
    brier_path = output_dir / f"landmark_xgboost{suffix}_brier.csv"
    risks_path = output_dir / f"landmark_xgboost{suffix}_patient_risks.csv"
    features_path = output_dir / f"landmark_xgboost{suffix}_feature_selection.csv"
    importance_path = output_dir / f"landmark_xgboost{suffix}_feature_importance.csv"
    cv_folds_path = output_dir / f"landmark_xgboost{suffix}_cv_folds.csv"
    cv_summary_path = output_dir / f"landmark_xgboost{suffix}_cv_summary.csv"
    fold_labs_path = output_dir / f"landmark_xgboost{suffix}_canonical_labs_folds.csv"

    saved = []
    if out["metrics"]:
        pd.concat(out["metrics"], ignore_index=True).to_csv(metrics_path, index=False)
        saved.append(metrics_path)
    if out["auc_t"]:
        pd.concat(out["auc_t"], ignore_index=True).to_csv(auc_path, index=False)
        saved.append(auc_path)
    if out["risks"]:
        pd.concat(out["risks"], ignore_index=True).to_csv(risks_path, index=False)
        saved.append(risks_path)
    if out["features"]:
        pd.concat(out["features"], ignore_index=True, sort=False).to_csv(features_path, index=False)
        saved.append(features_path)
    if out["importance"]:
        pd.concat(out["importance"], ignore_index=True, sort=False).to_csv(importance_path, index=False)
        saved.append(importance_path)
    if out["brier"]:
        pd.concat(out["brier"], ignore_index=True).to_csv(brier_path, index=False)
        saved.append(brier_path)
    if out["cv_folds"]:
        pd.concat(out["cv_folds"], ignore_index=True).to_csv(cv_folds_path, index=False)
        saved.append(cv_folds_path)
    if out["cv_summary"]:
        pd.concat(out["cv_summary"], ignore_index=True).to_csv(cv_summary_path, index=False)
        saved.append(cv_summary_path)
    if out["canonical_labs_folds"]:
        pd.concat(out["canonical_labs_folds"], ignore_index=True).to_csv(fold_labs_path, index=False)
        saved.append(fold_labs_path)
    print("\nSaved:")
    for path in saved:
        print(f"  {path}")
