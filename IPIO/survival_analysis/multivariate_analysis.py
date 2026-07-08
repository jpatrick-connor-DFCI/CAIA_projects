"""
Multivariate IPIO survival analysis.

Runs either elastic-net Cox (`--model elastic-net`) via the shared Cox runner or
XGBoost survival:cox (`--model xgboost`) via the XGBoost functions in this file.
The XGBoost path is adapted to IPIO's schema/endpoint/covariates and shares the
same landmark cohort / canonical-lab / feature-selection setup through
cox_aggregated.prepare_landmark_context.

For XGBoost, two modes mirror the elastic-net baseline flag:
  default (baseline+labs): 5-fold stratified CV on train_val over a
    (max_depth x eta x min_child_weight) grid to pick hyperparameters, then
    fit the chosen configuration on the full train_val with the CV-selected
    boosting round count and evaluate on the held-out test fold. Per-fold
    canonical labs and the fixed horizon grid keep test data fully isolated
    from selection.
  --baseline: age(+baseline-covariate)-only model (no lab features, no CV)
    on the same horizon grid for benchmarking against the lab model.

Baseline covariates (GENDER_MALE, pd1pdl1, ctla4, CANCER_TYPE_*, from
cox_aggregated.baseline_covariate_columns) are always included (unconditionally,
not gated behind a CLI flag) as XGBoost input features -- this is IPIO's
replacement for COMPASS's opt-in `--with-gleason` covariate mechanism.

Outputs (under --output-dir; `_baseline` suffix in --baseline mode):
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

# .../IPIO/survival_analysis -- import sibling IPIO modules (cox_aggregated, etc.).
SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from _paths import ensure_survival_common_on_path  # noqa: E402

# IPIO reuses the repo-level shared package for generic survival-analysis
# mechanics. Resolve the repo root before importing survival_common.* below.
ensure_survival_common_on_path()

import cox_aggregated as _ca  # noqa: E402
from cox_aggregated import (  # noqa: E402
    AGE_COL,
    DEFAULT_AUC_MAX_TIME_UNITS,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    ENDPOINTS,
    ID_COL,
    RESULTS,
    _load_build_manifest,
    baseline_covariate_columns,
    build_endpoint_horizon_grids,
    compute_ipcw_auc_t,
    normalize_endpoints,
    normalize_landmark_days,
    prepare_landmark_context,
    select_feature_columns,
)
from survival_common.cox_runners import run_multivariable  # noqa: E402
from survival_common.helper import (  # noqa: E402
    assert_disjoint_folds,
    assert_no_test_leakage,
    compute_brier,
    iter_stratified_folds,
    select_canonical_labs,
)
from survival_common.xgboost_engine import (  # noqa: E402
    best_iteration,
    chosen_from_best_row,
    feature_importance_frame,
    fit_xgb_cox,
    predict_risk,
    require_xgboost,
    truncate_features_by_rank as _truncate_features_by_rank,
    xgb_survival_at_horizons,
)
from survival_common.projects.ipio import CONFIG  # noqa: E402

DEFAULT_CV_MAX_DEPTHS = [2, 3, 4, 5, 6]
DEFAULT_CV_ETAS = [0.01, 0.03, 0.05, 0.10, 0.15]
DEFAULT_CV_MIN_CHILD_WEIGHTS = [1.0, 3.0, 5.0, 10.0, 15.0]


def require_lifelines() -> None:
    if concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to compute concordance indices."
        ) from LIFELINES_IMPORT_ERROR


def tune_xgboost_model(
    train_val: pd.DataFrame,
    *,
    raw_feature_cols: list[str],
    endpoint: str,
    pre_treatment_lab_df: pd.DataFrame,
    horizon_grid: np.ndarray,
    min_patient_coverage: float,
    static_covariate_cols: tuple[str, ...] = (),
    always_include_feature_cols: tuple[str, ...] = (),
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """5-fold stratified CV over (max_depth x eta x min_child_weight).

    Mirrors cox_aggregated.tune_multivariable_model's per-fold no-leakage
    pattern (canonical labs + per-stat feature selection recomputed from
    fold_train MRNs only), adapted for XGBoost's hyperparameter grid. Early
    stopping uses the held-out fold (fold_val) as the watch — same set used
    for metric reporting, the standard CV-with-early-stopping convention.

    `always_include_feature_cols` (e.g. non-lab baseline/genomic indicators)
    are exempt from the per-fold canonical-lab restriction -- see
    cox_aggregated.select_feature_columns's `always_include` parameter.

    Returns (fold_df, cv_df, best_row, fold_canonical_labs_df). Rows carry
    'endpoint' but not a landmark-day/feature-set label; the caller inserts
    that after the fact (mirrors tune_multivariable_model's contract).
    """
    require_xgboost()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
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
    # grid traverses identical fold partitions.
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
            id_col=ID_COL,
        )
        fold_canonical_labs[fold] = canonical
        fold_train = train_val.iloc[tr_idx]
        selected, fold_feature_meta = select_feature_columns(
            fold_train,
            raw_feature_cols,
            min_patient_coverage=min_patient_coverage,
            restrict_to_labs=canonical,
            always_include=list(always_include_feature_cols),
        )
        selected = _truncate_features_by_rank(selected, fold_feature_meta, max_features)
        fold_selected_features[fold] = selected
        for lab in canonical:
            fold_canonical_labs_rows.append(
                {"endpoint": endpoint, "fold": fold, "lab_name": lab}
            )

    grid = list(
        product(args.cv_max_depths, args.cv_etas, args.cv_min_child_weights)
    )
    fold_rows: list[dict] = []
    total_runs = len(grid) * len(fold_partitions)
    cv_bar = tqdm(
        total=total_runs,
        desc=f"xgb CV[{endpoint}]",
        dynamic_ncols=True,
    )
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
                    age_col=AGE_COL,
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
                        concordance_index(
                            duration[valid], -risk[valid], event[valid]
                        )
                    )
                mean_auc_val, auc_df_val = compute_ipcw_auc_t(
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
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit on full train_val and evaluate on the held-out test fold.

    `chosen` is the dict returned by chosen_from_best_row (CV-selected
    max_depth/eta/min_child_weight and round count), or None to fit with the
    args.* defaults directly (the --baseline / --no-cv path — mirrors
    cox_aggregated.fit_final_multivariable_model's un-tuned baseline usage).

    Mirrors fit_final_multivariable_model's contract: returns
    (metrics_row_df, importance_df, predictions_df, test_auc_df, test_brier_df).
    """
    require_lifelines()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

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
        age_col=AGE_COL,
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

    mean_auc, auc_t = compute_ipcw_auc_t(
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

    predictions = pd.DataFrame(
        {
            ID_COL: test.index,
            "endpoint": endpoint,
            "dataset": "test",
            "duration": duration,
            "event": event,
            "risk_score": risk,
        }
    )
    importance = feature_importance_frame(
        model,
        covariate_cols=covariate_cols,
        xgb_feature_names=preprocessor["xgb_feature_names"],
        endpoint=endpoint,
    )
    return metrics, importance, predictions, auc_t, brier_t


def _run_multivariable_landmark(
    ctx,
    endpoint_horizon_grids,
    *,
    landmark_day,
    endpoints,
    args,
    auc_time_unit_days,
    auc_max_time_units,
    min_patient_coverage,
    out,
):
    """Baseline+labs XGBoost arm for one landmark; appends to ``out`` lists."""
    static_covariate_cols = tuple(baseline_covariate_columns(ctx.merged))
    print("\n##### XGBOOST ARM: BASELINE+LABS (survival:cox, all endpoints) #####")
    if static_covariate_cols:
        print(f"  always-included covariates: age + {', '.join(static_covariate_cols)}")
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(ENDPOINTS[endpoint]["description"])
        horizon_grid = endpoint_horizon_grids[endpoint]

        chosen = None
        fold_df = pd.DataFrame()
        cv_df = pd.DataFrame()
        fold_canonical_labs_df = pd.DataFrame()
        if not args.no_cv:
            fold_df, cv_df, best_row, fold_canonical_labs_df = tune_xgboost_model(
                ctx.train_val,
                raw_feature_cols=ctx.raw_feature_cols,
                endpoint=endpoint,
                pre_treatment_lab_df=ctx.pre_treatment_lab_df,
                horizon_grid=horizon_grid,
                min_patient_coverage=min_patient_coverage,
                static_covariate_cols=static_covariate_cols,
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

        # Final feature set: reuse ctx.selected_feature_cols (already computed
        # on train_val with the canonical labs in scope by
        # prepare_landmark_context), optionally truncated by rank.
        selected_features = list(ctx.selected_feature_cols)
        feature_meta = ctx.feature_meta_selected
        if args.max_features is not None and len(selected_features) > args.max_features:
            selected_features = _truncate_features_by_rank(
                selected_features, feature_meta, args.max_features
            )
            feature_meta = feature_meta.loc[feature_meta["feature"].isin(selected_features)]

        metrics, importance, predictions, auc_t, brier_t = fit_final_xgboost_model(
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

        for df in (metrics, importance, predictions, auc_t, brier_t, fold_df, cv_df, fold_canonical_labs_df):
            if not df.empty:
                df.insert(0, "landmark_day", landmark_day)

        # ctx.feature_meta_selected already carries its own "landmark_days"
        # (plural) column from cox_aggregated.prepare_landmark_context; drop it
        # here and use "landmark_day"/"endpoint" for consistency with the rest
        # of this script's output schema (matches COMPASS's XGBoost path).
        feature_meta = feature_meta.drop(columns=["landmark_days"], errors="ignore").copy()
        feature_meta.insert(0, "endpoint", endpoint)
        feature_meta.insert(0, "landmark_day", landmark_day)

        out["metrics"].append(metrics)
        out["auc_t"].append(auc_t)
        if not brier_t.empty:
            out["brier"].append(brier_t)
        out["risks"].append(predictions)
        out["features"].append(feature_meta)
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
    ctx,
    endpoint_horizon_grids,
    *,
    landmark_day,
    endpoints,
    args,
    auc_time_unit_days,
    auc_max_time_units,
    out,
):
    """Age(+baseline-covariate)-only XGBoost arm for one landmark; appends to ``out``.

    No lab features, no CV / feature selection (mirrors
    multivariate_analysis.py --model elastic-net --baseline). Reuses fit_final_xgboost_model
    with chosen=None so it fits with the args.* hyperparameter defaults
    directly, on the identical horizon grid as the labs model.
    """
    static_covariate_cols = tuple(baseline_covariate_columns(ctx.merged))
    print("\n##### XGBOOST ARM: AGE(+BASELINE COVARIATES)-ONLY (all endpoints) #####")
    print(
        "  covariates: age"
        + (f" + {', '.join(static_covariate_cols)}" if static_covariate_cols else " (no baseline covariates found)")
    )
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(ENDPOINTS[endpoint]["description"])
        horizon_grid = endpoint_horizon_grids[endpoint]

        metrics, importance, predictions, auc_t, brier_t = fit_final_xgboost_model(
            ctx.train_val,
            ctx.test,
            feature_cols=[],
            endpoint=endpoint,
            chosen=None,
            static_covariate_cols=static_covariate_cols,
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grid=horizon_grid,
            canonical_labs=[],
            args=args,
        )
        for df in (metrics, importance, predictions, auc_t, brier_t):
            if not df.empty:
                df.insert(0, "landmark_day", landmark_day)

        feature_meta = pd.DataFrame(
            {
                "landmark_day": landmark_day,
                "endpoint": endpoint,
                "feature": list(static_covariate_cols),
                "lab_name": list(static_covariate_cols),
                "feature_stat": "static",
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


def run_xgboost(args: argparse.Namespace) -> None:
    global ID_COL, AGE_COL
    ID_COL = args.id_col
    AGE_COL = args.age_col
    _ca.ID_COL = ID_COL
    _ca.AGE_COL = AGE_COL
    require_xgboost()
    require_lifelines()
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
    auc_max_time_units = (
        args.auc_max_time_units
        if args.auc_max_time_units is not None
        else DEFAULT_AUC_MAX_TIME_UNITS
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
        ctx = prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
        )
        endpoint_horizon_grids, _horizon_grid_df = build_endpoint_horizon_grids(
            landmark_day,
            endpoints=endpoints,
            auc_horizons_by_landmark=auc_horizons_by_landmark,
            auc_quantiles=args.auc_quantiles,
            auc_time_unit_days=args.auc_time_unit_days,
        )

        if args.baseline:
            _run_baseline_landmark(
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


def main(args: argparse.Namespace) -> None:
    if args.model == "elastic-net":
        run_multivariable(CONFIG, _ca, args)
    elif args.model == "xgboost":
        run_xgboost(args)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported multivariate model: {args.model!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "IPIO multivariate survival analysis: elastic-net Cox or XGBoost survival:cox."
        )
    )
    parser.add_argument(
        "--model",
        choices=["elastic-net", "xgboost"],
        default="elastic-net",
        help="Multivariate model family to run.",
    )
    parser.add_argument("--id-col", default=ID_COL,
                        help="Patient identifier column name (default DFCI_MRN).")
    parser.add_argument("--age-col", default=AGE_COL,
                        help="Age covariate column name (default AGE_AT_TREATMENTSTART).")
    parser.add_argument(
        "--inputs-dir",
        default=str(RESULTS / "prediction_inputs"),
        help="Directory containing prebuilt inputs from build_prediction_inputs.py.",
    )
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["irae"],
        choices=list(ENDPOINTS),
        help="Endpoints to analyze.",
    )
    parser.add_argument("--landmark-days", nargs="+", type=int, default=DEFAULT_LANDMARK_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Fit an age(+baseline-covariates)-only XGBoost model: skip lab feature "
            "selection and CV; covariates = age + baseline covariates "
            "(GENDER_MALE, pd1pdl1, ctla4, CANCER_TYPE_*). Writes "
            "landmark_xgboost_baseline_*.csv on the same horizon grid."
        ),
    )
    parser.add_argument(
        "--auc-max-time-units",
        type=int,
        default=None,
        help=(
            f"Cap (in time-units) for the IPCW AUC(t)/Brier evaluation horizons "
            f"(default {DEFAULT_AUC_MAX_TIME_UNITS}). Caps evaluation only, not fitting."
        ),
    )
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
    parser.add_argument(
        "--verbose-eval",
        type=int,
        default=0,
        help="Print xgboost's per-round eval every N rounds (0 = silent, the default).",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument(
        "--cv-penalizers",
        nargs="+",
        type=float,
        default=list(_ca.DEFAULT_CV_PENALIZERS),
        help="Elastic-net Cox penalizer values searched during cross-validation.",
    )
    parser.add_argument(
        "--cv-l1-ratios",
        nargs="+",
        type=float,
        default=list(_ca.DEFAULT_CV_L1_RATIOS),
        help="Elastic-net Cox L1 mixing values searched during cross-validation.",
    )
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
