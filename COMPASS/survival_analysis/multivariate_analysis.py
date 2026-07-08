"""
Multivariate PROFILE survival analysis.

Runs either elastic-net Cox (`--model elastic-net`) via the shared Cox runner or
XGBoost survival:cox (`--model xgboost`) via the XGBoost functions in this file.
Both paths reuse cox_aggregated.py's landmarked feature engineering, held-out
split, canonical labs, and fixed horizon grid.

XGBoost outputs:
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

SURVIVAL_DIR = Path(__file__).resolve().parent
SURVIVAL_PARENT = SURVIVAL_DIR.parent
REPO_ROOT = SURVIVAL_DIR.parents[2]
for _p in (str(REPO_ROOT), str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cox_aggregated as _ca  # noqa: E402
from cox_aggregated import (  # noqa: E402
    AGE_COL,
    DEFAULT_AUC_MAX_TIME_UNITS,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_SEED,
    ENDPOINTS,
    ID_COL,
    RESULTS,
    _load_build_manifest,
    _load_prebuilt_landmark,
    compute_ipcw_auc_t,
    normalize_endpoints,
    normalize_landmark_days,
    select_feature_columns,
    stage_available_mask,
    stage_feature_columns,
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
from survival_common.projects.compass_profile import CONFIG  # noqa: E402

DEFAULT_CV_MAX_DEPTHS = [2, 3, 4, 5]
DEFAULT_CV_ETAS = [0.02, 0.03, 0.05, 0.075, 0.10]
DEFAULT_CV_MIN_CHILD_WEIGHTS = [1.0, 3.0, 5.0, 7.5, 10.0]
DEFAULT_N_FOLDS = 5


def require_lifelines() -> None:
    if concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to compute concordance indices."
        ) from LIFELINES_IMPORT_ERROR


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
            min_coverage=args.min_patient_coverage,
            id_col=ID_COL,
        )
        fold_canonical_labs[fold] = canonical
        fold_train = train_val.iloc[tr_idx]
        selected, fold_feature_meta = select_feature_columns(
            fold_train,
            raw_feature_cols,
            min_patient_coverage=args.min_patient_coverage,
            restrict_to_labs=canonical,
        )
        selected = _truncate_features_by_rank(selected, fold_feature_meta, args.max_features)
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
                    age_col=AGE_COL,
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
    baseline: bool = False,
    stage_cols: list[str] | None = None,
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
    if not args.no_cv and not baseline:
        cv_fold_df, cv_summary_df, best_row, fold_canonical_labs_df = cv_one_endpoint(
            train_val=train_val,
            raw_feature_cols=raw_feature_cols,
            pre_treatment_lab_df=pre_treatment_lab_df,
            horizon_grid=horizon_grid,
            endpoint=endpoint,
            landmark_day=landmark_day,
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

    if baseline:
        # Age(+stage)-only baseline: no lab features, no per-fold selection.
        # Empty `stage_cols` leaves an age-only model (age is added by
        # fit_preprocessor regardless of the feature set).
        selected_features = list(stage_cols or [])
        feature_meta = pd.DataFrame(
            {"feature": selected_features, "lab_name": selected_features,
             "feature_stat": "stage", "selected": True}
        )
    else:
        # Final selection on full train_val with the canonical labs already in scope.
        selected_features, feature_meta = select_feature_columns(
            train_val,
            raw_feature_cols,
            min_patient_coverage=args.min_patient_coverage,
            restrict_to_labs=canonical_labs,
        )
    if args.max_features is not None and len(selected_features) > args.max_features:
        feature_meta = feature_meta.copy()
        selected_features = _truncate_features_by_rank(selected_features, feature_meta, args.max_features)
        feature_meta["selected"] = feature_meta["feature"].isin(selected_features)

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
        feature_cols=selected_features,
        duration_col=duration_col,
        event_col=event_col,
        args=args,
        age_col=AGE_COL,
        eta=final_eta,
        max_depth=final_max_depth,
        min_child_weight=final_min_child_weight,
        num_boost_round=final_num_boost_round,
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

    mean_auc, auc_t = compute_ipcw_auc_t(
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

    risks = pd.DataFrame(
        {
            "landmark_day": landmark_day,
            "endpoint": endpoint,
            ID_COL: test.index,
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
        xgb_feature_names=preprocessor["xgb_feature_names"],
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
    args.min_patient_coverage = float(build_manifest["min_patient_coverage"])
    args.auc_time_unit_days = int(build_manifest["auc_time_unit_days"])
    args.auc_quantiles = tuple(build_manifest["auc_quantiles"])
    if getattr(args, "auc_max_time_units", None) is None:
        args.auc_max_time_units = DEFAULT_AUC_MAX_TIME_UNITS
    auc_horizons_by_landmark = build_manifest["auc_horizons_by_landmark"]
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={args.min_patient_coverage}, "
        f"auc_time_unit_days={args.auc_time_unit_days} per build manifest)"
    )

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
        merged, train_val, test, pre_treatment_lab_df = _load_prebuilt_landmark(
            inputs_dir, landmark_day
        )
        # Admin censoring removed (DeepHit silenced) — train/test use full follow-up.

        if args.restrict_to_stage:
            stage_cols_avail = stage_feature_columns(merged)
            if not stage_cols_avail:
                raise SystemExit(
                    "--restrict-to-stage requires CANCER_STAGE_* columns in the aggregated "
                    "inputs (build with --stage-file; PROFILE only)."
                )
            keep = merged.index[stage_available_mask(merged, stage_cols_avail)]
            n_before = len(merged)
            merged = merged.loc[merged.index.intersection(keep)]
            train_val = train_val.loc[train_val.index.intersection(keep)]
            test = test.loc[test.index.intersection(keep)]
            print(
                f"  [restrict-to-stage] complete-case (stage-available) cohort: "
                f"{len(merged)}/{n_before} patients "
                f"(train_val={len(train_val)}, test={len(test)})"
            )

        assert_no_test_leakage(
            test_mrns=test.index,
            train_mrns=train_val.index,
            context=f"landmark_xgboost.main[+{landmark_day}d]",
        )

        raw_feature_cols = [
            col for col in merged.columns if col not in _ca.outcome_columns()
        ]
        stage_cols = stage_feature_columns(merged) if args.baseline else []
        print(
            f"\nLandmark +{landmark_day}d: train_val={len(train_val)} test={len(test)} "
            f"raw_features={len(raw_feature_cols)}"
        )
        if args.baseline:
            print(
                "  baseline mode: age"
                + (f" + {', '.join(stage_cols)}" if stage_cols else " (no stage source)")
            )
        canonical_labs = select_canonical_labs(
            pre_treatment_lab_df,
            mrns=train_val.index,
            min_coverage=args.min_patient_coverage,
            id_col=ID_COL,
        )
        print(f"Canonical labs (train_val): {len(canonical_labs)}")
        landmark_horizons = auc_horizons_by_landmark.get(str(int(landmark_day)))
        if landmark_horizons is None:
            raise KeyError(
                f"build_manifest.json has no auc_horizons_by_landmark entry for landmark +{landmark_day}d."
            )
        for endpoint in endpoints:
            print(f"Fitting XGBoost Cox endpoint={endpoint} ...")
            if endpoint not in landmark_horizons:
                raise KeyError(
                    f"build_manifest.json missing horizons for endpoint {endpoint!r} at landmark +{landmark_day}d."
                )
            horizon_grid = np.asarray(landmark_horizons[endpoint], dtype=float)
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
                baseline=args.baseline,
                stage_cols=stage_cols,
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
        description="PROFILE multivariate survival analysis: elastic-net Cox or XGBoost survival:cox."
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
    parser.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS), choices=list(ENDPOINTS))
    parser.add_argument("--landmark-days", nargs="+", type=int, default=DEFAULT_LANDMARK_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Fit an age(+cancer-stage)-only baseline: skip lab feature selection "
            "and CV; covariates = CANCER_STAGE_* (if present) + age. Writes "
            "landmark_xgboost_baseline_*.csv on the same horizon grid."
        ),
    )
    parser.add_argument(
        "--restrict-to-stage",
        action="store_true",
        help=(
            "Restrict the cohort to stage-available patients (non-missing "
            "CANCER_STAGE_*) before fitting/evaluating, for a complete-case "
            "comparison on a matched population. Errors if no stage columns "
            "are present (PROFILE only)."
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
