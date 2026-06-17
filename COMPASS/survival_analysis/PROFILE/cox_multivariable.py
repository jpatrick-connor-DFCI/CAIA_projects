"""
Multivariable elastic-net Cox on landmarked lab summary features (+ age baseline).

Arm 2 (multivariable elastic-net Cox):
  - 80% train/val + 20% held-out test (split fixed upstream by
    build_prediction_inputs.py).
  - 5-fold CV over the (penalizer x l1_ratio) grid on the 80% block; AGE is
    unpenalized. Refit on the full 80% with the chosen hyperparameters and
    evaluate on the 20% test: C-index and IPCW cumulative/dynamic AUC(t)/Brier.

--baseline: fit an age(+cancer-stage)-only Cox model (no lab features, no CV /
  feature selection) on the same horizon grid for benchmarking. Reuses the
  multivariable final-fit + evaluation path so the metrics schema matches.

Reads prebuilt inputs from build_prediction_inputs.py (run that first). Shares
the cohort / canonical-lab / feature-selection setup with cox_univariate.py via
cox_aggregated.prepare_landmark_context. AUC(t) horizons come from
build_manifest.json so Cox / XGBoost / DeepHit evaluate on an identical grid.

Outputs (under --output-dir):
  cox_agg_feature_selection.csv         selected lab features + coverage
  cox_agg_horizon_grid.csv              per-(landmark, endpoint) AUC(t) horizons
  multivariable mode:
    cox_agg_canonical_labs_folds.csv    per-fold canonical labs
    cox_agg_multivariable.csv           coefs (+ landmark_days)
    cox_agg_multivariable_test_auc_t.csv
    cox_agg_multivariable_test_brier.csv
    cox_agg_multivariable_metrics.csv   C-index / mean AUC(t) / integrated Brier
  baseline mode:
    cox_agg_baseline.csv / _test_auc_t.csv / _test_brier.csv / _metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent           # .../survival_analysis/PROFILE
SURVIVAL_PARENT = SURVIVAL_DIR.parent                    # .../survival_analysis
for _p in (str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cox_aggregated as _ca  # noqa: E402
from cox_aggregated import (  # noqa: E402
    AGE_COL,
    CANONICAL_LABS_FOLDS_FILENAME,
    DEFAULT_AUC_MAX_TIME_UNITS,
    DEFAULT_CV_L1_RATIOS,
    DEFAULT_CV_PENALIZERS,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    ENDPOINTS,
    GLEASON_COL,
    HORIZON_GRID_FILENAME,
    ID_COL,
    RESULTS,
    _load_build_manifest,
    build_endpoint_horizon_grids,
    fit_final_multivariable_model,
    normalize_endpoints,
    normalize_landmark_days,
    prepare_landmark_context,
    stage_feature_columns,
    tune_multivariable_model,
)


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
    """Multivariable elastic-net arm for one landmark; appends to ``out`` lists."""
    static_covariate_cols = (GLEASON_COL,) if args.with_gleason else ()
    multivariable_train_val = ctx.train_val.copy()
    multivariable_test = ctx.test.copy()
    print("\n##### ARM 2: MULTIVARIABLE ELASTIC-NET (all endpoints) #####")
    if static_covariate_cols:
        print(f"  always-included covariates: age + {', '.join(static_covariate_cols)} (unpenalized)")
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(ENDPOINTS[endpoint]["description"])
        horizon_grid = endpoint_horizon_grids[endpoint]
        _, _, best_row, fold_canonical_labs_df = tune_multivariable_model(
            multivariable_train_val,
            raw_feature_cols=ctx.raw_feature_cols,
            endpoint=endpoint,
            penalizers=args.cv_penalizers,
            l1_ratios=args.cv_l1_ratios,
            n_folds=args.n_folds,
            seed=args.seed,
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            pre_treatment_lab_df=ctx.pre_treatment_lab_df,
            horizon_grid=horizon_grid,
            min_patient_coverage=min_patient_coverage,
            static_covariate_cols=static_covariate_cols,
        )
        if not fold_canonical_labs_df.empty:
            fold_canonical_labs_df.insert(0, "landmark_days", landmark_day)
            out["canonical_labs_fold_rows"].append(fold_canonical_labs_df)

        (
            metrics_row,
            summary_df,
            _,
            test_auc_df,
            test_brier_df,
        ) = fit_final_multivariable_model(
            multivariable_train_val,
            multivariable_test,
            feature_cols=ctx.selected_feature_cols,
            endpoint=endpoint,
            penalizer=float(best_row["penalizer"]),
            l1_ratio=float(best_row["l1_ratio"]),
            split_stratification=ctx.split_stratification,
            cv_stratification=str(best_row["cv_stratification"]),
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grid=horizon_grid,
            canonical_labs=ctx.canonical_labs,
            static_covariate_cols=static_covariate_cols,
        )
        metrics_row["landmark_days"] = landmark_day
        summary_df.insert(0, "landmark_days", landmark_day)
        out["metric_rows"].append(metrics_row)
        out["frames"].append(summary_df)
        if not test_auc_df.empty:
            test_auc_df = test_auc_df.copy()
            test_auc_df.insert(0, "landmark_days", landmark_day)
            out["test_auc_frames"].append(test_auc_df)
        if not test_brier_df.empty:
            test_brier_df = test_brier_df.copy()
            test_brier_df.insert(0, "landmark_days", landmark_day)
            out["test_brier_frames"].append(test_brier_df)

        top_cols = [c for c in ["feature", "coef", "exp(coef)"] if c in summary_df.columns]
        top = summary_df.loc[~summary_df["is_age_covariate"], top_cols].head(10)
        print("\nChosen hyperparameters (elastic-net, age unpenalized):")
        print(
            f"  penalizer={best_row['penalizer']}  l1_ratio={best_row['l1_ratio']}  "
            f"cv_mean C-index={best_row['cv_mean']:.4f}"
        )
        print(f"  CV mean AUC(t)={best_row['mean_auc_t_cv_mean']:.4f}")
        print(f"  CV mean integrated Brier={best_row['integrated_brier_cv_mean']:.4f}")
        print(
            f"  train/val C-index={metrics_row['train_val_c_index']:.4f}  "
            f"mean AUC(t)={metrics_row['train_val_mean_auc_t']:.4f}"
        )
        print(f"  held-out test C-index={metrics_row['test_c_index']:.4f}")
        print(f"  held-out test mean AUC(t)={metrics_row['test_mean_auc_t']:.4f}")
        print(f"  held-out test integrated Brier={metrics_row['test_integrated_brier']:.4f}")
        print("Top multivariable coefficients:")
        print(top.to_string(index=False))


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
    """Age(+stage)-only baseline arm for one landmark; appends to ``out`` lists.

    No lab features, no CV / feature selection. Reuses the multivariable
    final-fit + evaluation path so C-index / AUC(t) / Brier are on the identical
    horizon grid and the metrics CSV schema matches the multivariable arm.
    """
    stage_cols = stage_feature_columns(ctx.merged)
    static_covariate_cols = (GLEASON_COL,) if args.with_gleason else ()
    baseline_penalizer = float(args.cv_penalizers[0])
    baseline_l1_ratio = float(args.cv_l1_ratios[0])
    extra = list(stage_cols) + list(static_covariate_cols)
    print("\n##### BASELINE: AGE(+STAGE+GLEASON)-ONLY (all endpoints) #####")
    print(
        "  covariates: age" + (f" + {', '.join(extra)}" if extra else " (no stage/gleason source)")
    )
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(ENDPOINTS[endpoint]["description"])
        (
            metrics_row,
            summary_df,
            _,
            test_auc_df,
            test_brier_df,
        ) = fit_final_multivariable_model(
            ctx.train_val.copy(),
            ctx.test.copy(),
            feature_cols=stage_cols,
            endpoint=endpoint,
            penalizer=baseline_penalizer,
            l1_ratio=baseline_l1_ratio,
            split_stratification=ctx.split_stratification,
            cv_stratification="baseline_no_cv",
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grid=endpoint_horizon_grids[endpoint],
            canonical_labs=[],
            static_covariate_cols=static_covariate_cols,
        )
        metrics_row["landmark_days"] = landmark_day
        metrics_row["n_stage_cols"] = len(stage_cols)
        summary_df.insert(0, "landmark_days", landmark_day)
        out["metric_rows"].append(metrics_row)
        out["frames"].append(summary_df)
        if not test_auc_df.empty:
            test_auc_df = test_auc_df.copy()
            test_auc_df.insert(0, "landmark_days", landmark_day)
            out["test_auc_frames"].append(test_auc_df)
        if not test_brier_df.empty:
            test_brier_df = test_brier_df.copy()
            test_brier_df.insert(0, "landmark_days", landmark_day)
            out["test_brier_frames"].append(test_brier_df)
        print(f"  held-out test C-index={metrics_row['test_c_index']:.4f}")
        print(f"  held-out test mean AUC(t)={metrics_row['test_mean_auc_t']:.4f}")
        print(f"  held-out test integrated Brier={metrics_row['test_integrated_brier']:.4f}")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Push the runtime schema into the shared library so prepare_landmark_context
    # / _load_prebuilt_landmark read the right id/age columns (CAIA: person_id).
    _ca.ID_COL = args.id_col
    _ca.AGE_COL = args.age_col
    endpoints = normalize_endpoints(args.endpoints)
    landmark_days = normalize_landmark_days(args.landmark_days)
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(
            f"Inputs dir {inputs_dir} not found. Run build_prediction_inputs.py first."
        )
    build_manifest = _load_build_manifest(inputs_dir)
    min_patient_coverage = float(build_manifest["min_patient_coverage"])
    auc_time_unit_days = int(build_manifest["auc_time_unit_days"])
    auc_quantiles = tuple(build_manifest["auc_quantiles"])
    auc_max_time_units = (
        args.auc_max_time_units
        if args.auc_max_time_units is not None
        else DEFAULT_AUC_MAX_TIME_UNITS
    )
    auc_horizons_by_landmark = build_manifest["auc_horizons_by_landmark"]
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={min_patient_coverage}, "
        f"auc_time_unit_days={auc_time_unit_days} per build manifest)"
    )

    feature_selection_frames: list[pd.DataFrame] = []
    horizon_grid_frames: list[pd.DataFrame] = []
    # Per-arm accumulators (shared schema; only one arm runs per invocation).
    out = {
        "frames": [],
        "metric_rows": [],
        "test_auc_frames": [],
        "test_brier_frames": [],
        "canonical_labs_fold_rows": [],
    }

    for landmark_day in landmark_days:
        ctx = prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
            restrict_to_stage=args.restrict_to_stage,
            restrict_to_gleason=args.restrict_to_gleason,
        )
        if args.with_gleason and GLEASON_COL not in ctx.merged.columns:
            raise SystemExit(
                "--with-gleason requires a GLEASON_GROUP column in the aggregated inputs "
                "(build with --gleason-file; PROFILE only)."
            )
        feature_selection_frames.append(ctx.feature_meta_selected)

        endpoint_horizon_grids, horizon_grid_df = build_endpoint_horizon_grids(
            landmark_day,
            endpoints=endpoints,
            auc_horizons_by_landmark=auc_horizons_by_landmark,
            auc_quantiles=auc_quantiles,
            auc_time_unit_days=auc_time_unit_days,
        )
        if not horizon_grid_df.empty:
            horizon_grid_frames.append(horizon_grid_df)

        if args.baseline:
            _run_baseline_landmark(
                ctx,
                endpoint_horizon_grids,
                landmark_day=landmark_day,
                endpoints=endpoints,
                args=args,
                auc_time_unit_days=auc_time_unit_days,
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
                auc_time_unit_days=auc_time_unit_days,
                auc_max_time_units=auc_max_time_units,
                min_patient_coverage=min_patient_coverage,
                out=out,
            )

    if feature_selection_frames:
        pd.concat(feature_selection_frames, ignore_index=True).to_csv(
            output_dir / "cox_agg_feature_selection.csv", index=False
        )
    if horizon_grid_frames:
        pd.concat(horizon_grid_frames, ignore_index=True).to_csv(
            output_dir / HORIZON_GRID_FILENAME, index=False
        )

    prefix = "cox_agg_baseline" if args.baseline else "cox_agg_multivariable"
    if not args.baseline and out["canonical_labs_fold_rows"]:
        pd.concat(out["canonical_labs_fold_rows"], ignore_index=True).to_csv(
            output_dir / CANONICAL_LABS_FOLDS_FILENAME, index=False
        )
    if out["frames"]:
        pd.concat(out["frames"], ignore_index=True).to_csv(
            output_dir / f"{prefix}.csv", index=False
        )
    if out["test_auc_frames"]:
        pd.concat(out["test_auc_frames"], ignore_index=True).to_csv(
            output_dir / f"{prefix}_test_auc_t.csv", index=False
        )
    if out["test_brier_frames"]:
        pd.concat(out["test_brier_frames"], ignore_index=True).to_csv(
            output_dir / f"{prefix}_test_brier.csv", index=False
        )
    if out["metric_rows"]:
        pd.DataFrame(out["metric_rows"]).to_csv(
            output_dir / f"{prefix}_metrics.csv", index=False
        )

    print("\nSaved:")
    print("  cox_agg_feature_selection.csv")
    if horizon_grid_frames:
        print(f"  {HORIZON_GRID_FILENAME}")
    if not args.baseline and out["canonical_labs_fold_rows"]:
        print(f"  {CANONICAL_LABS_FOLDS_FILENAME}")
    if out["frames"]:
        print(f"  {prefix}.csv")
    if out["test_auc_frames"]:
        print(f"  {prefix}_test_auc_t.csv")
    if out["test_brier_frames"]:
        print(f"  {prefix}_test_brier.csv")
    if out["metric_rows"]:
        print(f"  {prefix}_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multivariable elastic-net Cox (+ age(+stage) baseline) on landmarked lab features."
    )
    parser.add_argument(
        "--id-col",
        default=ID_COL,
        help="Patient identifier column name (default: DFCI_MRN for PROFILE; e.g. person_id for CAIA).",
    )
    parser.add_argument(
        "--age-col",
        default=AGE_COL,
        help="Age covariate column name (default: AGE_AT_TREATMENTSTART for PROFILE; e.g. AGE_AT_DIAGNOSIS for CAIA).",
    )
    parser.add_argument(
        "--inputs-dir",
        default=str(RESULTS / "prediction_inputs"),
        help="Directory containing prebuilt inputs from build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS),
        help="Directory for Cox multivariable / baseline result CSVs.",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["platinum", "death"],
        choices=list(ENDPOINTS),
        help="Endpoints to analyze.",
    )
    parser.add_argument(
        "--landmark-days",
        nargs="+",
        type=int,
        default=DEFAULT_LANDMARK_DAYS,
        help="Landmark offsets to analyze. Each must have prebuilt inputs in --inputs-dir.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Fit an age(+cancer-stage)-only Cox model (no labs, no CV) on the same "
            "horizon grid for benchmarking, instead of the elastic-net lab model."
        ),
    )
    parser.add_argument(
        "--restrict-to-stage",
        action="store_true",
        help=(
            "Restrict the cohort to stage-available patients (non-missing "
            "CANCER_STAGE_*) before fitting/evaluating, for a complete-case "
            "age+stage-baseline vs. labs comparison on a matched population. "
            "Errors if no stage columns are present (PROFILE only)."
        ),
    )
    parser.add_argument(
        "--restrict-to-gleason",
        action="store_true",
        help=(
            "Restrict the cohort to Gleason-available patients (non-missing "
            "GLEASON_GROUP) before fitting/evaluating — the siloed Gleason analysis. "
            "Errors if no GLEASON_GROUP column is present (build with --gleason-file)."
        ),
    )
    parser.add_argument(
        "--with-gleason",
        action="store_true",
        help=(
            "Include GLEASON_GROUP as an always-present, unpenalized covariate in the "
            "multivariable (and baseline) Cox model, alongside age (and stage)."
        ),
    )
    parser.add_argument(
        "--auc-max-time-units",
        type=int,
        default=None,
        help=(
            "Cap (in time-units) for the IPCW AUC(t)/Brier evaluation horizons. "
            f"Defaults to {DEFAULT_AUC_MAX_TIME_UNITS} if unset. Caps evaluation only, not fitting."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for cross-validation. The patient split is fixed by build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of cross-validation folds within the train/validation cohort.",
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
    # AUC(t) time unit, quantiles, and horizons all come from build_manifest.json
    # so Cox / XGBoost / DeepHit evaluate on the identical horizon set.
    main(parser.parse_args())
