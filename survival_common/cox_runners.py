"""Shared CLI and orchestration for Cox survival analyses."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from survival_common.config import CoxProjectConfig


UNIVARIATE_KEEP_COLS = [
    "landmark_days",
    "endpoint",
    "feature",
    "lab_name",
    "feature_stat",
    "n_obs_feature",
    "coverage",
    "n_obs_coverage",
    "n_patients_used",
    "n_patients_observed",
    "n_patients_imputed",
    "n_patients_n_obs_observed",
    "n_patients_n_obs_imputed",
    "n_events_used",
    "coef_feature",
    "hazard_ratio_per_sd",
    "ci_lower",
    "ci_upper",
    "p_value",
    "q_value",
    "coef_n_obs",
    "hazard_ratio_n_obs_per_sd",
    "ci_lower_n_obs",
    "ci_upper_n_obs",
    "p_value_n_obs",
    "coef_missing",
    "p_value_missing",
    "note",
    "model_type",
]


def _set_runtime_schema(cox: Any, args: Namespace) -> None:
    cox.ID_COL = args.id_col
    cox.AGE_COL = args.age_col


def _load_common_inputs(cox: Any, args: Namespace) -> tuple[list[str], list[int], Path, dict]:
    endpoints = cox.normalize_endpoints(args.endpoints)
    landmark_days = cox.normalize_landmark_days(args.landmark_days)
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(
            f"Inputs dir {inputs_dir} not found. Run build_prediction_inputs.py first."
        )
    return endpoints, landmark_days, inputs_dir, cox._load_build_manifest(inputs_dir)


def add_common_cox_args(parser: argparse.ArgumentParser, config: CoxProjectConfig, cox: Any) -> None:
    parser.add_argument(
        "--id-col",
        default=cox.ID_COL,
        help=f"Patient identifier column name (default: {cox.ID_COL}).",
    )
    parser.add_argument(
        "--age-col",
        default=cox.AGE_COL,
        help=f"Age covariate column name (default: {cox.AGE_COL}).",
    )
    parser.add_argument(
        "--inputs-dir",
        default=str(cox.RESULTS / "prediction_inputs"),
        help="Directory containing prebuilt inputs from build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(cox.RESULTS),
        help="Directory for Cox result CSVs.",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=list(config.default_endpoints),
        choices=list(cox.ENDPOINTS),
        help="Endpoints to analyze.",
    )
    parser.add_argument(
        "--landmark-days",
        nargs="+",
        type=int,
        default=cox.DEFAULT_LANDMARK_DAYS,
        help="Landmark offsets to analyze. Each must have prebuilt inputs in --inputs-dir.",
    )


def build_univariate_parser(config: CoxProjectConfig, cox: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=config.univariate_description)
    add_common_cox_args(parser, config, cox)
    parser.add_argument(
        "--min-events-per-feature",
        type=int,
        default=cox.DEFAULT_MIN_EVENTS_PER_FEATURE,
        help="Skip univariate associations when too few endpoint events remain after outcome filtering.",
    )
    parser.add_argument(
        "--univariate-penalizer",
        type=float,
        default=0.05,
        help="Fallback penalizer used only when a univariate Cox model does not converge without regularization.",
    )
    config.add_cli_args(parser, cox)
    return parser


def build_multivariable_parser(config: CoxProjectConfig, cox: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=config.multivariable_description)
    add_common_cox_args(parser, config, cox)
    parser.add_argument("--baseline", action="store_true", help=config.baseline_help)
    config.add_cli_args(parser, cox)
    parser.add_argument(
        "--auc-max-time-units",
        type=int,
        default=None,
        help=(
            "Cap (in time-units) for the IPCW AUC(t)/Brier evaluation horizons. "
            f"Defaults to {cox.DEFAULT_AUC_MAX_TIME_UNITS} if unset. Caps evaluation only, not fitting."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cox.DEFAULT_SEED,
        help="Random seed for cross-validation. The patient split is fixed by build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=cox.DEFAULT_N_FOLDS,
        help="Number of cross-validation folds within the train/validation cohort.",
    )
    parser.add_argument(
        "--cv-penalizers",
        nargs="+",
        type=float,
        default=cox.DEFAULT_CV_PENALIZERS,
        help="Penalizer values searched during cross-validation on the train/validation block.",
    )
    parser.add_argument(
        "--cv-l1-ratios",
        nargs="+",
        type=float,
        default=cox.DEFAULT_CV_L1_RATIOS,
        help="Elastic-net L1 mixing values (0=ridge, 1=lasso) searched during cross-validation.",
    )
    return parser


def run_univariate(config: CoxProjectConfig, cox: Any, args: Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_runtime_schema(cox, args)
    endpoints, landmark_days, inputs_dir, build_manifest = _load_common_inputs(cox, args)
    min_patient_coverage = float(build_manifest["min_patient_coverage"])
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={min_patient_coverage})"
    )

    feature_selection_frames: list[pd.DataFrame] = []
    univariate_frames: list[pd.DataFrame] = []

    for landmark_day in landmark_days:
        ctx = cox.prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
            **config.prepare_context_kwargs(args),
        )
        static_covariate_cols = config.static_covariates(ctx, args, cox)
        feature_selection_frames.append(ctx.feature_meta_selected)

        print("\n##### ARM 1: UNIVARIATE (n_obs-adjusted, full follow-up, all endpoints) #####")
        for endpoint in endpoints:
            print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
            print(cox.ENDPOINTS[endpoint]["description"])
            adjusted_frames = [
                cox.run_univariate_nobs_adjusted_associations(
                    ctx.univariate_data,
                    feature_cols=ctx.selected_feature_cols,
                    endpoint=endpoint,
                    min_events_per_feature=args.min_events_per_feature,
                    fallback_penalizer=args.univariate_penalizer,
                    static_covariate_cols=static_covariate_cols,
                    model_type="cox",
                )
            ]
            # When the endpoint declares a competing event (e.g. death for
            # platinum/irAE), also fit the Fine-Gray subdistribution-hazard
            # arm and emit it alongside the cause-specific rows in the same
            # output file, distinguished by model_type.
            competing = cox.endpoint_competing(endpoint)
            if competing is not None:
                event_type_col, event_of_interest, competing_event = competing
                adjusted_frames.append(
                    cox.run_univariate_nobs_adjusted_associations(
                        ctx.univariate_data,
                        feature_cols=ctx.selected_feature_cols,
                        endpoint=endpoint,
                        min_events_per_feature=args.min_events_per_feature,
                        fallback_penalizer=args.univariate_penalizer,
                        static_covariate_cols=static_covariate_cols,
                        model_type="finegray",
                        event_type_col=event_type_col,
                        event_of_interest=event_of_interest,
                        competing_event=competing_event,
                    )
                )
            for adjusted_df in adjusted_frames:
                adjusted_df.insert(0, "landmark_days", landmark_day)
                univariate_frames.append(adjusted_df[UNIVARIATE_KEEP_COLS].copy())
                model_label = adjusted_df["model_type"].iloc[0] if len(adjusted_df) else "cox"
                cox.print_top_hits(
                    adjusted_df,
                    endpoint=endpoint,
                    label=f"n_obs-adjusted univariate ({model_label})",
                )

    if feature_selection_frames:
        pd.concat(feature_selection_frames, ignore_index=True).to_csv(
            output_dir / "cox_agg_feature_selection.csv", index=False
        )
    if univariate_frames:
        pd.concat(univariate_frames, ignore_index=True).to_csv(
            output_dir / "cox_agg_univariate_nobs_adjusted.csv", index=False
        )

    print("\nSaved:")
    print("  cox_agg_feature_selection.csv")
    if univariate_frames:
        print("  cox_agg_univariate_nobs_adjusted.csv")


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
    static_covariate_cols = config.static_covariates(ctx, args, cox)
    print("\n##### ARM 2: MULTIVARIABLE ELASTIC-NET (all endpoints) #####")
    if static_covariate_cols:
        print(f"  always-included covariates: age + {', '.join(static_covariate_cols)} (unpenalized)")
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(cox.ENDPOINTS[endpoint]["description"])
        horizon_grid = endpoint_horizon_grids[endpoint]
        _, _, best_row, fold_canonical_labs_df = cox.tune_multivariable_model(
            ctx.train_val.copy(),
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
            always_include_feature_cols=tuple(
                getattr(ctx, "always_include_feature_cols", ())
            ),
        )
        if not fold_canonical_labs_df.empty:
            fold_canonical_labs_df.insert(0, "landmark_days", landmark_day)
            out["canonical_labs_fold_rows"].append(fold_canonical_labs_df)

        metrics_row, summary_df, _, test_auc_df, test_brier_df = cox.fit_final_multivariable_model(
            ctx.train_val.copy(),
            ctx.test.copy(),
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
        _collect_multivariable_outputs(out, landmark_day, metrics_row, summary_df, test_auc_df, test_brier_df)
        _print_multivariable_summary(best_row, metrics_row, summary_df)


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
    static_covariate_cols = config.static_covariates(ctx, args, cox)
    feature_cols = config.baseline_feature_cols(ctx, args, cox)
    baseline_penalizer = float(args.cv_penalizers[0])
    baseline_l1_ratio = float(args.cv_l1_ratios[0])
    extra = list(feature_cols) + list(static_covariate_cols)
    print("\n##### BASELINE: AGE(+STATIC COVARIATES)-ONLY (all endpoints) #####")
    print("  covariates: age" + (f" + {', '.join(extra)}" if extra else " (no static covariates found)"))
    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
        print(cox.ENDPOINTS[endpoint]["description"])
        metrics_row, summary_df, _, test_auc_df, test_brier_df = cox.fit_final_multivariable_model(
            ctx.train_val.copy(),
            ctx.test.copy(),
            feature_cols=feature_cols,
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
        if config.baseline_feature_count_column:
            metrics_row[config.baseline_feature_count_column] = len(feature_cols)
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


def _collect_multivariable_outputs(
    out: dict[str, list],
    landmark_day: int,
    metrics_row: dict,
    summary_df: pd.DataFrame,
    test_auc_df: pd.DataFrame,
    test_brier_df: pd.DataFrame,
) -> None:
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


def _print_multivariable_summary(best_row: dict, metrics_row: dict, summary_df: pd.DataFrame) -> None:
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


def run_multivariable(config: CoxProjectConfig, cox: Any, args: Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_runtime_schema(cox, args)
    endpoints, landmark_days, inputs_dir, build_manifest = _load_common_inputs(cox, args)
    min_patient_coverage = float(build_manifest["min_patient_coverage"])
    auc_time_unit_days = int(build_manifest["auc_time_unit_days"])
    auc_quantiles = tuple(build_manifest["auc_quantiles"])
    auc_max_time_units = (
        args.auc_max_time_units
        if args.auc_max_time_units is not None
        else cox.DEFAULT_AUC_MAX_TIME_UNITS
    )
    auc_horizons_by_landmark = build_manifest["auc_horizons_by_landmark"]
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={min_patient_coverage}, "
        f"auc_time_unit_days={auc_time_unit_days} per build manifest)"
    )

    feature_selection_frames: list[pd.DataFrame] = []
    horizon_grid_frames: list[pd.DataFrame] = []
    out = {
        "frames": [],
        "metric_rows": [],
        "test_auc_frames": [],
        "test_brier_frames": [],
        "canonical_labs_fold_rows": [],
    }

    for landmark_day in landmark_days:
        ctx = cox.prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
            **config.prepare_context_kwargs(args),
        )
        feature_selection_frames.append(ctx.feature_meta_selected)
        endpoint_horizon_grids, horizon_grid_df = cox.build_endpoint_horizon_grids(
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
                config,
                cox,
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
                config,
                cox,
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

    _write_multivariable_outputs(cox, output_dir, args.baseline, feature_selection_frames, horizon_grid_frames, out)


def _write_multivariable_outputs(
    cox: Any,
    output_dir: Path,
    baseline: bool,
    feature_selection_frames: list[pd.DataFrame],
    horizon_grid_frames: list[pd.DataFrame],
    out: dict[str, list],
) -> None:
    if feature_selection_frames:
        pd.concat(feature_selection_frames, ignore_index=True).to_csv(
            output_dir / "cox_agg_feature_selection.csv", index=False
        )
    if horizon_grid_frames:
        pd.concat(horizon_grid_frames, ignore_index=True).to_csv(
            output_dir / cox.HORIZON_GRID_FILENAME, index=False
        )

    prefix = "cox_agg_baseline" if baseline else "cox_agg_multivariable"
    if not baseline and out["canonical_labs_fold_rows"]:
        pd.concat(out["canonical_labs_fold_rows"], ignore_index=True).to_csv(
            output_dir / cox.CANONICAL_LABS_FOLDS_FILENAME, index=False
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
        print(f"  {cox.HORIZON_GRID_FILENAME}")
    if not baseline and out["canonical_labs_fold_rows"]:
        print(f"  {cox.CANONICAL_LABS_FOLDS_FILENAME}")
    if out["frames"]:
        print(f"  {prefix}.csv")
    if out["test_auc_frames"]:
        print(f"  {prefix}_test_auc_t.csv")
    if out["test_brier_frames"]:
        print(f"  {prefix}_test_brier.csv")
    if out["metric_rows"]:
        print(f"  {prefix}_metrics.csv")
