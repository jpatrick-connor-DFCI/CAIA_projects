"""Shared CLI and orchestration for Cox survival analyses."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from survival_common.config import CoxProjectConfig
from survival_common.helper import resolve_auc_max_time_units
from survival_common.metrics_schema import (
    DEFAULT_COHORT,
    MODEL_ELASTIC_NET_COX,
    canonical_identity,
    order_canonical_first,
)


UNIVARIATE_KEEP_COLS = [
    "landmark_days",
    "endpoint",
    "feature",
    "lab_name",
    "feature_stat",
    "coverage",
    "n_patients_used",
    "n_patients_observed",
    "n_patients_imputed",
    "n_events_used",
    "coef_feature",
    "hazard_ratio_per_sd",
    "ci_lower",
    "ci_upper",
    "p_value",
    "q_value",
    "coef_missing",
    "p_value_missing",
    "note",
    "model_type",
]


def _set_runtime_schema(cox: Any, args: Namespace) -> None:
    cox.ID_COL = args.id_col
    cox.AGE_COL = args.age_col


def _per_landmark_path(output_dir: Path, prefix: str, landmark_day: int) -> Path:
    return output_dir / f"{prefix}_landmark{landmark_day}.csv"


def _csv_stem(filename: str) -> str:
    """Remove one trailing .csv without requiring Python 3.9 removesuffix()."""
    return filename[:-4] if filename.endswith(".csv") else filename


def _combine_per_landmark(
    output_dir: Path, prefix: str, landmark_days: list[int], combined_filename: str
) -> bool:
    """Rebuild the combined CSV from whichever per-landmark files exist on disk.

    Reads from disk rather than from in-memory frames so a resumed run's
    combined output reflects landmarks fit in earlier invocations too, not
    just the ones (re)computed this call.
    """
    frames = []
    for landmark_day in landmark_days:
        path = _per_landmark_path(output_dir, prefix, landmark_day)
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return False
    pd.concat(frames, ignore_index=True).to_csv(output_dir / combined_filename, index=False)
    return True


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
    parser.add_argument(
        "--cohort",
        default=DEFAULT_COHORT,
        help=(
            "Patient-subset label stamped onto the canonical `cohort` metrics "
            "column. The restriction itself is applied upstream at input-build "
            "time via --restrict-to-mrns; this only records which subset the "
            f"fit ran on (default: {DEFAULT_COHORT})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help=(
            "Refit every --landmark-days even if its per-landmark output files "
            "already exist in --output-dir. Default: pick up where left off, "
            "skipping landmarks whose per-landmark files are already present."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip landmarks whose per-landmark output files already exist (the default).",
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
    parser.add_argument(
        "--shared-canonical-labs",
        action="store_true",
        help=(
            "Test one lab set shared across all --landmark-days (their canonical-"
            "lab intersection) instead of re-deriving canonical labs per landmark. "
            "Makes per-lab associations comparable across landmarks (e.g. +0d vs "
            "+90d). Requires >=2 landmarks to be meaningful."
        ),
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
            "Defaults to the build manifest's auc_max_time_units, which the horizon "
            "grid was clamped to. Caps evaluation only, not fitting."
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

    shared_canonical_labs = None
    if getattr(args, "shared_canonical_labs", False):
        if not hasattr(cox, "compute_shared_canonical_labs"):
            raise SystemExit(
                "--shared-canonical-labs is not supported by this project "
                f"({cox.__name__}): it has no compute_shared_canonical_labs / "
                "canonical_labs_override support."
            )
        print("\n##### SHARED CANONICAL LABS: intersect across landmarks #####")
        shared_canonical_labs = cox.compute_shared_canonical_labs(
            inputs_dir,
            landmark_days,
            min_patient_coverage=min_patient_coverage,
        )

    overwrite = getattr(args, "overwrite", False)

    for landmark_day in landmark_days:
        feature_selection_path = _per_landmark_path(output_dir, "cox_agg_feature_selection", landmark_day)
        univariate_path = _per_landmark_path(output_dir, "cox_agg_univariate_nobs_adjusted", landmark_day)
        if not overwrite and feature_selection_path.exists() and univariate_path.exists():
            print(f"\n[skip] landmark +{landmark_day}d: per-landmark outputs already exist (pass --overwrite to refit)")
            continue

        context_kwargs = dict(config.prepare_context_kwargs(args))
        if shared_canonical_labs is not None:
            context_kwargs["canonical_labs_override"] = shared_canonical_labs
        ctx = cox.prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
            **context_kwargs,
        )
        ctx.feature_meta_selected.to_csv(feature_selection_path, index=False)

        univariate_frames: list[pd.DataFrame] = []
        print("\n##### ARM 1: UNIVARIATE (n_obs-adjusted, full follow-up, all endpoints) #####")
        for endpoint in endpoints:
            print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
            print(
                build_manifest.get(
                    "endpoint_description", cox.ENDPOINTS[endpoint]["description"]
                )
            )
            # Univariate testing runs on the full-cohort candidate universe
            # (ctx.raw_feature_cols), matching ctx.univariate_data's full-cohort
            # scope -- not ctx.selected_feature_cols, which is a train_val-only
            # coverage/prevalence filter used to gate the *multivariable* arm's
            # feature set (Finding 5: using the train-filtered list here would
            # under-correct the BH q-values relative to the number of hypotheses
            # actually tested on the full cohort). Cancer type, treatment, and
            # gender are never tested as their own feature, but are always
            # included as adjustment covariates (alongside age) in every
            # feature test.
            genomic_feature_cols = getattr(ctx, "always_include_feature_cols", ())
            baseline_covariate_cols = config.static_covariates(ctx, args, cox)
            adjusted_frames = [
                cox.run_univariate_nobs_adjusted_associations(
                    ctx.univariate_data,
                    feature_cols=ctx.raw_feature_cols,
                    endpoint=endpoint,
                    min_events_per_feature=args.min_events_per_feature,
                    fallback_penalizer=args.univariate_penalizer,
                    model_type="cox",
                    genomic_feature_cols=genomic_feature_cols,
                    baseline_covariate_cols=baseline_covariate_cols,
                )
            ]
            for adjusted_df in adjusted_frames:
                adjusted_df.insert(0, "landmark_days", landmark_day)
                univariate_frames.append(adjusted_df[UNIVARIATE_KEEP_COLS].copy())
                model_label = adjusted_df["model_type"].iloc[0] if len(adjusted_df) else "cox"
                cox.print_top_hits(
                    adjusted_df,
                    endpoint=endpoint,
                    label=f"n_obs-adjusted univariate ({model_label})",
                )
        if univariate_frames:
            pd.concat(univariate_frames, ignore_index=True).to_csv(univariate_path, index=False)
        print(f"\n[done] landmark +{landmark_day}d: saved {feature_selection_path.name}, {univariate_path.name}")

    print("\nSaved (combined):")
    if _combine_per_landmark(output_dir, "cox_agg_feature_selection", landmark_days, "cox_agg_feature_selection.csv"):
        print("  cox_agg_feature_selection.csv")
    if _combine_per_landmark(
        output_dir, "cox_agg_univariate_nobs_adjusted", landmark_days, "cox_agg_univariate_nobs_adjusted.csv"
    ):
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
    if getattr(args, "shared_canonical_labs", False):
        # The elastic-net arm re-selects canonical labs per CV fold inside
        # tune_multivariable_model (see below), so a cross-landmark shared set
        # cannot be honored end-to-end here without also gating fold selection.
        # The shared-canonical-labs arm is therefore univariate-only; fail loudly
        # rather than silently ignore the flag on the multivariable path.
        raise SystemExit(
            "--shared-canonical-labs is only supported for the univariate arm "
            "(univariate_analysis.py). The multivariable elastic-net re-selects "
            "canonical labs per CV fold, so a shared set does not apply."
        )
    endpoints, landmark_days, inputs_dir, build_manifest = _load_common_inputs(cox, args)
    min_patient_coverage = float(build_manifest["min_patient_coverage"])
    auc_time_unit_days = int(build_manifest["auc_time_unit_days"])
    auc_quantiles = tuple(build_manifest["auc_quantiles"])
    auc_max_time_units = resolve_auc_max_time_units(build_manifest, args.auc_max_time_units)
    auc_horizons_by_landmark = build_manifest["auc_horizons_by_landmark"]
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={min_patient_coverage}, "
        f"auc_time_unit_days={auc_time_unit_days} per build manifest)"
    )

    prefix = "cox_agg_baseline" if args.baseline else "cox_agg_multivariable"
    overwrite = getattr(args, "overwrite", False)

    for landmark_day in landmark_days:
        metrics_path = _per_landmark_path(output_dir, f"{prefix}_metrics", landmark_day)
        if not overwrite and metrics_path.exists():
            print(f"\n[skip] landmark +{landmark_day}d: {metrics_path.name} already exists (pass --overwrite to refit)")
            continue

        ctx = cox.prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
            **config.prepare_context_kwargs(args),
        )
        ctx.feature_meta_selected.to_csv(
            _per_landmark_path(output_dir, "cox_agg_feature_selection", landmark_day), index=False
        )
        endpoint_horizon_grids, horizon_grid_df = cox.build_endpoint_horizon_grids(
            landmark_day,
            endpoints=endpoints,
            auc_horizons_by_landmark=auc_horizons_by_landmark,
            auc_quantiles=auc_quantiles,
            auc_time_unit_days=auc_time_unit_days,
        )
        if not horizon_grid_df.empty:
            horizon_grid_df.to_csv(
                _per_landmark_path(
                    output_dir, _csv_stem(cox.HORIZON_GRID_FILENAME), landmark_day
                ),
                index=False,
            )

        out: dict[str, list] = {
            "frames": [],
            "metric_rows": [],
            "test_auc_frames": [],
            "test_brier_frames": [],
            "canonical_labs_fold_rows": [],
        }
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

        _write_multivariable_landmark_outputs(
            cox, output_dir, prefix, landmark_day, out, args
        )
        print(f"\n[done] landmark +{landmark_day}d: saved {metrics_path.name}")

    _combine_multivariable_outputs(cox, output_dir, prefix, landmark_days)


def _write_multivariable_landmark_outputs(
    cox: Any,
    output_dir: Path,
    prefix: str,
    landmark_day: int,
    out: dict[str, list],
    args: Namespace,
) -> None:
    if prefix == "cox_agg_multivariable" and out["canonical_labs_fold_rows"]:
        pd.concat(out["canonical_labs_fold_rows"], ignore_index=True).to_csv(
            _per_landmark_path(
                output_dir,
                _csv_stem(cox.CANONICAL_LABS_FOLDS_FILENAME),
                landmark_day,
            ),
            index=False,
        )
    if out["frames"]:
        pd.concat(out["frames"], ignore_index=True).to_csv(
            _per_landmark_path(output_dir, prefix, landmark_day), index=False
        )
    if out["test_auc_frames"]:
        pd.concat(out["test_auc_frames"], ignore_index=True).to_csv(
            _per_landmark_path(output_dir, f"{prefix}_test_auc_t", landmark_day), index=False
        )
    if out["test_brier_frames"]:
        pd.concat(out["test_brier_frames"], ignore_index=True).to_csv(
            _per_landmark_path(output_dir, f"{prefix}_test_brier", landmark_day), index=False
        )
    if out["metric_rows"]:
        metrics = pd.DataFrame(
            [
                # `endpoint` and `landmark_days` are already on the row; the
                # identity block restates them so every family stamps the same
                # five columns from the same helper.
                {
                    **row,
                    **canonical_identity(
                        model=MODEL_ELASTIC_NET_COX,
                        cohort=getattr(args, "cohort", None),
                        endpoint=row["endpoint"],
                        landmark_days=landmark_day,
                        config="baseline" if args.baseline else "both",
                    ),
                }
                for row in out["metric_rows"]
            ]
        )
        order_canonical_first(metrics).to_csv(
            _per_landmark_path(output_dir, f"{prefix}_metrics", landmark_day), index=False
        )


def _combine_multivariable_outputs(
    cox: Any, output_dir: Path, prefix: str, landmark_days: list[int]
) -> None:
    print("\nSaved (combined):")
    if _combine_per_landmark(output_dir, "cox_agg_feature_selection", landmark_days, "cox_agg_feature_selection.csv"):
        print("  cox_agg_feature_selection.csv")
    if _combine_per_landmark(
        output_dir,
        _csv_stem(cox.HORIZON_GRID_FILENAME),
        landmark_days,
        cox.HORIZON_GRID_FILENAME,
    ):
        print(f"  {cox.HORIZON_GRID_FILENAME}")
    if prefix == "cox_agg_multivariable" and _combine_per_landmark(
        output_dir,
        _csv_stem(cox.CANONICAL_LABS_FOLDS_FILENAME),
        landmark_days,
        cox.CANONICAL_LABS_FOLDS_FILENAME,
    ):
        print(f"  {cox.CANONICAL_LABS_FOLDS_FILENAME}")
    if _combine_per_landmark(output_dir, prefix, landmark_days, f"{prefix}.csv"):
        print(f"  {prefix}.csv")
    if _combine_per_landmark(output_dir, f"{prefix}_test_auc_t", landmark_days, f"{prefix}_test_auc_t.csv"):
        print(f"  {prefix}_test_auc_t.csv")
    if _combine_per_landmark(output_dir, f"{prefix}_test_brier", landmark_days, f"{prefix}_test_brier.csv"):
        print(f"  {prefix}_test_brier.csv")
    if _combine_per_landmark(output_dir, f"{prefix}_metrics", landmark_days, f"{prefix}_metrics.csv"):
        print(f"  {prefix}_metrics.csv")
