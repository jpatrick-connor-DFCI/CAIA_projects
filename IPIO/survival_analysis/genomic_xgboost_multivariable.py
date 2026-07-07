"""
XGBoost survival:cox on the genomic-eligible cohort (build_genomic_inputs.py).

New file (no direct COMPASS precedent -- COMPASS never had a genomics arm).
Mirrors cox_genomic_multivariable.py's structure (no landmark sweep,
--feature-sets genomics/labs/labs_genomics) but fits XGBoost survival:cox
models instead of elastic-net Coxnet, reusing the XGBoost engine
(tune_xgboost_model / fit_final_xgboost_model / chosen_from_best_row) from
landmark_xgboost.py -- exactly how cox_genomic_multivariable.py reuses
cox_aggregated.tune_multivariable_model / fit_final_multivariable_model.

Three feature-set configs (--feature-sets):
  genomics       : age + baseline covariates + genomic indicators (no labs)
  labs           : age + baseline covariates + labs (no genomics)
  labs_genomics  : age + baseline covariates + labs + genomic indicators

Anchored to IO_START (t_first_treatment = 0), the SAME time origin as the main
cohort's landmark 0 -- NOT the somatic sample collection date (see
build_genomic_inputs.py's module docstring for why). There is no 0/90-day
landmark sweep here -- one run per feature-set. Genomic indicator columns are
exempted from the per-fold canonical-lab gate via always_include_feature_cols
(see cox_aggregated.select_feature_columns's `always_include` parameter),
since they have no "canonical lab" concept and would otherwise be incorrectly
dropped by it -- same fix cox_genomic_multivariable.py already needed for Cox.

Outputs (under --output-dir), one set per --feature-set value:
  xgboost_genomic_multivariable_<feature_set>_metrics.csv
  xgboost_genomic_multivariable_<feature_set>_feature_importance.csv
  xgboost_genomic_multivariable_<feature_set>_patient_risks.csv
  xgboost_genomic_multivariable_<feature_set>_auc_t.csv
  xgboost_genomic_multivariable_<feature_set>_brier.csv
  xgboost_genomic_multivariable_<feature_set>_cv_folds.csv
  xgboost_genomic_multivariable_<feature_set>_cv_summary.csv
  xgboost_genomic_multivariable_<feature_set>_canonical_labs_folds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent           # .../survival_analysis (IPIO)
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

import cox_aggregated as _ca  # noqa: E402
from cox_aggregated import (  # noqa: E402
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    ENDPOINTS,
    ID_COL,
    OUTCOME_COLUMNS,
    RESULTS,
    normalize_endpoints,
    select_canonical_labs,
    select_feature_columns,
)
from cox_genomic_univariate import _load_genomic_inputs  # noqa: E402
from build_genomic_inputs import detect_genomic_feature_cols  # noqa: E402
from build_prediction_inputs import DEFAULT_OUTPUT_SUBDIR  # noqa: E402
from landmark_xgboost import (  # noqa: E402
    DEFAULT_CV_ETAS,
    DEFAULT_CV_MAX_DEPTHS,
    DEFAULT_CV_MIN_CHILD_WEIGHTS,
    chosen_from_best_row,
    fit_final_xgboost_model,
    require_lifelines,
    require_xgboost,
    tune_xgboost_model,
)

FEATURE_SETS = ("genomics", "labs", "labs_genomics")


def _prepare_genomic_context(inputs_dir: Path, *, min_genomic_prevalence: float):
    """Load the genomic cohort and derive the shared feature universe.

    Identical to cox_genomic_multivariable.py's helper of the same name --
    duplicated here (rather than imported) because the two scripts are meant
    to work standalone and this cohort-prep logic is model-agnostic. Returns
    (aggregated, train_val, test, pre_sample_lab_df, manifest,
    min_patient_coverage, static_covariate_cols, genomic_feature_cols,
    raw_lab_feature_cols).
    """
    aggregated, pre_sample_lab_df, manifest = _load_genomic_inputs(inputs_dir)
    min_patient_coverage = float(manifest["min_patient_coverage"])
    print(
        f"Loaded genomic inputs from {inputs_dir / 'genomic'} "
        f"(min_patient_coverage={min_patient_coverage} per build manifest)"
    )

    train_val = aggregated.loc[aggregated["split"].isin(["train", "valid"])].copy()
    test = aggregated.loc[aggregated["split"].eq("test")].copy()
    print(f"Cohort: train+valid={len(train_val)} test={len(test)} total={len(aggregated)}")

    static_covariate_cols = tuple(_ca.baseline_covariate_columns(aggregated))
    static_set = set(static_covariate_cols)

    genomic_feature_cols = detect_genomic_feature_cols(aggregated.columns)
    prevalence = aggregated[genomic_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).mean()
    below_threshold = prevalence.loc[prevalence < min_genomic_prevalence].index.tolist()
    if below_threshold:
        print(
            f"Dropping {len(below_threshold)}/{len(genomic_feature_cols)} genomic indicators "
            f"below {min_genomic_prevalence:.1%} prevalence for the multivariable arm too: "
            f"{', '.join(sorted(below_threshold))}"
        )
    genomic_feature_cols = [c for c in genomic_feature_cols if c not in set(below_threshold)]
    print(f"Genomic indicator candidates: {len(genomic_feature_cols)}")

    # Raw lab feature universe: everything not an outcome/split column, not a
    # baseline covariate, and not a genomic indicator. Baseline covariates must
    # be excluded explicitly here (rather than relying on the canonical-lab
    # gate to implicitly drop them) because always_include_feature_cols below
    # would otherwise let them leak back in as if they were genomic indicators.
    genomic_set = set(genomic_feature_cols) | set(below_threshold)
    raw_lab_feature_cols = [
        c for c in aggregated.columns
        if c not in OUTCOME_COLUMNS and c not in genomic_set and c not in static_set
    ]

    return (
        aggregated,
        train_val,
        test,
        pre_sample_lab_df,
        manifest,
        min_patient_coverage,
        static_covariate_cols,
        genomic_feature_cols,
        raw_lab_feature_cols,
    )


def _run_feature_set(
    *,
    feature_set,
    aggregated,
    train_val,
    test,
    pre_sample_lab_df,
    min_patient_coverage,
    static_covariate_cols,
    genomic_feature_cols,
    raw_lab_feature_cols,
    endpoints,
    args,
    auc_time_unit_days,
    auc_max_time_units,
    horizon_grids,
    out,
):
    if feature_set == "genomics":
        raw_feature_cols = list(genomic_feature_cols)
        always_include_feature_cols = tuple(genomic_feature_cols)
    elif feature_set == "labs":
        raw_feature_cols = list(raw_lab_feature_cols)
        always_include_feature_cols = ()
    elif feature_set == "labs_genomics":
        raw_feature_cols = list(raw_lab_feature_cols) + list(genomic_feature_cols)
        always_include_feature_cols = tuple(genomic_feature_cols)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown feature_set {feature_set!r}")

    print(f"\n##### GENOMIC XGBOOST ARM [{feature_set}] (survival:cox, all endpoints) #####")
    if static_covariate_cols:
        print(f"  always-included covariates: age + {', '.join(static_covariate_cols)}")
    print(f"  candidate features: {len(raw_feature_cols)} ({feature_set})")

    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | FEATURE-SET {feature_set} (anchor=IO_START) ===")
        print(ENDPOINTS[endpoint]["description"])
        horizon_grid = horizon_grids[endpoint]

        chosen = None
        fold_df = pd.DataFrame()
        cv_df = pd.DataFrame()
        fold_canonical_labs_df = pd.DataFrame()
        if not args.no_cv:
            fold_df, cv_df, best_row, fold_canonical_labs_df = tune_xgboost_model(
                train_val,
                raw_feature_cols=raw_feature_cols,
                endpoint=endpoint,
                pre_treatment_lab_df=pre_sample_lab_df,
                horizon_grid=horizon_grid,
                min_patient_coverage=min_patient_coverage,
                static_covariate_cols=static_covariate_cols,
                always_include_feature_cols=always_include_feature_cols,
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

        # Final selected feature set for refit: re-derive on the full
        # train_val (same no-leak pattern as cox_genomic_multivariable.py),
        # exempting genomic indicators from the canonical-lab gate.
        canonical_labs = select_canonical_labs(
            pre_sample_lab_df,
            mrns=train_val.index,
            min_coverage=min_patient_coverage,
            id_col=ID_COL,
        )
        selected_feature_cols, _ = select_feature_columns(
            train_val,
            raw_feature_cols,
            min_patient_coverage=min_patient_coverage,
            restrict_to_labs=canonical_labs,
            always_include=list(genomic_feature_cols),
        )
        if args.max_features is not None and len(selected_feature_cols) > args.max_features:
            # Simple coverage-ranked cap; genomic indicators are exempt from
            # neither this cap nor the canonical-lab gate is more nuanced than
            # needed here -- rank by coverage like the landmark script.
            coverage = train_val[selected_feature_cols].notna().mean().sort_values(ascending=False)
            selected_feature_cols = coverage.head(args.max_features).index.tolist()

        metrics, importance, predictions, auc_t, brier_t = fit_final_xgboost_model(
            train_val,
            test,
            feature_cols=selected_feature_cols,
            endpoint=endpoint,
            chosen=chosen,
            static_covariate_cols=static_covariate_cols,
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grid=horizon_grid,
            canonical_labs=canonical_labs,
            args=args,
        )

        for df in (metrics, importance, predictions, auc_t, brier_t, fold_df, cv_df, fold_canonical_labs_df):
            if not df.empty:
                df.insert(0, "feature_set", feature_set)

        out["metrics"].append(metrics)
        out["auc_t"].append(auc_t)
        if not brier_t.empty:
            out["brier"].append(brier_t)
        out["risks"].append(predictions)
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


def main(args: argparse.Namespace) -> None:
    require_xgboost()
    require_lifelines()
    inputs_dir = Path(args.inputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = normalize_endpoints(args.endpoints)
    feature_sets = list(dict.fromkeys(args.feature_sets))  # de-dup, preserve order

    (
        aggregated,
        train_val,
        test,
        pre_sample_lab_df,
        manifest,
        min_patient_coverage,
        static_covariate_cols,
        genomic_feature_cols,
        raw_lab_feature_cols,
    ) = _prepare_genomic_context(inputs_dir, min_genomic_prevalence=args.min_genomic_prevalence)

    args.auc_time_unit_days = int(manifest["auc_time_unit_days"])
    args.auc_quantiles = tuple(manifest.get("auc_quantiles", ()))
    auc_time_unit_days = args.auc_time_unit_days
    auc_max_time_units = args.auc_max_time_units
    auc_horizons = manifest["auc_horizons"]
    horizon_grids = {}
    for endpoint in endpoints:
        if endpoint not in auc_horizons:
            raise KeyError(
                f"genomic_build_manifest.json has no auc_horizons entry for endpoint {endpoint!r}. "
                "Re-run build_genomic_inputs.py."
            )
        horizon_grids[endpoint] = np.asarray(auc_horizons[endpoint], dtype=float)

    for feature_set in feature_sets:
        output_dir_fs = output_dir / feature_set
        output_dir_fs.mkdir(parents=True, exist_ok=True)
        out = {
            "metrics": [],
            "auc_t": [],
            "brier": [],
            "risks": [],
            "importance": [],
            "cv_folds": [],
            "cv_summary": [],
            "canonical_labs_folds": [],
        }
        _run_feature_set(
            feature_set=feature_set,
            aggregated=aggregated,
            train_val=train_val,
            test=test,
            pre_sample_lab_df=pre_sample_lab_df,
            min_patient_coverage=min_patient_coverage,
            static_covariate_cols=static_covariate_cols,
            genomic_feature_cols=genomic_feature_cols,
            raw_lab_feature_cols=raw_lab_feature_cols,
            endpoints=endpoints,
            args=args,
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grids=horizon_grids,
            out=out,
        )

        prefix = "xgboost_genomic_multivariable"
        saved = []
        if out["metrics"]:
            pd.concat(out["metrics"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_metrics.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_metrics.csv")
        if out["importance"]:
            pd.concat(out["importance"], ignore_index=True, sort=False).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_feature_importance.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_feature_importance.csv")
        if out["risks"]:
            pd.concat(out["risks"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_patient_risks.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_patient_risks.csv")
        if out["auc_t"]:
            pd.concat(out["auc_t"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_auc_t.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_auc_t.csv")
        if out["brier"]:
            pd.concat(out["brier"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_brier.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_brier.csv")
        if out["cv_folds"]:
            pd.concat(out["cv_folds"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_cv_folds.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_cv_folds.csv")
        if out["cv_summary"]:
            pd.concat(out["cv_summary"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_cv_summary.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_cv_summary.csv")
        if out["canonical_labs_folds"]:
            pd.concat(out["canonical_labs_folds"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_canonical_labs_folds.csv", index=False
            )
            saved.append(f"{prefix}_{feature_set}_canonical_labs_folds.csv")
        print(f"\nSaved [{feature_set}] under {output_dir_fs}:")
        for name in saved:
            print(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "XGBoost survival:cox on the genomic-eligible cohort, anchored to "
            "IO_START (genomics, labs, and labs+genomics feature sets)."
        )
    )
    parser.add_argument(
        "--inputs-dir",
        default=str(RESULTS / DEFAULT_OUTPUT_SUBDIR),
        help="Existing prediction_inputs dir (genomic subdir is read from <inputs-dir>/genomic).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS),
        help="Where to write xgboost_genomic_multivariable_<feature_set>_*.csv outputs (one subdir per feature-set).",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["irae"],
        choices=list(ENDPOINTS),
        help="Endpoints to analyze.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(FEATURE_SETS),
        choices=list(FEATURE_SETS),
        help="Which genomics-involving feature sets to fit (default: all three).",
    )
    parser.add_argument(
        "--min-genomic-prevalence",
        type=float,
        default=0.025,
        help="Only include genomic indicators with at least this positive-call frequency (default 2.5%%).",
    )
    parser.add_argument(
        "--auc-max-time-units",
        type=int,
        default=None,
        help="Cap (in time-units) for the IPCW AUC(t)/Brier evaluation horizons.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=None)
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
    main(parser.parse_args())
