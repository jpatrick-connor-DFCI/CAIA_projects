"""
Multivariable elastic-net Cox on the genomic (sample-anchored) cohort.

Two feature-set configs (--feature-set):
  genomics       : age + baseline covariates + genomic indicators (no labs)
  labs_genomics  : age + baseline covariates + labs + genomic indicators

This is the genomics half of the requested 4-way multivariable comparison
(baseline / baseline+labs / baseline+genomics / baseline+labs+genomics).
baseline and baseline+labs are unchanged and still come from cox_multivariable.py
(--baseline / default) on the treatment-anchored 0/90-day landmark cohort.
baseline+genomics and baseline+labs+genomics run here instead, on the
sample-anchored genomic cohort (genomic_aggregated.csv from
build_genomic_inputs.py) -- genomics are only known for patients with an
actual somatic sample, so this is evaluated on that cohort rather than being
merged (with a 0-fill-as-negative ambiguity) into the main landmark cohort.
There is no 0/90-day landmark sweep here -- one run per feature-set, anchored
to each patient's sample date.

Reuses cox_aggregated.tune_multivariable_model / fit_final_multivariable_model
(the same CV-tuned elastic-net engine cox_multivariable.py uses) and
cox_genomic_univariate._load_genomic_inputs for the cohort loader. Labs and
genomics share one raw_feature_cols universe for CV-tuning; genomic indicator
columns are exempted from the per-fold canonical-lab gate via
always_include_feature_cols (see cox_aggregated.select_feature_columns), since
they have no "canonical lab" concept and would otherwise be incorrectly
dropped by it.

Outputs (under --output-dir), one set per --feature-set value:
  cox_genomic_multivariable_<feature_set>.csv             coefs
  cox_genomic_multivariable_<feature_set>_test_auc_t.csv
  cox_genomic_multivariable_<feature_set>_test_brier.csv
  cox_genomic_multivariable_<feature_set>_metrics.csv
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
    DEFAULT_CV_L1_RATIOS,
    DEFAULT_CV_PENALIZERS,
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    ENDPOINTS,
    ID_COL,
    OUTCOME_COLUMNS,
    RESULTS,
    fit_final_multivariable_model,
    normalize_endpoints,
    select_canonical_labs,
    select_feature_columns,
    tune_multivariable_model,
)
from cox_genomic_univariate import _load_genomic_inputs  # noqa: E402
from build_genomic_inputs import detect_genomic_feature_cols  # noqa: E402
from build_prediction_inputs import DEFAULT_OUTPUT_SUBDIR  # noqa: E402

FEATURE_SETS = ("genomics", "labs_genomics")


def _prepare_genomic_context(inputs_dir: Path, *, min_genomic_prevalence: float):
    """Load the genomic cohort and derive the shared feature universe.

    Returns (aggregated, train_val, test, pre_sample_lab_df, manifest,
    static_covariate_cols, genomic_feature_cols, raw_lab_feature_cols).
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
    # be excluded explicitly here (unlike cox_genomic_univariate.py's lab-only
    # sweep, which relies on select_feature_columns's canonical-lab gate to
    # implicitly drop them) because always_include_feature_cols below would
    # otherwise let them leak back in as if they were genomic indicators.
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
    elif feature_set == "labs_genomics":
        raw_feature_cols = list(raw_lab_feature_cols) + list(genomic_feature_cols)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown feature_set {feature_set!r}")

    print(f"\n##### GENOMIC ARM 2: MULTIVARIABLE ELASTIC-NET [{feature_set}] (all endpoints) #####")
    if static_covariate_cols:
        print(f"  always-included covariates: age + {', '.join(static_covariate_cols)} (unpenalized)")
    print(f"  candidate features: {len(raw_feature_cols)} ({feature_set})")

    for endpoint in endpoints:
        print(f"\n=== {endpoint.upper()} | FEATURE-SET {feature_set} (anchor=t_sample) ===")
        print(ENDPOINTS[endpoint]["description"])
        horizon_grid = horizon_grids[endpoint]

        _, _, best_row, fold_canonical_labs_df = tune_multivariable_model(
            train_val,
            raw_feature_cols=raw_feature_cols,
            endpoint=endpoint,
            penalizers=args.cv_penalizers,
            l1_ratios=args.cv_l1_ratios,
            n_folds=args.n_folds,
            seed=args.seed,
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            pre_treatment_lab_df=pre_sample_lab_df,
            horizon_grid=horizon_grid,
            min_patient_coverage=min_patient_coverage,
            static_covariate_cols=static_covariate_cols,
            always_include_feature_cols=tuple(genomic_feature_cols),
        )
        if not fold_canonical_labs_df.empty:
            fold_canonical_labs_df = fold_canonical_labs_df.copy()
            fold_canonical_labs_df.insert(0, "feature_set", feature_set)
            out["canonical_labs_fold_rows"].append(fold_canonical_labs_df)

        # Final selected feature set for refit: re-derive on the full
        # train_val (same no-leak pattern as prepare_landmark_context /
        # cox_multivariable.py, just against this cohort's raw_feature_cols).
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

        (
            metrics_row,
            summary_df,
            _,
            test_auc_df,
            test_brier_df,
        ) = fit_final_multivariable_model(
            train_val,
            test,
            feature_cols=selected_feature_cols,
            endpoint=endpoint,
            penalizer=float(best_row["penalizer"]),
            l1_ratio=float(best_row["l1_ratio"]),
            split_stratification="prebuilt",
            cv_stratification=str(best_row["cv_stratification"]),
            auc_time_unit_days=auc_time_unit_days,
            auc_max_time_units=auc_max_time_units,
            horizon_grid=horizon_grid,
            canonical_labs=canonical_labs,
            static_covariate_cols=static_covariate_cols,
        )
        metrics_row["feature_set"] = feature_set
        summary_df.insert(0, "feature_set", feature_set)
        out["metric_rows"].append(metrics_row)
        out["frames"].append(summary_df)
        if not test_auc_df.empty:
            test_auc_df = test_auc_df.copy()
            test_auc_df.insert(0, "feature_set", feature_set)
            out["test_auc_frames"].append(test_auc_df)
        if not test_brier_df.empty:
            test_brier_df = test_brier_df.copy()
            test_brier_df.insert(0, "feature_set", feature_set)
            out["test_brier_frames"].append(test_brier_df)

        top_cols = [c for c in ["feature", "coef", "exp(coef)"] if c in summary_df.columns]
        top = summary_df.loc[~summary_df["is_age_covariate"], top_cols].head(10)
        print("\nChosen hyperparameters (elastic-net, age unpenalized):")
        print(
            f"  penalizer={best_row['penalizer']}  l1_ratio={best_row['l1_ratio']}  "
            f"cv_mean C-index={best_row['cv_mean']:.4f}"
        )
        print(f"  held-out test C-index={metrics_row['test_c_index']:.4f}")
        print(f"  held-out test mean AUC(t)={metrics_row['test_mean_auc_t']:.4f}")
        print(f"  held-out test integrated Brier={metrics_row['test_integrated_brier']:.4f}")
        print(f"Top {feature_set} coefficients:")
        print(top.to_string(index=False))


def main(args: argparse.Namespace) -> None:
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

    auc_time_unit_days = int(manifest["auc_time_unit_days"])
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
            "frames": [],
            "metric_rows": [],
            "test_auc_frames": [],
            "test_brier_frames": [],
            "canonical_labs_fold_rows": [],
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

        prefix = "cox_genomic_multivariable"
        if out["canonical_labs_fold_rows"]:
            pd.concat(out["canonical_labs_fold_rows"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_canonical_labs_folds.csv", index=False
            )
        if out["frames"]:
            pd.concat(out["frames"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}.csv", index=False
            )
        if out["test_auc_frames"]:
            pd.concat(out["test_auc_frames"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_test_auc_t.csv", index=False
            )
        if out["test_brier_frames"]:
            pd.concat(out["test_brier_frames"], ignore_index=True).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_test_brier.csv", index=False
            )
        if out["metric_rows"]:
            pd.DataFrame(out["metric_rows"]).to_csv(
                output_dir_fs / f"{prefix}_{feature_set}_metrics.csv", index=False
            )
        print(f"\nSaved [{feature_set}] under {output_dir_fs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Multivariable elastic-net Cox on the sample-anchored genomic cohort "
            "(baseline+genomics and baseline+labs+genomics feature sets)."
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
        help="Where to write cox_genomic_multivariable_<feature_set>_*.csv outputs (one subdir per feature-set).",
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
        help="Which genomics-involving feature sets to fit (default: both).",
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
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument(
        "--cv-penalizers", nargs="+", type=float, default=DEFAULT_CV_PENALIZERS,
    )
    parser.add_argument(
        "--cv-l1-ratios", nargs="+", type=float, default=DEFAULT_CV_L1_RATIOS,
    )
    main(parser.parse_args())
