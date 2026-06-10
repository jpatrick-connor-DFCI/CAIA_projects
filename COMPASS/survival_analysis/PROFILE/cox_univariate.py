"""
Univariate (n_obs-adjusted) Cox associations on landmarked lab summary features.

Arm 1b: for each selected lab summary feature, fit Cox on
``[AGE + matching LAB__n_observations + feature]`` using all patients at the
landmark, and report the log HR, HR-per-SD, 95% CI, and p-/q-values.

Reads prebuilt inputs from build_prediction_inputs.py (run that first). Shares
the cohort / canonical-lab / feature-selection setup with cox_multivariable.py
via cox_aggregated.prepare_landmark_context, so the two arms operate on an
identical patient set and feature set (only the model differs).

Outputs (under --output-dir):
  cox_agg_feature_selection.csv         selected lab features + coverage
  cox_agg_univariate_nobs_adjusted.csv  n_obs-adjusted log HRs, p, q
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
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_MIN_EVENTS_PER_FEATURE,
    ENDPOINTS,
    ID_COL,
    RESULTS,
    _load_build_manifest,
    normalize_endpoints,
    normalize_landmark_days,
    prepare_landmark_context,
    print_top_hits,
    run_univariate_nobs_adjusted_associations,
)

# Column subset (and order) written to cox_agg_univariate_nobs_adjusted.csv.
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
]


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
    print(
        f"Loading prebuilt prediction inputs from {inputs_dir} "
        f"(min_patient_coverage={min_patient_coverage})"
    )

    feature_selection_frames: list[pd.DataFrame] = []
    univariate_frames: list[pd.DataFrame] = []

    for landmark_day in landmark_days:
        ctx = prepare_landmark_context(
            inputs_dir,
            landmark_day,
            min_patient_coverage=min_patient_coverage,
            restrict_to_stage=args.restrict_to_stage,
        )
        feature_selection_frames.append(ctx.feature_meta_selected)

        print("\n##### ARM 1: UNIVARIATE (n_obs-adjusted, full follow-up, all endpoints) #####")
        for endpoint in endpoints:
            print(f"\n=== {endpoint.upper()} | LANDMARK +{landmark_day}D ===")
            print(ENDPOINTS[endpoint]["description"])
            adjusted_df = run_univariate_nobs_adjusted_associations(
                ctx.univariate_data,
                feature_cols=ctx.selected_feature_cols,
                endpoint=endpoint,
                min_events_per_feature=args.min_events_per_feature,
                fallback_penalizer=args.univariate_penalizer,
            )
            adjusted_df.insert(0, "landmark_days", landmark_day)
            univariate_frames.append(adjusted_df[UNIVARIATE_KEEP_COLS].copy())
            print_top_hits(
                adjusted_df,
                endpoint=endpoint,
                label="n_obs-adjusted univariate",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Univariate (n_obs-adjusted) Cox associations on landmarked lab features."
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
        help="Directory for Cox univariate result CSVs.",
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
        "--restrict-to-stage",
        action="store_true",
        help=(
            "Restrict the cohort to stage-available patients (non-missing "
            "CANCER_STAGE_*) before fitting, for a complete-case comparison on a "
            "matched population. Errors if no stage columns are present (PROFILE only)."
        ),
    )
    parser.add_argument(
        "--min-events-per-feature",
        type=int,
        default=DEFAULT_MIN_EVENTS_PER_FEATURE,
        help="Skip univariate associations when too few endpoint events remain after outcome filtering.",
    )
    parser.add_argument(
        "--univariate-penalizer",
        type=float,
        default=0.05,
        help="Fallback penalizer used only when a univariate Cox model does not converge without regularization.",
    )
    main(parser.parse_args())
