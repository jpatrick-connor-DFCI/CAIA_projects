"""
Univariate Cox associations for the genomic-landmark survival arm.

Two sweeps per endpoint:
  1) n_obs-adjusted lab features (Arm 1b style):
       Cox on [AGE + LAB__n_observations + feature], filtered to canonical labs.
  2) Genomic indicators ({TP53,RB1,PTEN} x {SV,DEL,AMP,SNV}):
       Cox on [AGE + indicator]. No SD scaling, no n_obs term.

Reads inputs from <inputs-dir>/genomic written by build_genomic_inputs.py.
Reuses cox_aggregated helpers; the patient split is whatever was assigned by the
main pipeline's split_assignments.csv (decision §0.2).

Outputs (under --output-dir):
  cox_genomic_univariate_lab_features.csv
  cox_genomic_univariate_genomic_features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from cox_aggregated import (  # noqa: E402
    AGE_COL,
    ENDPOINTS,
    OUTCOME_COLUMNS,
    RESULTS,
    benjamini_hochberg,
    normalize_endpoints,
    print_top_hits,
    require_lifelines,
    run_univariate_nobs_adjusted_associations,
    select_canonical_labs,
    select_feature_columns,
)
from build_genomic_inputs import (  # noqa: E402
    GENOMIC_AGGREGATED_FILENAME,
    GENOMIC_BUILD_MANIFEST_FILENAME,
    GENOMIC_FEATURE_COLS,
    GENOMIC_OUTPUT_SUBDIR,
    GENOMIC_PRE_SAMPLE_LAB_FILENAME,
)
from build_prediction_inputs import DEFAULT_OUTPUT_SUBDIR  # noqa: E402

try:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError
    LIFELINES_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    CoxPHFitter = None
    ConvergenceError = RuntimeError
    LIFELINES_IMPORT_ERROR = exc

LAB_FEATURES_FILENAME = "cox_genomic_univariate_lab_features.csv"
GENOMIC_FEATURES_FILENAME = "cox_genomic_univariate_genomic_features.csv"


def _load_genomic_inputs(inputs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    genomic_dir = inputs_dir / GENOMIC_OUTPUT_SUBDIR
    manifest_path = genomic_dir / GENOMIC_BUILD_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run build_genomic_inputs.py first."
        )
    manifest = json.loads(manifest_path.read_text())

    agg_path = genomic_dir / GENOMIC_AGGREGATED_FILENAME
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}.")
    aggregated = pd.read_csv(agg_path).set_index("DFCI_MRN")
    if "split" not in aggregated.columns:
        raise ValueError(f"{agg_path} missing 'split' column.")

    pre_path = genomic_dir / GENOMIC_PRE_SAMPLE_LAB_FILENAME
    if not pre_path.exists():
        raise FileNotFoundError(f"Missing {pre_path}.")
    pre_sample_lab_df = pd.read_csv(pre_path)
    return aggregated, pre_sample_lab_df, manifest


def run_genomic_indicator_univariate(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> pd.DataFrame:
    """Cox on [AGE + indicator] per binary genomic feature, per endpoint."""
    require_lifelines()
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
    rows: list[dict] = []
    total_patients = len(data)
    for feature in feature_cols:
        result = {
            "endpoint": endpoint,
            "feature": feature,
            "n_patients_total": total_patients,
            "n_patients_used": 0,
            "n_indicator_positive": 0,
            "n_events_used": 0,
            "n_events_indicator_positive": 0,
            "coef_indicator": np.nan,
            "hazard_ratio": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "coef_age": np.nan,
            "p_value_age": np.nan,
            "fit_penalizer": np.nan,
            "note": "",
        }
        if feature not in data.columns:
            result["note"] = "missing_feature_column"
            rows.append(result)
            continue
        keep_cols = [AGE_COL, feature, duration_col, event_col]
        sub = data[keep_cols].dropna()
        if sub.empty:
            result["note"] = "empty_after_dropna"
            rows.append(result)
            continue
        sub = sub.copy()
        sub[event_col] = sub[event_col].astype(int)
        sub[feature] = sub[feature].astype(int)
        n_events = int(sub[event_col].sum())
        result["n_patients_used"] = int(len(sub))
        result["n_indicator_positive"] = int(sub[feature].sum())
        result["n_events_used"] = n_events
        result["n_events_indicator_positive"] = int(sub.loc[sub[feature].eq(1), event_col].sum())
        if n_events < min_events_per_feature:
            result["note"] = f"too_few_events({n_events}<{min_events_per_feature})"
            rows.append(result)
            continue
        if sub[feature].nunique() < 2:
            result["note"] = "indicator_constant"
            rows.append(result)
            continue
        last_error = ""
        for penalizer in (0.0, float(fallback_penalizer)):
            try:
                cph = CoxPHFitter(penalizer=float(penalizer))
                cph.fit(sub, duration_col=duration_col, event_col=event_col)
            except (ConvergenceError, ValueError, np.linalg.LinAlgError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            summary = cph.summary
            if feature not in summary.index:
                last_error = "feature_dropped_from_summary"
                continue
            result["coef_indicator"] = float(summary.loc[feature, "coef"])
            result["hazard_ratio"] = float(summary.loc[feature, "exp(coef)"])
            result["ci_lower"] = float(summary.loc[feature, "exp(coef) lower 95%"])
            result["ci_upper"] = float(summary.loc[feature, "exp(coef) upper 95%"])
            result["p_value"] = float(summary.loc[feature, "p"])
            if AGE_COL in summary.index:
                result["coef_age"] = float(summary.loc[AGE_COL, "coef"])
                result["p_value_age"] = float(summary.loc[AGE_COL, "p"])
            result["fit_penalizer"] = float(penalizer)
            result["note"] = "fit_ok" if penalizer == 0.0 else f"fit_ok_fallback_penalizer_{penalizer:g}"
            last_error = ""
            break
        if not result["note"]:
            result["note"] = f"fit_failed: {last_error}"
        rows.append(result)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = benjamini_hochberg(out["p_value"])
    return out


def main(args: argparse.Namespace) -> None:
    require_lifelines()
    inputs_dir = Path(args.inputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = normalize_endpoints(args.endpoints)

    aggregated, pre_sample_lab_df, manifest = _load_genomic_inputs(inputs_dir)
    min_patient_coverage = float(manifest["min_patient_coverage"])
    print(
        f"Loaded genomic inputs from {inputs_dir / GENOMIC_OUTPUT_SUBDIR} "
        f"(min_patient_coverage={min_patient_coverage} per build manifest)"
    )
    print(
        "Using full post-sample follow-up for univariate Cox; no DeepHit-style "
        "admin censor is applied."
    )

    train_val = aggregated.loc[aggregated["split"].isin(["train", "valid"])].copy()
    test = aggregated.loc[aggregated["split"].eq("test")].copy()
    print(f"Cohort: train+valid={len(train_val)} test={len(test)} total={len(aggregated)}")

    # raw lab features = aggregated cols not in OUTCOME_COLUMNS, not genomic indicators
    genomic_set = set(GENOMIC_FEATURE_COLS)
    raw_lab_feature_cols = [
        c for c in aggregated.columns
        if c not in OUTCOME_COLUMNS and c not in genomic_set
    ]

    canonical_labs = select_canonical_labs(
        pre_sample_lab_df,
        mrns=train_val.index,
        min_coverage=min_patient_coverage,
    )
    print(f"Canonical labs (train+valid pre-sample): {len(canonical_labs)}")

    selected_lab_features, _ = select_feature_columns(
        train_val,
        raw_lab_feature_cols,
        min_patient_coverage=min_patient_coverage,
        restrict_to_labs=canonical_labs,
    )
    print(f"Selected lab features (train+valid pre-filter): {len(selected_lab_features)}")

    lab_frames: list[pd.DataFrame] = []
    genomic_frames: list[pd.DataFrame] = []

    keep_lab_cols = [
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

    for endpoint in endpoints:
        cfg = ENDPOINTS[endpoint]
        print(f"\n=== {endpoint.upper()} (anchor=t_sample) ===")
        print(cfg["description"])

        # Lab features (n_obs-adjusted)
        lab_df = run_univariate_nobs_adjusted_associations(
            aggregated,
            feature_cols=selected_lab_features,
            endpoint=endpoint,
            min_events_per_feature=args.min_events_per_feature,
            fallback_penalizer=args.univariate_penalizer,
        )
        present = [c for c in keep_lab_cols if c in lab_df.columns]
        lab_frames.append(lab_df[present].copy())
        print_top_hits(lab_df, endpoint=endpoint, label="lab univariate (n_obs adj)")

        # Genomic indicators
        gen_df = run_genomic_indicator_univariate(
            aggregated,
            feature_cols=GENOMIC_FEATURE_COLS,
            endpoint=endpoint,
            min_events_per_feature=args.min_events_per_feature,
            fallback_penalizer=args.univariate_penalizer,
        )
        genomic_frames.append(gen_df)
        if not gen_df.empty:
            top = (
                gen_df.dropna(subset=["p_value"])
                .sort_values("p_value")
                .head(10)[["feature", "hazard_ratio", "p_value", "q_value"]]
            )
            print(f"Top genomic-indicator associations for {endpoint}:")
            print(top.to_string(index=False) if not top.empty else "  (no estimable fits)")

    if lab_frames:
        lab_path = output_dir / LAB_FEATURES_FILENAME
        pd.concat(lab_frames, ignore_index=True).to_csv(lab_path, index=False)
        print(f"\nWrote {lab_path}")
    if genomic_frames:
        gen_path = output_dir / GENOMIC_FEATURES_FILENAME
        pd.concat(genomic_frames, ignore_index=True).to_csv(gen_path, index=False)
        print(f"Wrote {gen_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs-dir",
        default=str(RESULTS / DEFAULT_OUTPUT_SUBDIR),
        help="Existing prediction_inputs dir (genomic subdir is read from <inputs-dir>/genomic).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS),
        help="Where to write cox_genomic_univariate_*.csv outputs.",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["platinum", "death"],
        choices=list(ENDPOINTS),
    )
    parser.add_argument("--min-events-per-feature", type=int, default=10)
    parser.add_argument(
        "--univariate-penalizer",
        type=float,
        default=0.05,
        help="Fallback penalizer when an unpenalized Cox fit doesn't converge.",
    )
    main(parser.parse_args())
