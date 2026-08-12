"""
Build prediction inputs for the genomic-landmark univariate survival arm.

Index time = earliest SAMPLE_COLLECTION_DT (per patient, from the raw
per-sample somatic matrix SOMATIC_WIDE_BY_SAMPLE.parquet).
Predicts platinum exposure from sample collection forward, with
features derived from labs measured strictly before t_sample plus 12 binary
genomic indicators ({TP53, RB1, PTEN} x {SV, DEL, AMP, SNV}).

Cohort = longitudinal cohort INTERSECTED with patients that have a genomic
sample AND have a split label in the existing
prediction_inputs/split_assignments.csv (so test stays test across arms).

Outputs (under <inputs-dir>/genomic):
  genomic_aggregated.csv             one row per MRN: lab features + genomic
                                     indicators + outcomes rebased to t_sample +
                                     split column
  pre_sample_lab_long.csv            long-format pre-sample labs for per-fold
                                     canonical-lab selection
  genomic_canonical_labs_train_val.csv  landmark='sample', lab_name
  genomic_build_manifest.json        provenance + AUC horizons + cohort sizes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SURVIVAL_DIR = PROJECT_DIR / "survival_analysis"
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(REPO_ROOT), str(PROJECT_DIR), str(SURVIVAL_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cox_aggregated import (  # noqa: E402
    DATA_PATH,
    DEFAULT_MIN_PATIENT_COVERAGE,
    ENDPOINTS,
    RESULTS,
)
from survival_common.cohort import (  # noqa: E402
    AGE_COL,
    ID_COL,
    build_feature_matrix,
    build_pre_treatment_lab_long,
    make_outcome_df,
)
from survival_common.helper import (  # noqa: E402
    AUC_TIMELINE_SCHEMA_VERSION,
    DEFAULT_AUC_MAX_GRID_POINTS,
    DEFAULT_AUC_MAX_TIME_UNITS,
    DEFAULT_AUC_QUANTILES,
    assert_no_test_leakage,
    compute_horizon_grid,
    select_canonical_labs,
)
from build_prediction_inputs import (  # noqa: E402
    DEFAULT_OUTPUT_SUBDIR,
    SPLIT_ASSIGNMENTS_FILENAME,
)

import os  # noqa: E402

# The raw per-sample somatic matrix, not the clinical text embedding project's
# complete_somatic_data_df.csv.gz. That derived file is gated on a pan-cancer,
# note-filtered cohort and additionally drops any patient sequenced after their
# first treatment -- which removes most prostate patients, since sequencing
# usually follows ADT. Reading the raw matrix and doing our own per-patient
# sample selection keeps the COMPASS cohort intact.
PROFILE_DATA_ROOT = Path(
    os.environ.get("PROFILE_DATA_PATH", "/data/gusev/USERS/jpconnor/data/PROFILE_DATA/")
)
DEFAULT_SOMATIC_PATH = PROFILE_DATA_ROOT / "SOMATIC_WIDE_BY_SAMPLE.parquet"
DEFAULT_TIME_UNIT_DAYS = 7
GENOMIC_GENES = ("TP53", "RB1", "PTEN")
GENOMIC_VARIANT_TYPES = ("SV", "DEL", "AMP", "SNV")
GENOMIC_FEATURE_COLS = [f"{g}_{v}" for g in GENOMIC_GENES for v in GENOMIC_VARIANT_TYPES]

GENOMIC_OUTPUT_SUBDIR = "genomic"
GENOMIC_AGGREGATED_FILENAME = "genomic_aggregated.csv"
GENOMIC_PRE_SAMPLE_LAB_FILENAME = "pre_sample_lab_long.csv"
GENOMIC_CANONICAL_LABS_FILENAME = "genomic_canonical_labs_train_val.csv"
GENOMIC_BUILD_MANIFEST_FILENAME = "genomic_build_manifest.json"


def load_somatic(path: Path) -> pd.DataFrame:
    """Load the per-sample somatic matrix and collapse it to one row per patient.

    The raw matrix carries one row per (patient, sample, test type), so a
    patient with several specimens appears several times. The earliest dated
    sample is used as the index time, and the alteration indicators are OR-ed
    across that patient's samples: a call made on any specimen is a call.
    """
    needed_cols = [ID_COL, "SAMPLE_COLLECTION_DT", *GENOMIC_FEATURE_COLS]
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path, columns=None)
        raw = raw[[c for c in raw.columns if c in needed_cols]]
    else:
        raw = pd.read_csv(path, usecols=lambda c: c in needed_cols)
    missing = [c for c in needed_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Somatic matrix {path} missing columns: {missing}")
    raw[ID_COL] = pd.to_numeric(raw[ID_COL], errors="coerce")
    raw = raw.loc[raw[ID_COL].notna()].copy()
    raw[ID_COL] = raw[ID_COL].astype(int)
    raw["SAMPLE_COLLECTION_DT"] = pd.to_datetime(raw["SAMPLE_COLLECTION_DT"], errors="coerce")
    for col in GENOMIC_FEATURE_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)

    n_rows, n_patients = len(raw), raw[ID_COL].nunique()
    if n_rows != n_patients:
        print(
            f"Somatic matrix has {n_rows:,} sample rows for {n_patients:,} patients; "
            "collapsing to earliest sample date with alterations OR-ed across samples."
        )
    collapsed = raw.groupby(ID_COL).agg(
        {
            "SAMPLE_COLLECTION_DT": "min",
            **{col: "max" for col in GENOMIC_FEATURE_COLS},
        }
    )
    return collapsed


def attach_t_sample(df: pd.DataFrame, somatic: pd.DataFrame) -> pd.DataFrame:
    """Merge SAMPLE_COLLECTION_DT into df and compute t_sample (days from FIRST_RECORD_DATE)."""
    if "FIRST_RECORD_DATE" not in df.columns:
        raise ValueError("Longitudinal df missing FIRST_RECORD_DATE; cannot derive t_sample.")
    out = df.copy()
    out["FIRST_RECORD_DATE"] = pd.to_datetime(out["FIRST_RECORD_DATE"], errors="coerce")
    sample_dt = somatic["SAMPLE_COLLECTION_DT"]
    out["SAMPLE_COLLECTION_DT"] = out[ID_COL].map(sample_dt)
    out["t_sample"] = (
        out["SAMPLE_COLLECTION_DT"] - out["FIRST_RECORD_DATE"]
    ).dt.days.astype(float)
    return out


def main(args: argparse.Namespace) -> None:
    inputs_dir = Path(args.inputs_dir)
    output_dir = inputs_dir / GENOMIC_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = inputs_dir / SPLIT_ASSIGNMENTS_FILENAME
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}. Run build_prediction_inputs.py first; the "
            "genomic arm reuses its split."
        )
    split_assignments = (
        pd.read_csv(split_path).set_index(ID_COL)["split"].astype(str)
    )

    print(f"Loading longitudinal data from {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False)
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce")
    df = df.loc[df[ID_COL].notna()].copy()
    df[ID_COL] = df[ID_COL].astype(int)
    print(f"Loaded cohort: {df[ID_COL].nunique()} unique MRNs")

    somatic = load_somatic(Path(args.somatic_path))
    print(f"Somatic patients: {len(somatic)}")

    n_before_tsample = df[ID_COL].nunique()
    df = attach_t_sample(df, somatic)
    has_tsample = df["t_sample"].notna()
    n_with_tsample = df.loc[has_tsample, ID_COL].nunique()
    n_negative = df.loc[has_tsample & (df["t_sample"] < 0), ID_COL].nunique()
    df = df.loc[has_tsample].copy()
    print(
        f"Cohort after t_sample join: {n_with_tsample} patients with a sample date "
        f"(dropped {n_before_tsample - n_with_tsample} without one). "
        f"{n_negative} have a sample dated before their first record and will be "
        f"dropped by the landmark filter below."
    )

    # Outcome table rebased to t_sample (drops t_sample<0 and any sample at/after an event)
    outcome_df = make_outcome_df(
        df,
        landmark_offset_days=0,
        anchor_col="t_sample",
        extra_anchor_cols=("t_sample", "SAMPLE_COLLECTION_DT"),
        require_first_treatment=False,
    )
    print(
        f"Outcome cohort (post t_sample landmark filter): {len(outcome_df)} patients "
        f"(dropped {n_with_tsample - len(outcome_df)} with t_sample<0 or an event at/before the sample)"
    )

    # Per-patient lab summary features (pre-sample window)
    feature_df = build_feature_matrix(
        df,
        landmark_offset_days=0,
        anchor_col="t_sample",
    )
    print(f"Feature matrix: {feature_df.shape[0]} patients x {feature_df.shape[1]} lab features")

    # Inner join + attach genomics
    merged = feature_df.join(outcome_df, how="inner")
    merged = merged.loc[merged[AGE_COL].notna()].copy()
    if merged.empty:
        raise ValueError("No patients survived feature+outcome join in the genomic arm.")

    genomics = somatic.loc[somatic.index.intersection(merged.index), GENOMIC_FEATURE_COLS]
    # This arm is anchored on t_sample, so every surviving patient has a somatic
    # row by construction; untested patients were dropped by the t_sample join
    # rather than filled. The fill below is a guard, not the missingness policy.
    n_missing_genomics = int(merged.index.difference(somatic.index).size)
    if n_missing_genomics:
        raise ValueError(
            f"{n_missing_genomics} patients reached the genomic join without a somatic "
            "row despite being anchored on t_sample; the somatic index is inconsistent."
        )
    merged = merged.join(genomics, how="left")
    for col in GENOMIC_FEATURE_COLS:
        merged[col] = merged[col].fillna(0).astype(int)
    print(f"Cohort with genomics joined: {len(merged)} patients")

    # Reuse main split — drop patients without a label there
    aligned_split = split_assignments.reindex(merged.index)
    n_no_split = int(aligned_split.isna().sum())
    if n_no_split:
        print(
            f"Dropping {n_no_split} MRNs that lack a label in {split_path.name} "
            "(genomic cohort > main cohort)"
        )
    merged = merged.loc[aligned_split.notna()].copy()
    merged["split"] = aligned_split.loc[merged.index].astype(str)

    train_val = merged.loc[merged["split"].isin(["train", "valid"])]
    test = merged.loc[merged["split"].eq("test")]
    print(
        f"Final genomic cohort: train+valid={len(train_val)}  test={len(test)} "
        f"(total {len(merged)})"
    )
    if train_val.empty or test.empty:
        raise ValueError("Genomic cohort has empty train_val or test after split alignment.")
    # Guard: reused split must keep test disjoint from train+valid.
    assert_no_test_leakage(
        test_mrns=set(test.index),
        train_mrns=set(train_val.index),
        context="build_genomic_inputs: test vs train+valid",
    )

    agg_path = output_dir / GENOMIC_AGGREGATED_FILENAME
    merged.rename_axis(ID_COL).reset_index().to_csv(agg_path, index=False)
    print(f"Wrote {agg_path}")

    # Pre-sample lab long for per-fold canonical labs
    anchor_series = outcome_df["t_sample"].astype(float)
    pre_sample_lab_df = build_pre_treatment_lab_long(
        df,
        cohort_index=merged.index,
        landmark_offset_days=0,
        anchor_col="t_sample",
        anchor_series=anchor_series,
    )
    pre_path = output_dir / GENOMIC_PRE_SAMPLE_LAB_FILENAME
    pre_sample_lab_df.to_csv(pre_path, index=False)
    print(f"Wrote {pre_path} ({len(pre_sample_lab_df)} rows)")

    # Canonical labs (train+valid, pre-sample coverage)
    canonical_labs = select_canonical_labs(
        pre_sample_lab_df,
        mrns=train_val.index,
        min_coverage=args.min_patient_coverage,
        id_col=ID_COL,
    )
    canonical_path = output_dir / GENOMIC_CANONICAL_LABS_FILENAME
    pd.DataFrame({"landmark": "sample", "lab_name": canonical_labs}).to_csv(
        canonical_path, index=False
    )
    print(f"Canonical labs: {len(canonical_labs)} -> {canonical_path}")

    # Per-endpoint AUC horizons (independent from main pipeline since cohort + landmark differ)
    auc_quantiles = tuple(args.auc_quantiles)
    auc_horizons: dict[str, list[int]] = {}
    train_val_block = merged.loc[merged["split"].isin(["train", "valid"])]
    for endpoint, cfg in ENDPOINTS.items():
        grid = compute_horizon_grid(
            train_val_block,
            duration_col=cfg["duration_col"],
            event_col=cfg["event_col"],
            quantiles=auc_quantiles,
            time_unit_days=args.time_unit_days,
            admin_censor_days=args.auc_max_time_units * args.time_unit_days,
        )
        auc_horizons[endpoint] = [int(h) for h in grid]
        print(
            f"AUC horizons ({endpoint}): "
            + ", ".join(str(h) for h in auc_horizons[endpoint])
            + f" {args.time_unit_days}-day units"
        )

    manifest = {
        "data": str(args.data),
        "somatic_path": str(args.somatic_path),
        "anchor": "t_sample",
        "sample_pick_rule": (
            "earliest SAMPLE_COLLECTION_DT per DFCI_MRN; alteration indicators "
            "OR-ed across that patient's samples"
        ),
        "min_patient_coverage": float(args.min_patient_coverage),
        "time_unit_days": int(args.time_unit_days),
        "auc_quantiles": list(auc_quantiles),
        # Only the outer two quantiles bound the timeline; the interior ones
        # are recorded above for provenance but do not become horizons.
        "auc_timeline_bounding_quantiles": [float(min(auc_quantiles)), float(max(auc_quantiles))],
        "auc_max_grid_points": int(DEFAULT_AUC_MAX_GRID_POINTS),
        # Runners read this back so the evaluation cap matches the cap the grid
        # was built under; they must not fall back to their own default.
        "auc_max_time_units": int(args.auc_max_time_units),
        "auc_timeline_schema_version": AUC_TIMELINE_SCHEMA_VERSION,
        "auc_time_unit_days": int(args.time_unit_days),
        "auc_horizons": auc_horizons,
        "auc_max_horizon": int(max((h for hs in auc_horizons.values() for h in hs), default=0)),
        "genomic_features": GENOMIC_FEATURE_COLS,
        "n_patients_total": int(len(merged)),
        "n_patients_train_val": int(len(train_val)),
        "n_patients_test": int(len(test)),
        "n_dropped_no_split": n_no_split,
        "split_source": str(split_path),
    }
    manifest_path = output_dir / GENOMIC_BUILD_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print("\nGenomic prediction inputs ready under the preprocessing inputs directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument("--somatic-path", type=str, default=str(DEFAULT_SOMATIC_PATH))
    parser.add_argument(
        "--inputs-dir",
        default=str(RESULTS / DEFAULT_OUTPUT_SUBDIR),
        help="Existing prediction_inputs dir (genomic outputs go in <inputs-dir>/genomic).",
    )
    parser.add_argument(
        "--min-patient-coverage",
        type=float,
        default=DEFAULT_MIN_PATIENT_COVERAGE,
        help="Minimum train+valid lab coverage for the canonical lab set.",
    )
    parser.add_argument(
        "--time-unit-days",
        type=int,
        default=DEFAULT_TIME_UNIT_DAYS,
        help="Time unit for AUC(t) horizon grid.",
    )
    parser.add_argument(
        "--auc-quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_AUC_QUANTILES),
    )
    parser.add_argument(
        "--auc-max-time-units",
        type=int,
        default=DEFAULT_AUC_MAX_TIME_UNITS,
        help=(
            "Administrative censoring horizon (in time units) for the AUC(t)/Brier "
            f"timeline; default {DEFAULT_AUC_MAX_TIME_UNITS}. The grid stays strictly "
            "inside it and runners reuse it from the manifest."
        ),
    )
    main(parser.parse_args())
