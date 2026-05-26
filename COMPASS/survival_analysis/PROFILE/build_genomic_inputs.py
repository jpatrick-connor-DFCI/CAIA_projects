"""
Build prediction inputs for the genomic-landmark univariate survival arm.

Index time = SAMPLE_COLLECTION_DT (per patient, from the somatic table).
Predicts platinum / death from sample collection forward, with
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

SURVIVAL_DIR = Path(__file__).resolve().parent           # .../survival_analysis/PROFILE
SURVIVAL_PARENT = SURVIVAL_DIR.parent                    # .../survival_analysis
for _p in (str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cox_aggregated import (  # noqa: E402
    AGE_COL,
    DATA_PATH,
    DEFAULT_MIN_PATIENT_COVERAGE,
    ENDPOINTS,
    ID_COL,
    RESULTS,
    build_feature_matrix,
    build_pre_treatment_lab_long,
    make_outcome_df,
)
from helpers.helper import (  # noqa: E402
    DEFAULT_AUC_QUANTILES,
    compute_horizon_grid,
    select_canonical_labs,
)
from build_prediction_inputs import (  # noqa: E402
    DEFAULT_OUTPUT_SUBDIR,
    SPLIT_ASSIGNMENTS_FILENAME,
)

DEFAULT_SOMATIC_PATH = Path(
    "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
    "clinical_and_genomic_features/complete_somatic_data_df.csv.gz"
)
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
    needed_cols = [ID_COL, "SAMPLE_COLLECTION_DT", *GENOMIC_FEATURE_COLS]
    raw = pd.read_csv(path, usecols=lambda c: c in needed_cols)
    missing = [c for c in needed_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Somatic CSV missing columns: {missing}")
    raw[ID_COL] = pd.to_numeric(raw[ID_COL], errors="coerce")
    raw = raw.loc[raw[ID_COL].notna()].copy()
    raw[ID_COL] = raw[ID_COL].astype(int)
    if raw[ID_COL].nunique() != len(raw):
        n_dup = len(raw) - raw[ID_COL].nunique()
        raise ValueError(
            f"Somatic CSV is not deduplicated by DFCI_MRN ({n_dup} duplicate rows). "
            "Decision §0.1 assumed one row per patient."
        )
    raw["SAMPLE_COLLECTION_DT"] = pd.to_datetime(raw["SAMPLE_COLLECTION_DT"], errors="coerce")
    for col in GENOMIC_FEATURE_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)
    return raw.set_index(ID_COL)


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

    df = attach_t_sample(df, somatic)
    df = df.loc[df["t_sample"].notna()].copy()
    print(f"Cohort after t_sample join (notna): {df[ID_COL].nunique()} patients")

    # Outcome table rebased to t_sample
    outcome_df = make_outcome_df(
        df,
        landmark_offset_days=0,
        anchor_col="t_sample",
        extra_anchor_cols=("t_sample", "SAMPLE_COLLECTION_DT"),
        require_first_treatment=False,
    )
    print(f"Outcome cohort (post t_sample landmark filter): {len(outcome_df)} patients")

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
        "sample_pick_rule": "input csv assumed deduplicated by DFCI_MRN",
        "min_patient_coverage": float(args.min_patient_coverage),
        "time_unit_days": int(args.time_unit_days),
        "auc_quantiles": list(auc_quantiles),
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
    print("\nGenomic prediction inputs ready. Run cox_genomic_univariate.py next.")


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
    main(parser.parse_args())
