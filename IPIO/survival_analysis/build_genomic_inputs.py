"""
Build prediction inputs for the genomic-landmark univariate survival arm (IPIO).

Index time = SAMPLE_COLLECTION_DT (per patient, from the somatic table).
Predicts irAE from sample collection forward (death/censor are right-censoring),
with features derived from labs measured strictly before t_sample plus dynamic
per-gene x per-variant-type binary indicators (pan-cancer cohort -- all genes
present in the somatic table, not a fixed 3-gene prostate panel).

Cohort = main IPIO cohort INTERSECTED with patients that have a genomic sample
AND have a split label in the existing prediction_inputs/split_assignments.csv
(so test stays test across arms).

Outputs (under <inputs-dir>/genomic):
  genomic_aggregated.csv             one row per MRN: lab features + genomic
                                     indicators + outcome rebased to t_sample +
                                     split column
  pre_sample_lab_long.csv            long-format pre-sample labs for per-fold
                                     canonical-lab selection
  genomic_canonical_labs_train_val.csv  landmark='sample', lab_name
  genomic_build_manifest.json        provenance + AUC horizons + cohort sizes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent  # .../IPIO/survival_analysis
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

sys.path.insert(
    0,
    "/data/gusev/USERS/jpconnor/code/CAIA/COMPASS/survival_analysis",
)

from helpers.cohort import (  # noqa: E402
    AGE_COL,
    ID_COL,
    build_feature_matrix,
    build_pre_treatment_lab_long,
)
from helpers.helper import (  # noqa: E402
    DEFAULT_AUC_QUANTILES,
    assert_no_test_leakage,
    compute_horizon_grid,
    select_canonical_labs,
)

from ipio_cohort import make_irae_outcome_df  # noqa: E402
from build_prediction_inputs import (  # noqa: E402
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_MIN_PATIENT_COVERAGE,
    DATA_PATH,
    ENDPOINTS,
    RESULTS,
    SPLIT_ASSIGNMENTS_FILENAME,
)

DEFAULT_SOMATIC_PATH = Path(
    "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
    "clinical_and_genomic_features/complete_somatic_data_df.csv.gz"
)
DEFAULT_TIME_UNIT_DAYS = 7

# Pan-cancer cohort: detect all <GENE>_<VARIANT> columns dynamically rather
# than hardcoding a fixed {TP53,RB1,PTEN} x {SV,DEL,AMP,SNV} prostate panel.
GENE_VARIANT_RE = re.compile(r"^[A-Za-z0-9]+_(SV|SNV|AMP|DEL)$")


def detect_genomic_feature_cols(columns) -> list[str]:
    return sorted(c for c in columns if GENE_VARIANT_RE.match(c))


GENOMIC_OUTPUT_SUBDIR = "genomic"
GENOMIC_AGGREGATED_FILENAME = "genomic_aggregated.csv"
GENOMIC_PRE_SAMPLE_LAB_FILENAME = "pre_sample_lab_long.csv"
GENOMIC_CANONICAL_LABS_FILENAME = "genomic_canonical_labs_train_val.csv"
GENOMIC_BUILD_MANIFEST_FILENAME = "genomic_build_manifest.json"


def load_somatic(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read the somatic table, restricted to ID_COL + SAMPLE_COLLECTION_DT +
    whatever <GENE>_<VARIANT> columns are present (pan-cancer -- data-dependent
    gene set). Reads the full header first to detect the gene-variant columns,
    then restricts `usecols` to just what's needed.
    """
    header = pd.read_csv(path, nrows=0)
    feature_cols = detect_genomic_feature_cols(header.columns)
    if not feature_cols:
        raise ValueError(f"Somatic CSV {path} has no <GENE>_<SV|SNV|AMP|DEL> columns.")
    needed_cols = [ID_COL, "SAMPLE_COLLECTION_DT", *feature_cols]
    missing = [c for c in needed_cols if c not in header.columns]
    if missing:
        raise ValueError(f"Somatic CSV missing columns: {missing}")

    raw = pd.read_csv(path, usecols=lambda c: c in needed_cols)
    raw[ID_COL] = pd.to_numeric(raw[ID_COL], errors="coerce")
    raw = raw.loc[raw[ID_COL].notna()].copy()
    raw[ID_COL] = raw[ID_COL].astype(int)
    if raw[ID_COL].nunique() != len(raw):
        n_dup = len(raw) - raw[ID_COL].nunique()
        raise ValueError(
            f"Somatic CSV is not deduplicated by {ID_COL} ({n_dup} duplicate rows). "
            "One row per patient is assumed."
        )
    raw["SAMPLE_COLLECTION_DT"] = pd.to_datetime(raw["SAMPLE_COLLECTION_DT"], errors="coerce")
    for col in feature_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)
    return raw.set_index(ID_COL), feature_cols


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
    for date_col in ("FIRST_RECORD_DATE", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE", "LAB_DATE"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "t_lab" not in df.columns:
        # Tolerate a source file without a precomputed t_lab (mirrors
        # load_ipio_longitudinal in build_prediction_inputs.py): recompute it
        # from LAB_DATE - FIRST_RECORD_DATE.
        if not {"LAB_DATE", "FIRST_RECORD_DATE"}.issubset(df.columns):
            raise ValueError(
                f"{args.data}: missing t_lab and cannot recompute it (need LAB_DATE + FIRST_RECORD_DATE)."
            )
        df["t_lab"] = (df["LAB_DATE"] - df["FIRST_RECORD_DATE"]).dt.days.astype(float)
    else:
        df["t_lab"] = pd.to_numeric(df["t_lab"], errors="coerce")
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce")
    df = df.loc[df[ID_COL].notna()].copy()
    df[ID_COL] = df[ID_COL].astype(int)
    print(f"Loaded cohort: {df[ID_COL].nunique()} unique MRNs")

    somatic, genomic_feature_cols = load_somatic(Path(args.somatic_path))
    print(f"Somatic patients: {len(somatic)} ({len(genomic_feature_cols)} gene-variant features detected)")

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

    # Outcome table rebased to t_sample (drops t_sample<0 and any sample at/after
    # an event). No require_first_treatment concept here -- IPIO's outcome
    # doesn't depend on FIRST_TREATMENT the way COMPASS's did (the invariant
    # FIRST_TREATMENT_DATE == FIRST_RECORD_DATE == IO_START already holds for
    # every patient in the source data).
    outcome_df = make_irae_outcome_df(
        df,
        landmark_offset_days=0,
        anchor_col="t_sample",
        extra_anchor_cols=("t_sample", "SAMPLE_COLLECTION_DT"),
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

    genomics = somatic.loc[somatic.index.intersection(merged.index), genomic_feature_cols]
    n_missing_genomics = int(merged.index.difference(somatic.index).size)
    if n_missing_genomics:
        print(
            f"WARNING: {n_missing_genomics} patients in the genomic cohort had no somatic row; "
            "their genomic indicators are being set to 0 (indistinguishable from a true "
            "all-negative profile)."
        )
    merged = merged.join(genomics, how="left")
    for col in genomic_feature_cols:
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

    # Drop the debug-only date/duration duplicates (mirrors build_prediction_inputs'
    # AGGREGATED_DROP_COLUMNS) but keep baseline covariates + genomic indicators.
    drop_cols = [
        c for c in ("FIRST_RECORD_DATE", "LAST_CONTACT_DATE", "t_irae_from_first_record")
        if c in merged.columns
    ]
    aggregated = merged.drop(columns=drop_cols) if drop_cols else merged

    agg_path = output_dir / GENOMIC_AGGREGATED_FILENAME
    aggregated.rename_axis(ID_COL).reset_index().to_csv(agg_path, index=False)
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
    train_val_block = aggregated.loc[aggregated["split"].isin(["train", "valid"])]
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
        "sample_pick_rule": f"input csv assumed deduplicated by {ID_COL}",
        "min_patient_coverage": float(args.min_patient_coverage),
        "time_unit_days": int(args.time_unit_days),
        "auc_quantiles": list(auc_quantiles),
        "auc_time_unit_days": int(args.time_unit_days),
        "auc_horizons": auc_horizons,
        "auc_max_horizon": int(max((h for hs in auc_horizons.values() for h in hs), default=0)),
        "genomic_features": genomic_feature_cols,
        "n_patients_total": int(len(aggregated)),
        "n_patients_train_val": int(len(train_val)),
        "n_patients_test": int(len(test)),
        "n_dropped_no_split": n_no_split,
        "split_source": str(split_path),
    }
    manifest_path = output_dir / GENOMIC_BUILD_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print("\nGenomic prediction inputs ready.")


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
