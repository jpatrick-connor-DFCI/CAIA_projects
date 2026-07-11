"""
Build prediction inputs for the genomic survival arm (IPIO).

Index time = IO_START (t_first_treatment = 0), the SAME anchor as the main
cohort's landmarks -- NOT the somatic sample collection date. Predicting from
the sequencing date isn't clinically actionable (sequencing can land before or
after treatment start); anchoring to treatment start instead lets this arm
isolate genomics' added predictive value over labs on a shared population and
time origin, rather than mixing in a second, less meaningful anchor.

Predicts irAE from treatment start forward (death/censor are right-censoring),
with features derived from labs measured strictly before each requested
landmark (identical window to the main cohort's landmarks) plus dynamic
per-gene x per-variant-type binary indicators (pan-cancer cohort -- all genes
present in the somatic table, not a fixed 3-gene prostate panel). Genomic
indicators themselves are static (not landmark-dependent) and are joined
unchanged at every landmark.

Cohort = main IPIO cohort INTERSECTED with patients that have an actual genomic
sample (no 0-fill for untested patients -- being untested is not the same as
testing negative) AND have a split label in the existing
prediction_inputs/split_assignments.csv (so test stays test across arms). The
cohort/split is derived once at the base (first requested) landmark and reused
at every other landmark, mirroring build_prediction_inputs.py's common-MRN
handling.

Outputs (under <inputs-dir>/genomic), written once per requested landmark day
<lm> (aliases below use the runner-compatible landmark-0 filenames only when
<lm> == 0):
  genomic_aggregated_landmark<lm>.csv one row per MRN: lab features + genomic
                                     indicators + outcome rebased to the
                                     landmark + split column
  aggregated_landmark<lm>.csv        runner-compatible alias
  pre_sample_lab_long_landmark<lm>.csv long-format pre-landmark labs (genomics-
                                     eligible subset) for per-fold canonical-lab
                                     selection
  pre_treatment_lab_long_landmark<lm>.csv
                                     runner-compatible alias
  genomic_canonical_labs_train_val.csv  landmark_days, lab_name (all landmarks)
  canonical_labs_train_val.csv       runner-compatible alias
  genomic_build_manifest.json        provenance + AUC horizons + cohort sizes
  build_manifest.json                runner-compatible alias
"""

from __future__ import annotations

import argparse
import json
import re
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

from _paths import ensure_survival_common_on_path  # noqa: E402

ensure_survival_common_on_path()

from survival_common.cohort import (  # noqa: E402
    AGE_COL,
    ID_COL,
    build_landmark_availability_table,
    build_pre_treatment_lab_long,
    configure_id_columns,
    normalize_landmark_days,
)
from survival_common.helper import (  # noqa: E402
    DEFAULT_AUC_QUANTILES,
    assert_no_test_leakage,
    compute_horizon_grid,
    select_canonical_labs,
)

# build_landmark_merged is IPIO-local (wraps ipio_cohort.make_irae_outcome_df) --
# NOT survival_common.cohort.build_landmark_merged, which is COMPASS-oriented
# (hard-wired PLATINUM/DEATH) and has no irAE outcome columns at all.
from build_prediction_inputs import (  # noqa: E402
    BUILD_MANIFEST_FILENAME,
    CANONICAL_LABS_FILENAME,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_OUTPUT_SUBDIR,
    DEFAULT_MIN_PATIENT_COVERAGE,
    DATA_PATH,
    ENDPOINTS,
    RESULTS,
    SPLIT_ASSIGNMENTS_FILENAME,
    aggregated_filename,
    build_landmark_merged,
    pre_treatment_lab_filename,
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
    """Read the somatic table, restricted to ID_COL + whatever <GENE>_<VARIANT>
    columns are present (pan-cancer -- data-dependent gene set). Reads the full
    header first to detect the gene-variant columns, then restricts `usecols`
    to just what's needed. SAMPLE_COLLECTION_DT is not read -- this arm anchors
    to IO_START (t_first_treatment), not the sequencing date.
    """
    header = pd.read_csv(path, nrows=0)
    feature_cols = detect_genomic_feature_cols(header.columns)
    if not feature_cols:
        raise ValueError(f"Somatic CSV {path} has no <GENE>_<SV|SNV|AMP|DEL> columns.")
    needed_cols = [ID_COL, *feature_cols]
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
    for col in feature_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)
    return raw.set_index(ID_COL), feature_cols


def main(args: argparse.Namespace) -> None:
    global ID_COL, AGE_COL
    ID_COL = args.id_col
    AGE_COL = args.age_col
    # Push the runtime schema into the shared cohort builders BEFORE any cohort/
    # split logic runs, so split_assignments.csv reuse and build_landmark_merged /
    # build_pre_treatment_lab_long all key on the requested id/age columns
    # (mirrors build_prediction_inputs.py's main()).
    configure_id_columns(ID_COL, AGE_COL)
    landmark_days = normalize_landmark_days(args.landmark_days)
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

    # Design invariant: FIRST_TREATMENT_DATE == FIRST_RECORD_DATE == IO_START for
    # every patient, so t_first_treatment (days from FIRST_RECORD_DATE) is 0 for
    # everyone (mirrors build_prediction_inputs.py's landmark setup).
    if "t_first_treatment" not in df.columns and {"FIRST_TREATMENT_DATE", "FIRST_RECORD_DATE"}.issubset(df.columns):
        df["t_first_treatment"] = (
            df["FIRST_TREATMENT_DATE"] - df["FIRST_RECORD_DATE"]
        ).dt.days.astype(float)

    somatic, genomic_feature_cols = load_somatic(Path(args.somatic_path))
    print(f"Somatic patients: {len(somatic)} ({len(genomic_feature_cols)} gene-variant features detected)")

    # Anchor to IO_START (t_first_treatment), identical to the main cohort's
    # landmarks -- NOT the sample collection date. Restrict to patients with an
    # actual genomic sample (no 0-fill for untested patients).
    n_before_genomics = df[ID_COL].nunique()
    df = df.loc[df[ID_COL].isin(somatic.index)].copy()
    n_with_genomics = df[ID_COL].nunique()
    print(
        f"Cohort restricted to patients with a genomic sample: {n_with_genomics} "
        f"(dropped {n_before_genomics - n_with_genomics} without one)."
    )

    # Build the per-landmark (outcome + lab feature) cohort using the IPIO-local
    # build_landmark_merged (irAE-aware), then attach the (landmark-invariant)
    # genomic indicators at each landmark.
    merged_by_landmark: dict[int, pd.DataFrame] = {}
    for landmark_day in landmark_days:
        print(f"\n##### GENOMIC COHORT BUILD: LANDMARK +{landmark_day} DAYS #####")
        _, _, merged = build_landmark_merged(
            df,
            landmark_offset_days=landmark_day,
            anchor_col="t_first_treatment",
        )
        genomics = somatic.loc[somatic.index.intersection(merged.index), genomic_feature_cols]
        merged = merged.join(genomics, how="left")
        for col in genomic_feature_cols:
            merged[col] = merged[col].fillna(0).astype(int)
        merged_by_landmark[landmark_day] = merged
        print(f"Cohort with genomics joined @ landmark +{landmark_day}d: {len(merged)} patients")

    _, common_mrns = build_landmark_availability_table(merged_by_landmark)
    common_mrns = merged_by_landmark[landmark_days[0]].index.intersection(common_mrns)
    if len(common_mrns) == 0:
        raise ValueError("No MRNs were eligible at every requested genomic landmark.")
    print(f"\nCommon MRN cohort across genomic landmarks {landmark_days}: {len(common_mrns)} patients")

    # Reuse main split — drop patients without a label there. Split membership
    # doesn't depend on landmark, so this is derived once from the common cohort.
    aligned_split = split_assignments.reindex(common_mrns)
    n_no_split = int(aligned_split.isna().sum())
    if n_no_split:
        print(
            f"Dropping {n_no_split} MRNs that lack a label in {split_path.name} "
            "(genomic cohort > main cohort)"
        )
    common_mrns = aligned_split.dropna().index
    aligned_split = aligned_split.loc[common_mrns].astype(str)

    train_val_mrns = set(aligned_split.index[aligned_split.isin(["train", "valid"])])
    test_mrns = set(aligned_split.index[aligned_split.eq("test")])
    print(
        f"Final genomic cohort: train+valid={len(train_val_mrns)}  test={len(test_mrns)} "
        f"(total {len(common_mrns)})"
    )
    if not train_val_mrns or not test_mrns:
        raise ValueError("Genomic cohort has empty train_val or test after split alignment.")
    # Guard: reused split must keep test disjoint from train+valid.
    assert_no_test_leakage(
        test_mrns=test_mrns,
        train_mrns=train_val_mrns,
        context="build_genomic_inputs: test vs train+valid",
    )

    canonical_labs_rows: list[dict] = []
    auc_horizons_by_landmark: dict[str, dict[str, list[int]]] = {}
    auc_quantiles = tuple(args.auc_quantiles)

    for landmark_day in landmark_days:
        print(f"\n##### GENOMIC LANDMARK +{landmark_day}d: BUILD INPUTS #####")
        merged = merged_by_landmark[landmark_day].loc[common_mrns].copy()
        merged["split"] = aligned_split.loc[merged.index]

        # Drop the debug-only date/duration duplicates (mirrors
        # build_prediction_inputs' AGGREGATED_DROP_COLUMNS) but keep baseline
        # covariates + genomic indicators.
        drop_cols = [
            c for c in ("FIRST_RECORD_DATE", "LAST_CONTACT_DATE", "t_irae_from_first_record")
            if c in merged.columns
        ]
        aggregated = merged.drop(columns=drop_cols) if drop_cols else merged

        agg_path = output_dir / (
            GENOMIC_AGGREGATED_FILENAME if landmark_day == 0
            else f"genomic_aggregated_landmark{landmark_day}.csv"
        )
        aggregated.rename_axis(ID_COL).reset_index().to_csv(agg_path, index=False)
        print(f"  aggregated:        {len(aggregated)} patients -> {agg_path}")
        compat_agg_path = output_dir / aggregated_filename(landmark_day)
        aggregated.rename_axis(ID_COL).reset_index().to_csv(compat_agg_path, index=False)
        print(f"  runner-compatible alias -> {compat_agg_path}")

        # Pre-landmark lab long (genomics-eligible subset) for per-fold canonical labs
        pre_sample_lab_df = build_pre_treatment_lab_long(
            df,
            cohort_index=merged.index,
            landmark_offset_days=landmark_day,
            anchor_col="t_first_treatment",
        )
        pre_path = output_dir / (
            GENOMIC_PRE_SAMPLE_LAB_FILENAME if landmark_day == 0
            else f"pre_sample_lab_long_landmark{landmark_day}.csv"
        )
        pre_sample_lab_df.to_csv(pre_path, index=False)
        print(f"  pre-landmark labs: {len(pre_sample_lab_df)} rows -> {pre_path}")
        compat_pre_path = output_dir / pre_treatment_lab_filename(landmark_day)
        pre_sample_lab_df.to_csv(compat_pre_path, index=False)
        print(f"  runner-compatible alias -> {compat_pre_path}")

        # Canonical labs (train+valid, pre-landmark coverage, genomics-eligible subset)
        canonical_labs = select_canonical_labs(
            pre_sample_lab_df,
            mrns=pd.Index(sorted(train_val_mrns)),
            min_coverage=args.min_patient_coverage,
            id_col=ID_COL,
        )
        for lab in canonical_labs:
            canonical_labs_rows.append({"landmark_days": landmark_day, "lab_name": lab})
        print(f"  canonical labs (train+valid): {len(canonical_labs)}")

        # Per-endpoint AUC horizons (independent from main pipeline since cohort + landmark differ)
        landmark_horizons: dict[str, list[int]] = {}
        train_val_block = aggregated.loc[aggregated["split"].isin(["train", "valid"])]
        for endpoint, cfg in ENDPOINTS.items():
            grid = compute_horizon_grid(
                train_val_block,
                duration_col=cfg["duration_col"],
                event_col=cfg["event_col"],
                quantiles=auc_quantiles,
                time_unit_days=args.time_unit_days,
            )
            landmark_horizons[endpoint] = [int(h) for h in grid]
            print(
                f"  AUC horizons ({endpoint}): "
                + ", ".join(str(h) for h in landmark_horizons[endpoint])
                + f" {args.time_unit_days}-day units"
            )
        auc_horizons_by_landmark[str(int(landmark_day))] = landmark_horizons

    canonical_labs_df = pd.DataFrame(canonical_labs_rows)
    canonical_path = output_dir / GENOMIC_CANONICAL_LABS_FILENAME
    canonical_labs_df.to_csv(canonical_path, index=False)
    print(f"\nWrote {canonical_path}")
    compat_canonical_path = output_dir / CANONICAL_LABS_FILENAME
    canonical_labs_df.to_csv(compat_canonical_path, index=False)
    print(f"Wrote runner-compatible alias {compat_canonical_path}")

    max_horizon = max(
        (h for endpoints in auc_horizons_by_landmark.values() for hs in endpoints.values() for h in hs),
        default=0,
    )
    manifest = {
        "data": str(args.data),
        "somatic_path": str(args.somatic_path),
        "anchor": "t_first_treatment",
        "sample_pick_rule": f"input csv assumed deduplicated by {ID_COL}",
        "min_patient_coverage": float(args.min_patient_coverage),
        "time_unit_days": int(args.time_unit_days),
        "auc_quantiles": list(auc_quantiles),
        "auc_time_unit_days": int(args.time_unit_days),
        "auc_horizons_by_landmark": auc_horizons_by_landmark,
        "auc_max_horizon": int(max_horizon),
        "landmark_days": [int(d) for d in landmark_days],
        "genomic_features": genomic_feature_cols,
        "n_patients_total": int(len(common_mrns)),
        "n_patients_train_val": int(len(train_val_mrns)),
        "n_patients_test": int(len(test_mrns)),
        "n_dropped_no_split": n_no_split,
        "split_source": str(split_path),
    }
    manifest_path = output_dir / GENOMIC_BUILD_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    compat_manifest_path = output_dir / BUILD_MANIFEST_FILENAME
    compat_manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote runner-compatible alias {compat_manifest_path}")
    print("\nGenomic prediction inputs ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-col", default=ID_COL,
                        help="Patient identifier column name (default DFCI_MRN).")
    parser.add_argument("--age-col", default=AGE_COL,
                        help="Age covariate column name (default AGE_AT_TREATMENTSTART).")
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument("--somatic-path", type=str, default=str(DEFAULT_SOMATIC_PATH))
    parser.add_argument(
        "--inputs-dir",
        default=str(RESULTS / DEFAULT_OUTPUT_SUBDIR),
        help="Existing prediction_inputs dir (genomic outputs go in <inputs-dir>/genomic).",
    )
    parser.add_argument(
        "--landmark-days",
        nargs="+",
        type=int,
        default=DEFAULT_LANDMARK_DAYS,
        help="Landmark offsets in days relative to treatment start.",
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
