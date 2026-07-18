"""
Single source of truth for IPIO (pan-cancer immunotherapy, time-to-irAE)
prediction inputs. Adapted from COMPASS's PROFILE/build_prediction_inputs.py.

Unlike COMPASS, the outcome (death vs. censor vs. irAE) is already clean in
the source `longitudinal_prediction_data.csv` -- there is no raw ICD/med-table
date inference. For the cause-specific (plain Cox) arm, `event == "irAE"` is
the event of interest and death and censoring are both right-censoring. The
aggregated table also carries DEATH/event_type so univariate runs can
additionally fit a Fine-Gray subdistribution-hazard arm where death is a
competing event for irAE (survival_common.finegray). See
`ipio_cohort.make_irae_outcome_df`.

For each requested landmark, this script:
  1. Loads the IPIO longitudinal lab CSV (long format; patient-level columns
     repeated per lab row) and splits it into patient-level / lab-level frames.
  2. Builds the landmarked patient cohort and intersects MRNs across all
     requested landmarks so every downstream model sees the same patients.
  3. Derives a 3-way train/valid/test split ONCE on the intersection cohort
     at the base (smallest) landmark, then reuses it for every landmark.
  4. Writes per-landmark:
       aggregated_landmark{D}.csv                 (1 row per MRN; features + outcomes + split)
       pre_treatment_lab_long_landmark{D}.csv     (long-format pre-landmark labs
                                                   for per-fold canonical-lab selection)
  5. Writes shared:
       split_assignments.csv                      DFCI_MRN, split (train/valid/test)
       landmark_mrn_availability.csv              per-landmark eligibility + split
       canonical_labs_train_val.csv               landmark_days, lab_name (train+valid)
       build_manifest.json

Older COMPASS experiments also built a wide person-period table for dynamic
sequence models. The current COMPASS/IPIO builders persist the aggregated
landmark table plus the pre-treatment long lab table consumed by per-fold
canonical-lab selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SURVIVAL_DIR = PROJECT_DIR / "survival_analysis"
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(REPO_ROOT), str(PROJECT_DIR), str(SURVIVAL_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import ensure_survival_common_on_path  # noqa: E402

# survival_common.cohort / survival_common.helper are schema-agnostic (configurable
# ID_COL/AGE_COL) and reused unchanged for the generic mechanics: pre-landmark
# lab feature engineering, landmark-availability bookkeeping, canonical-lab
# selection, AUC(t) horizon grid, and the test-leakage guard.
ensure_survival_common_on_path()

from survival_common.cohort import (  # noqa: E402
    AGE_COL,
    ID_COL,
    build_feature_matrix,
    build_landmark_availability_table,
    build_pre_treatment_lab_long,
    configure_id_columns,
    normalize_landmark_days,
)
from survival_common.helper import (  # noqa: E402
    DEFAULT_AUC_QUANTILES,
    assert_no_test_leakage,
    choose_stratification_labels as choose_shared_stratification_labels,
    compute_horizon_grid,
    select_canonical_labs,
)
from survival_common.plotting import (  # noqa: E402
    DROP as VITALS_DROP,
    VITALS,
    canonicalize_lab_name,
)

from ipio_cohort import make_irae_outcome_df  # noqa: E402

# Single source of truth for these constants is cox_aggregated.py (matches
# COMPASS's own build_prediction_inputs.py, which imports DATA_PATH,
# DEFAULT_LANDMARK_DAYS, DEFAULT_SEED, DEFAULT_TEST_FRAC,
# DEFAULT_MIN_PATIENT_COVERAGE, ENDPOINTS, RESULTS from cox_aggregated rather
# than redefining them here).
from cox_aggregated import (  # noqa: E402
    DATA_PATH,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_MIN_PATIENT_COVERAGE,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    ENDPOINTS,
    RESULTS,
)

# ---------------------------------------------------------------------------
# Constants genuinely local to the input-building step (not part of the
# shared cox_aggregated.py contract, same as in COMPASS's build_prediction_inputs.py).
# ---------------------------------------------------------------------------
DEFAULT_VAL_FRAC = 0.20
DEFAULT_TIME_UNIT_DAYS = 7

DEFAULT_OUTPUT_SUBDIR = "prediction_inputs"

SPLIT_ASSIGNMENTS_FILENAME = "split_assignments.csv"
LANDMARK_AVAILABILITY_FILENAME = "landmark_mrn_availability.csv"
LANDMARK_ATTRITION_FILENAME = "landmark_attrition.json"
CANONICAL_LABS_FILENAME = "canonical_labs_train_val.csv"
BUILD_MANIFEST_FILENAME = "build_manifest.json"


EXCLUDED_LAB_NAMES = VITALS | VITALS_DROP


def drop_vitals_rows(labs_df: pd.DataFrame) -> pd.DataFrame:
    """Drop vitals LAB_NAME rows so vitals never enter the feature set."""
    if "LAB_NAME" not in labs_df.columns:
        return labs_df
    canonical = labs_df["LAB_NAME"].astype(str).str.strip().map(canonicalize_lab_name)
    is_vital = canonical.isin(EXCLUDED_LAB_NAMES)
    n_dropped = int(is_vital.sum())
    if n_dropped:
        print(f"  Vitals exclusion: dropped {n_dropped} rows ({sorted(canonical[is_vital].unique())})")
    return labs_df.loc[~is_vital].copy()


def aggregated_filename(landmark_day: int) -> str:
    return f"aggregated_landmark{int(landmark_day)}.csv"


def pre_treatment_lab_filename(landmark_day: int) -> str:
    return f"pre_treatment_lab_long_landmark{int(landmark_day)}.csv"


# ---------------------------------------------------------------------------
# Local data loader for the IPIO longitudinal_prediction_data.csv schema.
# IPIO-specific because its patient-level column list (IRAE/DEATH-based, no
# PLATINUM) doesn't match COMPASS's schema.
# ---------------------------------------------------------------------------

def load_ipio_longitudinal(path: Path, *, id_col: str = "DFCI_MRN") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load IPIO's `longitudinal_prediction_data.csv` into (patient_df, labs_df).

    Long format, one row per (patient x lab observation); patient-level columns
    repeat per row. Patient-level: DFCI_MRN, AGE_AT_TREATMENTSTART, GENDER_MALE,
    CANCER_TYPE_* (dynamic one-hot set), pd1pdl1, ctla4, FIRST_RECORD_DATE,
    FIRST_TREATMENT_DATE, FIRST_TREATMENT, LAST_CONTACT_DATE, IRAE, DEATH.
    Lab-level: LAB_NAME, LAB_VALUE, LAB_UNIT, LAB_DATE, t_lab (recomputed from
    LAB_DATE - FIRST_RECORD_DATE if not already present).
    """
    df = pd.read_csv(path, low_memory=False)
    for date_col in ("FIRST_RECORD_DATE", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE", "LAB_DATE"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "t_lab" not in df.columns:
        if not {"LAB_DATE", "FIRST_RECORD_DATE"}.issubset(df.columns):
            raise ValueError(
                f"{path}: missing t_lab and cannot recompute it (need LAB_DATE + FIRST_RECORD_DATE)."
            )
        df["t_lab"] = (df["LAB_DATE"] - df["FIRST_RECORD_DATE"]).dt.days.astype(float)
    else:
        df["t_lab"] = pd.to_numeric(df["t_lab"], errors="coerce")

    cancer_type_cols = [c for c in df.columns if c.startswith("CANCER_TYPE_")]
    patient_level_candidates = [
        id_col,
        "AGE_AT_TREATMENTSTART",
        "GENDER_MALE",
        *cancer_type_cols,
        "pd1pdl1",
        "ctla4",
        "FIRST_RECORD_DATE",
        "FIRST_TREATMENT_DATE",
        "FIRST_TREATMENT",
        "LAST_CONTACT_DATE",
        "IRAE",
    ]
    pat_cols = [c for c in patient_level_candidates if c in df.columns]
    patient_df = df[pat_cols].drop_duplicates(subset=[id_col]).reset_index(drop=True)

    lab_col_candidates = [id_col, "LAB_NAME", "LAB_VALUE", "LAB_UNIT", "LAB_DATE", "t_lab"]
    lab_cols = [c for c in lab_col_candidates if c in df.columns]
    labs_df = df.loc[df["LAB_NAME"].notna(), lab_cols].reset_index(drop=True)
    labs_df = drop_vitals_rows(labs_df)
    return patient_df, labs_df


def load_mrn_subset(mrn_file: Path, id_col: str) -> set[int]:
    """Read the restrict-to-MRNs CSV and return the subset as a set of int ids."""
    subset = pd.read_csv(mrn_file)
    col = id_col if id_col in subset.columns else "DFCI_MRN"
    if col not in subset.columns:
        raise ValueError(
            f"{mrn_file} has no '{id_col}'/'DFCI_MRN' column; columns={list(subset.columns)[:8]}"
        )
    ids = pd.to_numeric(subset[col], errors="coerce").dropna().astype(int)
    if ids.empty:
        raise ValueError(f"{mrn_file} contains no valid MRNs in column '{col}'.")
    return set(ids.tolist())


# ---------------------------------------------------------------------------
# IPIO-local equivalent of survival_common.cohort.build_landmark_merged: that function
# calls COMPASS's make_outcome_df internally (hard-wired PLATINUM/DEATH logic),
# so it is reimplemented here around make_irae_outcome_df. build_feature_matrix
# itself is reused unchanged from survival_common.cohort.
# ---------------------------------------------------------------------------

def build_landmark_merged(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int,
    anchor_col: str = "t_first_treatment",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome_df = make_irae_outcome_df(
        df,
        landmark_offset_days=landmark_offset_days,
        anchor_col=anchor_col,
    )
    print(f"Outcome table @ landmark +{landmark_offset_days}d: {len(outcome_df)} patients")

    print(f"Building raw aggregated lab summary feature matrix through landmark +{landmark_offset_days}d...")
    feature_df = build_feature_matrix(
        df, landmark_offset_days=landmark_offset_days, anchor_col=anchor_col
    )

    merged = feature_df.join(outcome_df, how="inner")
    merged = merged.loc[merged[AGE_COL].notna()].copy()
    if merged.empty:
        raise ValueError("No patients have both engineered features and valid outcomes.")
    return outcome_df, feature_df, merged


# ---------------------------------------------------------------------------
# Single-endpoint split stratification. Keep the local name for this script's
# call sites, but delegate to the shared helper with IPIO's endpoint column.
# ---------------------------------------------------------------------------

def choose_stratification_labels(df: pd.DataFrame, *, min_count: int) -> tuple[np.ndarray | None, str]:
    return choose_shared_stratification_labels(
        df,
        min_count=min_count,
        event_col="IRAE",
    )


def derive_three_way_split(
    base_merged: pd.DataFrame,
    *,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.Series, str, str]:
    """Stratified 80/20 train_val/test, then 80/20 train/valid carved out of train_val.

    Returns (split_series indexed by DFCI_MRN with values train/valid/test,
             test stratification label, valid stratification label).
    """
    test_labels, test_stratification = choose_stratification_labels(base_merged, min_count=2)
    try:
        train_val_idx, test_idx = train_test_split(
            np.arange(len(base_merged)),
            test_size=test_frac,
            stratify=test_labels,
            random_state=seed,
        )
    except ValueError:
        test_stratification = "unstratified"
        train_val_idx, test_idx = train_test_split(
            np.arange(len(base_merged)),
            test_size=test_frac,
            random_state=seed,
        )

    train_val_block = base_merged.iloc[train_val_idx]
    val_labels, val_stratification = choose_stratification_labels(train_val_block, min_count=2)
    try:
        train_offsets, valid_offsets = train_test_split(
            np.arange(len(train_val_block)),
            test_size=val_frac,
            stratify=val_labels,
            random_state=seed,
        )
    except ValueError:
        val_stratification = "unstratified"
        train_offsets, valid_offsets = train_test_split(
            np.arange(len(train_val_block)),
            test_size=val_frac,
            random_state=seed,
        )

    split = pd.Series(index=base_merged.index, dtype=object, name="split")
    split.iloc[test_idx] = "test"
    train_val_mrns = train_val_block.index.to_numpy()
    split.loc[train_val_mrns[train_offsets]] = "train"
    split.loc[train_val_mrns[valid_offsets]] = "valid"
    if split.isna().any():
        raise ValueError("Split assignment failed to cover every MRN in the base cohort.")
    return split, test_stratification, val_stratification


# Debug-only metadata columns dropped from the final aggregated table. Must NOT
# include any baseline covariate (AGE_AT_TREATMENTSTART, GENDER_MALE, pd1pdl1,
# ctla4, CANCER_TYPE_*) -- those need to survive into aggregated_landmark{D}.csv
# for cox_aggregated.py's baseline-covariate mechanism.
AGGREGATED_DROP_COLUMNS = (
    "FIRST_RECORD_DATE",
    "FIRST_TREATMENT_DATE",
    "LAST_CONTACT_DATE",
    "t_irae_from_first_record",
    # Always 1 after the cohort's design invariant (FIRST_TREATMENT_DATE ==
    # FIRST_RECORD_DATE == IO_START for every patient).
    "FIRST_TREATMENT",
)


def build_aggregated_table(
    merged: pd.DataFrame,
    *,
    split: pd.Series,
) -> pd.DataFrame:
    """Attach the shared split column and drop debug-only metadata columns."""
    if "split" in merged.columns:
        merged = merged.drop(columns=["split"])
    aligned = split.reindex(merged.index)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()[:5]
        raise ValueError(
            f"Split assignment missing for {aligned.isna().sum()} aggregated MRNs; first few: {missing}"
        )
    out = merged.copy()
    drop = [c for c in AGGREGATED_DROP_COLUMNS if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    out["split"] = aligned.astype(str)
    return out


def main(args: argparse.Namespace) -> None:
    global ID_COL, AGE_COL
    ID_COL = args.id_col
    AGE_COL = args.age_col
    # Push the runtime schema into the shared cohort builders so
    # build_feature_matrix / build_pre_treatment_lab_long operate on the
    # requested id/age columns.
    configure_id_columns(ID_COL, AGE_COL)
    landmark_days = normalize_landmark_days(args.landmark_days)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    print(f"Loading data from {data_path} ...")
    patient_df, labs_df = load_ipio_longitudinal(data_path, id_col=ID_COL)
    df = labs_df.merge(patient_df, on=ID_COL, how="left")

    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce")
    df = df.loc[df[ID_COL].notna()].copy()
    df[ID_COL] = df[ID_COL].astype(int)
    n_loaded_cohort = df[ID_COL].nunique()
    print(f"Loaded cohort: {n_loaded_cohort} unique MRNs")

    # Design invariant: FIRST_TREATMENT_DATE == FIRST_RECORD_DATE == IO_START for
    # every patient, so t_first_treatment (days from FIRST_RECORD_DATE) is 0 for
    # everyone. Compute it explicitly (rather than hardcode 0) so any future
    # relaxation of the invariant is reflected automatically.
    if "t_first_treatment" not in df.columns and {"FIRST_TREATMENT_DATE", "FIRST_RECORD_DATE"}.issubset(df.columns):
        df["t_first_treatment"] = (
            df["FIRST_TREATMENT_DATE"] - df["FIRST_RECORD_DATE"]
        ).dt.days.astype(float)
    if "FIRST_TREATMENT" not in df.columns:
        df["FIRST_TREATMENT"] = 1

    # Optional restrict-to-MRNs subset: filter the raw cohort to the requested
    # MRN list BEFORE building landmark cohorts, so the train/valid/test split,
    # canonical lab set, and feature selection are all derived on the subset.
    if args.restrict_to_mrns:
        mrn_subset = load_mrn_subset(Path(args.restrict_to_mrns), ID_COL)
        n_before = df[ID_COL].nunique()
        df = df.loc[df[ID_COL].isin(mrn_subset)].copy()
        n_after = df[ID_COL].nunique()
        if n_after == 0:
            raise ValueError(
                f"--restrict-to-mrns {args.restrict_to_mrns}: none of the "
                f"{len(mrn_subset)} requested MRNs are present in the cohort."
            )
        print(
            f"  [restrict-to-mrns] subset cohort: {n_after}/{n_before} loaded MRNs "
            f"retained ({len(mrn_subset)} requested in {args.restrict_to_mrns})"
        )

    anchor_col = args.anchor_col
    if anchor_col not in df.columns:
        raise ValueError(
            f"--anchor-col {anchor_col!r} not found in {data_path}. Available t_* "
            f"columns: {[c for c in df.columns if c.startswith('t_')]}"
        )
    if anchor_col != "t_first_treatment":
        n_with_anchor = df.loc[df[anchor_col].notna(), ID_COL].nunique()
        print(
            f"Anchor column: {anchor_col} "
            f"({n_with_anchor} patients have a non-null anchor; the rest are dropped "
            f"by the landmark filter)"
        )

    merged_by_landmark: dict[int, pd.DataFrame] = {}
    for landmark_day in landmark_days:
        print(f"\n##### COHORT BUILD: LANDMARK +{landmark_day} DAYS #####")
        _, _, merged = build_landmark_merged(
            df,
            landmark_offset_days=landmark_day,
            anchor_col=anchor_col,
        )
        merged_by_landmark[landmark_day] = merged

    availability, common_mrns = build_landmark_availability_table(merged_by_landmark)
    common_mrns = merged_by_landmark[landmark_days[0]].index.intersection(common_mrns)
    if len(common_mrns) == 0:
        raise ValueError("No MRNs were eligible at every requested landmark.")
    print(f"\nCommon MRN cohort across landmarks {landmark_days}: {len(common_mrns)} patients")

    base_landmark_day = landmark_days[0]
    base_merged = merged_by_landmark[base_landmark_day].loc[common_mrns].copy()
    split, test_stratification, val_stratification = derive_three_way_split(
        base_merged,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(
        f"3-way split derived once on landmark +{base_landmark_day}d "
        f"(test stratification={test_stratification}, valid stratification={val_stratification})"
    )
    counts = split.value_counts().to_dict()
    print(f"Split sizes: train={counts.get('train', 0)} valid={counts.get('valid', 0)} test={counts.get('test', 0)}")

    split_path = output_dir / SPLIT_ASSIGNMENTS_FILENAME
    split.rename_axis(ID_COL).reset_index().to_csv(split_path, index=False)
    print(f"Wrote {split_path}")

    availability["included_all_landmarks"] = availability[ID_COL].isin(common_mrns)
    availability["split"] = availability[ID_COL].map(split)
    availability_path = output_dir / LANDMARK_AVAILABILITY_FILENAME
    availability.to_csv(availability_path, index=False)
    print(f"Wrote {availability_path}")

    # Structured attrition counts for the Figure 1 CONSORT diagram (ported from
    # COMPASS's build_prediction_inputs.py). Purely additive -- an extra JSON
    # alongside the existing outputs. COMPASS's `n_common_across_landmarks` /
    # `eligible_by_landmark` map to IPIO's `common_mrns` and the shared
    # `eligible_landmark_{lm}` availability columns.
    landmark_attrition = {
        "n_loaded_cohort": int(n_loaded_cohort),
        "eligible_by_landmark": {
            str(lm): int(availability[f"eligible_landmark_{lm}"].sum())
            for lm in landmark_days
        },
        "n_common_across_landmarks": int(len(common_mrns)),
        "split_sizes": {k: int(v) for k, v in counts.items()},
    }
    attrition_path = output_dir / LANDMARK_ATTRITION_FILENAME
    attrition_path.write_text(json.dumps(landmark_attrition, indent=2))
    print(f"Wrote {attrition_path}")

    train_val_mrns = set(split.index[split.isin(["train", "valid"])])
    test_mrns = set(split.index[split.eq("test")])
    # Guard: the test set must be disjoint from train+valid before any downstream
    # canonical-lab selection / horizon fitting uses train_val_mrns.
    assert_no_test_leakage(
        test_mrns=test_mrns,
        train_mrns=train_val_mrns,
        context="build_prediction_inputs: test vs train+valid",
    )

    canonical_labs_rows: list[dict] = []
    auc_horizons_by_landmark: dict[str, dict[str, list[int]]] = {}
    auc_quantiles = tuple(args.auc_quantiles)

    for landmark_day in landmark_days:
        print(f"\n##### LANDMARK +{landmark_day}d: BUILD INPUTS #####")
        merged = merged_by_landmark[landmark_day].loc[common_mrns].copy()
        aggregated = build_aggregated_table(merged, split=split)

        agg_path = output_dir / aggregated_filename(landmark_day)
        aggregated.rename_axis(ID_COL).reset_index().to_csv(agg_path, index=False)
        print(f"  aggregated:        {len(aggregated)} patients -> {agg_path}")

        pre_treatment_lab_df = build_pre_treatment_lab_long(
            df,
            cohort_index=aggregated.index,
            landmark_offset_days=landmark_day,
            anchor_col=anchor_col,
        )
        pre_treatment_path = output_dir / pre_treatment_lab_filename(landmark_day)
        pre_treatment_lab_df.to_csv(pre_treatment_path, index=False)
        print(f"  pre-landmark labs: {len(pre_treatment_lab_df)} rows -> {pre_treatment_path}")

        canonical_labs = select_canonical_labs(
            pre_treatment_lab_df,
            mrns=pd.Index(sorted(train_val_mrns)),
            min_coverage=args.min_patient_coverage,
            id_col=ID_COL,
        )
        for lab in canonical_labs:
            canonical_labs_rows.append({"landmark_days": landmark_day, "lab_name": lab})
        print(f"  canonical labs (train+valid): {len(canonical_labs)}")

        # AUC horizon grid, derived ONCE from train+valid event times (full
        # follow-up; no admin censoring). Downstream models read these from the
        # manifest so mean AUC(t) is on the same horizon set.
        train_val_block = aggregated.loc[aggregated["split"].isin(["train", "valid"])]
        landmark_horizons: dict[str, list[int]] = {}
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

        # NOTE: Older COMPASS experiments also built a wide person-period table
        # for dynamic sequence models. Current COMPASS/IPIO input builders
        # persist aggregated + pre-treatment long lab tables, which are the
        # inputs consumed by the survival models here.

    if canonical_labs_rows:
        canonical_path = output_dir / CANONICAL_LABS_FILENAME
        pd.DataFrame(canonical_labs_rows).to_csv(canonical_path, index=False)
        print(f"\nWrote {canonical_path}")

    max_horizon = max(
        (h for endpoints in auc_horizons_by_landmark.values() for hs in endpoints.values() for h in hs),
        default=0,
    )
    build_manifest = {
        "data": str(args.data),
        "anchor_col": str(args.anchor_col),
        "landmark_days": [int(d) for d in landmark_days],
        "seed": int(args.seed),
        "test_frac": float(args.test_frac),
        "val_frac": float(args.val_frac),
        "min_patient_coverage": float(args.min_patient_coverage),
        "restrict_to_mrns": str(args.restrict_to_mrns) if args.restrict_to_mrns else None,
        "time_unit_days": int(args.time_unit_days),
        "test_stratification": test_stratification,
        "val_stratification": val_stratification,
        "n_patients_common_cohort": int(len(common_mrns)),
        "auc_quantiles": list(auc_quantiles),
        "auc_time_unit_days": int(args.time_unit_days),
        "auc_horizons_by_landmark": auc_horizons_by_landmark,
        "auc_max_horizon": int(max_horizon),
        "longitudinal_wide_format": "out_of_scope_for_ipio",
    }
    manifest_path = output_dir / BUILD_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(build_manifest, indent=2))
    print(f"Wrote {manifest_path}")

    print("\nAll prediction inputs written. Downstream models read from this directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-col", default=ID_COL,
                        help="Patient identifier column name (default DFCI_MRN).")
    parser.add_argument("--age-col", default=AGE_COL,
                        help="Age covariate column name (default AGE_AT_TREATMENTSTART).")
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument(
        "--restrict-to-mrns",
        default=None,
        help=(
            "Optional CSV with a DFCI_MRN (or --id-col) column. When set, the raw "
            "cohort is filtered to this MRN subset before the landmark/split build, "
            "so association testing runs on only those patients."
        ),
    )
    parser.add_argument(
        "--anchor-col",
        default="t_first_treatment",
        help=(
            "Per-patient index-time column (days from FIRST_RECORD_DATE) that anchors "
            "the landmark. Default t_first_treatment, which is 0 for every patient by "
            "the IPIO cohort's design invariant (FIRST_TREATMENT_DATE == "
            "FIRST_RECORD_DATE == IO_START). The genomic arm passes t_sample."
        ),
    )
    parser.add_argument(
        "--landmark-days",
        nargs="+",
        type=int,
        default=DEFAULT_LANDMARK_DAYS,
        help="Landmark offsets in days relative to treatment start.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS / DEFAULT_OUTPUT_SUBDIR),
        help="Directory to write prediction inputs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Fraction of train+val carved out as the validation set.",
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
        help="Time unit (days) for the AUC(t)/Brier horizon grid.",
    )
    parser.add_argument(
        "--auc-quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_AUC_QUANTILES),
        help="Quantiles of train+valid event times used to derive the shared AUC(t) horizon grid.",
    )
    main(parser.parse_args())
