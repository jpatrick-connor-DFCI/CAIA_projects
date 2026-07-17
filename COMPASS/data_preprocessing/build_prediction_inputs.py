"""
Single source of truth for survival-analysis prediction inputs.

For each requested landmark, this script:
  1. Loads the broad row-level longitudinal prostate lab CSV and applies the
     requested downstream cohort filters.
  2. Builds the landmarked patient cohort and intersects MRNs across all
     requested landmarks so every downstream model sees the same patients.
  3. Derives a 3-way train/valid/test split ONCE on the intersection cohort
     at the base (smallest) landmark, then reuses it for every landmark.
     Cox / XGBoost union train+valid into their train_val block; DeepHit
     uses train and valid directly for early stopping.
  4. Writes per-landmark:
       aggregated_landmark{D}.csv                 (1 row per MRN; features + outcomes + split)
       pre_treatment_lab_long_landmark{D}.csv     (long-format pre-landmark labs
                                                   for per-fold canonical-lab selection)
  5. Writes shared:
       split_assignments.csv                      DFCI_MRN, split (train/valid/test)
       landmark_mrn_availability.csv              per-landmark eligibility + split
       canonical_labs_train_val.csv               landmark_days, lab_name (train+valid)
"""

from __future__ import annotations

import argparse
import json
import re
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

from cox_aggregated import (  # noqa: E402
    DATA_PATH,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_MIN_PATIENT_COVERAGE,
    ENDPOINTS,
    RESULTS,
)
from survival_common.cohort import (  # noqa: E402
    AGE_COL,
    ID_COL,
    build_landmark_availability_table,
    build_landmark_merged,
    build_pre_treatment_lab_long,
    normalize_landmark_days,
)
from survival_common.helper import (  # noqa: E402
    DEFAULT_AUC_QUANTILES,
    assert_no_test_leakage,
    choose_stratification_labels,
    compute_horizon_grid,
    select_canonical_labs,
)

DEFAULT_OUTPUT_SUBDIR = "prediction_inputs"
DEFAULT_VAL_FRAC = 0.20
DEFAULT_TIME_UNIT_DAYS = 7
DEFAULT_MIN_PSA_COUNT = 3

SPLIT_ASSIGNMENTS_FILENAME = "split_assignments.csv"
LANDMARK_AVAILABILITY_FILENAME = "landmark_mrn_availability.csv"
LANDMARK_ATTRITION_FILENAME = "landmark_attrition.json"
CANONICAL_LABS_FILENAME = "canonical_labs_train_val.csv"
BUILD_MANIFEST_FILENAME = "build_manifest.json"


def _is_none_anchor(value: str | None) -> bool:
    """Whether --anchor-col requests the pure-offset (no anchor column) landmark."""
    return value is None or str(value).strip().lower() in {"", "none"}


def aggregated_filename(landmark_day: int) -> str:
    return f"aggregated_landmark{int(landmark_day)}.csv"


def pre_treatment_lab_filename(landmark_day: int) -> str:
    return f"pre_treatment_lab_long_landmark{int(landmark_day)}.csv"


# Cancer-stage baseline covariate (PROFILE only; ported from the clinical-text
# embedding project). Stage is a static per-patient attribute derived from
# clinical text, keyed by DFCI_MRN. One-hot with Stage I as the reference, so the
# fixed dummy set is exactly these three columns.
STAGE_CATEGORIES = ["I", "II", "III", "IV"]
STAGE_COLUMNS = ["CANCER_STAGE_II", "CANCER_STAGE_III", "CANCER_STAGE_IV"]
_STAGE_TOKEN = re.compile(r"^(IV|III|II|I|4|3|2|1)[A-D]?$")
_ARABIC_TO_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV"}

def normalize_cancer_stage(raw) -> str | None:
    """Collapse a raw stage value to a major stage in {I,II,III,IV}.

    Mirrors the clinical-text-embedding project: strips a leading "STAGE",
    drops substage letters (IVA -> IV), maps arabic numerals (4 -> IV), and
    handles float reprs (2.0 -> II). Returns None for unknown / in-situ /
    unstageable values so they are treated as missing downstream.
    """
    if pd.isna(raw):
        return None
    s = str(raw).upper().strip().replace("STAGE", "").strip()
    s = re.sub(r"\.0+$", "", s)
    m = _STAGE_TOKEN.match(s)
    if not m:
        return None
    token = m.group(1)
    return _ARABIC_TO_ROMAN.get(token, token)


def _major_stage_from_pickle(stage_file: Path, id_col: str) -> pd.Series:
    """Read a ``{DFCI_MRN: stage}`` pickle into a major-stage Series."""
    raw = pd.read_pickle(stage_file)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a {{id: stage}} dict in {stage_file}, got {type(raw)}.")
    items: dict[int, str | None] = {}
    for key, value in raw.items():
        try:
            mrn = int(key)
        except (TypeError, ValueError):
            continue
        items[mrn] = normalize_cancer_stage(value)
    major = pd.Series(items, name="CANCER_STAGE")
    major.index.name = id_col
    return major


def _major_stage_from_dummy_csv(stage_file: Path, id_col: str) -> pd.Series:
    """Read the pre-encoded ``cancer_stage_df.csv.gz`` into a major-stage Series.

    That file is ``DFCI_MRN`` + one-hot ``CANCER_STAGE_<value>`` dummies built
    with ``drop_first=True`` (Stage I reference = all-zero) and WITHOUT substage
    normalization. We reconstruct each patient's raw stage from whichever dummy
    is set (all-zero rows -> the dropped Stage I reference) and collapse to a
    major stage in {I,II,III,IV}; suffixes that don't map to a major stage (e.g.
    in-situ / unknown) become NaN.
    """
    df = pd.read_csv(stage_file)
    id_name = next((c for c in (id_col, "DFCI_MRN") if c in df.columns), None)
    if id_name is None:
        raise ValueError(
            f"{stage_file} has no '{id_col}'/'DFCI_MRN' column; columns={list(df.columns)[:8]}"
        )
    df = df.loc[pd.to_numeric(df[id_name], errors="coerce").notna()].copy()
    df[id_name] = pd.to_numeric(df[id_name], errors="coerce").astype(int)
    df = df.drop_duplicates(subset=[id_name], keep="first").set_index(id_name)

    dummy_cols = [c for c in df.columns if c.startswith("CANCER_STAGE_")]
    if not dummy_cols:
        raise ValueError(f"{stage_file} has no CANCER_STAGE_* dummy columns.")
    onehot = df[dummy_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    any_hit = pd.Series(onehot.to_numpy().sum(axis=1) > 0, index=df.index)
    # idxmax gives the set dummy for one-hot rows; all-zero rows fall back to the
    # dropped Stage I reference.
    picked = onehot.idxmax(axis=1).str[len("CANCER_STAGE_"):]
    raw = picked.where(any_hit, "I")
    major = raw.map(normalize_cancer_stage)
    major.name = "CANCER_STAGE"
    major.index.name = id_col
    return major


def load_stage_dummies(stage_file: Path, cohort_index: pd.Index, id_col: str) -> pd.DataFrame:
    """Build fixed CANCER_STAGE_II/III/IV dummies indexed by ``id_col``.

    Accepts either the raw ``{DFCI_MRN: stage}`` pickle or the pre-encoded
    ``cancer_stage_df.csv[.gz]`` one-hot table; both are collapsed to a major
    stage and re-encoded with Stage I as the reference. Patients absent from the
    source (or with an unstageable value) get all-NaN dummies — NOT all-zero —
    so the downstream imputer treats them as missing rather than collapsing them
    into the Stage I reference. The column set is fixed via an ordered
    Categorical, so it is identical regardless of which stages happen to appear.
    """
    if stage_file.suffix.lower() in {".pkl", ".pickle"}:
        major = _major_stage_from_pickle(stage_file, id_col)
    elif ".csv" in "".join(stage_file.suffixes).lower():
        major = _major_stage_from_dummy_csv(stage_file, id_col)
    else:
        raise ValueError(
            f"Unsupported --stage-file type: {stage_file} (expected .pkl or .csv[.gz])."
        )

    normalized = pd.Series(
        {mrn: major.get(mrn) for mrn in cohort_index},
        name="CANCER_STAGE",
    )
    normalized.index = cohort_index
    normalized.index.name = id_col
    categorical = pd.Categorical(normalized, categories=STAGE_CATEGORIES, ordered=True)
    dummies = pd.get_dummies(
        pd.DataFrame({"CANCER_STAGE": categorical}, index=cohort_index),
        columns=["CANCER_STAGE"],
        drop_first=True,
    )
    # Guarantee the fixed column set even if a stage level is entirely absent.
    for col in STAGE_COLUMNS:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[STAGE_COLUMNS].astype(float)
    # Unknown/absent stage -> NaN dummies (missing), not a false Stage I.
    unknown = normalized.isna().to_numpy()
    dummies.loc[unknown, STAGE_COLUMNS] = np.nan
    dummies.index.name = id_col
    return dummies


def load_mrn_subset(mrn_file: Path, id_col: str) -> set[int]:
    """Read the restrict-to-MRNs CSV and return the subset as a set of int ids.

    Expects a CSV with an ``id_col`` (default DFCI_MRN) column; any other columns
    are ignored. MRNs are coerced to int to match the cohort id dtype set in
    ``main`` so the downstream ``isin`` filter aligns on type.
    """
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


def load_broad_psa_counts(total_psa_file: Path, id_col: str) -> pd.Series:
    """Per-patient count of *broad* PSA labs from total_psa_records.csv.

    Prevalence filtering counts any and all PSA assays (total, free, complexed,
    ultrasensitive, percentage, etc.) as written by compile_COMPASS_cohort_data.py
    -- i.e. the full TEST_TYPE_CD set, NOT the narrow OMOP-collapsed
    ``LAB_NAME == "PSA"`` set that drives the prediction features. Each row in
    this file is one PSA result, so the group size is the assay count per patient.

    Returns an int Series indexed by patient id.
    """
    if not total_psa_file.exists():
        raise FileNotFoundError(
            f"--min-psa-count requires the broad PSA records file, but "
            f"{total_psa_file} does not exist. Point --total-psa-file at the "
            f"total_psa_records.csv written by compile_COMPASS_cohort_data.py."
        )
    psa = pd.read_csv(total_psa_file, low_memory=False)
    col = id_col if id_col in psa.columns else "DFCI_MRN"
    if col not in psa.columns:
        raise ValueError(
            f"{total_psa_file} has no '{id_col}'/'DFCI_MRN' column; "
            f"columns={list(psa.columns)[:8]}"
        )
    ids = pd.to_numeric(psa[col], errors="coerce").dropna().astype(int)
    return ids.value_counts()


def apply_downstream_cohort_filters(
    df: pd.DataFrame,
    *,
    min_psa_count: int,
    exclude_parpi: bool,
    broad_psa_counts: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply patient-level cohort filters after loading the broad lab frame.

    Treated status is enforced downstream by make_outcome_df: durations are
    measured from the treatment anchor, so patients with no highlighted-drug anchor
    have all-NaN durations and fail its notna() validity checks. There is no
    first-treatment inclusion step here.
    """
    if min_psa_count < 0:
        raise ValueError(f"min_psa_count must be >= 0, got {min_psa_count}.")

    out = df.copy()
    n_start = out[ID_COL].nunique()
    print(f"Downstream cohort filter start: {n_start} patients")

    if min_psa_count > 0:
        # Prevalence gate uses the BROAD PSA set (all PSA assays from
        # total_psa_records.csv), not the narrow OMOP-collapsed LAB_NAME=="PSA"
        # rows in this frame. The latter still drives prediction features
        # downstream; only cohort inclusion counts the broad set here.
        if broad_psa_counts is None:
            raise ValueError(
                "--min-psa-count > 0 requires broad_psa_counts "
                "(loaded from --total-psa-file)."
            )
        keep_psa = broad_psa_counts.loc[broad_psa_counts >= min_psa_count].index
        out = out.loc[out[ID_COL].isin(keep_psa)].copy()
        n_after_psa = out[ID_COL].nunique()
        print(
            f"  PSA count filter (>= {min_psa_count}, broad PSA set): "
            f"kept {n_after_psa}/{n_start}"
        )
    else:
        n_after_psa = out[ID_COL].nunique()
        print("  PSA count filter: disabled")

    if exclude_parpi:
        if "PARPI_EXPOSED" in out.columns:
            parpi = pd.to_numeric(out["PARPI_EXPOSED"], errors="coerce").fillna(0).astype(int)
            out = out.loc[~parpi.eq(1)].copy()
            n_after_parpi = out[ID_COL].nunique()
            print(
                f"  PARPi exclusion: dropped {n_after_psa - n_after_parpi} "
                f"(remaining: {n_after_parpi})"
            )
        else:
            n_after_parpi = n_after_psa
            print("  PARPi exclusion requested, but PARPI_EXPOSED is missing; skipping")
    else:
        n_after_parpi = n_after_psa
        print("  PARPi exclusion: disabled")

    if out.empty:
        raise ValueError("No rows remain after downstream cohort filters.")

    attrition = {
        "n_before_downstream_cohort_filters": int(n_start),
        "min_psa_count": int(min_psa_count),
        "n_after_psa_count_filter": int(n_after_psa),
        "exclude_parpi": bool(exclude_parpi),
        "n_after_parpi_exclusion": int(n_after_parpi),
    }
    return out, attrition


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
    test_labels, test_stratification = choose_stratification_labels(
        base_merged,
        min_count=2,
        event_col="PLATINUM",
    )
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
    val_labels, val_stratification = choose_stratification_labels(
        train_val_block,
        min_count=2,
        event_col="PLATINUM",
    )
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


AGGREGATED_DROP_COLUMNS = (
    # Raw date strings — duplicate of the rebased durations; not consumed downstream.
    "FIRST_RECORD_DATE",
    "DIAGNOSIS_DATE",
    "LAST_CONTACT_DATE",
    "PLATINUM_DATE",
    "TREATMENT_ANCHOR_DATE",
    # Pre-rebase duration duplicates kept for debugging by make_outcome_df.
    "t_platinum_from_first_record",
    "t_last_contact_from_first_record",
    "t_death_from_first_record",
    # Shared make_outcome_df still derives these first-treatment fields (falling
    # back to last-contact when the columns are absent, as they are for the
    # treatment-anchored COMPASS pipeline); drop them so they never leak into the
    # feature matrix.
    "FIRST_TREATMENT_DATE",
    "FIRST_TREATMENT",
    "t_first_treatment",
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
    # Push the runtime schema into the shared cohort builders so make_outcome_df /
    # build_feature_matrix / build_landmark_merged operate on the requested id/age
    # columns.
    import survival_common.cohort as _cohort
    _cohort.configure_id_columns(ID_COL, AGE_COL)
    landmark_days = normalize_landmark_days(args.landmark_days)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    print(f"Loading PROFILE data from {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)

    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce")
    df = df.loc[df[ID_COL].notna()].copy()
    df[ID_COL] = df[ID_COL].astype(int)
    n_loaded_cohort = df[ID_COL].nunique()
    print(f"Loaded cohort: {n_loaded_cohort} unique MRNs")

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

    broad_psa_counts = None
    if args.min_psa_count > 0:
        total_psa_file = Path(args.total_psa_file)
        print(f"Loading broad PSA records for prevalence filter from {total_psa_file} ...")
        broad_psa_counts = load_broad_psa_counts(total_psa_file, ID_COL)
        print(
            f"  Broad PSA records: {int(broad_psa_counts.sum())} labs across "
            f"{broad_psa_counts.size} patients"
        )

    df, cohort_filter_attrition = apply_downstream_cohort_filters(
        df,
        min_psa_count=args.min_psa_count,
        exclude_parpi=args.exclude_parpi,
        broad_psa_counts=broad_psa_counts,
    )

    # COMPASS durations are already measured from the treatment anchor (time 0),
    # so anchor_col is None: the landmark is a pure offset and there is no separate
    # anchor index column. "Treated" is enforced by the anchor-relative durations
    # themselves -- patients with no highlighted-drug anchor have all-NaN durations
    # and are dropped by make_outcome_df's notna() validity checks.
    anchor_col = None if _is_none_anchor(args.anchor_col) else args.anchor_col
    if anchor_col is not None and anchor_col not in df.columns:
        raise ValueError(
            f"--anchor-col {anchor_col!r} not found in {data_path}. Available t_* "
            f"columns: {[c for c in df.columns if c.startswith('t_')]}"
        )
    if anchor_col is None:
        print(
            "Anchor column: none (durations are treatment-anchor-relative; "
            "landmark is a pure offset from the anchor)"
        )
    else:
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
            require_first_treatment=False,
        )
        merged_by_landmark[landmark_day] = merged

    availability, common_mrns = build_landmark_availability_table(merged_by_landmark)
    common_mrns = merged_by_landmark[landmark_days[0]].index.intersection(common_mrns)
    if len(common_mrns) == 0:
        raise ValueError("No MRNs were eligible at every requested landmark.")
    print(f"\nCommon MRN cohort across landmarks {landmark_days}: {len(common_mrns)} patients")

    if len(landmark_days) >= 2:
        for earlier, later in zip(landmark_days, landmark_days[1:]):
            idx_earlier = merged_by_landmark[earlier].index
            idx_later = merged_by_landmark[later].index
            only_earlier = idx_earlier.difference(idx_later)
            only_later = idx_later.difference(idx_earlier)
            print(
                f"[debug] merged landmark +{earlier}d ({len(idx_earlier)}) vs +{later}d "
                f"({len(idx_later)}): in +{earlier}d only={len(only_earlier)}, "
                f"in +{later}d only={len(only_later)}"
            )
            if len(only_earlier):
                print(f"[debug]   +{earlier}d-only MRNs (first 20): {list(only_earlier)[:20]}")

    # Optional static cancer-stage baseline covariate (PROFILE only). Built once
    # on the full common cohort so the CANCER_STAGE_* column set is identical
    # across the train/valid/test split and every landmark.
    stage_dummies = None
    if args.stage_file:
        stage_dummies = load_stage_dummies(Path(args.stage_file), common_mrns, ID_COL)
        n_known = int(stage_dummies.notna().any(axis=1).sum())
        print(
            f"Loaded cancer-stage dummies from {args.stage_file}: "
            f"{n_known}/{len(common_mrns)} patients have a known stage "
            f"(columns: {', '.join(STAGE_COLUMNS)})"
        )

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

    # Structured attrition counts for the Figure 1 CONSORT diagram, continuing
    # the chain persisted by longitudinal_data_processing.py's cohort_attrition.json.
    landmark_attrition = {
        "n_loaded_cohort": int(n_loaded_cohort),
        "downstream_cohort_filters": cohort_filter_attrition,
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

    train_mrns = set(split.index[split.eq("train")])
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
        if stage_dummies is not None:
            aggregated = aggregated.join(stage_dummies, how="left")

        agg_path = output_dir / aggregated_filename(landmark_day)
        aggregated.rename_axis(ID_COL).reset_index().to_csv(agg_path, index=False)
        print(f"  aggregated:        {len(aggregated)} patients -> {agg_path}")
        n_platinum = int(aggregated["PLATINUM"].eq(1).sum())
        n_platinum_by_split = (
            aggregated.loc[aggregated["PLATINUM"].eq(1), "split"].value_counts().to_dict()
        )
        print(
            f"  platinum events:   {n_platinum}/{len(aggregated)} patients "
            f"({', '.join(f'{k}={v}' for k, v in n_platinum_by_split.items())})"
        )

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

        # Per-endpoint AUC horizon grid, derived ONCE from train+valid event
        # times (full follow-up; no admin censoring). All downstream models
        # read these from the manifest so mean AUC(t) is on the same horizon set.
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
        "anchor_col": "none" if anchor_col is None else str(anchor_col),
        "downstream_cohort_filters": cohort_filter_attrition,
        "landmark_days": [int(d) for d in landmark_days],
        "seed": int(args.seed),
        "test_frac": float(args.test_frac),
        "val_frac": float(args.val_frac),
        "min_patient_coverage": float(args.min_patient_coverage),
        "stage_file": str(args.stage_file) if args.stage_file else None,
        "restrict_to_mrns": str(args.restrict_to_mrns) if args.restrict_to_mrns else None,
        "stage_columns": STAGE_COLUMNS if stage_dummies is not None else [],
        "time_unit_days": int(args.time_unit_days),
        "test_stratification": test_stratification,
        "val_stratification": val_stratification,
        "n_patients_common_cohort": int(len(common_mrns)),
        "auc_quantiles": list(auc_quantiles),
        "auc_time_unit_days": int(args.time_unit_days),
        "auc_horizons_by_landmark": auc_horizons_by_landmark,
        "auc_max_horizon": int(max_horizon),
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
        default="none",
        help=(
            "Per-patient index-time column that anchors the landmark. COMPASS "
            "durations (t_lab, t_platinum, ...) are already measured from the "
            "treatment anchor (first ARPI/taxane/radium-223 exposure = time 0), so "
            "the default 'none' makes the landmark a pure offset from the anchor "
            "with no anchor column. Pass a t_* column name to anchor on a different "
            "index-time instead (days from FIRST_RECORD_DATE)."
        ),
    )
    parser.add_argument(
        "--min-psa-count",
        type=int,
        default=DEFAULT_MIN_PSA_COUNT,
        help=(
            "Minimum number of PSA labs (broad set: total/free/complexed/"
            "ultrasensitive/etc. from --total-psa-file) required before landmark "
            "building. Set to 0 to disable this downstream cohort filter."
        ),
    )
    parser.add_argument(
        "--total-psa-file",
        default=str(DATA_PATH / "total_psa_records.csv"),
        help=(
            "Broad PSA records CSV (one row per PSA lab, any assay type) written "
            "by compile_COMPASS_cohort_data.py. Used only for the --min-psa-count "
            "prevalence gate; prediction features still use the OMOP-collapsed "
            "LAB_NAME == 'PSA' set."
        ),
    )
    parser.add_argument(
        "--exclude-parpi",
        dest="exclude_parpi",
        action="store_true",
        default=True,
        help="Exclude PARPi-exposed patients when PARPI_EXPOSED is present (default; on).",
    )
    parser.add_argument(
        "--include-parpi",
        dest="exclude_parpi",
        action="store_false",
        help="Do not exclude PARPi-exposed patients.",
    )
    parser.add_argument(
        "--landmark-days",
        nargs="+",
        type=int,
        default=DEFAULT_LANDMARK_DAYS,
        help="Landmark offsets in days relative to --anchor-col.",
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
        help="Width of the time bin (in days) used for the AUC(t) horizon grid.",
    )
    parser.add_argument(
        "--stage-file",
        default=None,
        help=(
            "Optional cancer-stage source for the age(+stage) baseline: either "
            "the pre-encoded cancer_stage_df.csv[.gz] (DFCI_MRN + one-hot "
            "CANCER_STAGE_* dummies) or the raw {DFCI_MRN: stage} pickle. When "
            "set, adds static CANCER_STAGE_II/III/IV dummies (Stage I reference) "
            "to each aggregated_landmark{D}.csv."
        ),
    )
    parser.add_argument(
        "--auc-quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_AUC_QUANTILES),
        help="Quantiles of train+valid event times used to derive the shared AUC(t) horizon grid.",
    )
    main(parser.parse_args())
