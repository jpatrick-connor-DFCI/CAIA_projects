"""
Single source of truth for survival-analysis prediction inputs.

For each requested landmark, this script:
  1. Loads the broad row-level longitudinal prostate lab CSV and applies the
     requested downstream cohort filters.
  2. Builds a separate eligible risk set for each requested landmark. Patients
     do not have to survive/event-free to a later landmark to enter an earlier
     one, avoiding an immortal-time restriction on the day-0 cohort.
  3. Derives an independent 3-way train/valid/test split within each landmark
     cohort.
     Cox / XGBoost union train+valid into their train_val block; DeepHit
     uses train and valid directly for early stopping.
  4. Writes per-landmark:
       aggregated_landmark{D}.csv                 (1 row per MRN; features + outcomes + split)
       pre_treatment_lab_long_landmark{D}.csv     (long-format pre-landmark labs
                                                   for per-fold canonical-lab selection)
  5. Writes shared:
       split_assignments_landmark{D}.csv           DFCI_MRN, split (per landmark)
       split_assignments.csv                       base-landmark compatibility copy
       landmark_mrn_availability.csv               per-landmark eligibility + splits
       canonical_labs_train_val.csv               landmark_days, lab_name (train+valid)
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
    build_person_period_wide,
    build_pre_treatment_lab_long,
    normalize_landmark_days,
)
from survival_common.helper import (  # noqa: E402
    AUC_TIMELINE_SCHEMA_VERSION,
    DEFAULT_AUC_MAX_GRID_POINTS,
    DEFAULT_AUC_MAX_TIME_UNITS,
    DEFAULT_AUC_QUANTILES,
    assert_no_test_leakage,
    choose_stratification_labels,
    compute_horizon_grid,
    select_canonical_labs,
)
DEFAULT_OUTPUT_SUBDIR = "prediction_inputs"
DEFAULT_VAL_FRAC = 0.20
DEFAULT_TIME_UNIT_DAYS = 7
DEFAULT_MIN_PSA_COUNT = 5
DEFAULT_LONG_MIN_COVERAGE = 0.1
DEFAULT_OUTLIER_LO = 0.005
DEFAULT_OUTLIER_HI = 0.995
# event_cols/time_to_event_cols carried in build_manifest.json (Cox/XGBoost
# ignore these; see survival_common.longitudinal_targets.LONGITUDINAL_CONFIGS
# for the config registry that consumes them). NEPC is appended only when the
# cohort actually carries it -- see longitudinal_event_columns().
LONGITUDINAL_EVENT_COLS = ["PLATINUM", "DEATH"]
LONGITUDINAL_TIME_TO_EVENT_COLS = ["t_platinum", "t_death"]
LONGITUDINAL_OPTIONAL_EVENT_COLS = {
    "NEPC": "t_nepc",
    "AVPC": "t_avpc",
}


def longitudinal_event_columns(frame) -> tuple[list[str], list[str]]:
    """(event_cols, time_to_event_cols) for the manifest, given a cohort frame.

    The base platinum/death pair is always declared. Optional endpoints are
    appended only when both their event and duration columns are present, so a
    cohort built without the LLM annotations mounted advertises exactly what it
    used to and ``resolve_config("nepc", ...)`` fails with a clear "rebuild
    prediction inputs" message rather than a KeyError deep in target building.
    """
    event_cols = list(LONGITUDINAL_EVENT_COLS)
    time_cols = list(LONGITUDINAL_TIME_TO_EVENT_COLS)
    for event_col, time_col in LONGITUDINAL_OPTIONAL_EVENT_COLS.items():
        if event_col in frame.columns and time_col in frame.columns:
            event_cols.append(event_col)
            time_cols.append(time_col)
    return event_cols, time_cols
LONGITUDINAL_SCHEMA_VERSION = 1
# ~10 years. Restores administrative censoring at the cohort level so it
# matches the AUC(t)/Brier evaluation timeline, which is already capped at
# DEFAULT_AUC_MAX_TIME_UNITS * DEFAULT_TIME_UNIT_DAYS = 260 * 7 = 1820 days by
# default (see survival_common.helper); kept as an explicit, recorded flag
# rather than a silent default because it changes every existing
# prediction-inputs result.
DEFAULT_MAX_FOLLOWUP_DAYS = 3650.0
# Broad PSA assay set (raw TEST_TYPE_CD, any assay type: total/free/complexed/
# ultrasensitive/etc.), read from longitudinal_prediction_data.csv's
# RAW_TEST_CODE passthrough column. This is deliberately NOT the narrow
# OMOP-collapsed LAB_NAME == "PSA" set, which drives prediction features.
# Kept in sync with BROAD_PSA_CODES in longitudinal_data_processing.py.
BROAD_PSA_CODES = ["PSA", "PSAR", "PSATOTSCRN", "CPSA", "PSAMON", "PSAULT", "PSAT"]

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


def split_assignments_filename(landmark_day: int) -> str:
    return f"split_assignments_landmark{int(landmark_day)}.csv"


def longitudinal_filename(landmark_day: int) -> str:
    return f"longitudinal_landmark{int(landmark_day)}.csv"


def longitudinal_manifest_filename(landmark_day: int) -> str:
    return f"longitudinal_landmark{int(landmark_day)}_manifest.json"


def validate_max_followup_days(
    max_followup_days: float | None,
    *,
    auc_max_time_units: int,
    time_unit_days: int,
) -> None:
    """Raise if the administrative-censoring horizon is shorter than the
    AUC(t)/Brier evaluation timeline it must cover.

    A cohort censored shorter than the horizons it is scored at would
    silently blank the tail of every model's evaluation timeline rather than
    erroring, so this is a hard fail rather than a warning.
    """
    if max_followup_days is None:
        return
    auc_horizon_days = auc_max_time_units * time_unit_days
    if max_followup_days < auc_horizon_days:
        raise ValueError(
            f"--max-followup-days {max_followup_days:g} is shorter than the "
            f"AUC(t)/Brier evaluation horizon (--auc-max-time-units "
            f"{auc_max_time_units} * --time-unit-days {time_unit_days} = "
            f"{auc_horizon_days:g} days). A cohort censored shorter than the "
            "horizons it is scored at would silently blank the tail of every "
            "model's evaluation timeline. Raise --max-followup-days or lower "
            "--auc-max-time-units."
        )


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


def load_broad_psa_counts(df: pd.DataFrame, id_col: str) -> pd.Series:
    """Per-patient count of *broad* PSA labs from the already-loaded
    longitudinal_prediction_data.csv frame's RAW_TEST_CODE column.

    Prevalence filtering counts any and all PSA assays (total, free, complexed,
    ultrasensitive, percentage, etc. -- BROAD_PSA_CODES) as carried through by
    longitudinal_data_processing.py's RAW_TEST_CODE passthrough, NOT the narrow
    OMOP-collapsed ``LAB_NAME == "PSA"`` set that drives the prediction
    features. Each matching row is one PSA result, so the group size is the
    assay count per patient.

    Returns an int Series indexed by patient id.
    """
    if "RAW_TEST_CODE" not in df.columns:
        raise ValueError(
            "--min-psa-count requires a RAW_TEST_CODE column in --data "
            "(written by longitudinal_data_processing.py); none found."
        )
    psa = df.loc[df["RAW_TEST_CODE"].isin(BROAD_PSA_CODES)]
    ids = pd.to_numeric(psa[id_col], errors="coerce").dropna().astype(int)
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
        # Prevalence gate uses the BROAD PSA set (all PSA assays, from
        # --data's RAW_TEST_CODE column via load_broad_psa_counts), not the
        # narrow OMOP-collapsed LAB_NAME=="PSA" rows in this frame. The latter
        # still drives prediction features downstream; only cohort inclusion
        # counts the broad set here.
        if broad_psa_counts is None:
            raise ValueError(
                "--min-psa-count > 0 requires broad_psa_counts "
                "(loaded from --data's RAW_TEST_CODE column)."
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
    "NEPC_DATE",
    "AVPC_NEPC_DATE",
    "AVPC_DATE",
    "NEPC_TIMELINE_DATE",
    "TREATMENT_ANCHOR_DATE",
    # Pre-rebase duration duplicates kept for debugging by make_outcome_df.
    "t_platinum_from_first_record",
    "t_nepc_from_first_record",
    "t_avpc_nepc_from_first_record",
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


def assert_split_agreement(
    aggregated: pd.DataFrame,
    longitudinal: pd.DataFrame,
    *,
    landmark_day: int,
) -> None:
    """Guard that the per-landmark longitudinal output agrees with the
    aggregated output on split assignment: one split per MRN, no MRN present
    in longitudinal but absent from aggregated, and no disagreement on the
    overlap. Cheap, and catches exactly the class of bug that would silently
    leak train/test across the two output families.
    """
    long_split = (
        longitudinal.groupby(ID_COL)["split"].agg(lambda s: s.iloc[0]).rename("split_long")
    )
    nunique = longitudinal.groupby(ID_COL)["split"].nunique()
    inconsistent = nunique[nunique > 1]
    if not inconsistent.empty:
        raise AssertionError(
            f"Landmark +{landmark_day}d: {len(inconsistent)} MRNs have multiple split "
            f"labels in the longitudinal output (first: {inconsistent.index[0]})."
        )

    agg_split = aggregated["split"].rename("split_agg")
    extra_in_long = long_split.index.difference(agg_split.index)
    if len(extra_in_long):
        raise AssertionError(
            f"Landmark +{landmark_day}d: {len(extra_in_long)} MRNs appear in longitudinal "
            f"but not aggregated (first: {extra_in_long[0]})."
        )

    overlap = long_split.index.intersection(agg_split.index)
    mismatched = (agg_split.loc[overlap] != long_split.loc[overlap]).sum()
    if mismatched:
        raise AssertionError(
            f"Landmark +{landmark_day}d: {mismatched} MRNs disagree on split between "
            "aggregated and longitudinal."
        )


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

    # Optional exclusion list, applied AFTER --restrict-to-mrns and before the
    # landmark build for the same reason: the split, canonical lab set, and
    # feature selection must all be derived on the final patient set. The two
    # lists are separate arguments so a cohort definition stays independent of
    # any exclusion rule layered on top of it.
    exclude_mrns = getattr(args, "exclude_mrns", None)
    if exclude_mrns:
        excluded = load_mrn_subset(Path(exclude_mrns), ID_COL)
        n_before = df[ID_COL].nunique()
        df = df.loc[~df[ID_COL].isin(excluded)].copy()
        n_after = df[ID_COL].nunique()
        if n_after == 0:
            raise ValueError(
                f"--exclude-mrns {exclude_mrns}: excluding the "
                f"{len(excluded)} listed MRNs left no patients in the cohort."
            )
        print(
            f"  [exclude-mrns] {n_before - n_after} of {n_before} MRNs excluded, "
            f"{n_after} retained ({len(excluded)} listed in {exclude_mrns})"
        )

    broad_psa_counts = None
    if args.min_psa_count > 0:
        print(f"Computing broad PSA prevalence counts from RAW_TEST_CODE in {data_path} ...")
        broad_psa_counts = load_broad_psa_counts(df, ID_COL)
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

    validate_max_followup_days(
        args.max_followup_days,
        auc_max_time_units=args.auc_max_time_units,
        time_unit_days=args.time_unit_days,
    )
    if args.max_followup_days is not None:
        auc_horizon_days = args.auc_max_time_units * args.time_unit_days
        print(
            f"Administrative censoring: {args.max_followup_days:g} days "
            f"(AUC(t)/Brier timeline cap is {auc_horizon_days:g} days)"
        )
    else:
        print("Administrative censoring: disabled (--no-max-followup-days)")

    # getattr default: callers (tests, notebooks) build the Namespace directly
    # rather than through the parser, so a bare attribute access would break them.
    require_nepc = bool(getattr(args, "require_nepc", False))
    endpoint = "nepc" if require_nepc else str(getattr(args, "endpoint", "platinum")).lower()
    if endpoint == "platinum":
        print(
            "Platinum endpoint cohort: ON (--endpoint platinum). t_platinum must be "
            "non-null and > 0, without applying any other endpoint's timing "
            "eligibility filter."
        )
    else:
        duration_col = ENDPOINTS[endpoint]["duration_col"]
        description = ENDPOINTS[endpoint]["description"]
        print(
            f"{endpoint!r} endpoint cohort: ON (--endpoint {endpoint}). "
            f"{duration_col} must be non-null and > 0, so an event at or before "
            "the landmark (prevalent) is excluded, without applying any "
            f"platinum-time eligibility filter. {description}."
        )
    merged_by_landmark: dict[int, pd.DataFrame] = {}
    for landmark_day in landmark_days:
        print(f"\n##### COHORT BUILD: LANDMARK +{landmark_day} DAYS #####")
        _, _, merged = build_landmark_merged(
            df,
            landmark_offset_days=landmark_day,
            anchor_col=anchor_col,
            require_first_treatment=False,
            max_followup_days=args.max_followup_days,
            endpoint=endpoint,
        )
        merged_by_landmark[landmark_day] = merged

    availability, common_mrns = build_landmark_availability_table(merged_by_landmark)
    print("\nIndependent landmark cohorts:")
    for landmark_day in landmark_days:
        print(f"  +{landmark_day}d: {len(merged_by_landmark[landmark_day])} patients")
    print(
        f"  descriptive overlap eligible at every landmark: {len(common_mrns)} patients "
        "(not used to restrict any cohort)"
    )

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

    base_landmark_day = landmark_days[0]
    splits_by_landmark: dict[int, pd.Series] = {}
    stratification_by_landmark: dict[str, dict[str, str]] = {}
    split_sizes_by_landmark: dict[str, dict[str, int]] = {}
    for landmark_day in landmark_days:
        split, test_stratification, val_stratification = derive_three_way_split(
            merged_by_landmark[landmark_day],
            test_frac=args.test_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        splits_by_landmark[landmark_day] = split
        counts = split.value_counts().to_dict()
        split_sizes_by_landmark[str(landmark_day)] = {
            key: int(counts.get(key, 0)) for key in ("train", "valid", "test")
        }
        stratification_by_landmark[str(landmark_day)] = {
            "test": test_stratification,
            "validation": val_stratification,
        }
        print(
            f"Landmark +{landmark_day}d split "
            f"(test={test_stratification}, validation={val_stratification}): "
            f"train={counts.get('train', 0)} valid={counts.get('valid', 0)} "
            f"test={counts.get('test', 0)}"
        )
        landmark_split_path = output_dir / split_assignments_filename(landmark_day)
        split.rename_axis(ID_COL).reset_index().to_csv(landmark_split_path, index=False)
        print(f"Wrote {landmark_split_path}")
        availability[f"split_landmark_{landmark_day}"] = availability[ID_COL].map(split)

    # Preserve the historical filename as a base-landmark copy for the optional
    # genomic arm and any external consumers that expect one split file.
    base_split = splits_by_landmark[base_landmark_day]
    split_path = output_dir / SPLIT_ASSIGNMENTS_FILENAME
    base_split.rename_axis(ID_COL).reset_index().to_csv(split_path, index=False)
    print(f"Wrote base-landmark compatibility split to {split_path}")

    availability["included_all_landmarks"] = availability[ID_COL].isin(common_mrns)
    availability["split"] = availability[ID_COL].map(base_split)
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
        "n_common_across_landmarks_descriptive_only": int(len(common_mrns)),
        "split_sizes_by_landmark": split_sizes_by_landmark,
    }
    attrition_path = output_dir / LANDMARK_ATTRITION_FILENAME
    attrition_path.write_text(json.dumps(landmark_attrition, indent=2))
    print(f"Wrote {attrition_path}")

    canonical_labs_rows: list[dict] = []
    auc_horizons_by_landmark: dict[str, dict[str, list[int]]] = {}
    auc_quantiles = tuple(args.auc_quantiles)

    for landmark_day in landmark_days:
        print(f"\n##### LANDMARK +{landmark_day}d: BUILD INPUTS #####")
        merged = merged_by_landmark[landmark_day].copy()
        split = splits_by_landmark[landmark_day]
        train_val_mrns = set(split.index[split.isin(["train", "valid"])])
        test_mrns = set(split.index[split.eq("test")])
        assert_no_test_leakage(
            test_mrns=test_mrns,
            train_mrns=train_val_mrns,
            context=f"build_prediction_inputs landmark +{landmark_day}d: test vs train+valid",
        )
        aggregated = build_aggregated_table(merged, split=split)

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
        # times under the same administrative cap the runners evaluate with,
        # so every requested horizon is estimable rather than dead weight in
        # the coverage denominator. All downstream models read these from the
        # manifest so mean AUC(t) is on the same horizon set.
        train_val_block = aggregated.loc[aggregated["split"].isin(["train", "valid"])]
        landmark_horizons: dict[str, list[int]] = {}
        for endpoint, cfg in ENDPOINTS.items():
            # Endpoints whose outcome columns this build doesn't carry (nepc,
            # when the cohort was compiled without the LLM annotations) get no
            # horizon grid rather than a KeyError. A run that later asks for
            # such an endpoint fails at fit time on the missing manifest key.
            missing = [
                col
                for col in (cfg["duration_col"], cfg["event_col"])
                if col not in train_val_block.columns
            ]
            if missing:
                print(
                    f"  AUC horizons ({endpoint}): skipped, cohort has no "
                    f"{', '.join(missing)}"
                )
                continue
            grid = compute_horizon_grid(
                train_val_block,
                duration_col=cfg["duration_col"],
                event_col=cfg["event_col"],
                quantiles=auc_quantiles,
                time_unit_days=args.time_unit_days,
                admin_censor_days=args.auc_max_time_units * args.time_unit_days,
            )
            landmark_horizons[endpoint] = [int(h) for h in grid]
            print(
                f"  AUC horizons ({endpoint}): "
                + ", ".join(str(h) for h in landmark_horizons[endpoint])
                + f" {args.time_unit_days}-day units"
            )
        auc_horizons_by_landmark[str(int(landmark_day))] = landmark_horizons

        if args.build_longitudinal:
            train_mrns = set(split.index[split.eq("train")])
            merged_with_split = merged.copy()
            merged_with_split["split"] = split.reindex(merged_with_split.index)
            wide, manifest_extras, selected_labs, bounds = build_person_period_wide(
                pre_treatment_lab_df,
                merged_with_split,
                landmark_day=landmark_day,
                train_mrns=train_mrns,
                canonical_labs=None if args.no_canonical_labs else canonical_labs,
                time_unit_days=args.time_unit_days,
                min_coverage=args.long_min_coverage,
                max_labs=args.max_longitudinal_labs,
                outlier_lo=args.outlier_lo,
                outlier_hi=args.outlier_hi,
                anchor_col=anchor_col,
            )
            n_without = manifest_extras["n_patients_without_selected_labs"]
            if n_without:
                print(f"  patients with landmark-only longitudinal rows: {n_without}")

            long_csv = output_dir / longitudinal_filename(landmark_day)
            wide.to_csv(long_csv, index=False)
            long_manifest_path = output_dir / longitudinal_manifest_filename(landmark_day)
            long_event_cols, long_time_cols = longitudinal_event_columns(merged)
            longitudinal_manifest = {
                "id_col": ID_COL,
                "time_col": "TIME",
                "event_cols": long_event_cols,
                "time_to_event_cols": long_time_cols,
                "feat_cont": selected_labs + [AGE_COL],
                "feat_cat": [],
                "feat_reconstr": selected_labs,
                "time_unit_days": int(args.time_unit_days),
                "time_origin": "first_selected_pre_landmark_lab_bin",
                "prediction_landmark": f"treatment_anchor_plus_{int(landmark_day)}d",
                "landmark_days": int(landmark_day),
                "anchor_col": "none" if anchor_col is None else str(anchor_col),
                "seed": int(args.seed),
                "test_frac": float(args.test_frac),
                "val_frac": float(args.val_frac),
                "test_stratification": stratification_by_landmark[str(landmark_day)]["test"],
                "val_stratification": stratification_by_landmark[str(landmark_day)]["validation"],
                "min_coverage": float(args.long_min_coverage),
                "outlier_quantiles": [float(args.outlier_lo), float(args.outlier_hi)],
                "clip_bounds": {k: list(v) for k, v in bounds.items()},
                "canonical_labs_used": not args.no_canonical_labs,
                "split_assignments": split_assignments_filename(landmark_day),
                "split_counts": {
                    "train": int(wide.loc[wide["split"] == "train", ID_COL].nunique()),
                    "valid": int(wide.loc[wide["split"] == "valid", ID_COL].nunique()),
                    "test": int(wide.loc[wide["split"] == "test", ID_COL].nunique()),
                },
                "n_rows": int(len(wide)),
                "longitudinal_schema_version": LONGITUDINAL_SCHEMA_VERSION,
                **manifest_extras,
            }
            long_manifest_path.write_text(json.dumps(longitudinal_manifest, indent=2))
            print(
                f"  longitudinal:      rows={len(wide)} patients={wide[ID_COL].nunique()} "
                f"-> {long_csv}"
            )

            assert_split_agreement(aggregated, wide, landmark_day=landmark_day)
            long_mrns = wide[ID_COL].nunique()
            agg_mrns = len(aggregated)
            if long_mrns < agg_mrns:
                print(
                    f"  note: longitudinal dropped {agg_mrns - long_mrns} MRNs "
                    "with no usable pre-event observations"
                )

    if canonical_labs_rows:
        canonical_path = output_dir / CANONICAL_LABS_FILENAME
        pd.DataFrame(canonical_labs_rows).to_csv(canonical_path, index=False)
        print(f"\nWrote {canonical_path}")

    max_horizon = max(
        (h for endpoints in auc_horizons_by_landmark.values() for hs in endpoints.values() for h in hs),
        default=0,
    )
    # Every landmark's outcome frame comes from the same cohort build, so the
    # available causes are identical across them; read the first.
    build_event_cols, build_time_cols = longitudinal_event_columns(
        merged_by_landmark[landmark_days[0]]
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
        "max_followup_days": (
            float(args.max_followup_days) if args.max_followup_days is not None else None
        ),
        "restrict_to_mrns": str(args.restrict_to_mrns) if args.restrict_to_mrns else None,
        "exclude_mrns": (
            str(getattr(args, "exclude_mrns", None))
            if getattr(args, "exclude_mrns", None)
            else None
        ),
        "endpoint": endpoint,
        # Retained for older readers of the manifest.
        "require_nepc": endpoint == "nepc",
        "time_unit_days": int(args.time_unit_days),
        "cohort_mode": "independent_by_landmark",
        "stratification_by_landmark": stratification_by_landmark,
        "n_patients_by_landmark": {
            str(lm): int(len(merged_by_landmark[lm])) for lm in landmark_days
        },
        "n_patients_common_across_landmarks_descriptive_only": int(len(common_mrns)),
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
        "auc_horizons_by_landmark": auc_horizons_by_landmark,
        "auc_max_horizon": int(max_horizon),
        # A property of the cohort (which causes it carries), not of the
        # longitudinal build -- Cox/XGBoost ignore these. Always platinum+death;
        # NEPC too when the cohort carries it. See
        # survival_common.longitudinal_targets.LONGITUDINAL_CONFIGS.
        "event_cols": build_event_cols,
        "time_to_event_cols": build_time_cols,
        "longitudinal_schema_version": LONGITUDINAL_SCHEMA_VERSION,
        "longitudinal_landmarks_built": (
            [int(d) for d in landmark_days] if args.build_longitudinal else []
        ),
    }
    if args.build_longitudinal:
        build_manifest["longitudinal_time_unit_days"] = int(args.time_unit_days)
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
        "--exclude-mrns",
        default=None,
        help=(
            "Optional CSV with a DFCI_MRN (or --id-col) column. When set, these "
            "MRNs are REMOVED from the cohort after --restrict-to-mrns is "
            "applied, before the landmark/split build."
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
            "ultrasensitive/etc., from --data's RAW_TEST_CODE column) required "
            "before landmark building. Set to 0 to disable this downstream "
            "cohort filter."
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
        "--auc-quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_AUC_QUANTILES),
        help="Quantiles of train+valid event times used to derive the shared AUC(t) horizon grid.",
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
    parser.add_argument(
        "--max-followup-days",
        dest="max_followup_days",
        type=float,
        default=DEFAULT_MAX_FOLLOWUP_DAYS,
        help=(
            f"Administrative censoring horizon in days (default {DEFAULT_MAX_FOLLOWUP_DAYS:g}, "
            "~10 years). Events (PLATINUM/DEATH) occurring after this horizon are "
            "censored and durations clipped, for every downstream model that reads "
            "this directory's cohort. Must be >= --auc-max-time-units * "
            "--time-unit-days, since the AUC(t)/Brier timeline is already capped "
            "there; use --no-max-followup-days to disable."
        ),
    )
    parser.add_argument(
        "--no-max-followup-days",
        dest="max_followup_days",
        action="store_const",
        const=None,
        help="Disable administrative censoring (no cap on follow-up).",
    )
    parser.add_argument(
        "--endpoint",
        choices=sorted(ENDPOINTS),
        default="platinum",
        help=(
            "Endpoint whose duration defines cohort eligibility. Only that endpoint's "
            "non-null and >0 checks are applied; the other endpoint never excludes a patient."
        ),
    )
    parser.add_argument(
        "--require-nepc",
        dest="require_nepc",
        action="store_true",
        default=False,
        help=(
            "Deprecated alias for --endpoint nepc."
        ),
    )
    parser.add_argument(
        "--build-longitudinal",
        dest="build_longitudinal",
        action="store_true",
        default=True,
        help=(
            "Build the person-period (wide, binned-time) longitudinal_landmark{D}.csv "
            "consumed by Dynamic-DeepHit/SurvLatent ODE, alongside the aggregated "
            "outputs (default; on)."
        ),
    )
    parser.add_argument(
        "--no-build-longitudinal",
        dest="build_longitudinal",
        action="store_false",
        help="Skip the longitudinal person-period build.",
    )
    parser.add_argument(
        "--long-min-coverage",
        type=float,
        default=DEFAULT_LONG_MIN_COVERAGE,
        help=(
            "Coverage threshold used by the longitudinal lab selector when "
            "--no-canonical-labs is set."
        ),
    )
    parser.add_argument(
        "--no-canonical-labs",
        action="store_true",
        help=(
            "Use the longitudinal builder's own coverage+variability lab selector "
            "instead of reusing the canonical Cox lab list."
        ),
    )
    parser.add_argument(
        "--max-longitudinal-labs",
        dest="max_longitudinal_labs",
        type=int,
        default=None,
        help="Optional cap on the number of labs in the longitudinal output.",
    )
    parser.add_argument(
        "--outlier-lo",
        type=float,
        default=DEFAULT_OUTLIER_LO,
        help="Lower quantile for the longitudinal per-lab clip bounds (train-fit).",
    )
    parser.add_argument(
        "--outlier-hi",
        type=float,
        default=DEFAULT_OUTLIER_HI,
        help="Upper quantile for the longitudinal per-lab clip bounds (train-fit).",
    )
    main(parser.parse_args())
