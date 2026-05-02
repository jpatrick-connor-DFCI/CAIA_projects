"""
Single source of truth for survival-analysis prediction inputs.

For each requested landmark, this script:
  1. Loads + v3-filters the raw longitudinal lab CSV.
  2. Builds the landmarked patient cohort and intersects MRNs across all
     requested landmarks so every downstream model sees the same patients.
  3. Derives a 3-way train/valid/test split ONCE on the intersection cohort
     at the base (smallest) landmark, then reuses it for every landmark.
     Cox / XGBoost union train+valid into their train_val block; DeepHit
     uses train and valid directly for early stopping.
  4. Writes per-landmark:
       aggregated_landmark{D}.csv                 (1 row per MRN; features + outcomes + split)
       longitudinal_landmark{D}.csv               (wide person-period frame + split)
       longitudinal_landmark{D}_manifest.json
       pre_treatment_lab_long_landmark{D}.csv     (long-format pre-landmark labs
                                                   for per-fold canonical-lab selection)
  5. Writes shared:
       split_assignments.csv                      DFCI_MRN, split (train/valid/test)
       landmark_mrn_availability.csv              per-landmark eligibility + split
       canonical_labs_train_val.csv               landmark_days, lab_name (train+valid)
  6. Asserts per-MRN split agrees between the aggregated and longitudinal outputs.
     Longitudinal MRNs may be a strict subset because the person-period filter
     drops patients whose only observation falls at/after their event time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from cox_aggregated import (  # noqa: E402
    AGE_COL,
    DATA_PATH,
    DEFAULT_LANDMARK_DAYS,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_MIN_PATIENT_COVERAGE,
    DEFAULT_V3_LABELS_PATH,
    ENDPOINTS,
    RESULTS,
    build_landmark_availability_table,
    build_landmark_merged,
    build_pre_treatment_lab_long,
    choose_stratification_labels,
    load_v3_label_mrns,
    normalize_landmark_days,
)
from helper import (  # noqa: E402
    DEFAULT_AUC_QUANTILES,
    compute_horizon_grid,
    select_canonical_labs,
)

DEFAULT_OUTPUT_SUBDIR = "prediction_inputs"
DEFAULT_VAL_FRAC = 0.20
DEFAULT_TIME_UNIT_DAYS = 7
DEFAULT_LONG_MIN_COVERAGE = 0.20
DEFAULT_OUTLIER_LO = 0.005
DEFAULT_OUTLIER_HI = 0.995

SPLIT_ASSIGNMENTS_FILENAME = "split_assignments.csv"
LANDMARK_AVAILABILITY_FILENAME = "landmark_mrn_availability.csv"
CANONICAL_LABS_FILENAME = "canonical_labs_train_val.csv"
BUILD_MANIFEST_FILENAME = "build_manifest.json"


def aggregated_filename(landmark_day: int) -> str:
    return f"aggregated_landmark{int(landmark_day)}.csv"


def longitudinal_filename(landmark_day: int) -> str:
    return f"longitudinal_landmark{int(landmark_day)}.csv"


def longitudinal_manifest_filename(landmark_day: int) -> str:
    return f"longitudinal_landmark{int(landmark_day)}_manifest.json"


def pre_treatment_lab_filename(landmark_day: int) -> str:
    return f"pre_treatment_lab_long_landmark{int(landmark_day)}.csv"


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


AGGREGATED_DROP_COLUMNS = (
    # Raw date strings — duplicate of the rebased durations; not consumed downstream.
    "FIRST_RECORD_DATE",
    "DIAGNOSIS_DATE",
    "FIRST_TREATMENT_DATE",
    "LAST_CONTACT_DATE",
    "PLATINUM_DATE",
    # Pre-rebase duration duplicates kept for debugging by make_outcome_df.
    "t_platinum_from_first_record",
    "t_last_contact_from_first_record",
    "t_death_from_first_record",
    # Always 1 after make_outcome_df's cohort filter.
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


def assemble_static_for_longitudinal(merged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        AGE_COL,
        "t_first_treatment",
        "t_platinum",
        "t_death",
        "PLATINUM",
        "DEATH",
        "EITHER",
        "split",
    ]
    missing = [c for c in cols if c not in merged.columns]
    if missing:
        raise ValueError(f"Aggregated table is missing columns required by the longitudinal builder: {missing}")
    static = merged[cols].copy()
    for c in ["t_first_treatment", "t_platinum", "t_death"]:
        static[c] = pd.to_numeric(static[c], errors="coerce").astype(float)
    static[AGE_COL] = pd.to_numeric(static[AGE_COL], errors="coerce")
    static["PLATINUM"] = pd.to_numeric(static["PLATINUM"], errors="coerce").fillna(0).astype(int)
    static["DEATH"] = pd.to_numeric(static["DEATH"], errors="coerce").fillna(0).astype(int)
    static["EITHER"] = pd.to_numeric(static["EITHER"], errors="coerce").fillna(0).astype(int)
    valid = (
        static[AGE_COL].notna()
        & static["t_first_treatment"].notna()
        & static["t_platinum"].notna()
        & static["t_death"].notna()
        & static["t_first_treatment"].ge(0)
        & static["t_platinum"].gt(0)
        & static["t_death"].gt(0)
    )
    return static.loc[valid].copy()


def build_longitudinal_lab_long(
    df: pd.DataFrame,
    static: pd.DataFrame,
    *,
    landmark_offset_days: int,
) -> pd.DataFrame:
    required = {"DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing lab columns: {sorted(missing)}")
    labs = df[["DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab"]].copy()
    labs["LAB_NAME"] = labs["LAB_NAME"].astype(str).str.strip()
    labs["LAB_VALUE"] = pd.to_numeric(labs["LAB_VALUE"], errors="coerce")
    labs["t_lab"] = pd.to_numeric(labs["t_lab"], errors="coerce").astype(float)
    labs = labs.dropna(subset=["DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab"])
    labs = labs.loc[labs["DFCI_MRN"].isin(static.index)]
    labs = labs.merge(
        static[["t_first_treatment"]].reset_index(),
        on="DFCI_MRN",
        how="inner",
    )
    landmark_t = labs["t_first_treatment"] + float(landmark_offset_days)
    return labs.loc[labs["t_lab"] < landmark_t].copy()


def select_longitudinal_labs(
    labs: pd.DataFrame,
    *,
    train_mrns: set,
    min_coverage: float,
    max_labs: int | None,
    canonical_labs: list[str] | None,
) -> list[str]:
    if canonical_labs is not None:
        present = set(labs["LAB_NAME"].astype(str))
        selected = [lab for lab in canonical_labs if lab in present]
        if not selected:
            raise ValueError("No canonical labs are present in the longitudinal lab table.")
        if max_labs is not None:
            selected = selected[:max_labs]
        return selected
    if not train_mrns:
        raise ValueError("Training set is empty; cannot select labs.")
    train_labs = labs.loc[labs["DFCI_MRN"].isin(train_mrns)]
    coverage = train_labs.groupby("LAB_NAME")["DFCI_MRN"].nunique() / len(train_mrns)
    variability = train_labs.groupby("LAB_NAME")["LAB_VALUE"].nunique()
    eligible = coverage.index[(coverage >= min_coverage) & (variability.reindex(coverage.index) > 1)]
    if not len(eligible):
        raise ValueError(f"No labs passed coverage >= {min_coverage} on training set.")
    ranked = coverage.loc[eligible].sort_values(ascending=False)
    if max_labs is not None and len(ranked) > max_labs:
        ranked = ranked.head(max_labs)
    return ranked.index.tolist()


def fit_clip_bounds(
    agg: pd.DataFrame,
    *,
    train_mrns: set,
    labs: list[str],
    q_lo: float,
    q_hi: float,
) -> dict[str, tuple[float, float]]:
    train_agg = agg.loc[agg["DFCI_MRN"].isin(train_mrns)]
    bounds: dict[str, tuple[float, float]] = {}
    for lab in labs:
        vals = train_agg.loc[train_agg["LAB_NAME"] == lab, "LAB_VALUE"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue
        lo = float(np.quantile(vals, q_lo))
        hi = float(np.quantile(vals, q_hi))
        if hi <= lo:
            continue
        bounds[lab] = (lo, hi)
    return bounds


def apply_clip_bounds(agg: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    if not bounds:
        return agg
    vals = agg["LAB_VALUE"].to_numpy(dtype=float)
    lo_series = agg["LAB_NAME"].map({k: v[0] for k, v in bounds.items()}).to_numpy(dtype=float)
    hi_series = agg["LAB_NAME"].map({k: v[1] for k, v in bounds.items()}).to_numpy(dtype=float)
    has_bound = np.isfinite(lo_series) & np.isfinite(hi_series)
    clipped = vals.copy()
    clipped[has_bound] = np.clip(vals[has_bound], lo_series[has_bound], hi_series[has_bound])
    out = agg.copy()
    out["LAB_VALUE"] = clipped
    return out


def build_longitudinal_wide(
    df: pd.DataFrame,
    static: pd.DataFrame,
    *,
    landmark_day: int,
    train_mrns: set,
    canonical_labs: list[str] | None,
    time_unit_days: int,
    min_coverage: float,
    max_labs: int | None,
    outlier_lo: float,
    outlier_hi: float,
) -> tuple[pd.DataFrame, dict, list[str], dict[str, tuple[float, float]], int]:
    labs = build_longitudinal_lab_long(df, static, landmark_offset_days=landmark_day)
    selected_labs = select_longitudinal_labs(
        labs,
        train_mrns=train_mrns,
        min_coverage=min_coverage,
        max_labs=max_labs,
        canonical_labs=canonical_labs,
    )
    selected_lab_rows = labs.loc[labs["LAB_NAME"].isin(selected_labs)].copy()

    selected_lab_rows["REL_BIN"] = np.floor(
        (
            selected_lab_rows["t_lab"].to_numpy(dtype=float)
            - (
                selected_lab_rows["t_first_treatment"].to_numpy(dtype=float)
                + float(landmark_day)
            )
        )
        / float(time_unit_days)
    ).astype(int)

    landmark_time = pd.Series(0, index=static.index, dtype=int, name="landmark_time")
    selected_lab_landmarks = (
        -selected_lab_rows.groupby("DFCI_MRN")["REL_BIN"].min()
    ).astype(int)
    landmark_time.loc[selected_lab_landmarks.index] = selected_lab_landmarks
    n_without_selected_labs = int((landmark_time == 0).sum())

    selected_lab_rows = selected_lab_rows.merge(
        landmark_time.reset_index(),
        on="DFCI_MRN",
        how="inner",
    )
    selected_lab_rows["TIME"] = selected_lab_rows["REL_BIN"] + selected_lab_rows["landmark_time"]

    landmark_time = (
        selected_lab_rows.groupby("DFCI_MRN")["landmark_time"].first()
        .reindex(static.index)
        .fillna(0)
        .astype(int)
        .rename("landmark_time")
    )

    agg = (
        selected_lab_rows.groupby(["DFCI_MRN", "TIME", "LAB_NAME"], sort=False)["LAB_VALUE"]
        .mean()
        .reset_index()
    )
    bounds = fit_clip_bounds(
        agg,
        train_mrns=train_mrns,
        labs=selected_labs,
        q_lo=outlier_lo,
        q_hi=outlier_hi,
    )
    agg = apply_clip_bounds(agg, bounds)

    wide = agg.pivot_table(
        index=["DFCI_MRN", "TIME"],
        columns="LAB_NAME",
        values="LAB_VALUE",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    for lab in selected_labs:
        if lab not in wide.columns:
            wide[lab] = np.nan
    wide = wide.merge(landmark_time.reset_index(), on="DFCI_MRN", how="inner")

    landmark_rows = pd.DataFrame(
        {
            "DFCI_MRN": landmark_time.index,
            "TIME": landmark_time.to_numpy(dtype=int),
            "landmark_time": landmark_time.to_numpy(dtype=int),
        }
    )
    for lab in selected_labs:
        landmark_rows[lab] = np.nan
    wide = pd.concat([wide, landmark_rows], ignore_index=True, sort=False)

    static_cols = [AGE_COL, "t_platinum", "t_death", "PLATINUM", "DEATH", "split"]
    wide = wide.merge(static[static_cols], left_on="DFCI_MRN", right_index=True, how="inner")

    t_platinum_after_landmark = np.ceil(
        wide["t_platinum"].to_numpy(dtype=float) / float(time_unit_days)
    )
    t_death_after_landmark = np.ceil(
        wide["t_death"].to_numpy(dtype=float) / float(time_unit_days)
    )
    wide["t_platinum"] = wide["landmark_time"].to_numpy(dtype=float) + t_platinum_after_landmark
    wide["t_death"] = wide["landmark_time"].to_numpy(dtype=float) + t_death_after_landmark

    event_time = wide[["t_platinum", "t_death"]].min(axis=1).to_numpy(dtype=float)
    wide = wide.loc[wide["TIME"].to_numpy(dtype=float) < event_time].copy()

    counts = wide.groupby("DFCI_MRN").size()
    surviving = counts[counts > 0].index
    wide = wide.loc[wide["DFCI_MRN"].isin(surviving)].copy()

    max_landmark_time = (
        int(wide.loc[wide["split"] == "train", "landmark_time"].max())
        if "landmark_time" in wide.columns and not wide.empty
        else 0
    )
    column_order = (
        ["DFCI_MRN", "TIME"]
        + selected_labs
        + [AGE_COL, "PLATINUM", "DEATH", "t_platinum", "t_death", "split"]
    )
    wide = wide.sort_values(["DFCI_MRN", "TIME"])[column_order]

    manifest_extras = {
        "max_landmark_time": max_landmark_time,
        "n_patients_without_selected_labs": n_without_selected_labs,
    }
    return wide, manifest_extras, selected_labs, bounds, n_without_selected_labs


def assert_split_agreement(
    aggregated: pd.DataFrame,
    longitudinal: pd.DataFrame,
    *,
    landmark_day: int,
) -> None:
    long_split = (
        longitudinal.groupby("DFCI_MRN")["split"]
        .agg(lambda s: s.iloc[0])
        .rename("split_long")
    )
    nunique = longitudinal.groupby("DFCI_MRN")["split"].nunique()
    inconsistent = nunique[nunique > 1]
    if not inconsistent.empty:
        raise AssertionError(
            f"Landmark +{landmark_day}d: {len(inconsistent)} MRNs have multiple split labels in the "
            f"longitudinal output (first: {inconsistent.index[0]})."
        )

    agg_split = aggregated["split"].rename("split_agg")
    extra_in_long = long_split.index.difference(agg_split.index)
    if len(extra_in_long):
        raise AssertionError(
            f"Landmark +{landmark_day}d: {len(extra_in_long)} MRNs appear in longitudinal but not aggregated "
            f"(first: {extra_in_long[0]})."
        )

    overlap = long_split.index.intersection(agg_split.index)
    mismatched = (agg_split.loc[overlap] != long_split.loc[overlap]).sum()
    if mismatched:
        raise AssertionError(
            f"Landmark +{landmark_day}d: {mismatched} MRNs disagree on split between aggregated and longitudinal."
        )


def main(args: argparse.Namespace) -> None:
    landmark_days = normalize_landmark_days(args.landmark_days)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False)

    v3_mrns = load_v3_label_mrns(Path(args.v3_labels_path))
    n_before = df["DFCI_MRN"].nunique()
    df["DFCI_MRN"] = pd.to_numeric(df["DFCI_MRN"], errors="coerce")
    df = df.loc[df["DFCI_MRN"].notna()].copy()
    df["DFCI_MRN"] = df["DFCI_MRN"].astype(int)
    df = df.loc[df["DFCI_MRN"].isin(v3_mrns)].copy()
    n_after = df["DFCI_MRN"].nunique()
    print(
        f"Restricted to MRNs with v3 LLM labels ({args.v3_labels_path}): "
        f"{n_after}/{n_before} patients retained"
    )
    if n_after == 0:
        raise ValueError("No MRNs remain after intersecting with v3 LLM label cohort.")

    merged_by_landmark: dict[int, pd.DataFrame] = {}
    for landmark_day in landmark_days:
        print(f"\n##### COHORT BUILD: LANDMARK +{landmark_day} DAYS #####")
        _, _, merged = build_landmark_merged(df, landmark_offset_days=landmark_day)
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
    split.rename_axis("DFCI_MRN").reset_index().to_csv(split_path, index=False)
    print(f"Wrote {split_path}")

    availability["included_all_landmarks"] = availability["DFCI_MRN"].isin(common_mrns)
    availability["split"] = availability["DFCI_MRN"].map(split)
    availability_path = output_dir / LANDMARK_AVAILABILITY_FILENAME
    availability.to_csv(availability_path, index=False)
    print(f"Wrote {availability_path}")

    train_mrns = set(split.index[split.eq("train")])
    train_val_mrns = set(split.index[split.isin(["train", "valid"])])

    canonical_labs_rows: list[dict] = []
    auc_horizons_by_landmark: dict[str, dict[str, list[int]]] = {}
    auc_quantiles = tuple(args.auc_quantiles)

    for landmark_day in landmark_days:
        print(f"\n##### LANDMARK +{landmark_day}d: BUILD INPUTS #####")
        merged = merged_by_landmark[landmark_day].loc[common_mrns].copy()
        aggregated = build_aggregated_table(merged, split=split)

        agg_path = output_dir / aggregated_filename(landmark_day)
        aggregated.rename_axis("DFCI_MRN").reset_index().to_csv(agg_path, index=False)
        print(f"  aggregated:        {len(aggregated)} patients -> {agg_path}")

        pre_treatment_lab_df = build_pre_treatment_lab_long(
            df,
            cohort_index=aggregated.index,
            landmark_offset_days=landmark_day,
        )
        pre_treatment_path = output_dir / pre_treatment_lab_filename(landmark_day)
        pre_treatment_lab_df.to_csv(pre_treatment_path, index=False)
        print(f"  pre-landmark labs: {len(pre_treatment_lab_df)} rows -> {pre_treatment_path}")

        canonical_labs = select_canonical_labs(
            pre_treatment_lab_df,
            mrns=pd.Index(sorted(train_val_mrns)),
            min_coverage=args.min_patient_coverage,
        )
        for lab in canonical_labs:
            canonical_labs_rows.append({"landmark_days": landmark_day, "lab_name": lab})
        print(f"  canonical labs (train+valid): {len(canonical_labs)}")

        # Per-endpoint AUC horizon grid, derived ONCE from train+valid event
        # times. All three downstream models read these from the manifest so
        # mean AUC(t) is on the same horizon set for the same test cohort.
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

        static = assemble_static_for_longitudinal(aggregated)
        wide, manifest_extras, selected_labs, bounds, n_without = build_longitudinal_wide(
            df,
            static,
            landmark_day=landmark_day,
            train_mrns=train_mrns,
            canonical_labs=None if args.no_canonical_labs else canonical_labs,
            time_unit_days=args.time_unit_days,
            min_coverage=args.long_min_coverage,
            max_labs=args.max_labs,
            outlier_lo=args.outlier_lo,
            outlier_hi=args.outlier_hi,
        )
        if n_without:
            print(f"  patients with landmark-only longitudinal rows: {n_without}")

        long_csv = output_dir / longitudinal_filename(landmark_day)
        wide.to_csv(long_csv, index=False)
        long_manifest = output_dir / longitudinal_manifest_filename(landmark_day)
        manifest = {
            "id_col": "DFCI_MRN",
            "time_col": "TIME",
            "event_cols": ["PLATINUM", "DEATH"],
            "time_to_event_cols": ["t_platinum", "t_death"],
            "feat_cont": selected_labs + [AGE_COL],
            "feat_cat": [],
            "feat_reconstr": selected_labs,
            "time_unit_days": args.time_unit_days,
            "time_origin": "first_selected_pre_landmark_lab_bin",
            "prediction_landmark": f"first_treatment_start_plus_{int(landmark_day)}d",
            "landmark_days": int(landmark_day),
            "event_times": "absolute_bins_from_time_origin",
            "seed": args.seed,
            "test_frac": args.test_frac,
            "val_frac": args.val_frac,
            "test_stratification": test_stratification,
            "val_stratification": val_stratification,
            "min_coverage": args.long_min_coverage,
            "outlier_quantiles": [args.outlier_lo, args.outlier_hi],
            "clip_bounds": {k: list(v) for k, v in bounds.items()},
            "canonical_labs_used": not args.no_canonical_labs,
            "split_assignments": str(split_path.name),
            "split_counts": {
                "train": int(wide.loc[wide["split"] == "train", "DFCI_MRN"].nunique()),
                "valid": int(wide.loc[wide["split"] == "valid", "DFCI_MRN"].nunique()),
                "test": int(wide.loc[wide["split"] == "test", "DFCI_MRN"].nunique()),
            },
            "n_rows": int(len(wide)),
            **manifest_extras,
        }
        long_manifest.write_text(json.dumps(manifest, indent=2))
        print(
            f"  longitudinal:      rows={len(wide)} patients={wide['DFCI_MRN'].nunique()} "
            f"-> {long_csv}"
        )

        assert_split_agreement(aggregated, wide, landmark_day=landmark_day)
        long_mrns = wide["DFCI_MRN"].nunique()
        agg_mrns = len(aggregated)
        if long_mrns < agg_mrns:
            print(
                f"  note: longitudinal dropped {agg_mrns - long_mrns} MRNs "
                f"with no usable pre-event observations"
            )

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
        "v3_labels_path": str(args.v3_labels_path),
        "landmark_days": [int(d) for d in landmark_days],
        "seed": int(args.seed),
        "test_frac": float(args.test_frac),
        "val_frac": float(args.val_frac),
        "min_patient_coverage": float(args.min_patient_coverage),
        "time_unit_days": int(args.time_unit_days),
        "long_min_coverage": float(args.long_min_coverage),
        "no_canonical_labs": bool(args.no_canonical_labs),
        "max_labs": args.max_labs,
        "outlier_quantiles": [float(args.outlier_lo), float(args.outlier_hi)],
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
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument(
        "--v3-labels-path",
        type=str,
        default=str(DEFAULT_V3_LABELS_PATH),
        help="TSV of v3 LLM labels (DFCI_MRN column). Cohort is restricted to its MRN set.",
    )
    parser.add_argument(
        "--landmark-days",
        nargs="+",
        type=int,
        default=DEFAULT_LANDMARK_DAYS,
        help="Landmark offsets in days relative to first treatment.",
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
        help="Width of the time bin used for the longitudinal person-period frame.",
    )
    parser.add_argument(
        "--long-min-coverage",
        type=float,
        default=DEFAULT_LONG_MIN_COVERAGE,
        help="Coverage threshold used by the longitudinal lab selector when --no-canonical-labs is set.",
    )
    parser.add_argument(
        "--no-canonical-labs",
        action="store_true",
        help="Use the longitudinal builder's own coverage filter instead of the canonical Cox lab list.",
    )
    parser.add_argument(
        "--max-labs",
        type=int,
        default=None,
        help="Optional cap on the number of labs in the longitudinal output.",
    )
    parser.add_argument("--outlier-lo", type=float, default=DEFAULT_OUTLIER_LO)
    parser.add_argument("--outlier-hi", type=float, default=DEFAULT_OUTLIER_HI)
    parser.add_argument(
        "--auc-quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_AUC_QUANTILES),
        help="Quantiles of train+valid event times used to derive the shared AUC(t) horizon grid.",
    )
    main(parser.parse_args())
