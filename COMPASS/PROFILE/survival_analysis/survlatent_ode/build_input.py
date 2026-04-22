"""
Build person-period input for SurvLatent ODE from the pre-treatment longitudinal
lab data used by cox_aggregated.py.

Output is a wide CSV with one row per (DFCI_MRN, binned time) carrying lab values
as columns (NaN where not measured in that bin), plus patient-level static and
event columns repeated on every row. Pre-treatment history is binned into
fixed-width windows (default 7 days), a synthetic first-treatment-start landmark
row is appended, and event/censoring times are encoded after that landmark.

Splits match cox_aggregated.py (seed=42, test_frac=0.20) plus an internal
train/validation split carved out of the 80% train_val block for early stopping.
Feature selection (coverage filter) and outlier clipping (0.005-0.995 quantiles)
are derived from the training fold only; the clip bounds are then applied to
validation and test.

Outputs (alongside cox outputs at --output-dir):
  survlatent_ode_input.csv       wide person-period frame
  survlatent_ode_manifest.json   column roles + time-unit metadata for run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SURVIVAL_DIR = Path(__file__).resolve().parents[1]
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from cox_aggregated import (  # noqa: E402
    AGE_COL,
    SPLIT_ASSIGNMENTS_FILENAME,
    build_aligned_cohort,
    choose_stratification_labels,
)

DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")

DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_VAL_FRAC = 0.20
DEFAULT_TIME_UNIT_DAYS = 7
DEFAULT_MIN_COVERAGE = 0.20
DEFAULT_OUTLIER_LO = 0.005
DEFAULT_OUTLIER_HI = 0.995


def _coerce_duration(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def assemble_static(outcome_df: pd.DataFrame, cohort_index: pd.Index) -> pd.DataFrame:
    cols = [
        AGE_COL,
        "t_first_treatment",
        "t_platinum",
        "t_death",
        "PLATINUM",
        "DEATH",
        "EITHER",
    ]
    missing = [c for c in cols if c not in outcome_df.columns]
    if missing:
        raise ValueError(f"Outcome table is missing required patient-level columns: {missing}")

    static = outcome_df.loc[outcome_df.index.intersection(cohort_index), cols].copy()
    static[AGE_COL] = pd.to_numeric(static[AGE_COL], errors="coerce")
    for c in ["t_first_treatment", "t_platinum", "t_death"]:
        static[c] = _coerce_duration(static[c])
    static["PLATINUM"] = pd.to_numeric(static["PLATINUM"], errors="coerce").fillna(0).astype(int)
    static["DEATH"] = pd.to_numeric(static["DEATH"], errors="coerce").fillna(0).astype(int)
    static["EITHER"] = pd.to_numeric(static["EITHER"], errors="coerce").fillna(0).astype(int)

    valid = (
        static[AGE_COL].notna()
        & static["t_first_treatment"].notna()
        & static["t_platinum"].notna()
        & static["t_death"].notna()
        & static["t_first_treatment"].gt(0)
        & static["t_platinum"].gt(0)
        & static["t_death"].gt(0)
    )
    return static.loc[valid].copy()


def assign_survlatent_splits(
    static: pd.DataFrame,
    cox_split_assignments: pd.Series,
    *,
    seed: int,
    val_frac: float,
) -> tuple[pd.Series, str]:
    """Reuse the Cox test split and carve validation out of Cox train_val."""
    aligned = static.index.intersection(cox_split_assignments.index)
    split = pd.Series(index=aligned, dtype=object, name="split")
    cox_split = cox_split_assignments.reindex(aligned)
    split.loc[cox_split.eq("test")] = "test"

    train_val_idx = cox_split.index[cox_split.eq("train_val")]
    train_val_static = static.loc[train_val_idx]
    if train_val_static.empty:
        raise ValueError("Cox train_val split is empty after SurvLatent static filtering.")

    labels, label_name = choose_stratification_labels(train_val_static, min_count=2)
    stratify = labels if labels is not None else None
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_static)),
            test_size=val_frac,
            stratify=stratify,
            random_state=seed,
        )
    except ValueError:
        label_name = "unstratified"
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_static)),
            test_size=val_frac,
            random_state=seed,
        )

    train_val_mrns = train_val_static.index.to_numpy()
    split.loc[train_val_mrns[train_idx]] = "train"
    split.loc[train_val_mrns[val_idx]] = "valid"
    return split, label_name


def build_lab_long(df: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    required = {"DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing lab columns: {sorted(missing)}")

    labs = df[["DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab"]].copy()
    labs["LAB_NAME"] = labs["LAB_NAME"].astype(str).str.strip()
    labs["LAB_VALUE"] = pd.to_numeric(labs["LAB_VALUE"], errors="coerce")
    labs["t_lab"] = _coerce_duration(labs["t_lab"])
    labs = labs.dropna(subset=["DFCI_MRN", "LAB_NAME", "LAB_VALUE", "t_lab"])
    labs = labs.loc[labs["DFCI_MRN"].isin(static.index)]

    # Restrict to pre-treatment observations so the endpoint is predicted from
    # information available before the treatment-start landmark.
    labs = labs.merge(
        static[["t_first_treatment"]].reset_index(),
        on="DFCI_MRN",
        how="inner",
    )
    labs = labs.loc[labs["t_lab"] < labs["t_first_treatment"]].copy()
    return labs


def select_labs(
    labs: pd.DataFrame,
    *,
    train_mrns: set,
    min_coverage: float,
    max_labs: int | None,
) -> list[str]:
    train_labs = labs.loc[labs["DFCI_MRN"].isin(train_mrns)]
    n_train = len(train_mrns)
    if n_train == 0:
        raise ValueError("Training set is empty; cannot select labs.")
    coverage = train_labs.groupby("LAB_NAME")["DFCI_MRN"].nunique() / n_train
    # require at least two distinct values in training
    variability = train_labs.groupby("LAB_NAME")["LAB_VALUE"].nunique()
    eligible = coverage.index[(coverage >= min_coverage) & (variability.reindex(coverage.index) > 1)]
    eligible = list(eligible)
    if not eligible:
        raise ValueError(
            f"No labs passed coverage >= {min_coverage} on training set (n={n_train})."
        )
    ranked = coverage.loc[eligible].sort_values(ascending=False)
    if max_labs is not None and len(ranked) > max_labs:
        ranked = ranked.head(max_labs)
    return ranked.index.tolist()


def clip_bounds(
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


def apply_clip(agg: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    if not bounds:
        return agg
    vals = agg["LAB_VALUE"].to_numpy(dtype=float)
    lo_series = agg["LAB_NAME"].map({k: v[0] for k, v in bounds.items()}).to_numpy(dtype=float)
    hi_series = agg["LAB_NAME"].map({k: v[1] for k, v in bounds.items()}).to_numpy(dtype=float)
    has_bound = np.isfinite(lo_series) & np.isfinite(hi_series)
    clipped = vals.copy()
    clipped[has_bound] = np.clip(vals[has_bound], lo_series[has_bound], hi_series[has_bound])
    agg = agg.copy()
    agg["LAB_VALUE"] = clipped
    return agg


def main(args: argparse.Namespace) -> None:
    print(f"Loading longitudinal data from {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False)

    print("Rebuilding Cox landmark cohort so the held-out test split is identical ...")
    outcome_df, _, merged, _, _, cox_split_assignments, cox_split_stratification = build_aligned_cohort(
        df,
        seed=args.seed,
        test_frac=args.test_frac,
    )

    static = assemble_static(outcome_df, merged.index)
    print(f"Patients with valid landmarked static info: {len(static)}")

    surv_split, val_stratification = assign_survlatent_splits(
        static,
        cox_split_assignments,
        seed=args.seed,
        val_frac=args.val_frac,
    )
    static = static.loc[surv_split.index].copy()
    static["split"] = surv_split
    train_mrns = set(static.index[static["split"] == "train"])
    val_mrns = set(static.index[static["split"] == "valid"])
    test_mrns = set(static.index[static["split"] == "test"])
    print(f"Splits: train={len(train_mrns)} valid={len(val_mrns)} test={len(test_mrns)}")
    print(f"Cox test split stratification: {cox_split_stratification}")
    print(f"SurvLatent internal validation stratification: {val_stratification}")

    labs = build_lab_long(df, static)
    print(f"Pre-treatment lab rows: {len(labs)}")

    selected_labs = select_labs(
        labs,
        train_mrns=train_mrns,
        min_coverage=args.min_coverage,
        max_labs=args.max_labs,
    )
    print(f"Selected labs ({len(selected_labs)}): {selected_labs}")
    selected_lab_rows = labs.loc[labs["LAB_NAME"].isin(selected_labs)].copy()

    # Bin pre-treatment history relative to first treatment start. The treatment
    # landmark itself is added below as the final observed row for each patient,
    # so SurvLatent's survival horizon starts at treatment start.
    selected_lab_rows["REL_BIN"] = np.floor(
        (
            selected_lab_rows["t_lab"].to_numpy(dtype=float)
            - selected_lab_rows["t_first_treatment"].to_numpy(dtype=float)
        )
        / float(args.time_unit_days)
    ).astype(int)
    landmark_time = pd.Series(0, index=static.index, dtype=int, name="landmark_time")
    selected_lab_landmarks = (
        -selected_lab_rows.groupby("DFCI_MRN")["REL_BIN"].min()
    ).astype(int)
    landmark_time.loc[selected_lab_landmarks.index] = selected_lab_landmarks
    n_without_selected_labs = int((landmark_time == 0).sum())
    if n_without_selected_labs:
        print(
            f"Patients represented by landmark-only rows because selected labs are absent: "
            f"{n_without_selected_labs}"
        )

    selected_lab_rows = selected_lab_rows.merge(
        landmark_time.reset_index(),
        on="DFCI_MRN",
        how="inner",
    )
    selected_lab_rows["TIME"] = selected_lab_rows["REL_BIN"] + selected_lab_rows["landmark_time"]

    if selected_lab_rows.empty:
        raise ValueError("No selected pre-treatment lab rows remain after filtering.")

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

    bounds = clip_bounds(
        agg,
        train_mrns=train_mrns,
        labs=selected_labs,
        q_lo=args.outlier_lo,
        q_hi=args.outlier_hi,
    )
    agg = apply_clip(agg, bounds)
    print(f"Outlier clipping bounds fit on train for {len(bounds)}/{len(selected_labs)} labs")

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

    # Attach per-patient static and event info; SurvLatent ODE expects these
    # repeated on every row for a patient.
    static_cols = [
        AGE_COL,
        "t_platinum",
        "t_death",
        "PLATINUM",
        "DEATH",
        "split",
    ]
    wide = wide.merge(static[static_cols], left_on="DFCI_MRN", right_index=True, how="inner")

    # Convert post-treatment event/censoring durations into the same integer time
    # unit as TIME. Ceil keeps any positive post-landmark duration strictly after
    # the synthetic treatment-start row.
    t_platinum_after_landmark = np.ceil(
        wide["t_platinum"].to_numpy(dtype=float) / float(args.time_unit_days)
    )
    t_death_after_landmark = np.ceil(
        wide["t_death"].to_numpy(dtype=float) / float(args.time_unit_days)
    )
    wide["t_platinum"] = wide["landmark_time"].to_numpy(dtype=float) + t_platinum_after_landmark
    wide["t_death"] = wide["landmark_time"].to_numpy(dtype=float) + t_death_after_landmark

    # Drop rows whose observation time equals or exceeds the patient's event time
    # (the model excludes t_event == t_last_observation samples; we enforce here).
    event_time = wide[["t_platinum", "t_death"]].min(axis=1).to_numpy(dtype=float)
    wide = wide.loc[wide["TIME"].to_numpy(dtype=float) < event_time].copy()

    # Drop patients who lose all observations to the filter above.
    counts = wide.groupby("DFCI_MRN").size()
    surviving = counts[counts > 0].index
    wide = wide.loc[wide["DFCI_MRN"].isin(surviving)].copy()
    dropped = len(static) - wide["DFCI_MRN"].nunique()
    if dropped:
        print(f"Dropped {dropped} patients with no usable pre-event observations.")
    max_landmark_time = int(wide["landmark_time"].max()) if not wide.empty else 0

    column_order = (
        ["DFCI_MRN", "TIME"]
        + selected_labs
        + [AGE_COL, "PLATINUM", "DEATH", "t_platinum", "t_death", "split"]
    )
    wide = wide.sort_values(["DFCI_MRN", "TIME"])[column_order]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "survlatent_ode_input.csv"
    out_manifest = output_dir / "survlatent_ode_manifest.json"
    split_path = output_dir / SPLIT_ASSIGNMENTS_FILENAME

    wide.to_csv(out_csv, index=False)
    cox_split_assignments.rename_axis("DFCI_MRN").reset_index().to_csv(split_path, index=False)
    manifest = {
        "id_col": "DFCI_MRN",
        "time_col": "TIME",
        "event_cols": ["PLATINUM", "DEATH"],
        "time_to_event_cols": ["t_platinum", "t_death"],
        "feat_cont": selected_labs + [AGE_COL],
        "feat_cat": [],
        "feat_reconstr": selected_labs,
        "time_unit_days": args.time_unit_days,
        "time_origin": "first_selected_pre_treatment_lab_bin",
        "prediction_landmark": "first_treatment_start",
        "event_times": "absolute_bins_from_time_origin",
        "max_landmark_time": max_landmark_time,
        "n_patients_without_selected_labs": n_without_selected_labs,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "cox_split_stratification": cox_split_stratification,
        "survlatent_validation_stratification": val_stratification,
        "min_coverage": args.min_coverage,
        "outlier_quantiles": [args.outlier_lo, args.outlier_hi],
        "clip_bounds": {k: list(v) for k, v in bounds.items()},
        "cox_split_assignments": str(split_path),
        "split_counts": {
            "train": int(wide.loc[wide["split"] == "train", "DFCI_MRN"].nunique()),
            "valid": int(wide.loc[wide["split"] == "valid", "DFCI_MRN"].nunique()),
            "test": int(wide.loc[wide["split"] == "test", "DFCI_MRN"].nunique()),
        },
        "n_rows": int(len(wide)),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved:\n  {out_csv}\n  {out_manifest}\n  {split_path}")
    print(f"Rows: {len(wide)}  Patients: {wide['DFCI_MRN'].nunique()}")
    print(f"Time unit: {args.time_unit_days} day(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Fraction of the train+val block held out as an internal validation set.",
    )
    parser.add_argument(
        "--time-unit-days",
        type=int,
        default=DEFAULT_TIME_UNIT_DAYS,
        help="Width of the time bin used to aggregate lab observations and integerize event times.",
    )
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument(
        "--max-labs",
        type=int,
        default=None,
        help="Optional cap on number of labs kept, ranked by training-set coverage.",
    )
    parser.add_argument("--outlier-lo", type=float, default=DEFAULT_OUTLIER_LO)
    parser.add_argument("--outlier-hi", type=float, default=DEFAULT_OUTLIER_HI)
    main(parser.parse_args())
