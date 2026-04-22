"""
Build person-period input for SurvLatent ODE from the pre-treatment longitudinal
lab data used by cox_aggregated.py.

Output is a wide CSV with one row per (DFCI_MRN, binned time) carrying lab values
as columns (NaN where not measured in that bin), plus patient-level static and
event columns repeated on every row. Time is binned into fixed-width windows
(default 7 days) and values are aggregated (mean within bin) to remove duplicate
time points for the same patient.

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
from pathlib import Path

import numpy as np
import pandas as pd

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


def assemble_static(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "DFCI_MRN",
        "AGE_AT_TREATMENTSTART",
        "t_first_treatment",
        "t_platinum",
        "t_death",
        "PLATINUM",
        "DEATH",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required patient-level columns: {missing}")

    static = df[cols].drop_duplicates("DFCI_MRN").set_index("DFCI_MRN").copy()
    static["AGE_AT_TREATMENTSTART"] = pd.to_numeric(static["AGE_AT_TREATMENTSTART"], errors="coerce")
    for c in ["t_first_treatment", "t_platinum", "t_death"]:
        static[c] = _coerce_duration(static[c])
    static["PLATINUM"] = pd.to_numeric(static["PLATINUM"], errors="coerce").fillna(0).astype(int)
    static["DEATH"] = pd.to_numeric(static["DEATH"], errors="coerce").fillna(0).astype(int)

    valid = (
        static["AGE_AT_TREATMENTSTART"].notna()
        & static["t_first_treatment"].notna()
        & static["t_platinum"].notna()
        & static["t_death"].notna()
        & static["t_first_treatment"].gt(0)
        & static["t_platinum"].ge(0)
        & static["t_death"].ge(0)
    )
    return static.loc[valid].copy()


def assign_splits(
    mrns: list[str], *, seed: int, test_frac: float, val_frac: float
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    mrns = list(mrns)
    rng.shuffle(mrns)
    n_test = int(len(mrns) * test_frac)
    test_mrns = set(mrns[:n_test])
    remaining = mrns[n_test:]
    n_val = int(len(remaining) * val_frac)
    val_mrns = set(remaining[:n_val])
    train_mrns = set(remaining[n_val:])
    split = {}
    for m in train_mrns:
        split[m] = "train"
    for m in val_mrns:
        split[m] = "valid"
    for m in test_mrns:
        split[m] = "test"
    return split


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

    # Restrict to pre-treatment observations so the platinum endpoint is not trivially leaked.
    labs = labs.merge(
        static[["t_first_treatment"]].reset_index(),
        on="DFCI_MRN",
        how="inner",
    )
    labs = labs.loc[labs["t_lab"] < labs["t_first_treatment"]].drop(columns="t_first_treatment")
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

    static = assemble_static(df)
    print(f"Patients with valid static info: {len(static)}")

    split = assign_splits(
        static.index.tolist(),
        seed=args.seed,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
    )
    static["split"] = static.index.map(split)
    train_mrns = set(static.index[static["split"] == "train"])
    val_mrns = set(static.index[static["split"] == "valid"])
    test_mrns = set(static.index[static["split"] == "test"])
    print(f"Splits: train={len(train_mrns)} valid={len(val_mrns)} test={len(test_mrns)}")

    labs = build_lab_long(df, static)
    print(f"Pre-treatment lab rows: {len(labs)}")

    selected_labs = select_labs(
        labs,
        train_mrns=train_mrns,
        min_coverage=args.min_coverage,
        max_labs=args.max_labs,
    )
    print(f"Selected labs ({len(selected_labs)}): {selected_labs}")
    labs = labs.loc[labs["LAB_NAME"].isin(selected_labs)].copy()

    # Bin time into fixed-width units and aggregate (mean) within each bin.
    labs["TIME"] = (labs["t_lab"] // args.time_unit_days).astype(int)
    agg = (
        labs.groupby(["DFCI_MRN", "TIME", "LAB_NAME"], sort=False)["LAB_VALUE"]
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

    # Attach per-patient static and event info; SurvLatent ODE expects these
    # repeated on every row for a patient.
    static_cols = [
        "AGE_AT_TREATMENTSTART",
        "t_platinum",
        "t_death",
        "PLATINUM",
        "DEATH",
        "split",
    ]
    wide = wide.merge(static[static_cols], left_on="DFCI_MRN", right_index=True, how="inner")

    # Convert event times into the same integer time unit as TIME.
    wide["t_platinum"] = np.round(wide["t_platinum"].to_numpy(dtype=float) / args.time_unit_days)
    wide["t_death"] = np.round(wide["t_death"].to_numpy(dtype=float) / args.time_unit_days)

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

    column_order = (
        ["DFCI_MRN", "TIME"]
        + selected_labs
        + ["AGE_AT_TREATMENTSTART", "PLATINUM", "DEATH", "t_platinum", "t_death", "split"]
    )
    wide = wide.sort_values(["DFCI_MRN", "TIME"])[column_order]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "survlatent_ode_input.csv"
    out_manifest = output_dir / "survlatent_ode_manifest.json"

    wide.to_csv(out_csv, index=False)
    manifest = {
        "id_col": "DFCI_MRN",
        "time_col": "TIME",
        "event_cols": ["PLATINUM", "DEATH"],
        "time_to_event_cols": ["t_platinum", "t_death"],
        "feat_cont": selected_labs + ["AGE_AT_TREATMENTSTART"],
        "feat_cat": [],
        "feat_reconstr": selected_labs,
        "time_unit_days": args.time_unit_days,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "min_coverage": args.min_coverage,
        "outlier_quantiles": [args.outlier_lo, args.outlier_hi],
        "clip_bounds": {k: list(v) for k, v in bounds.items()},
        "split_counts": {
            "train": int(wide.loc[wide["split"] == "train", "DFCI_MRN"].nunique()),
            "valid": int(wide.loc[wide["split"] == "valid", "DFCI_MRN"].nunique()),
            "test": int(wide.loc[wide["split"] == "test", "DFCI_MRN"].nunique()),
        },
        "n_rows": int(len(wide)),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved:\n  {out_csv}\n  {out_manifest}")
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
