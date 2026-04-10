"""
Step 1: CoxPH with aggregated lab features.

For each patient, labs observed strictly before FIRST_TREATMENT_DATE are
summarized into static features (last value, mean, slope, n measurements,
days since last measurement).  Separate CoxPH models are fit for:
  - time-to-platinum (PLATINUM event)
  - time-to-death    (DEATH event)

Outputs:
  results/cox_agg_platinum_summary.csv
  results/cox_agg_death_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import linregress
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_PATIENT_COVERAGE = 0.20   # keep labs present in >= 20% of patients
MIN_OBS_FOR_SLOPE = 3         # need at least 3 time points to estimate slope


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------
def _slope(days: np.ndarray, values: np.ndarray) -> float:
    """Linear slope of values over time (per day).  Returns NaN if < MIN_OBS."""
    mask = np.isfinite(values)
    if mask.sum() < MIN_OBS_FOR_SLOPE:
        return np.nan
    slope, *_ = linregress(days[mask], values[mask])
    return float(slope)


def aggregate_patient_labs(grp: pd.DataFrame, t0: pd.Timestamp) -> dict:
    """For one patient+lab group, return summary stats relative to t0."""
    pre = grp[grp["LAB_DATE"] < t0].copy()
    if pre.empty:
        return {}
    pre = pre.sort_values("LAB_DATE")
    days = (pre["LAB_DATE"] - t0).dt.days.values.astype(float)   # negative = before t0
    vals = pre["LAB_VALUE"].values.astype(float)
    return {
        "last":       float(vals[-1]),
        "mean":       float(np.nanmean(vals)),
        "std":        float(np.nanstd(vals)) if len(vals) > 1 else np.nan,
        "slope":      _slope(days, vals),
        "n":          float(len(vals)),
        "days_since": float(-days[-1]),   # positive = days before t0
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: longitudinal_prediction_data (long format).
    Returns wide patient-level feature matrix.
    """
    df = df.copy()
    df["LAB_DATE"] = pd.to_datetime(df["LAB_DATE"])
    df["FIRST_TREATMENT_DATE"] = pd.to_datetime(df["FIRST_TREATMENT_DATE"])

    records = []
    for (mrn, lab), grp in df.groupby(["DFCI_MRN", "LAB_NAME"]):
        t0 = grp["FIRST_TREATMENT_DATE"].iloc[0]
        stats = aggregate_patient_labs(grp, t0)
        if not stats:
            continue
        row = {"DFCI_MRN": mrn}
        for stat_name, val in stats.items():
            row[f"{lab}__{stat_name}"] = val
        records.append(row)

    feat = pd.DataFrame(records).groupby("DFCI_MRN").first()

    # Drop features from labs with < MIN_PATIENT_COVERAGE patients having data
    n_patients = df["DFCI_MRN"].nunique()
    coverage = feat.notna().mean()
    feat = feat.loc[:, coverage >= MIN_PATIENT_COVERAGE]

    print(f"Feature matrix: {feat.shape[0]} patients × {feat.shape[1]} features")
    print(f"  ({(coverage >= MIN_PATIENT_COVERAGE).sum()} / {coverage.shape[0]} "
          f"lab×stat columns retained at {MIN_PATIENT_COVERAGE:.0%} coverage)")
    return feat


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------
def make_outcome_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with time and event columns."""
    pat = (df[["DFCI_MRN", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE",
               "DEATH", "PLATINUM_DATE", "PLATINUM", "AGE_AT_TREATMENTSTART"]]
           .drop_duplicates("DFCI_MRN")
           .set_index("DFCI_MRN"))

    pat["FIRST_TREATMENT_DATE"] = pd.to_datetime(pat["FIRST_TREATMENT_DATE"])
    pat["LAST_CONTACT_DATE"]    = pd.to_datetime(pat["LAST_CONTACT_DATE"])
    pat["PLATINUM_DATE"]        = pd.to_datetime(pat["PLATINUM_DATE"])

    # Time to platinum (days from T=0)
    pat["t_platinum"] = (pat["PLATINUM_DATE"] - pat["FIRST_TREATMENT_DATE"]).dt.days
    # Time to death / censoring
    pat["t_death"]    = (pat["LAST_CONTACT_DATE"] - pat["FIRST_TREATMENT_DATE"]).dt.days

    # Sanity: drop patients with non-positive follow-up
    pat = pat[(pat["t_platinum"] > 0) & (pat["t_death"] > 0)]
    return pat


# ---------------------------------------------------------------------------
# CoxPH fitting
# ---------------------------------------------------------------------------
def fit_cox(feature_df: pd.DataFrame, outcome_df: pd.DataFrame,
            duration_col: str, event_col: str, label: str) -> pd.DataFrame:
    """
    Merge features + outcomes, impute, scale, fit penalized CoxPH.
    Returns summary DataFrame sorted by |coef|.
    """
    merged = feature_df.join(outcome_df[[duration_col, event_col,
                                         "AGE_AT_TREATMENTSTART"]], how="inner")
    merged = merged.dropna(subset=[duration_col, event_col])

    feature_cols = [c for c in merged.columns
                    if c not in {duration_col, event_col, "AGE_AT_TREATMENTSTART"}]

    X = merged[feature_cols].values
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X))

    model_df = pd.DataFrame(X, columns=feature_cols, index=merged.index)
    model_df["AGE_AT_TREATMENTSTART"] = merged["AGE_AT_TREATMENTSTART"].values
    model_df[duration_col] = merged[duration_col].values
    model_df[event_col]    = merged[event_col].values

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(model_df, duration_col=duration_col, event_col=event_col)

    # Concordance on training data (reporting only; no CV here)
    pred = cph.predict_partial_hazard(model_df)
    c_idx = concordance_index(model_df[duration_col], -pred, model_df[event_col])
    print(f"\n[{label}] C-index (train): {c_idx:.4f}")
    print(f"  N={len(model_df)}, events={int(model_df[event_col].sum())}")

    summary = cph.summary.copy()
    summary = summary.sort_values("coef", key=abs, ascending=False)
    summary.to_csv(RESULTS / f"cox_agg_{label}_summary.csv")
    print(f"  Saved: results/cox_agg_{label}_summary.csv")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)

    print("Building outcome table...")
    outcome_df = make_outcome_df(df)
    print(f"  {len(outcome_df)} patients after outcome filtering")

    print("\nBuilding feature matrix (pre-treatment labs only)...")
    feat_df = build_feature_matrix(df)

    print("\n--- Fitting CoxPH: time-to-platinum ---")
    fit_cox(feat_df, outcome_df,
            duration_col="t_platinum", event_col="PLATINUM", label="platinum")

    print("\n--- Fitting CoxPH: time-to-death ---")
    fit_cox(feat_df, outcome_df,
            duration_col="t_death", event_col="DEATH", label="death")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(DATA_PATH / "longitudinal_prediction_data.csv"),
        help="Path to longitudinal_prediction_data.csv",
    )
    main(parser.parse_args())
