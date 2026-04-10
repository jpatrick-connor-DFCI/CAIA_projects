"""
Step 2: CoxPH with time-varying lab covariates (counting process / start-stop).

Labs are carried forward (LOCF) between measurements.  Each interval
(tstart, tstop] carries the most recently observed lab values.

The counting process setup means labs can update throughout follow-up —
we are NOT restricted to pre-treatment observations.  To avoid leakage,
we still start the clock at FIRST_TREATMENT_DATE (T=0).

Outputs:
  results/cox_tv_platinum_summary.csv
  results/cox_tv_death_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

MIN_PATIENT_COVERAGE = 0.20
TOP_N_LABS = 20          # keep only the N most covered labs to limit width


# ---------------------------------------------------------------------------
# Build counting-process dataset
# ---------------------------------------------------------------------------
def select_labs(df: pd.DataFrame, t0_col: str = "FIRST_TREATMENT_DATE") -> list[str]:
    """Return labs with >= MIN_PATIENT_COVERAGE patients having any observation."""
    df = df.copy()
    df[t0_col] = pd.to_datetime(df[t0_col])
    # Count distinct patients per lab
    coverage = (df.groupby("LAB_NAME")["DFCI_MRN"].nunique()
                / df["DFCI_MRN"].nunique())
    eligible = coverage[coverage >= MIN_PATIENT_COVERAGE].sort_values(ascending=False)
    labs = eligible.index.tolist()[:TOP_N_LABS]
    print(f"Selected {len(labs)} labs (coverage >= {MIN_PATIENT_COVERAGE:.0%}, "
          f"top {TOP_N_LABS}): {labs}")
    return labs


def build_counting_process(df: pd.DataFrame, outcome_df: pd.DataFrame,
                            labs: list[str],
                            duration_col: str, event_col: str) -> pd.DataFrame:
    """
    For each patient, construct (id, tstart, tstop, event, lab_values...) rows.

    Strategy:
      - Change-points = sorted unique LAB_DATEs for the patient (clipped to [T0, T_end])
      - Within each interval, lab value = last observed value (LOCF) or NaN
      - Final interval ends at T_end with event indicator
    """
    df = df.copy()
    df["LAB_DATE"] = pd.to_datetime(df["LAB_DATE"])
    df["FIRST_TREATMENT_DATE"] = pd.to_datetime(df["FIRST_TREATMENT_DATE"])

    lab_df = df[df["LAB_NAME"].isin(labs)].copy()

    rows = []
    for mrn, pat_out in outcome_df.iterrows():
        t0   = pd.to_datetime(pat_out["FIRST_TREATMENT_DATE"])
        tend = float(pat_out[duration_col])
        evt  = int(pat_out[event_col])
        age  = float(pat_out["AGE_AT_TREATMENTSTART"])

        if tend <= 0:
            continue

        pat_labs = lab_df[lab_df["DFCI_MRN"] == mrn].copy()
        # Days from T0
        pat_labs["t"] = (pat_labs["LAB_DATE"] - t0).dt.days.astype(float)
        # Only keep observations in (0, tend]  — strictly after T0
        pat_labs = pat_labs[(pat_labs["t"] > 0) & (pat_labs["t"] <= tend)]

        # Change-points: 0 + unique lab dates + tend
        change_pts = sorted({0.0} | set(pat_labs["t"].unique()) | {tend})

        # Build LOCF values at each change point per lab
        locf: dict[str, float] = {}
        # Seed with pre-T0 last value if available
        pre_t0 = df[(df["DFCI_MRN"] == mrn) & (df["LAB_NAME"].isin(labs))].copy()
        pre_t0["t"] = (pre_t0["LAB_DATE"] - t0).dt.days
        pre_t0 = pre_t0[pre_t0["t"] <= 0]
        for lab in labs:
            sub = pre_t0[pre_t0["LAB_NAME"] == lab].sort_values("t")
            locf[lab] = float(sub["LAB_VALUE"].iloc[-1]) if not sub.empty else np.nan

        # Walk through intervals
        lab_updates = (pat_labs.sort_values("t")
                       .groupby("t")
                       .apply(lambda g: dict(zip(g["LAB_NAME"], g["LAB_VALUE"])))
                       .to_dict())

        for i in range(len(change_pts) - 1):
            tstart = change_pts[i]
            tstop  = change_pts[i + 1]
            if tstop in lab_updates:
                locf.update(lab_updates[tstop])
            is_last = (i == len(change_pts) - 2)
            row = {"id": mrn, "tstart": tstart, "tstop": tstop,
                   event_col: int(is_last) * evt,
                   "AGE_AT_TREATMENTSTART": age}
            row.update({lab: locf[lab] for lab in labs})
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fit time-varying Cox
# ---------------------------------------------------------------------------
def fit_tv_cox(df: pd.DataFrame, outcome_df: pd.DataFrame,
               labs: list[str], duration_col: str, event_col: str,
               label: str) -> pd.DataFrame:

    print(f"\nBuilding counting-process dataset for [{label}]...")
    cp_df = build_counting_process(df, outcome_df, labs, duration_col, event_col)
    print(f"  {cp_df['id'].nunique()} patients, {len(cp_df)} intervals")

    feature_cols = labs + ["AGE_AT_TREATMENTSTART"]

    X = cp_df[feature_cols].values
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X))
    cp_df[feature_cols] = X

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(cp_df, id_col="id", start_col="tstart", stop_col="tstop",
            event_col=event_col, formula=" + ".join(feature_cols))

    print(f"  Concordance (Harrell): {ctv.concordance_index_:.4f}")
    summary = ctv.summary.sort_values("coef", key=abs, ascending=False)
    summary.to_csv(RESULTS / f"cox_tv_{label}_summary.csv")
    print(f"  Saved: results/cox_tv_{label}_summary.csv")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)
    df["LAB_DATE"] = pd.to_datetime(df["LAB_DATE"])
    df["FIRST_TREATMENT_DATE"] = pd.to_datetime(df["FIRST_TREATMENT_DATE"])
    df["LAST_CONTACT_DATE"]    = pd.to_datetime(df["LAST_CONTACT_DATE"])
    df["PLATINUM_DATE"]        = pd.to_datetime(df["PLATINUM_DATE"])

    # Outcome table (one row per patient)
    pat = (df[["DFCI_MRN", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE",
               "DEATH", "PLATINUM_DATE", "PLATINUM", "AGE_AT_TREATMENTSTART"]]
           .drop_duplicates("DFCI_MRN").set_index("DFCI_MRN"))
    pat["t_platinum"] = (pat["PLATINUM_DATE"] - pat["FIRST_TREATMENT_DATE"]).dt.days
    pat["t_death"]    = (pat["LAST_CONTACT_DATE"] - pat["FIRST_TREATMENT_DATE"]).dt.days
    pat = pat[(pat["t_platinum"] > 0) & (pat["t_death"] > 0)]
    print(f"{len(pat)} patients in outcome table")

    labs = select_labs(df)

    fit_tv_cox(df, pat, labs, "t_platinum", "PLATINUM", "platinum")
    fit_tv_cox(df, pat, labs, "t_death",    "DEATH",    "death")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(DATA_PATH / "longitudinal_prediction_data.csv"),
    )
    main(parser.parse_args())
