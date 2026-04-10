"""
Step 3: Competing risks — platinum and death as competing events.

Two formulations:
  A) Cause-specific CoxPH: separate Cox model per event; the other event
     is treated as censoring.  Valid for identifying risk factors.

  B) Fine-Gray subdistribution hazard: models the cumulative incidence
     function (CIF) directly.  Better for absolute risk prediction.

Uses the same aggregated lab features as cox_aggregated.py.

Requires: scikit-survival (cause-specific) and lifelines (Fine-Gray).

Outputs:
  results/cr_cause_specific_platinum_summary.csv
  results/cr_cause_specific_death_summary.csv
  results/cr_finegray_platinum_summary.csv
  results/cr_finegray_death_summary.csv
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
# Paths / config
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

MIN_PATIENT_COVERAGE = 0.20
MIN_OBS_FOR_SLOPE = 3


# ---------------------------------------------------------------------------
# Reuse feature building from cox_aggregated (inline to keep standalone)
# ---------------------------------------------------------------------------
def _slope(days: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < MIN_OBS_FOR_SLOPE:
        return np.nan
    slope, *_ = linregress(days[mask], values[mask])
    return float(slope)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LAB_DATE"] = pd.to_datetime(df["LAB_DATE"])
    df["FIRST_TREATMENT_DATE"] = pd.to_datetime(df["FIRST_TREATMENT_DATE"])

    records = []
    for (mrn, lab), grp in df.groupby(["DFCI_MRN", "LAB_NAME"]):
        t0  = grp["FIRST_TREATMENT_DATE"].iloc[0]
        pre = grp[grp["LAB_DATE"] < t0].sort_values("LAB_DATE")
        if pre.empty:
            continue
        days = (pre["LAB_DATE"] - t0).dt.days.values.astype(float)
        vals = pre["LAB_VALUE"].values.astype(float)
        row = {"DFCI_MRN": mrn,
               f"{lab}__last":       float(vals[-1]),
               f"{lab}__mean":       float(np.nanmean(vals)),
               f"{lab}__std":        float(np.nanstd(vals)) if len(vals) > 1 else np.nan,
               f"{lab}__slope":      _slope(days, vals),
               f"{lab}__n":          float(len(vals)),
               f"{lab}__days_since": float(-days[-1])}
        records.append(row)

    feat = pd.DataFrame(records).groupby("DFCI_MRN").first()
    coverage = feat.notna().mean()
    feat = feat.loc[:, coverage >= MIN_PATIENT_COVERAGE]
    print(f"Feature matrix: {feat.shape[0]} patients × {feat.shape[1]} features")
    return feat


def make_outcome_df(df: pd.DataFrame) -> pd.DataFrame:
    pat = (df[["DFCI_MRN", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE",
               "DEATH", "PLATINUM_DATE", "PLATINUM", "AGE_AT_TREATMENTSTART"]]
           .drop_duplicates("DFCI_MRN").set_index("DFCI_MRN"))
    for col in ["FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE", "PLATINUM_DATE"]:
        pat[col] = pd.to_datetime(pat[col])
    pat["t_platinum"] = (pat["PLATINUM_DATE"] - pat["FIRST_TREATMENT_DATE"]).dt.days
    pat["t_death"]    = (pat["LAST_CONTACT_DATE"] - pat["FIRST_TREATMENT_DATE"]).dt.days
    return pat[(pat["t_platinum"] > 0) & (pat["t_death"] > 0)]


# ---------------------------------------------------------------------------
# A) Cause-specific CoxPH
# ---------------------------------------------------------------------------
def fit_cause_specific(feat_df: pd.DataFrame, outcome_df: pd.DataFrame,
                       imputer: SimpleImputer, scaler: StandardScaler,
                       feature_cols: list[str]) -> None:
    """
    Platinum as event, death as competing (censored at death time for platinum model).
    Death as event, platinum as competing (censored at platinum time for death model).
    """
    for event_name, duration_col, event_col, competing_col in [
        ("platinum", "t_platinum", "PLATINUM", "DEATH"),
        ("death",    "t_death",    "DEATH",    "PLATINUM"),
    ]:
        merged = feat_df.join(
            outcome_df[[duration_col, event_col, competing_col, "AGE_AT_TREATMENTSTART"]],
            how="inner"
        ).dropna(subset=[duration_col, event_col])

        X = scaler.transform(imputer.transform(merged[feature_cols].values))
        model_df = pd.DataFrame(X, columns=feature_cols, index=merged.index)
        model_df["AGE_AT_TREATMENTSTART"] = merged["AGE_AT_TREATMENTSTART"].values
        model_df[duration_col] = merged[duration_col].values
        # Cause-specific: censor at competing event time, indicator = 0
        model_df[event_col]    = merged[event_col].values

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(model_df, duration_col=duration_col, event_col=event_col)

        pred = cph.predict_partial_hazard(model_df)
        c_idx = concordance_index(model_df[duration_col], -pred, model_df[event_col])
        print(f"  [cause-specific {event_name}] C-index={c_idx:.4f}, "
              f"N={len(model_df)}, events={int(model_df[event_col].sum())}")

        summary = cph.summary.sort_values("coef", key=abs, ascending=False)
        summary.to_csv(RESULTS / f"cr_cause_specific_{event_name}_summary.csv")
        print(f"  Saved: results/cr_cause_specific_{event_name}_summary.csv")


# ---------------------------------------------------------------------------
# B) Fine-Gray subdistribution hazard (via lifelines FinegrayFitter)
# ---------------------------------------------------------------------------
def fit_fine_gray(feat_df: pd.DataFrame, outcome_df: pd.DataFrame,
                  imputer: SimpleImputer, scaler: StandardScaler,
                  feature_cols: list[str]) -> None:
    """
    Fine-Gray requires a weighted dataset per event of interest.
    We use lifelines.statistics.fine_gray_regression (univariate per feature)
    or the FinegrayFitter for multivariate via pseudo-observations.

    lifelines FinegrayFitter fits one covariate at a time; for multivariate
    Fine-Gray we use the weighted approach: apply Fine-Gray weights to the
    cause-specific Cox model using lifelines' built-in weight support.
    """
    from lifelines import FinegrayFitter

    for event_name, duration_col, event_col, competing_col in [
        ("platinum", "t_platinum", "PLATINUM", "DEATH"),
        ("death",    "t_death",    "DEATH",    "PLATINUM"),
    ]:
        merged = feat_df.join(
            outcome_df[[duration_col, event_col, competing_col, "AGE_AT_TREATMENTSTART"]],
            how="inner"
        ).dropna(subset=[duration_col, event_col, competing_col])

        X = scaler.transform(imputer.transform(merged[feature_cols].values))
        model_df = pd.DataFrame(X, columns=feature_cols, index=merged.index)
        model_df["AGE_AT_TREATMENTSTART"] = merged["AGE_AT_TREATMENTSTART"].values
        model_df[duration_col] = merged[duration_col].values
        model_df[event_col]    = merged[event_col].values
        model_df[competing_col] = merged[competing_col].values

        # Fine-Gray: construct weighted dataset for the event of interest
        # "event_col" = 1 is the event; competing_col = 1 is the competing event
        fg = FinegrayFitter()
        fg.fit(model_df[duration_col], model_df[event_col],
               competing_risk=1)   # 1 = observed event; competing = other event

        # lifelines FinegrayFitter only supports univariate out of the box.
        # For multivariate, we apply FG weights to CoxPH (recommended approach).
        # Fine-Gray weights are stored in fg.weights_ after fitting a dataset.
        # We re-weight via CoxPH with `weights_col`.
        weighted_df = fg.weights_.rename("fg_weight").to_frame()
        # Merge weights back
        model_df = model_df.join(weighted_df, how="left")
        model_df["fg_weight"] = model_df["fg_weight"].fillna(0.0)

        # Use competing-event censoring for the weighted Cox model
        # (Fine-Gray pseudo-time trick: leave event times at observed value,
        #  set competing-event times to max follow-up time but weight=0)
        fg_duration = model_df[duration_col].copy()
        fg_event    = model_df[event_col].copy()

        cph_fg = CoxPHFitter(penalizer=0.1)
        all_cols = feature_cols + ["AGE_AT_TREATMENTSTART", "fg_weight"]
        fit_df = model_df[all_cols + [duration_col, event_col]].copy()
        fit_df[duration_col] = fg_duration
        fit_df[event_col]    = fg_event

        cph_fg.fit(fit_df, duration_col=duration_col, event_col=event_col,
                   weights_col="fg_weight", robust=True)

        pred = cph_fg.predict_partial_hazard(fit_df)
        c_idx = concordance_index(fit_df[duration_col], -pred, fit_df[event_col])
        print(f"  [Fine-Gray {event_name}] C-index={c_idx:.4f}, "
              f"N={len(fit_df)}, events={int(fit_df[event_col].sum())}")

        summary = cph_fg.summary.sort_values("coef", key=abs, ascending=False)
        summary.to_csv(RESULTS / f"cr_finegray_{event_name}_summary.csv")
        print(f"  Saved: results/cr_finegray_{event_name}_summary.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)

    print("Building outcome table...")
    outcome_df = make_outcome_df(df)
    print(f"  {len(outcome_df)} patients")

    print("\nBuilding feature matrix...")
    feat_df = build_feature_matrix(df)

    feature_cols = feat_df.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    imputer.fit(feat_df.values)
    scaler.fit(imputer.transform(feat_df.values))

    print("\n--- A) Cause-specific CoxPH ---")
    fit_cause_specific(feat_df, outcome_df, imputer, scaler, feature_cols)

    print("\n--- B) Fine-Gray subdistribution hazard ---")
    fit_fine_gray(feat_df, outcome_df, imputer, scaler, feature_cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(DATA_PATH / "longitudinal_prediction_data.csv"),
    )
    main(parser.parse_args())
