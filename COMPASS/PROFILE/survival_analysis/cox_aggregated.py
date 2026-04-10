"""
Step 1: CoxPH with aggregated lab features.

For each patient, labs observed strictly before FIRST_TREATMENT_DATE are
summarized into static features (last value, mean, slope, n measurements,
days since last measurement).  Separate CoxPH models are fit for:
  - time-to-platinum (PLATINUM event)
  - time-to-death    (DEATH event)

Evaluation:
  - 20% of patients held out as test set (never touched during CV)
  - 5-fold CV on the remaining 80% (train/val) reports mean ± std C-index
  - Final model fit on full 80%, evaluated on 20% test set

Outputs:
  results/cox_agg_cv_metrics.csv
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
TEST_FRAC = 0.20
MIN_PATIENT_COVERAGE = 0.20
MIN_OBS_FOR_SLOPE = 3


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _slope(days: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < MIN_OBS_FOR_SLOPE:
        return np.nan
    slope, *_ = linregress(days[mask], values[mask])
    return float(slope)


def aggregate_patient_labs(grp: pd.DataFrame, t0: pd.Timestamp) -> dict:
    pre = grp[grp["LAB_DATE"] < t0].sort_values("LAB_DATE")
    if pre.empty:
        return {}
    days = (pre["LAB_DATE"] - t0).dt.days.values.astype(float)
    vals = pre["LAB_VALUE"].values.astype(float)
    return {
        "last":       float(vals[-1]),
        "mean":       float(np.nanmean(vals)),
        "std":        float(np.nanstd(vals)) if len(vals) > 1 else np.nan,
        "slope":      _slope(days, vals),
        "n":          float(len(vals)),
        "days_since": float(-days[-1]),
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
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
# Train/test split (stratified by combined event label)
# ---------------------------------------------------------------------------
def stratified_label(outcome_df: pd.DataFrame) -> np.ndarray:
    """0=neither, 1=platinum only, 2=death only, 3=both."""
    p = outcome_df["PLATINUM"].values.astype(int)
    d = outcome_df["DEATH"].values.astype(int)
    return p + 2 * d


def split_train_test(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = stratified_label(merged)
    train_idx, test_idx = train_test_split(
        np.arange(len(merged)), test_size=TEST_FRAC,
        stratify=strat, random_state=SEED)
    return merged.iloc[train_idx], merged.iloc[test_idx]


# ---------------------------------------------------------------------------
# CoxPH helpers
# ---------------------------------------------------------------------------
def prepare_model_df(data: pd.DataFrame, feature_cols: list[str],
                     duration_col: str, event_col: str,
                     imputer: SimpleImputer, scaler: StandardScaler) -> pd.DataFrame:
    X = scaler.transform(imputer.transform(data[feature_cols].values))
    mdf = pd.DataFrame(X, columns=feature_cols, index=data.index)
    mdf["AGE_AT_TREATMENTSTART"] = data["AGE_AT_TREATMENTSTART"].values
    mdf[duration_col] = data[duration_col].values
    mdf[event_col]    = data[event_col].values
    return mdf


def fit_and_score(train_mdf: pd.DataFrame, val_mdf: pd.DataFrame,
                  duration_col: str, event_col: str) -> float:
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_mdf, duration_col=duration_col, event_col=event_col)
    pred = cph.predict_partial_hazard(val_mdf)
    return concordance_index(val_mdf[duration_col], -pred, val_mdf[event_col])


# ---------------------------------------------------------------------------
# Main per-outcome pipeline
# ---------------------------------------------------------------------------
def run_outcome(merged: pd.DataFrame, train_val: pd.DataFrame, test: pd.DataFrame,
                duration_col: str, event_col: str, label: str) -> None:
    feature_cols = [c for c in merged.columns
                    if c not in {"t_platinum", "t_death", "PLATINUM", "DEATH",
                                 "AGE_AT_TREATMENTSTART"}]

    # ---- 5-fold CV on train_val ----------------------------------------
    strat = stratified_label(train_val)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = []

    print(f"\n[{label}] 5-fold CV on {len(train_val)} train/val patients "
          f"({int(train_val[event_col].sum())} events)")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_val, strat), 1):
        tr  = train_val.iloc[tr_idx]
        val = train_val.iloc[val_idx]

        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        imputer.fit(tr[feature_cols].values)
        scaler.fit(imputer.transform(tr[feature_cols].values))

        tr_mdf  = prepare_model_df(tr,  feature_cols, duration_col, event_col, imputer, scaler)
        val_mdf = prepare_model_df(val, feature_cols, duration_col, event_col, imputer, scaler)

        c = fit_and_score(tr_mdf, val_mdf, duration_col, event_col)
        cv_scores.append(c)
        print(f"  Fold {fold}: C={c:.4f}")

    print(f"  CV C-index: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # ---- Final model on full train_val, evaluate on test ----------------
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    imputer.fit(train_val[feature_cols].values)
    scaler.fit(imputer.transform(train_val[feature_cols].values))

    train_mdf = prepare_model_df(train_val, feature_cols, duration_col, event_col, imputer, scaler)
    test_mdf  = prepare_model_df(test,      feature_cols, duration_col, event_col, imputer, scaler)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_mdf, duration_col=duration_col, event_col=event_col)

    pred_test = cph.predict_partial_hazard(test_mdf)
    c_test = concordance_index(test_mdf[duration_col], -pred_test, test_mdf[event_col])
    print(f"  Test C-index: {c_test:.4f}  (N={len(test_mdf)}, "
          f"events={int(test_mdf[event_col].sum())})")

    summary = cph.summary.sort_values("coef", key=abs, ascending=False)
    summary.to_csv(RESULTS / f"cox_agg_{label}_summary.csv")
    print(f"  Saved: results/cox_agg_{label}_summary.csv")

    return {
        "label":    label,
        "cv_mean":  np.mean(cv_scores),
        "cv_std":   np.std(cv_scores),
        "test_c":   c_test,
        "n_train":  len(train_val),
        "n_test":   len(test),
        "n_events_train": int(train_val[event_col].sum()),
        "n_events_test":  int(test[event_col].sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)

    print("Building outcome table...")
    outcome_df = make_outcome_df(df)
    print(f"  {len(outcome_df)} patients")

    print("\nBuilding feature matrix (pre-treatment labs only)...")
    feat_df = build_feature_matrix(df)

    merged = feat_df.join(
        outcome_df[["t_platinum", "t_death", "PLATINUM", "DEATH",
                    "AGE_AT_TREATMENTSTART"]],
        how="inner"
    ).dropna(subset=["t_platinum", "t_death"])

    print(f"\n{len(merged)} patients with features and outcomes")

    train_val, test = split_train_test(merged)
    print(f"Train/val: {len(train_val)}  |  Test (held-out): {len(test)}")

    rows = []
    for duration_col, event_col, label in [
        ("t_platinum", "PLATINUM", "platinum"),
        ("t_death",    "DEATH",    "death"),
    ]:
        row = run_outcome(merged, train_val, test, duration_col, event_col, label)
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(RESULTS / "cox_agg_cv_metrics.csv", index=False)
    print(f"\nSaved: results/cox_agg_cv_metrics.csv")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    main(parser.parse_args())
