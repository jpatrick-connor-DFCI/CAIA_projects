"""
Step 3: Competing risks — platinum and death as competing events.

Two formulations on the same aggregated pre-treatment lab features:
  A) Cause-specific CoxPH: separate model per event; other event censored.
  B) Fine-Gray subdistribution hazard: weighted CoxPH on CIF directly.

Evaluation:
  - 20% of patients held out as test set (patient-level split)
  - 5-fold CV on the remaining 80% reports mean ± std C-index per model type
  - Final models fit on full 80%, evaluated on 20% test set

Outputs:
  results/cr_cv_metrics.csv
  results/cr_cause_specific_{platinum,death}_summary.csv
  results/cr_finegray_{platinum,death}_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, FinegrayFitter
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
# Feature / outcome building (same as cox_aggregated)
# ---------------------------------------------------------------------------
def _slope(days, values):
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


def stratified_label(outcome_df: pd.DataFrame) -> np.ndarray:
    p = outcome_df["PLATINUM"].values.astype(int)
    d = outcome_df["DEATH"].values.astype(int)
    return p + 2 * d


def split_train_test(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = stratified_label(merged)
    tr_idx, te_idx = train_test_split(
        np.arange(len(merged)), test_size=TEST_FRAC,
        stratify=strat, random_state=SEED)
    return merged.iloc[tr_idx].copy(), merged.iloc[te_idx].copy()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def fit_transforms(data: pd.DataFrame, feature_cols: list[str]):
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    imputer.fit(data[feature_cols].values)
    scaler.fit(imputer.transform(data[feature_cols].values))
    return imputer, scaler


def apply_transforms(data: pd.DataFrame, feature_cols: list[str],
                     imputer: SimpleImputer, scaler: StandardScaler) -> pd.DataFrame:
    out = data.copy()
    out[feature_cols] = scaler.transform(imputer.transform(data[feature_cols].values))
    return out


def make_model_df(data: pd.DataFrame, feature_cols: list[str],
                  duration_col: str, event_col: str) -> pd.DataFrame:
    cols = feature_cols + ["AGE_AT_TREATMENTSTART", duration_col, event_col]
    return data[cols].copy()


# ---------------------------------------------------------------------------
# A) Cause-specific CoxPH
# ---------------------------------------------------------------------------
def _fit_cause_specific(train: pd.DataFrame, val: pd.DataFrame,
                         feature_cols: list[str],
                         duration_col: str, event_col: str) -> float:
    all_cols = feature_cols + ["AGE_AT_TREATMENTSTART", duration_col, event_col]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train[all_cols], duration_col=duration_col, event_col=event_col)
    pred = cph.predict_partial_hazard(val[all_cols])
    return concordance_index(val[duration_col], -pred, val[event_col])


def run_cause_specific(train_val: pd.DataFrame, test: pd.DataFrame,
                        feature_cols: list[str],
                        duration_col: str, event_col: str, label: str) -> dict:
    strat = stratified_label(train_val)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = []

    print(f"\n  [cause-specific {label}] 5-fold CV  N={len(train_val)}, "
          f"events={int(train_val[event_col].sum())}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_val, strat), 1):
        tr  = train_val.iloc[tr_idx]
        val = train_val.iloc[val_idx]

        imputer, scaler = fit_transforms(tr, feature_cols)
        tr_t  = apply_transforms(tr,  feature_cols, imputer, scaler)
        val_t = apply_transforms(val, feature_cols, imputer, scaler)

        c = _fit_cause_specific(tr_t, val_t, feature_cols, duration_col, event_col)
        cv_scores.append(c)
        print(f"    Fold {fold}: C={c:.4f}")

    print(f"    CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Final model
    imputer, scaler = fit_transforms(train_val, feature_cols)
    tv_t   = apply_transforms(train_val, feature_cols, imputer, scaler)
    test_t = apply_transforms(test,      feature_cols, imputer, scaler)

    all_cols = feature_cols + ["AGE_AT_TREATMENTSTART", duration_col, event_col]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tv_t[all_cols], duration_col=duration_col, event_col=event_col)
    pred = cph.predict_partial_hazard(test_t[all_cols])
    c_test = concordance_index(test_t[duration_col], -pred, test_t[event_col])
    print(f"    Test C: {c_test:.4f}  (N={len(test_t)}, "
          f"events={int(test_t[event_col].sum())})")

    summary = cph.summary.sort_values("coef", key=abs, ascending=False)
    summary.to_csv(RESULTS / f"cr_cause_specific_{label}_summary.csv")

    return {
        "model": "cause_specific", "label": label,
        "cv_mean": np.mean(cv_scores), "cv_std": np.std(cv_scores),
        "test_c": c_test,
        "n_train": len(train_val), "n_test": len(test),
        "n_events_train": int(train_val[event_col].sum()),
        "n_events_test":  int(test[event_col].sum()),
    }


# ---------------------------------------------------------------------------
# B) Fine-Gray subdistribution hazard
# ---------------------------------------------------------------------------
def _apply_finegray_weights(data: pd.DataFrame,
                             duration_col: str, event_col: str) -> pd.DataFrame:
    """
    Compute Fine-Gray subdistribution weights for the given event of interest.
    Returns the data with a 'fg_weight' column appended.
    """
    fg = FinegrayFitter()
    fg.fit(data[duration_col], data[event_col], competing_risk=1)
    # fg.weights_ is indexed by the original row index
    data = data.copy()
    data["fg_weight"] = fg.weights_.reindex(data.index).fillna(0.0).values
    return data


def _fit_fine_gray(train: pd.DataFrame, val: pd.DataFrame,
                    feature_cols: list[str],
                    duration_col: str, event_col: str) -> float:
    train_w = _apply_finegray_weights(train, duration_col, event_col)
    # Weighted CoxPH on training data
    all_cols = feature_cols + ["AGE_AT_TREATMENTSTART",
                                duration_col, event_col, "fg_weight"]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_w[all_cols], duration_col=duration_col, event_col=event_col,
            weights_col="fg_weight", robust=True)
    # Score on val (unweighted — Fine-Gray weights are a training device)
    pred = cph.predict_partial_hazard(val[feature_cols + ["AGE_AT_TREATMENTSTART",
                                                           duration_col, event_col]])
    return concordance_index(val[duration_col], -pred, val[event_col])


def run_fine_gray(train_val: pd.DataFrame, test: pd.DataFrame,
                   feature_cols: list[str],
                   duration_col: str, event_col: str, label: str) -> dict:
    strat = stratified_label(train_val)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = []

    print(f"\n  [Fine-Gray {label}] 5-fold CV  N={len(train_val)}, "
          f"events={int(train_val[event_col].sum())}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_val, strat), 1):
        tr  = train_val.iloc[tr_idx]
        val = train_val.iloc[val_idx]

        imputer, scaler = fit_transforms(tr, feature_cols)
        tr_t  = apply_transforms(tr,  feature_cols, imputer, scaler)
        val_t = apply_transforms(val, feature_cols, imputer, scaler)

        c = _fit_fine_gray(tr_t, val_t, feature_cols, duration_col, event_col)
        cv_scores.append(c)
        print(f"    Fold {fold}: C={c:.4f}")

    print(f"    CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Final model
    imputer, scaler = fit_transforms(train_val, feature_cols)
    tv_t   = apply_transforms(train_val, feature_cols, imputer, scaler)
    test_t = apply_transforms(test,      feature_cols, imputer, scaler)

    tv_w = _apply_finegray_weights(tv_t, duration_col, event_col)
    all_cols = feature_cols + ["AGE_AT_TREATMENTSTART",
                                duration_col, event_col, "fg_weight"]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tv_w[all_cols], duration_col=duration_col, event_col=event_col,
            weights_col="fg_weight", robust=True)

    pred = cph.predict_partial_hazard(
        test_t[feature_cols + ["AGE_AT_TREATMENTSTART", duration_col, event_col]])
    c_test = concordance_index(test_t[duration_col], -pred, test_t[event_col])
    print(f"    Test C: {c_test:.4f}  (N={len(test_t)}, "
          f"events={int(test_t[event_col].sum())})")

    summary = cph.summary.sort_values("coef", key=abs, ascending=False)
    summary.to_csv(RESULTS / f"cr_finegray_{label}_summary.csv")

    return {
        "model": "fine_gray", "label": label,
        "cv_mean": np.mean(cv_scores), "cv_std": np.std(cv_scores),
        "test_c": c_test,
        "n_train": len(train_val), "n_test": len(test),
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

    print("\nBuilding feature matrix...")
    feat_df = build_feature_matrix(df)

    merged = feat_df.join(
        outcome_df[["t_platinum", "t_death", "PLATINUM", "DEATH",
                    "AGE_AT_TREATMENTSTART"]],
        how="inner"
    ).dropna(subset=["t_platinum", "t_death"])
    print(f"{len(merged)} patients with features and outcomes")

    train_val, test = split_train_test(merged)
    print(f"Train/val: {len(train_val)}  |  Test (held-out): {len(test)}")

    feature_cols = [c for c in merged.columns
                    if c not in {"t_platinum", "t_death", "PLATINUM", "DEATH",
                                 "AGE_AT_TREATMENTSTART"}]

    rows = []
    for duration_col, event_col, label in [
        ("t_platinum", "PLATINUM", "platinum"),
        ("t_death",    "DEATH",    "death"),
    ]:
        print(f"\n=== {label.upper()} ===")
        rows.append(run_cause_specific(train_val, test, feature_cols,
                                        duration_col, event_col, label))
        rows.append(run_fine_gray(train_val, test, feature_cols,
                                   duration_col, event_col, label))

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(RESULTS / "cr_cv_metrics.csv", index=False)
    print(f"\nSaved: results/cr_cv_metrics.csv")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    main(parser.parse_args())
