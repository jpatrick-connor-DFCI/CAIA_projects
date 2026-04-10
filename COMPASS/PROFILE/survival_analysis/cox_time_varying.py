"""
Step 2: CoxPH with time-varying lab covariates (counting process / start-stop).

Labs are carried forward (LOCF) between measurements.  Each interval
(tstart, tstop] carries the most recently observed lab values.

Evaluation:
  - 20% of patients held out as test set (patient-level split)
  - 5-fold CV on the remaining 80% reports mean ± std C-index
  - Final model fit on full 80%, evaluated on 20% test set

Outputs:
  results/cox_tv_cv_metrics.csv
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
TOP_N_LABS = 20


# ---------------------------------------------------------------------------
# Lab selection
# ---------------------------------------------------------------------------
def select_labs(df: pd.DataFrame) -> list[str]:
    coverage = df.groupby("LAB_NAME")["DFCI_MRN"].nunique() / df["DFCI_MRN"].nunique()
    eligible = coverage[coverage >= MIN_PATIENT_COVERAGE].sort_values(ascending=False)
    labs = eligible.index.tolist()[:TOP_N_LABS]
    print(f"Selected {len(labs)} labs: {labs}")
    return labs


# ---------------------------------------------------------------------------
# Outcome table
# ---------------------------------------------------------------------------
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
    p = outcome_df["PLATINUM"].values.astype(int)
    d = outcome_df["DEATH"].values.astype(int)
    return p + 2 * d


def split_mrns(outcome_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mrns  = outcome_df.index.values
    strat = stratified_label(outcome_df)
    tr_idx, te_idx = train_test_split(
        np.arange(len(mrns)), test_size=TEST_FRAC,
        stratify=strat, random_state=SEED)
    return mrns[tr_idx], mrns[te_idx]


# ---------------------------------------------------------------------------
# Counting-process dataset builder (for a given patient subset)
# ---------------------------------------------------------------------------
def build_counting_process(df: pd.DataFrame, outcome_df: pd.DataFrame,
                            mrns: np.ndarray, labs: list[str],
                            duration_col: str, event_col: str) -> pd.DataFrame:
    lab_df = df[df["LAB_NAME"].isin(labs) & df["DFCI_MRN"].isin(mrns)].copy()
    lab_df["t"] = (lab_df["LAB_DATE"] - lab_df["FIRST_TREATMENT_DATE"]).dt.days.astype(float)

    rows = []
    for mrn in mrns:
        pat_out = outcome_df.loc[mrn]
        t0   = pat_out["FIRST_TREATMENT_DATE"]
        tend = float(pat_out[duration_col])
        evt  = int(pat_out[event_col])
        age  = float(pat_out["AGE_AT_TREATMENTSTART"])

        if tend <= 0:
            continue

        pat_labs = lab_df[lab_df["DFCI_MRN"] == mrn]
        pat_labs = pat_labs[(pat_labs["t"] > 0) & (pat_labs["t"] <= tend)]

        change_pts = sorted({0.0} | set(pat_labs["t"].unique()) | {tend})

        # Seed LOCF with pre-T0 last value per lab
        pre_df = df[(df["DFCI_MRN"] == mrn) & (df["LAB_NAME"].isin(labs))].copy()
        pre_df["t"] = (pre_df["LAB_DATE"] - t0).dt.days
        pre_df = pre_df[pre_df["t"] <= 0]
        locf: dict[str, float] = {}
        for lab in labs:
            sub = pre_df[pre_df["LAB_NAME"] == lab].sort_values("t")
            locf[lab] = float(sub["LAB_VALUE"].iloc[-1]) if not sub.empty else np.nan

        lab_updates = (pat_labs.sort_values("t")
                       .groupby("t")
                       .apply(lambda g: dict(zip(g["LAB_NAME"], g["LAB_VALUE"])))
                       .to_dict())

        for i in range(len(change_pts) - 1):
            tstart = change_pts[i]
            tstop  = change_pts[i + 1]
            if tstop in lab_updates:
                locf.update(lab_updates[tstop])
            is_last = i == len(change_pts) - 2
            row = {"id": mrn, "tstart": tstart, "tstop": tstop,
                   event_col: int(is_last) * evt,
                   "AGE_AT_TREATMENTSTART": age}
            row.update({lab: locf[lab] for lab in labs})
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fit / score helpers
# ---------------------------------------------------------------------------
def impute_scale_fit(cp_df: pd.DataFrame, feature_cols: list[str],
                     duration_col: str, event_col: str,
                     imputer: SimpleImputer | None = None,
                     scaler: StandardScaler | None = None,
                     fit_transforms: bool = True):
    if fit_transforms:
        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        imputer.fit(cp_df[feature_cols].values)
        scaler.fit(imputer.transform(cp_df[feature_cols].values))

    cp = cp_df.copy()
    cp[feature_cols] = scaler.transform(imputer.transform(cp[feature_cols].values))
    return cp, imputer, scaler


def score_ctv(model: CoxTimeVaryingFitter, cp_df: pd.DataFrame,
              duration_col: str, event_col: str) -> float:
    return model.concordance_index_


# ---------------------------------------------------------------------------
# Main per-outcome pipeline
# ---------------------------------------------------------------------------
def run_outcome(df: pd.DataFrame, outcome_df: pd.DataFrame,
                labs: list[str], train_mrns: np.ndarray, test_mrns: np.ndarray,
                duration_col: str, event_col: str, label: str) -> dict:
    feature_cols = labs + ["AGE_AT_TREATMENTSTART"]

    # Stratify train_mrns for K-fold
    tr_outcomes = outcome_df.loc[train_mrns]
    strat = stratified_label(tr_outcomes)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = []

    print(f"\n[{label}] 5-fold CV  (train/val N={len(train_mrns)}, "
          f"events={int(tr_outcomes[event_col].sum())})")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_mrns, strat), 1):
        fold_tr_mrns  = train_mrns[tr_idx]
        fold_val_mrns = train_mrns[val_idx]

        tr_cp  = build_counting_process(df, outcome_df, fold_tr_mrns,
                                         labs, duration_col, event_col)
        val_cp = build_counting_process(df, outcome_df, fold_val_mrns,
                                         labs, duration_col, event_col)
        if tr_cp.empty or val_cp.empty:
            continue

        tr_cp, imputer, scaler = impute_scale_fit(tr_cp, feature_cols,
                                                   duration_col, event_col,
                                                   fit_transforms=True)
        val_cp, _, _ = impute_scale_fit(val_cp, feature_cols,
                                         duration_col, event_col,
                                         imputer=imputer, scaler=scaler,
                                         fit_transforms=False)

        ctv = CoxTimeVaryingFitter(penalizer=0.1)
        ctv.fit(tr_cp, id_col="id", start_col="tstart", stop_col="tstop",
                event_col=event_col,
                formula=" + ".join(feature_cols))

        # Score on val: use concordance_index from lifelines on val predictions
        # CoxTimeVaryingFitter doesn't have predict_partial_hazard on new data easily,
        # so we use the training concordance as a proxy on the fitted val set by
        # re-fitting with combined data — instead we compute on the val dataset
        # using the model's linear predictor via the log partial hazard approximation.
        # Workaround: get partial hazard from summary coefficients dot features.
        coef = ctv.params_
        val_feats = val_cp[feature_cols].values
        # For missing columns not in val, use 0
        lp = val_cp[feature_cols].fillna(0).values @ coef[feature_cols].values
        # C-index on val: for each patient take their last-interval linear predictor
        val_last = (val_cp.sort_values("tstop")
                    .groupby("id")
                    .last()
                    .reset_index())
        val_last["lp"] = val_cp.groupby("id").apply(
            lambda g: (g[feature_cols].fillna(0).values @
                       coef[feature_cols].values).mean()
        ).values

        from lifelines.utils import concordance_index as ci
        val_outcomes = outcome_df.loc[fold_val_mrns]
        val_last = val_last.set_index("id")
        shared = val_last.index.intersection(val_outcomes.index)
        c = ci(val_outcomes.loc[shared, duration_col],
               -val_last.loc[shared, "lp"],
               val_outcomes.loc[shared, event_col])
        cv_scores.append(c)
        print(f"  Fold {fold}: C={c:.4f}  (tr={len(fold_tr_mrns)}, "
              f"val={len(fold_val_mrns)})")

    print(f"  CV C-index: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # ---- Final model on full train_mrns, evaluate on test_mrns ----------
    train_cp = build_counting_process(df, outcome_df, train_mrns,
                                       labs, duration_col, event_col)
    test_cp  = build_counting_process(df, outcome_df, test_mrns,
                                       labs, duration_col, event_col)

    train_cp, imputer, scaler = impute_scale_fit(train_cp, feature_cols,
                                                   duration_col, event_col,
                                                   fit_transforms=True)
    test_cp, _, _ = impute_scale_fit(test_cp, feature_cols,
                                      duration_col, event_col,
                                      imputer=imputer, scaler=scaler,
                                      fit_transforms=False)

    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(train_cp, id_col="id", start_col="tstart", stop_col="tstop",
            event_col=event_col, formula=" + ".join(feature_cols))

    coef = ctv.params_
    test_outcomes = outcome_df.loc[test_mrns]
    test_lp = test_cp.groupby("id").apply(
        lambda g: (g[feature_cols].fillna(0).values @ coef[feature_cols].values).mean()
    )
    shared = test_lp.index.intersection(test_outcomes.index)
    from lifelines.utils import concordance_index as ci
    c_test = ci(test_outcomes.loc[shared, duration_col],
                -test_lp.loc[shared],
                test_outcomes.loc[shared, event_col])
    print(f"  Test C-index: {c_test:.4f}  (N={len(shared)}, "
          f"events={int(test_outcomes.loc[shared, event_col].sum())})")

    summary = ctv.summary.sort_values("coef", key=abs, ascending=False)
    summary.to_csv(RESULTS / f"cox_tv_{label}_summary.csv")
    print(f"  Saved: results/cox_tv_{label}_summary.csv")

    return {
        "label":   label,
        "cv_mean": np.mean(cv_scores),
        "cv_std":  np.std(cv_scores),
        "test_c":  c_test,
        "n_train": len(train_mrns),
        "n_test":  len(test_mrns),
        "n_events_train": int(tr_outcomes[event_col].sum()),
        "n_events_test":  int(test_outcomes[event_col].sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("Loading data...")
    df = pd.read_csv(args.data, low_memory=False)
    for col in ["LAB_DATE", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE", "PLATINUM_DATE"]:
        df[col] = pd.to_datetime(df[col])

    print("Building outcome table...")
    outcome_df = make_outcome_df(df)
    print(f"  {len(outcome_df)} patients")

    train_mrns, test_mrns = split_mrns(outcome_df)
    print(f"Train/val: {len(train_mrns)}  |  Test (held-out): {len(test_mrns)}")

    labs = select_labs(df)

    rows = []
    for duration_col, event_col, label in [
        ("t_platinum", "PLATINUM", "platinum"),
        ("t_death",    "DEATH",    "death"),
    ]:
        row = run_outcome(df, outcome_df, labs, train_mrns, test_mrns,
                          duration_col, event_col, label)
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(RESULTS / "cox_tv_cv_metrics.csv", index=False)
    print(f"\nSaved: results/cox_tv_cv_metrics.csv")
    print(cv_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    main(parser.parse_args())
