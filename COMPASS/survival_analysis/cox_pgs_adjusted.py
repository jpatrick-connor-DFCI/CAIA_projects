"""
PGS-adjusted univariate Cox for the lab-feature arm.

For each (endpoint, landmark, target lab feature, PGS), fits three Cox models
on the same cohort (rows with valid outcomes + age):

    baseline:  Surv(t) ~ z(AGE) + z(LAB__n_observations) + z(LAB__feature)
    joint:     Surv(t) ~ z(AGE) + z(PGS) + z(LAB__n_observations) + z(LAB__feature)
    pgs-alone: Surv(t) ~ z(AGE) + z(PGS)

    [+ feature_missing if any] [+ pgs_missing if any]

The pgs-alone fit doesn't depend on which target_lab is being compared, so it
is computed once per (endpoint, pgs) and broadcast across the target-lab rows.
Together these implement the nested-model comparison: pgs-alone vs baseline-lab
(head-to-head), and joint vs baseline-lab (does PGS add on top of clinical?).

Compared to cox_aggregated's run_univariate_nobs_adjusted_associations, this
module sweeps over PGS columns instead of over all lab features: target labs
are fixed to {Testosterone, PSA} with feature_stat=mean. PGS columns are
selected from the germline file by name match ('testosterone' or 'prostate').

Reads:
  * aggregated_landmark{N}.csv  (one per landmark; from build_prediction_inputs.py)
  * complete_germline_data_df.csv.gz  (DFCI_MRN-keyed, deduplicated)

Writes one CSV per landmark:
  <output-dir>/cox/landmark_{N}/both/cox_pgs_adjusted.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from cox_aggregated import (  # noqa: E402
    AGE_COL,
    ENDPOINTS,
    RESULTS,
    benjamini_hochberg,
    fit_cox_with_fallback,
    matching_n_obs_feature,
    normalize_endpoints,
    normalize_landmark_days,
    require_lifelines,
)

DEFAULT_AGGREGATED_PATTERN = (
    "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis/"
    "prediction_inputs/aggregated_landmark{landmark}.csv"
)
DEFAULT_GERMLINE_PATH = (
    "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
    "clinical_and_genomic_features/complete_germline_data_df.csv.gz"
)
DEFAULT_TARGET_LABS = ("Testosterone", "PSA")
DEFAULT_FEATURE_STAT = "mean"
PGS_NAME_TOKENS = ("testosterone", "prostate")
OUTPUT_FILENAME = "cox_pgs_adjusted.csv"


def select_pgs_columns(germline: pd.DataFrame, tokens: tuple[str, ...]) -> list[str]:
    """Columns whose name (lowercased) contains any of the given tokens.

    Preserves germline column order; de-duplicates.
    """
    seen: set[str] = set()
    selected: list[str] = []
    for col in germline.columns:
        lc = col.lower()
        if any(tok in lc for tok in tokens) and col not in seen:
            seen.add(col)
            selected.append(col)
    return selected


def load_germline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "DFCI_MRN" not in df.columns:
        raise ValueError(f"{path} missing DFCI_MRN column.")
    df["DFCI_MRN"] = pd.to_numeric(df["DFCI_MRN"], errors="coerce")
    df = df.loc[df["DFCI_MRN"].notna()].copy()
    df["DFCI_MRN"] = df["DFCI_MRN"].astype(int)
    if df["DFCI_MRN"].duplicated().any():
        n_dup = int(df["DFCI_MRN"].duplicated().sum())
        raise ValueError(f"{path} has {n_dup} duplicate DFCI_MRN rows; expected deduplicated.")
    return df.set_index("DFCI_MRN")


def load_aggregated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "DFCI_MRN" not in df.columns:
        raise ValueError(f"{path} missing DFCI_MRN column.")
    df["DFCI_MRN"] = pd.to_numeric(df["DFCI_MRN"], errors="coerce")
    df = df.loc[df["DFCI_MRN"].notna()].copy()
    df["DFCI_MRN"] = df["DFCI_MRN"].astype(int)
    return df.set_index("DFCI_MRN")


def _zscore(values: np.ndarray) -> tuple[np.ndarray, float]:
    sd = float(np.std(values, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return values, sd
    return (values - float(np.mean(values))) / sd, sd


def fit_baseline_lab(
    data: pd.DataFrame,
    *,
    feature: str,
    endpoint: str,
    fallback_penalizer: float,
    min_events_per_feature: int,
) -> dict:
    """Cox on [z(AGE) + z(n_obs) + z(lab)] — same cohort, no PGS term.

    Provides the reference effect size for the lab feature so callers can compute
    PGS-induced attenuation. Mirrors fit_one's missing-indicator and z-scoring
    conventions so the two are directly comparable.
    """
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
    n_obs_feature = matching_n_obs_feature(feature)

    out = {
        "coef_feature_baseline": np.nan,
        "hazard_ratio_baseline_per_sd": np.nan,
        "ci_lower_baseline": np.nan,
        "ci_upper_baseline": np.nan,
        "p_value_baseline": np.nan,
        "n_patients_baseline": 0,
        "n_events_baseline": 0,
        "fit_penalizer_baseline": np.nan,
        "note_baseline": "",
    }
    for required in (feature, n_obs_feature, duration_col, event_col, AGE_COL):
        if required not in data.columns:
            out["note_baseline"] = f"missing_column:{required}"
            return out

    sub = data[[feature, n_obs_feature, duration_col, event_col, AGE_COL]].copy()
    sub = sub.dropna(subset=[duration_col, event_col, AGE_COL])
    out["n_patients_baseline"] = int(len(sub))
    out["n_events_baseline"] = int(sub[event_col].sum()) if len(sub) else 0
    if sub.empty:
        out["note_baseline"] = "no_rows_with_outcomes"
        return out
    if int(sub[feature].notna().sum()) == 0:
        out["note_baseline"] = "no_non_missing_feature_rows"
        return out
    if int(sub[n_obs_feature].notna().sum()) == 0:
        out["note_baseline"] = "no_non_missing_n_obs_rows"
        return out
    if out["n_events_baseline"] < min_events_per_feature:
        out["note_baseline"] = f"too_few_events_lt_{min_events_per_feature}"
        return out

    feature_missing = sub[feature].isna().astype(float).to_numpy()
    include_feature_missing = bool(np.unique(feature_missing).size > 1)

    feature_vals = SimpleImputer(strategy="mean").fit_transform(sub[[feature]]).reshape(-1)
    n_obs_vals = SimpleImputer(strategy="mean").fit_transform(sub[[n_obs_feature]]).reshape(-1)
    feature_z, feature_sd = _zscore(feature_vals)
    if feature_sd <= 0 or not np.isfinite(feature_sd):
        out["note_baseline"] = "feature_has_no_variation"
        return out
    n_obs_z, n_obs_sd = _zscore(n_obs_vals)
    if n_obs_sd <= 0 or not np.isfinite(n_obs_sd):
        out["note_baseline"] = "n_obs_has_no_variation"
        return out
    age_vals = sub[AGE_COL].to_numpy(dtype=float)
    age_z, age_sd = _zscore(age_vals)
    if age_sd <= 0 or not np.isfinite(age_sd):
        age_z = age_vals - float(np.mean(age_vals))

    model_df = pd.DataFrame(
        {
            "feature_z": feature_z,
            "n_obs_z": n_obs_z,
            "age": age_z,
            duration_col: sub[duration_col].to_numpy(dtype=float),
            event_col: sub[event_col].astype(int).to_numpy(),
        },
        index=sub.index,
    )
    if include_feature_missing:
        model_df["feature_missing"] = feature_missing

    model, used_penalizer, note = fit_cox_with_fallback(
        model_df,
        duration_col=duration_col,
        event_col=event_col,
        penalizers=[0.0, fallback_penalizer],
        l1_ratio=0.0,
    )
    out["fit_penalizer_baseline"] = used_penalizer
    out["note_baseline"] = note
    if model is None:
        return out

    s = model.summary
    if "feature_z" in s.index:
        row = s.loc["feature_z"]
        out["coef_feature_baseline"] = float(row["coef"])
        out["hazard_ratio_baseline_per_sd"] = float(row["exp(coef)"])
        out["ci_lower_baseline"] = float(row["exp(coef) lower 95%"])
        out["ci_upper_baseline"] = float(row["exp(coef) upper 95%"])
        out["p_value_baseline"] = float(row["p"])
    return out


def fit_pgs_only(
    data: pd.DataFrame,
    *,
    pgs: str,
    endpoint: str,
    fallback_penalizer: float,
    min_events_per_feature: int,
) -> dict:
    """Cox on [z(AGE) + z(PGS)] — no lab term, no n_obs term.

    Cohort matches fit_baseline_lab / fit_one (rows with valid outcomes + age).
    PGS missingness is mean-imputed with a missing indicator, mirroring fit_one.
    Run once per (endpoint, pgs); the result is broadcast across the target-lab
    rows in run_for_landmark since the fit doesn't depend on which lab is being
    evaluated against PGS.
    """
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]

    out = {
        "coef_pgs_alone": np.nan,
        "hazard_ratio_pgs_alone_per_sd": np.nan,
        "ci_lower_pgs_alone": np.nan,
        "ci_upper_pgs_alone": np.nan,
        "p_value_pgs_alone": np.nan,
        "coef_age_pgs_alone": np.nan,
        "p_value_age_pgs_alone": np.nan,
        "coef_pgs_missing_alone": np.nan,
        "p_value_pgs_missing_alone": np.nan,
        "n_patients_pgs_alone": 0,
        "n_patients_pgs_observed_alone": 0,
        "n_patients_pgs_imputed_alone": 0,
        "n_events_pgs_alone": 0,
        "fit_penalizer_pgs_alone": np.nan,
        "note_pgs_alone": "",
    }
    for required in (pgs, duration_col, event_col, AGE_COL):
        if required not in data.columns:
            out["note_pgs_alone"] = f"missing_column:{required}"
            return out

    sub = data[[pgs, duration_col, event_col, AGE_COL]].copy()
    sub = sub.dropna(subset=[duration_col, event_col, AGE_COL])
    out["n_patients_pgs_alone"] = int(len(sub))
    out["n_events_pgs_alone"] = int(sub[event_col].sum()) if len(sub) else 0
    out["n_patients_pgs_observed_alone"] = int(sub[pgs].notna().sum()) if len(sub) else 0
    out["n_patients_pgs_imputed_alone"] = int(len(sub) - out["n_patients_pgs_observed_alone"])
    if sub.empty:
        out["note_pgs_alone"] = "no_rows_with_outcomes"
        return out
    if out["n_patients_pgs_observed_alone"] == 0:
        out["note_pgs_alone"] = "no_non_missing_pgs_rows"
        return out
    if out["n_events_pgs_alone"] < min_events_per_feature:
        out["note_pgs_alone"] = f"too_few_events_lt_{min_events_per_feature}"
        return out

    pgs_missing = sub[pgs].isna().astype(float).to_numpy()
    include_pgs_missing = bool(np.unique(pgs_missing).size > 1)

    pgs_vals = SimpleImputer(strategy="mean").fit_transform(sub[[pgs]]).reshape(-1)
    pgs_z, pgs_sd = _zscore(pgs_vals)
    if pgs_sd <= 0 or not np.isfinite(pgs_sd):
        out["note_pgs_alone"] = "pgs_has_no_variation"
        return out

    age_vals = sub[AGE_COL].to_numpy(dtype=float)
    age_z, age_sd = _zscore(age_vals)
    if age_sd <= 0 or not np.isfinite(age_sd):
        age_z = age_vals - float(np.mean(age_vals))

    model_df = pd.DataFrame(
        {
            "pgs_z": pgs_z,
            "age": age_z,
            duration_col: sub[duration_col].to_numpy(dtype=float),
            event_col: sub[event_col].astype(int).to_numpy(),
        },
        index=sub.index,
    )
    if include_pgs_missing:
        model_df["pgs_missing"] = pgs_missing

    model, used_penalizer, note = fit_cox_with_fallback(
        model_df,
        duration_col=duration_col,
        event_col=event_col,
        penalizers=[0.0, fallback_penalizer],
        l1_ratio=0.0,
    )
    out["fit_penalizer_pgs_alone"] = used_penalizer
    out["note_pgs_alone"] = note
    if model is None:
        return out

    s = model.summary
    if "pgs_z" in s.index:
        row = s.loc["pgs_z"]
        out["coef_pgs_alone"] = float(row["coef"])
        out["hazard_ratio_pgs_alone_per_sd"] = float(row["exp(coef)"])
        out["ci_lower_pgs_alone"] = float(row["exp(coef) lower 95%"])
        out["ci_upper_pgs_alone"] = float(row["exp(coef) upper 95%"])
        out["p_value_pgs_alone"] = float(row["p"])
    if "age" in s.index:
        row = s.loc["age"]
        out["coef_age_pgs_alone"] = float(row["coef"])
        out["p_value_age_pgs_alone"] = float(row["p"])
    if include_pgs_missing and "pgs_missing" in s.index:
        row = s.loc["pgs_missing"]
        out["coef_pgs_missing_alone"] = float(row["coef"])
        out["p_value_pgs_missing_alone"] = float(row["p"])
    return out


def fit_one(
    data: pd.DataFrame,
    *,
    feature: str,
    pgs: str,
    endpoint: str,
    fallback_penalizer: float,
    min_events_per_feature: int,
) -> dict:
    duration_col = ENDPOINTS[endpoint]["duration_col"]
    event_col = ENDPOINTS[endpoint]["event_col"]
    n_obs_feature = matching_n_obs_feature(feature)

    result = {
        "endpoint": endpoint,
        "lab_name": feature.split("__", 1)[0],
        "feature": feature,
        "n_obs_feature": n_obs_feature,
        "pgs": pgs,
        "n_patients_total": len(data),
        "n_patients_used": 0,
        "n_patients_observed": 0,
        "n_patients_imputed": 0,
        "n_patients_pgs_observed": 0,
        "n_patients_pgs_imputed": 0,
        "n_events_used": 0,
        "coef_feature": np.nan,
        "hazard_ratio_per_sd": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "coef_pgs": np.nan,
        "hazard_ratio_pgs_per_sd": np.nan,
        "ci_lower_pgs": np.nan,
        "ci_upper_pgs": np.nan,
        "p_value_pgs": np.nan,
        "coef_n_obs": np.nan,
        "hazard_ratio_n_obs_per_sd": np.nan,
        "ci_lower_n_obs": np.nan,
        "ci_upper_n_obs": np.nan,
        "p_value_n_obs": np.nan,
        "coef_age": np.nan,
        "p_value_age": np.nan,
        "coef_missing": np.nan,
        "p_value_missing": np.nan,
        "coef_pgs_missing": np.nan,
        "p_value_pgs_missing": np.nan,
        "fit_penalizer": np.nan,
        "note": "",
    }

    for required in (feature, n_obs_feature, pgs, duration_col, event_col, AGE_COL):
        if required not in data.columns:
            result["note"] = f"missing_column:{required}"
            return result

    sub = data[[feature, n_obs_feature, pgs, duration_col, event_col, AGE_COL]].copy()
    sub = sub.dropna(subset=[duration_col, event_col, AGE_COL])
    if sub.empty:
        result["note"] = "no_rows_with_outcomes"
        return result

    n_feature_obs = int(sub[feature].notna().sum())
    n_pgs_obs = int(sub[pgs].notna().sum())
    result["n_patients_used"] = int(len(sub))
    result["n_patients_observed"] = n_feature_obs
    result["n_patients_imputed"] = int(len(sub) - n_feature_obs)
    result["n_patients_pgs_observed"] = n_pgs_obs
    result["n_patients_pgs_imputed"] = int(len(sub) - n_pgs_obs)
    result["n_events_used"] = int(sub[event_col].sum())

    if n_feature_obs == 0:
        result["note"] = "no_non_missing_feature_rows"
        return result
    if n_pgs_obs == 0:
        result["note"] = "no_non_missing_pgs_rows"
        return result
    if int(sub[n_obs_feature].notna().sum()) == 0:
        result["note"] = "no_non_missing_n_obs_rows"
        return result
    if result["n_events_used"] < min_events_per_feature:
        result["note"] = f"too_few_events_lt_{min_events_per_feature}"
        return result

    feature_missing = sub[feature].isna().astype(float).to_numpy()
    pgs_missing = sub[pgs].isna().astype(float).to_numpy()
    include_feature_missing = bool(np.unique(feature_missing).size > 1)
    include_pgs_missing = bool(np.unique(pgs_missing).size > 1)

    feature_vals = SimpleImputer(strategy="mean").fit_transform(sub[[feature]]).reshape(-1)
    n_obs_vals = SimpleImputer(strategy="mean").fit_transform(sub[[n_obs_feature]]).reshape(-1)
    pgs_vals = SimpleImputer(strategy="mean").fit_transform(sub[[pgs]]).reshape(-1)

    feature_z, feature_sd = _zscore(feature_vals)
    if feature_sd <= 0 or not np.isfinite(feature_sd):
        result["note"] = "feature_has_no_variation"
        return result
    n_obs_z, n_obs_sd = _zscore(n_obs_vals)
    if n_obs_sd <= 0 or not np.isfinite(n_obs_sd):
        result["note"] = "n_obs_has_no_variation"
        return result
    pgs_z, pgs_sd = _zscore(pgs_vals)
    if pgs_sd <= 0 or not np.isfinite(pgs_sd):
        result["note"] = "pgs_has_no_variation"
        return result

    age_vals = sub[AGE_COL].to_numpy(dtype=float)
    age_z, age_sd = _zscore(age_vals)
    if age_sd <= 0 or not np.isfinite(age_sd):
        age_z = age_vals - float(np.mean(age_vals))

    model_df = pd.DataFrame(
        {
            "feature_z": feature_z,
            "pgs_z": pgs_z,
            "n_obs_z": n_obs_z,
            "age": age_z,
            duration_col: sub[duration_col].to_numpy(dtype=float),
            event_col: sub[event_col].astype(int).to_numpy(),
        },
        index=sub.index,
    )
    if include_feature_missing:
        model_df["feature_missing"] = feature_missing
    if include_pgs_missing:
        model_df["pgs_missing"] = pgs_missing

    model, used_penalizer, note = fit_cox_with_fallback(
        model_df,
        duration_col=duration_col,
        event_col=event_col,
        penalizers=[0.0, fallback_penalizer],
        l1_ratio=0.0,
    )
    result["fit_penalizer"] = used_penalizer
    result["note"] = note
    if model is None:
        return result

    s = model.summary
    if "feature_z" in s.index:
        row = s.loc["feature_z"]
        result["coef_feature"] = float(row["coef"])
        result["hazard_ratio_per_sd"] = float(row["exp(coef)"])
        result["ci_lower"] = float(row["exp(coef) lower 95%"])
        result["ci_upper"] = float(row["exp(coef) upper 95%"])
        result["p_value"] = float(row["p"])
    if "pgs_z" in s.index:
        row = s.loc["pgs_z"]
        result["coef_pgs"] = float(row["coef"])
        result["hazard_ratio_pgs_per_sd"] = float(row["exp(coef)"])
        result["ci_lower_pgs"] = float(row["exp(coef) lower 95%"])
        result["ci_upper_pgs"] = float(row["exp(coef) upper 95%"])
        result["p_value_pgs"] = float(row["p"])
    if "n_obs_z" in s.index:
        row = s.loc["n_obs_z"]
        result["coef_n_obs"] = float(row["coef"])
        result["hazard_ratio_n_obs_per_sd"] = float(row["exp(coef)"])
        result["ci_lower_n_obs"] = float(row["exp(coef) lower 95%"])
        result["ci_upper_n_obs"] = float(row["exp(coef) upper 95%"])
        result["p_value_n_obs"] = float(row["p"])
    if "age" in s.index:
        row = s.loc["age"]
        result["coef_age"] = float(row["coef"])
        result["p_value_age"] = float(row["p"])
    if include_feature_missing and "feature_missing" in s.index:
        row = s.loc["feature_missing"]
        result["coef_missing"] = float(row["coef"])
        result["p_value_missing"] = float(row["p"])
    if include_pgs_missing and "pgs_missing" in s.index:
        row = s.loc["pgs_missing"]
        result["coef_pgs_missing"] = float(row["coef"])
        result["p_value_pgs_missing"] = float(row["p"])
    return result


def run_for_landmark(
    *,
    landmark: int,
    aggregated_path: Path,
    germline: pd.DataFrame,
    pgs_cols: list[str],
    target_features: list[str],
    endpoints: list[str],
    output_path: Path,
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> None:
    print(f"\n=== landmark={landmark} ===")
    print(f"Reading aggregated: {aggregated_path}")
    agg = load_aggregated(aggregated_path)
    n_agg = len(agg)

    overlap = agg.index.intersection(germline.index)
    n_overlap = len(overlap)
    print(
        f"Cohort: aggregated={n_agg} germline={len(germline)} "
        f"intersection={n_overlap} (drop {n_agg - n_overlap} for missing germline)"
    )
    if n_overlap == 0:
        raise RuntimeError("No MRN overlap between aggregated and germline files.")

    merged = agg.loc[overlap].join(germline.loc[overlap, pgs_cols], how="left")

    missing_target = [f for f in target_features if f not in merged.columns]
    if missing_target:
        raise ValueError(f"Aggregated CSV missing target features: {missing_target}")

    # PGS-alone fits don't depend on which target_lab we're comparing against, so
    # compute once per (endpoint, pgs) and broadcast across the target-lab rows.
    pgs_only_cache: dict[tuple[str, str], dict] = {}
    for endpoint in endpoints:
        for pgs in pgs_cols:
            pgs_only_cache[(endpoint, pgs)] = fit_pgs_only(
                merged,
                pgs=pgs,
                endpoint=endpoint,
                fallback_penalizer=fallback_penalizer,
                min_events_per_feature=min_events_per_feature,
            )

    rows: list[dict] = []
    for endpoint in endpoints:
        for feature in target_features:
            baseline = fit_baseline_lab(
                merged,
                feature=feature,
                endpoint=endpoint,
                fallback_penalizer=fallback_penalizer,
                min_events_per_feature=min_events_per_feature,
            )
            print(
                f"  baseline {endpoint}/{feature}: "
                f"HR={baseline['hazard_ratio_baseline_per_sd']:.3f} "
                f"p={baseline['p_value_baseline']:.2e} "
                f"n={baseline['n_patients_baseline']} "
                f"events={baseline['n_events_baseline']} "
                f"({baseline['note_baseline']})"
            )
            for pgs in pgs_cols:
                row = fit_one(
                    merged,
                    feature=feature,
                    pgs=pgs,
                    endpoint=endpoint,
                    fallback_penalizer=fallback_penalizer,
                    min_events_per_feature=min_events_per_feature,
                )
                row["landmark_days"] = landmark
                row.update(baseline)
                row.update(pgs_only_cache[(endpoint, pgs)])
                row["coef_feature_delta"] = (
                    row["coef_feature"] - baseline["coef_feature_baseline"]
                    if np.isfinite(row["coef_feature"])
                    and np.isfinite(baseline["coef_feature_baseline"])
                    else np.nan
                )
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No rows produced.")
        return

    # BH-adjust PGS and lab-feature p-values within each (endpoint, lab_name) sweep.
    # The lab-feature p-value varies across PGS rows because each fit adjusts for
    # a different PGS, so its q reflects sensitivity to PGS choice. The PGS-alone
    # p-value is constant across the two target_lab rows but BH is still applied
    # within the per-lab group so each output slice carries a self-consistent q.
    out["q_value_pgs"] = np.nan
    out["q_value"] = np.nan
    out["q_value_pgs_alone"] = np.nan
    for (endpoint, lab_name), idx in out.groupby(["endpoint", "lab_name"]).groups.items():
        out.loc[idx, "q_value_pgs"] = benjamini_hochberg(out.loc[idx, "p_value_pgs"]).values
        out.loc[idx, "q_value"] = benjamini_hochberg(out.loc[idx, "p_value"]).values
        out.loc[idx, "q_value_pgs_alone"] = benjamini_hochberg(
            out.loc[idx, "p_value_pgs_alone"]
        ).values

    out = out.sort_values(["endpoint", "lab_name", "p_value_pgs"], na_position="last")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path}  ({len(out)} rows)")

    for endpoint in endpoints:
        for lab_name in sorted({f.split('__', 1)[0] for f in target_features}):
            top = (
                out[(out["endpoint"] == endpoint) & (out["lab_name"] == lab_name)]
                .dropna(subset=["p_value_pgs"])
                .sort_values("p_value_pgs")
                .head(5)
            )
            if top.empty:
                continue
            print(f"  Top PGS for {endpoint} / {lab_name}:")
            cols = [
                "pgs",
                "n_patients_used",
                "n_events_used",
                "hazard_ratio_pgs_per_sd",
                "p_value_pgs",
                "q_value_pgs",
                "hazard_ratio_pgs_alone_per_sd",
                "p_value_pgs_alone",
                "q_value_pgs_alone",
                "hazard_ratio_baseline_per_sd",
                "hazard_ratio_per_sd",
                "p_value",
                "q_value",
                "coef_feature_delta",
            ]
            print(top[cols].to_string(index=False))


def main(args: argparse.Namespace) -> None:
    require_lifelines()

    landmarks = normalize_landmark_days(args.landmarks)
    endpoints = normalize_endpoints(args.endpoints)
    target_features = [f"{lab}__{args.feature_stat}" for lab in args.target_labs]
    print(f"Landmarks: {landmarks}")
    print(f"Endpoints: {endpoints}")
    print(f"Target lab features: {target_features}")

    germline_path = Path(args.germline_path)
    print(f"Loading germline: {germline_path}")
    germline = load_germline(germline_path)
    pgs_cols = select_pgs_columns(germline, PGS_NAME_TOKENS)
    if not pgs_cols:
        raise RuntimeError(f"No PGS columns in {germline_path} match tokens {PGS_NAME_TOKENS}.")
    print(f"Selected {len(pgs_cols)} PGS columns matching {PGS_NAME_TOKENS}:")
    for c in pgs_cols:
        print(f"  {c}")

    output_root = Path(args.output_dir)
    for landmark in landmarks:
        agg_path = Path(args.aggregated_csv_pattern.format(landmark=landmark))
        out_path = output_root / "cox" / f"landmark_{landmark}" / "both" / OUTPUT_FILENAME
        run_for_landmark(
            landmark=landmark,
            aggregated_path=agg_path,
            germline=germline,
            pgs_cols=pgs_cols,
            target_features=target_features,
            endpoints=endpoints,
            output_path=out_path,
            min_events_per_feature=args.min_events_per_feature,
            fallback_penalizer=args.univariate_penalizer,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregated-csv-pattern",
        default=DEFAULT_AGGREGATED_PATTERN,
        help="Path with '{landmark}' placeholder for the aggregated_landmark{N}.csv files.",
    )
    parser.add_argument("--germline-path", default=DEFAULT_GERMLINE_PATH)
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument("--landmarks", nargs="+", type=int, default=[0, 90])
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["platinum", "death"],
        choices=list(ENDPOINTS),
    )
    parser.add_argument("--target-labs", nargs="+", default=list(DEFAULT_TARGET_LABS))
    parser.add_argument("--feature-stat", default=DEFAULT_FEATURE_STAT)
    parser.add_argument("--min-events-per-feature", type=int, default=10)
    parser.add_argument("--univariate-penalizer", type=float, default=0.05)
    main(parser.parse_args())