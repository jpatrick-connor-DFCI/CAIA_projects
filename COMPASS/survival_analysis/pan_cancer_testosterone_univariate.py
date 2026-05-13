"""
Pan-cancer univariate Cox for testosterone features vs overall survival.

Sanity check for the time-to-platinum work: are the testosterone aggregations
that drive associations within prostate cancer also generally prognostic across
cancer types? Any pan-cancer signal raises a confounding concern for the
prostate-specific time-to-platinum interpretation.

Cohort: pan-cancer males (GENDER == 'M') from the clinical_text_embedding_project
survival cohort. For each landmark in {0, 90} days post-treatment-start, fits
univariate Cox per testosterone feature:

    Surv(tt_death, death) ~ z(AGE) + cancer_type_dummies
                          + z(Testosterone__n_observations) + z(Testosterone__feat)
                          [+ feature_missing if any]

mirroring cox_aggregated.run_univariate_nobs_adjusted_associations, except the
cohort is pan-cancer males and the cancer-type dummies stand in for the
prostate-only setting.

Inputs (defaults set for cluster paths used by the clinical_text_embedding_project
preprocessing pipeline):
  * OUTPT_LAB_RESULTS_LABS.csv          raw pan-cancer labs (DFCI_MRN, TEST_TYPE_DESCR,
                                        NUMERIC_RESULT, SPECIMEN_COLLECT_DT)
  * death_met_surv_df.csv.gz            survival cohort (DFCI_MRN, GENDER,
                                        AGE_AT_TREATMENTSTART, tt_death, death)
  * cancer_type_df.csv.gz               per-MRN cancer type
  * follow_up_vte_df_cohort.csv         per-MRN first_treatment_date

Output:
  <output-dir>/pan_cancer_testosterone_univariate_landmark{N}.csv

Notes / caveats:
  * Testosterone units are NOT canonicalized here (PROFILE has a separate
    consolidation pipeline). Filtering is by TEST_TYPE_DESCR substring match;
    inspect the descriptors actually present before trusting effect sizes.
  * The sentinel value 9999999.00 (consistent with the clinical_text_embedding_project
    preprocessing) is dropped.
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
    benjamini_hochberg,
    build_feature_matrix,
    fit_cox_with_fallback,
    matching_n_obs_feature,
    normalize_landmark_days,
    parse_feature_name,
    require_lifelines,
)

DEFAULT_LABS_CSV = (
    "/data/gusev/USERS/jpconnor/data/PROFILE/OncDRS/ALL_2025_03/OUTPT_LAB_RESULTS_LABS.csv"
)
DEFAULT_SURVIVAL_CSV = (
    "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
    "time-to-event_analysis/death_met_surv_df.csv.gz"
)
DEFAULT_CANCER_TYPE_CSV = (
    "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
    "clinical_and_genomic_features/cancer_type_df.csv.gz"
)
DEFAULT_FIRST_TREATMENT_CSV = (
    "/data/gusev/USERS/jpconnor/data/PROFILE/robust_VTE_pred_project_2025_03_cohort/"
    "data/follow_up_vte_df_cohort.csv"
)
DEFAULT_OUTPUT_DIR = SURVIVAL_DIR / "results" / "pan_cancer_testosterone"

DURATION_COL = "tt_death"
EVENT_COL = "death"
AGE_COL = "AGE_AT_TREATMENTSTART"
CANONICAL_LAB_NAME = "Testosterone"
LAB_SENTINEL = 9999999.00
TESTOSTERONE_NAME_TOKEN = "testosterone"
DEFAULT_MIN_CANCER_TYPE_TOTAL = 30
RARE_CANCER_TYPE_BUCKET = "OTHER"


def load_survival_cohort_males(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"DFCI_MRN", "GENDER", AGE_COL, DURATION_COL, EVENT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.loc[df["GENDER"].astype(str).str.upper().str.startswith("M")].copy()
    df["DFCI_MRN"] = pd.to_numeric(df["DFCI_MRN"], errors="coerce")
    df = df.dropna(subset=["DFCI_MRN", AGE_COL, DURATION_COL, EVENT_COL])
    df["DFCI_MRN"] = df["DFCI_MRN"].astype(int)
    df[DURATION_COL] = df[DURATION_COL].astype(float)
    df[EVENT_COL] = df[EVENT_COL].astype(int)
    df[AGE_COL] = df[AGE_COL].astype(float)
    return df[["DFCI_MRN", AGE_COL, DURATION_COL, EVENT_COL]].drop_duplicates("DFCI_MRN")


def build_cancer_type_dummies(
    cancer_type_path: Path,
    mrn_index: pd.Index,
    *,
    min_total: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Per-MRN cancer-type one-hot DataFrame with rare types collapsed to OTHER.

    Returns (dummies_df indexed by DFCI_MRN, list_of_used_dummy_cols).
    Drops the first level to avoid colinearity in the Cox design matrix.
    """
    ct = pd.read_csv(cancer_type_path)
    if "DFCI_MRN" not in ct.columns:
        raise ValueError(f"{cancer_type_path} missing DFCI_MRN")
    type_col_candidates = [c for c in ct.columns if c != "DFCI_MRN"]
    if "CANCER_TYPE" in ct.columns:
        type_col = "CANCER_TYPE"
    elif len(type_col_candidates) == 1:
        type_col = type_col_candidates[0]
    else:
        # cancer_type_df from the clinical text project is already wide one-hot:
        # detect this and reduce back to long for the rare-merge step.
        ct_cols = [c for c in ct.columns if c.startswith("CANCER_TYPE_")]
        if ct_cols:
            long = ct[["DFCI_MRN"] + ct_cols].melt(
                id_vars="DFCI_MRN", var_name="CANCER_TYPE_COL", value_name="present"
            )
            long = long.loc[long["present"].astype(bool)]
            long["CANCER_TYPE"] = long["CANCER_TYPE_COL"].str.replace("CANCER_TYPE_", "", regex=False)
            ct = long[["DFCI_MRN", "CANCER_TYPE"]].copy()
            type_col = "CANCER_TYPE"
        else:
            raise ValueError(
                f"Cannot infer cancer-type column in {cancer_type_path}; columns={list(ct.columns)}"
            )

    ct = ct.dropna(subset=["DFCI_MRN", type_col])
    ct["DFCI_MRN"] = pd.to_numeric(ct["DFCI_MRN"], errors="coerce").astype("Int64")
    ct = ct.dropna(subset=["DFCI_MRN"])
    ct["DFCI_MRN"] = ct["DFCI_MRN"].astype(int)
    ct = ct.drop_duplicates("DFCI_MRN")
    ct = ct.loc[ct["DFCI_MRN"].isin(mrn_index)].copy()

    counts = ct[type_col].value_counts()
    rare = set(counts.index[counts < min_total])
    if rare:
        print(
            f"Merging {len(rare)} rare cancer types into {RARE_CANCER_TYPE_BUCKET} "
            f"(threshold={min_total}): {sorted(rare)}"
        )
    ct[type_col] = ct[type_col].where(~ct[type_col].isin(rare), RARE_CANCER_TYPE_BUCKET)

    dummies = pd.get_dummies(ct.set_index("DFCI_MRN")[type_col], prefix="CANCER_TYPE", drop_first=True)
    dummies = dummies.astype(float)
    return dummies, list(dummies.columns)


def load_first_treatment(path: Path, mrn_index: pd.Index) -> pd.Series:
    df = pd.read_csv(path, usecols=["DFCI_MRN", "first_treatment_date"])
    df["DFCI_MRN"] = pd.to_numeric(df["DFCI_MRN"], errors="coerce")
    df = df.dropna(subset=["DFCI_MRN", "first_treatment_date"])
    df["DFCI_MRN"] = df["DFCI_MRN"].astype(int)
    df = df.drop_duplicates("DFCI_MRN")
    df["first_treatment_date"] = pd.to_datetime(df["first_treatment_date"], errors="coerce")
    df = df.dropna(subset=["first_treatment_date"])
    return df.loc[df["DFCI_MRN"].isin(mrn_index)].set_index("DFCI_MRN")["first_treatment_date"]


def load_testosterone_long(
    labs_path: Path,
    mrn_keep: pd.Index,
    *,
    name_token: str,
) -> pd.DataFrame:
    """Return (DFCI_MRN, LAB_NAME='Testosterone', LAB_VALUE, LAB_DATE) for males."""
    df = pd.read_csv(
        labs_path,
        usecols=["DFCI_MRN", "TEST_TYPE_DESCR", "NUMERIC_RESULT", "SPECIMEN_COLLECT_DT"],
    )
    df["DFCI_MRN"] = pd.to_numeric(df["DFCI_MRN"], errors="coerce")
    df = df.dropna(subset=["DFCI_MRN", "TEST_TYPE_DESCR", "NUMERIC_RESULT", "SPECIMEN_COLLECT_DT"])
    df["DFCI_MRN"] = df["DFCI_MRN"].astype(int)
    df = df.loc[df["DFCI_MRN"].isin(mrn_keep)]
    df = df.loc[df["NUMERIC_RESULT"] != LAB_SENTINEL]

    mask = df["TEST_TYPE_DESCR"].astype(str).str.contains(name_token, case=False, na=False)
    df = df.loc[mask].copy()
    descrs = df["TEST_TYPE_DESCR"].value_counts()
    print(
        f"Filtered to {len(df)} testosterone lab rows from "
        f"{df['DFCI_MRN'].nunique()} male patients across "
        f"{len(descrs)} distinct TEST_TYPE_DESCR values:"
    )
    for descr, n in descrs.items():
        print(f"  {descr}: {n}")

    df = df.rename(
        columns={
            "NUMERIC_RESULT": "LAB_VALUE",
            "SPECIMEN_COLLECT_DT": "LAB_DATE",
        }
    )
    df["LAB_NAME"] = CANONICAL_LAB_NAME
    df["LAB_DATE"] = pd.to_datetime(df["LAB_DATE"], errors="coerce")
    df = df.dropna(subset=["LAB_DATE"])
    return df[["DFCI_MRN", "LAB_NAME", "LAB_VALUE", "LAB_DATE"]]


def build_features_for_landmark(
    labs_long: pd.DataFrame,
    *,
    first_treatment: pd.Series,
    landmark_offset_days: int,
) -> pd.DataFrame:
    """Aggregate testosterone features per MRN through landmark + offset.

    Uses cox_aggregated.build_feature_matrix so feature definitions exactly
    mirror the time-to-platinum pipeline.
    """
    working = labs_long.copy()
    first_record = working.groupby("DFCI_MRN")["LAB_DATE"].transform("min")
    working["FIRST_RECORD_DATE"] = first_record
    working["t_lab"] = (working["LAB_DATE"] - working["FIRST_RECORD_DATE"]).dt.days.astype(float)

    first_record_per_mrn = working.groupby("DFCI_MRN")["FIRST_RECORD_DATE"].first()
    treatment_offset_days = (
        first_treatment.reindex(first_record_per_mrn.index) - first_record_per_mrn
    ).dt.days.astype(float)
    treatment_offset_days = treatment_offset_days.dropna()

    working = working.loc[working["DFCI_MRN"].isin(treatment_offset_days.index)].copy()
    features = build_feature_matrix(
        working,
        landmark_offset_days=landmark_offset_days,
        anchor_col="t_first_treatment",
        anchor_series=treatment_offset_days,
    )
    return features


def _zscore(values: np.ndarray) -> tuple[np.ndarray, float]:
    sd = float(np.std(values, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return values, sd
    return (values - float(np.mean(values))) / sd, sd


def fit_univariate(
    data: pd.DataFrame,
    *,
    feature: str,
    cancer_type_cols: list[str],
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> dict:
    lab_name, feature_stat = parse_feature_name(feature)
    n_obs_feature = matching_n_obs_feature(feature)

    result = {
        "feature": feature,
        "lab_name": lab_name,
        "feature_stat": feature_stat,
        "n_obs_feature": n_obs_feature,
        "n_patients_total": len(data),
        "n_patients_used": 0,
        "n_patients_observed": 0,
        "n_patients_imputed": 0,
        "n_events_used": 0,
        "coef_feature": np.nan,
        "hazard_ratio_per_sd": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "coef_n_obs": np.nan,
        "hazard_ratio_n_obs_per_sd": np.nan,
        "p_value_n_obs": np.nan,
        "coef_age": np.nan,
        "p_value_age": np.nan,
        "coef_missing": np.nan,
        "p_value_missing": np.nan,
        "fit_penalizer": np.nan,
        "note": "",
    }

    if feature_stat == "n_observations":
        result["note"] = "target_is_n_observations"
        return result

    needed = [feature, n_obs_feature, DURATION_COL, EVENT_COL, AGE_COL] + cancer_type_cols
    missing_cols = [c for c in needed if c not in data.columns]
    if missing_cols:
        result["note"] = f"missing_columns:{missing_cols[:3]}"
        return result

    sub = data[needed].copy()
    sub = sub.dropna(subset=[DURATION_COL, EVENT_COL, AGE_COL])
    result["n_patients_used"] = int(len(sub))
    result["n_patients_observed"] = int(sub[feature].notna().sum())
    result["n_patients_imputed"] = int(len(sub) - result["n_patients_observed"])
    result["n_events_used"] = int(sub[EVENT_COL].sum()) if len(sub) else 0

    if sub.empty:
        result["note"] = "no_rows_with_outcomes"
        return result
    if result["n_patients_observed"] == 0:
        result["note"] = "no_non_missing_feature_rows"
        return result
    if int(sub[n_obs_feature].notna().sum()) == 0:
        result["note"] = "no_non_missing_n_obs_rows"
        return result
    if result["n_events_used"] < min_events_per_feature:
        result["note"] = f"too_few_events_lt_{min_events_per_feature}"
        return result

    feature_missing = sub[feature].isna().astype(float).to_numpy()
    include_feature_missing = bool(np.unique(feature_missing).size > 1)

    feature_vals = SimpleImputer(strategy="mean").fit_transform(sub[[feature]]).reshape(-1)
    n_obs_vals = SimpleImputer(strategy="mean").fit_transform(sub[[n_obs_feature]]).reshape(-1)
    feature_z, feature_sd = _zscore(feature_vals)
    if feature_sd <= 0 or not np.isfinite(feature_sd):
        result["note"] = "feature_has_no_variation"
        return result
    n_obs_z, n_obs_sd = _zscore(n_obs_vals)
    if n_obs_sd <= 0 or not np.isfinite(n_obs_sd):
        result["note"] = "n_obs_has_no_variation"
        return result
    age_vals = sub[AGE_COL].to_numpy(dtype=float)
    age_z, age_sd = _zscore(age_vals)
    if age_sd <= 0 or not np.isfinite(age_sd):
        age_z = age_vals - float(np.mean(age_vals))

    model_df = pd.DataFrame(
        {
            "feature_z": feature_z,
            "n_obs_z": n_obs_z,
            "age": age_z,
            DURATION_COL: sub[DURATION_COL].to_numpy(dtype=float),
            EVENT_COL: sub[EVENT_COL].astype(int).to_numpy(),
        },
        index=sub.index,
    )
    for ct_col in cancer_type_cols:
        col_vals = sub[ct_col].astype(float).fillna(0.0).to_numpy()
        if np.unique(col_vals).size > 1:
            model_df[ct_col] = col_vals
    if include_feature_missing:
        model_df["feature_missing"] = feature_missing

    model, used_penalizer, note = fit_cox_with_fallback(
        model_df,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
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
    if "n_obs_z" in s.index:
        row = s.loc["n_obs_z"]
        result["coef_n_obs"] = float(row["coef"])
        result["hazard_ratio_n_obs_per_sd"] = float(row["exp(coef)"])
        result["p_value_n_obs"] = float(row["p"])
    if "age" in s.index:
        row = s.loc["age"]
        result["coef_age"] = float(row["coef"])
        result["p_value_age"] = float(row["p"])
    if include_feature_missing and "feature_missing" in s.index:
        row = s.loc["feature_missing"]
        result["coef_missing"] = float(row["coef"])
        result["p_value_missing"] = float(row["p"])
    return result


def run_for_landmark(
    *,
    landmark: int,
    labs_long: pd.DataFrame,
    first_treatment: pd.Series,
    survival: pd.DataFrame,
    cancer_dummies: pd.DataFrame,
    cancer_type_cols: list[str],
    output_path: Path,
    min_events_per_feature: int,
    fallback_penalizer: float,
) -> None:
    print(f"\n=== landmark={landmark} ===")
    features = build_features_for_landmark(
        labs_long,
        first_treatment=first_treatment,
        landmark_offset_days=landmark,
    )
    print(
        f"Feature matrix: {features.shape[0]} patients x {features.shape[1]} testosterone features"
    )

    merged = (
        survival.set_index("DFCI_MRN")
        .join(features, how="inner")
        .join(cancer_dummies, how="left")
    )
    if merged.empty:
        print("No patients after merging survival x features x cancer-type. Skipping.")
        return
    # patients with no cancer-type assignment fall into the dropped reference level
    merged[cancer_type_cols] = merged[cancer_type_cols].fillna(0.0)
    print(
        f"Merged cohort: {len(merged)} male patients, "
        f"{int(merged[EVENT_COL].sum())} deaths, "
        f"{len(cancer_type_cols)} non-reference cancer-type dummies"
    )

    feature_cols = [c for c in features.columns if not c.endswith("__n_observations")]
    rows = []
    for feature in feature_cols:
        rows.append(
            fit_univariate(
                merged,
                feature=feature,
                cancer_type_cols=cancer_type_cols,
                min_events_per_feature=min_events_per_feature,
                fallback_penalizer=fallback_penalizer,
            )
        )

    out = pd.DataFrame(rows)
    out.insert(0, "landmark_days", landmark)
    out["q_value"] = benjamini_hochberg(out["p_value"])
    out = out.sort_values("p_value", na_position="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path}  ({len(out)} rows)")
    top = out.dropna(subset=["p_value"]).head(10)
    if not top.empty:
        cols = [
            "feature_stat",
            "n_patients_used",
            "n_patients_observed",
            "n_events_used",
            "hazard_ratio_per_sd",
            "ci_lower",
            "ci_upper",
            "p_value",
            "q_value",
        ]
        print(top[cols].to_string(index=False))


def main(args: argparse.Namespace) -> None:
    require_lifelines()

    landmarks = normalize_landmark_days(args.landmarks)
    print(f"Landmarks: {landmarks}")

    survival = load_survival_cohort_males(Path(args.survival_csv))
    print(f"Loaded {len(survival)} male patients with outcomes from {args.survival_csv}")
    mrn_set = pd.Index(survival["DFCI_MRN"].unique())

    cancer_dummies, cancer_type_cols = build_cancer_type_dummies(
        Path(args.cancer_type_csv),
        mrn_set,
        min_total=args.min_cancer_type_total,
    )
    print(
        f"Cancer-type dummies: {len(cancer_type_cols)} columns "
        f"covering {len(cancer_dummies)} patients"
    )

    first_treatment = load_first_treatment(Path(args.first_treatment_csv), mrn_set)
    print(f"first_treatment_date available for {len(first_treatment)} patients")

    labs_long = load_testosterone_long(
        Path(args.labs_csv),
        first_treatment.index,
        name_token=args.name_token,
    )

    output_dir = Path(args.output_dir)
    for landmark in landmarks:
        out_path = output_dir / f"pan_cancer_testosterone_univariate_landmark{landmark}.csv"
        run_for_landmark(
            landmark=landmark,
            labs_long=labs_long,
            first_treatment=first_treatment,
            survival=survival,
            cancer_dummies=cancer_dummies,
            cancer_type_cols=cancer_type_cols,
            output_path=out_path,
            min_events_per_feature=args.min_events_per_feature,
            fallback_penalizer=args.fallback_penalizer,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labs-csv", default=DEFAULT_LABS_CSV)
    parser.add_argument("--survival-csv", default=DEFAULT_SURVIVAL_CSV)
    parser.add_argument("--cancer-type-csv", default=DEFAULT_CANCER_TYPE_CSV)
    parser.add_argument("--first-treatment-csv", default=DEFAULT_FIRST_TREATMENT_CSV)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--landmarks", nargs="+", type=int, default=[0, 90])
    parser.add_argument("--name-token", default=TESTOSTERONE_NAME_TOKEN)
    parser.add_argument("--min-cancer-type-total", type=int, default=DEFAULT_MIN_CANCER_TYPE_TOTAL)
    parser.add_argument("--min-events-per-feature", type=int, default=10)
    parser.add_argument("--fallback-penalizer", type=float, default=0.05)
    main(parser.parse_args())
