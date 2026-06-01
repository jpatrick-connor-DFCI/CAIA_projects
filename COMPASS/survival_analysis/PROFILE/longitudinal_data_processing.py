from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent           # .../survival_analysis/PROFILE
SURVIVAL_PARENT = SURVIVAL_DIR.parent                    # .../survival_analysis
for _p in (str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from helpers.consolidate_dfci_labs import consolidate_dfci_labs  # noqa: E402

ID_COL = "DFCI_MRN"
AGE_COL = "AGE_AT_TREATMENTSTART"


DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data")
EMBED_PROJ_PATH = DATA_ROOT / "clinical_text_embedding_project"
NEPC_PROJ_PATH = DATA_ROOT / "CAIA" / "COMPASS"
PROFILE_PATH = Path("/data/gusev/PROFILE/CLINICAL")
SURV_PATH = EMBED_PROJ_PATH / "time-to-event_analysis"

PROFILE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_CSV = PROFILE_DIR / "OMOP_to_DFCI_lab_ids.csv"
DEFAULT_UNIQUE_LABS_CSV = PROFILE_DIR / "unique_lab_ids_w_units.csv"

PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}
PARPI_MEDS = {"OLAPARIB", "RUCAPARIB", "NIRAPARIB", "TALAZOPARIB", "VELIPARIB"}
MIN_PSA_COUNT = 5


def generate_new_test_name(code: object, descr: object) -> str:
    if pd.isna(code):
        return str(descr)
    if code == descr:
        return str(code)
    return f"{code} ({descr})"


def build_raw_longitudinal_data(
    health_df: pd.DataFrame,
    labs_df: pd.DataFrame,
) -> pd.DataFrame:
    vital_signs_df = health_df.loc[
        health_df["CODE_TYPE"] == "Vital Signs",
        [ID_COL, "START_DT", "HEALTH_HISTORY_TYPE", "RESULTS", "UNITS_CD"],
    ].copy()

    labs_df_col_sub = labs_df[
        [
            ID_COL,
            "SPECIMEN_COLLECT_DT",
            "TEST_TYPE_CD",
            "TEST_TYPE_DESCR",
            "NUMERIC_RESULT",
            "RESULT_UOM_NM",
        ]
    ].copy()

    labs_df_col_sub["TEST_NAME"] = labs_df_col_sub.apply(
        lambda row: generate_new_test_name(row["TEST_TYPE_CD"], row["TEST_TYPE_DESCR"]),
        axis=1,
    )

    vital_signs_df = (
        vital_signs_df.rename(
            columns={
                "START_DT": "DATE",
                "RESULTS": "LAB_VALUE",
                "HEALTH_HISTORY_TYPE": "LAB_NAME",
                "UNITS_CD": "LAB_UNIT",
            }
        )[[ID_COL, "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE"]]
    )

    labs_df_col_sub = (
        labs_df_col_sub.rename(
            columns={
                "SPECIMEN_COLLECT_DT": "DATE",
                "NUMERIC_RESULT": "LAB_VALUE",
                "RESULT_UOM_NM": "LAB_UNIT",
                "TEST_NAME": "LAB_NAME",
            }
        )[[ID_COL, "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE"]]
    )

    return pd.concat([vital_signs_df, labs_df_col_sub], ignore_index=True)


def mark_non_prostate_primary_icd(icds: pd.DataFrame) -> pd.DataFrame:
    icds = icds.copy()
    codes = icds["DIAGNOSIS_ICD10_CD"].astype(str).str.upper().str.strip()

    letter = codes.str.extract(r"^([A-Z])", expand=False)
    num = pd.to_numeric(codes.str.extract(r"^[A-Z](\d{2,3})", expand=False), errors="coerce")

    is_c00_c76 = (letter == "C") & (num >= 0) & (num <= 76)
    is_c81_c96 = (letter == "C") & (num >= 81) & (num <= 96)
    is_c97 = codes.str.startswith("C97")
    is_c7a = codes.str.startswith("C7A")
    is_c801 = codes.str.startswith("C801") | codes.str.startswith("C80.1")

    is_primary = is_c00_c76 | is_c81_c96 | is_c97 | is_c7a | is_c801
    is_prostate = codes.str.startswith("C61")
    is_secondary = ((letter == "C") & (num >= 77) & (num <= 79)) | codes.str.startswith("C7B")
    is_nmsc = codes.str.startswith("C44")
    is_nos = codes.str.startswith("C80.9") | codes.str.startswith("C809")

    icds["NON_PROSTATE_PRIMARY_ICD10"] = (
        is_primary
        & ~is_prostate
        & ~is_secondary
        & ~is_nmsc
        & ~is_nos
    )
    return icds


def compute_first_prostate_diagnosis(icds: pd.DataFrame) -> pd.DataFrame:
    icds = mark_non_prostate_primary_icd(icds)
    mrns_to_include = icds.loc[~icds["NON_PROSTATE_PRIMARY_ICD10"], ID_COL].unique()

    return (
        icds.loc[
            (icds["DIAGNOSIS_ICD10_CD"] == "C61") & icds[ID_COL].isin(mrns_to_include),
            [ID_COL, "START_DT"],
        ]
        .groupby(ID_COL, as_index=False)["START_DT"]
        .min()
        .rename(columns={"START_DT": "DIAGNOSIS_DATE"})
    )


def build_longitudinal_prediction_data(
    consolidated_df: pd.DataFrame,
    first_prostate_diagnosis: pd.DataFrame,
    death_df: pd.DataFrame,
    platinum_df: pd.DataFrame,
) -> pd.DataFrame:
    prediction_df = consolidated_df[
        [ID_COL, "DATE", "collapsed_measurement", "numeric_result_standardized"]
    ].dropna().copy()

    # Collapse the platinum table to one row per patient (earliest platinum start).
    # Without this, the left-join below fans out every lab row by the number of
    # platinum medication records the patient has.
    platinum_first = platinum_df[[ID_COL, "medication", "medication_start_time"]].copy()
    platinum_first["_plat_dt"] = pd.to_datetime(
        platinum_first["medication_start_time"], errors="coerce"
    )
    platinum_first = (
        platinum_first.sort_values("_plat_dt")
        .drop_duplicates(subset=ID_COL, keep="first")
        .drop(columns="_plat_dt")
    )

    n_lab_mrns = prediction_df[ID_COL].nunique()
    after_dx = prediction_df.merge(first_prostate_diagnosis, on=ID_COL, how="inner")
    n_after_dx = after_dx[ID_COL].nunique()
    after_death = after_dx.merge(death_df, on=ID_COL, how="inner")
    n_after_death = after_death[ID_COL].nunique()
    print(
        f"[build_longitudinal_prediction_data] patients with labs={n_lab_mrns}; "
        f"after first-prostate-diagnosis inner-join={n_after_dx} "
        f"(dropped {n_lab_mrns - n_after_dx}); "
        f"after death-table inner-join={n_after_death} "
        f"(dropped {n_after_dx - n_after_death})."
    )
    pred_df = after_death.merge(platinum_first, on=ID_COL, how="left")

    pred_df = (
        pred_df.rename(
            columns={
                "collapsed_measurement": "LAB_NAME",
                "numeric_result_standardized": "LAB_VALUE",
                "DATE": "LAB_DATE",
                "first_treatment_date": "FIRST_TREATMENT_DATE",
                "last_contact_date": "LAST_CONTACT_DATE",
                "death": "DEATH",
                "medication": "PLATINUM_MEDICATION",
                "medication_start_time": "PLATINUM_DATE",
            }
        )
    )

    date_cols = [
        "LAB_DATE",
        "DIAGNOSIS_DATE",
        "FIRST_TREATMENT_DATE",
        "LAST_CONTACT_DATE",
        "PLATINUM_DATE",
    ]
    for col in date_cols:
        pred_df[col] = pd.to_datetime(pred_df[col], errors="coerce").dt.floor("D")

    pred_df["DEATH"] = pd.to_numeric(pred_df["DEATH"], errors="coerce").fillna(0).astype(int)
    pred_df["AGE_AT_TREATMENTSTART"] = pd.to_numeric(
        pred_df["AGE_AT_TREATMENTSTART"],
        errors="coerce",
    )
    pred_df["FIRST_TREATMENT"] = pred_df["FIRST_TREATMENT_DATE"].notna().astype(int)
    pred_df["PLATINUM"] = (
        pred_df["PLATINUM_MEDICATION"]
        .astype(str)
        .str.upper()
        .isin(PLATINUM_MEDS)
        .astype(int)
    )
    pred_df["PLATINUM_DATE"] = pred_df["PLATINUM_DATE"].fillna(pred_df["LAST_CONTACT_DATE"])

    first_lab_date = pred_df.groupby(ID_COL)["LAB_DATE"].transform("min")
    pred_df["FIRST_RECORD_DATE"] = pd.concat(
        [first_lab_date, pred_df["DIAGNOSIS_DATE"], pred_df["FIRST_TREATMENT_DATE"]],
        axis=1,
    ).min(axis=1)
    pred_df["t_lab"] = (pred_df["LAB_DATE"] - pred_df["FIRST_RECORD_DATE"]).dt.days.astype(float)
    pred_df["t_last_contact"] = (
        pred_df["LAST_CONTACT_DATE"] - pred_df["FIRST_RECORD_DATE"]
    ).dt.days.astype(float)
    # NOTE: the death source table (death_met_surv_df) carries only `last_contact_date`
    # and a `death` flag — there is no true date-of-death. We therefore use last-contact
    # time as the death-endpoint duration for everyone. The DEATH *event* indicator is
    # real, but dead and censored patients share the same duration. Replace this with a
    # true death date if one becomes available.
    pred_df["t_death"] = pred_df["t_last_contact"]

    pred_df["t_diagnosis"] = (
        (pred_df["DIAGNOSIS_DATE"] - pred_df["FIRST_RECORD_DATE"]).dt.days.astype(float)
    )
    first_treatment_days = (
        pred_df["FIRST_TREATMENT_DATE"] - pred_df["FIRST_RECORD_DATE"]
    ).dt.days
    platinum_days = (pred_df["PLATINUM_DATE"] - pred_df["FIRST_RECORD_DATE"]).dt.days

    pred_df["t_first_treatment"] = np.where(
        pred_df["FIRST_TREATMENT"].eq(1),
        first_treatment_days,
        pred_df["t_last_contact"],
    ).astype(float)
    pred_df["t_platinum"] = np.where(
        pred_df["PLATINUM"].eq(1),
        platinum_days,
        pred_df["t_last_contact"],
    ).astype(float)

    ordered_cols = [
        ID_COL,
        "AGE_AT_TREATMENTSTART",
        "FIRST_RECORD_DATE",
        "DIAGNOSIS_DATE",
        "FIRST_TREATMENT_DATE",
        "FIRST_TREATMENT",
        "LAST_CONTACT_DATE",
        "DEATH",
        "PLATINUM_MEDICATION",
        "PLATINUM_DATE",
        "PLATINUM",
        "LAB_DATE",
        "t_lab",
        "t_diagnosis",
        "t_first_treatment",
        "t_platinum",
        "t_last_contact",
        "t_death",
        "LAB_NAME",
        "LAB_VALUE",
    ]
    return pred_df[ordered_cols].copy()


def apply_cohort_filters(
    pred_df: pd.DataFrame,
    *,
    medications_df: pd.DataFrame,
    min_psa_count: int = MIN_PSA_COUNT,
) -> pd.DataFrame:
    """Patient-level inclusion/exclusion on the row-level prediction frame.

    Inclusion:
      - recorded first treatment (FIRST_TREATMENT == 1); never-treated
        patients are excluded so that the "pre-treatment" window used by
        downstream feature engineering is well-defined.
      - >= ``min_psa_count`` PSA rows in the longitudinal prediction frame
        (LAB_NAME == "PSA", after lab consolidation)
    Exclusion:
      - any PARPi exposure, identified from the pre-existing prostate
        medications table via NCI_PREFERRED_MED_NM membership in PARPI_MEDS

    Prostate-diagnosis inclusion and non-prostate-primary exclusion are
    already enforced upstream (ICD-based, via compute_first_prostate_diagnosis).
    """
    n_before = pred_df[ID_COL].nunique()
    print(f"Cohort before treatment/PSA/PARPi filters: {n_before} patients")

    pred_df = pred_df.loc[pred_df["FIRST_TREATMENT"].eq(1)].copy()
    n_after_treated = pred_df[ID_COL].nunique()
    print(f"  First-treatment inclusion: kept {n_after_treated}/{n_before}")

    psa_counts = (
        pred_df.loc[pred_df["LAB_NAME"].eq("PSA")]
        .groupby(ID_COL)
        .size()
    )
    keep_psa = psa_counts.loc[psa_counts >= min_psa_count].index
    pred_df = pred_df.loc[pred_df[ID_COL].isin(keep_psa)].copy()
    n_after_psa = pred_df[ID_COL].nunique()
    print(
        f"  PSA count filter (>= {min_psa_count}): "
        f"kept {n_after_psa}/{n_after_treated}"
    )

    parpi_mrns = set(
        medications_df.loc[
            medications_df["NCI_PREFERRED_MED_NM"].astype(str).str.upper().isin(PARPI_MEDS),
            ID_COL,
        ].unique()
    )
    pred_df = pred_df.loc[~pred_df[ID_COL].isin(parpi_mrns)].copy()
    n_after_parpi = pred_df[ID_COL].nunique()
    print(
        f"  PARPi exclusion: dropped {n_after_psa - n_after_parpi} "
        f"(remaining: {n_after_parpi})"
    )

    return pred_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build row-level longitudinal survival input from the COMPASS/Profile exports.",
    )
    parser.add_argument(
        "--health-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "prostate_health_history_data.csv",
    )
    parser.add_argument(
        "--labs-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "prostate_labs_data.csv",
    )
    parser.add_argument(
        "--icd-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "prostate_icd_data.csv",
    )
    parser.add_argument(
        "--platinum-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "platinum_chemo_records.csv",
    )
    parser.add_argument(
        "--medications-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "prostate_medications_data.csv",
        help="Pre-compiled prostate-cohort medications table (used for PARPi exclusion).",
    )
    parser.add_argument(
        "--death-csv",
        type=Path,
        default=SURV_PATH / "death_met_surv_df.csv.gz",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
    )
    parser.add_argument(
        "--unique-labs-csv",
        type=Path,
        default=DEFAULT_UNIQUE_LABS_CSV,
    )
    parser.add_argument(
        "--uncondensed-output-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "uncondensed_longitudinal_data.csv",
    )
    parser.add_argument(
        "--consolidated-output-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "consolidated_longitudinal_data.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "longitudinal_prediction_data.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for output_path in [
        args.unique_labs_csv,
        args.uncondensed_output_csv,
        args.consolidated_output_csv,
        args.output_csv,
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    health_df = pd.read_csv(args.health_csv)
    labs_df = pd.read_csv(args.labs_csv)
    raw_longitudinal_df = build_raw_longitudinal_data(health_df, labs_df)

    unique_labs_df = (
        raw_longitudinal_df[["LAB_NAME", "LAB_UNIT"]]
        .value_counts()
        .reset_index(name="count")
    )
    unique_labs_df.to_csv(args.unique_labs_csv, index=False)
    raw_longitudinal_df.to_csv(args.uncondensed_output_csv, index=False)

    mapping_df = pd.read_csv(args.mapping_csv)
    consolidated_df = consolidate_dfci_labs(raw_longitudinal_df, mapping_df)
    consolidated_df.to_csv(args.consolidated_output_csv, index=False)

    icds = pd.read_csv(args.icd_csv)
    platinum_df = pd.read_csv(args.platinum_csv)
    death_df = pd.read_csv(args.death_csv)[
        [
            ID_COL,
            "first_treatment_date",
            "last_contact_date",
            "death",
            "AGE_AT_TREATMENTSTART",
        ]
    ].copy()

    first_prostate_diagnosis = compute_first_prostate_diagnosis(icds)

    longitudinal_prediction_df = build_longitudinal_prediction_data(
        consolidated_df,
        first_prostate_diagnosis,
        death_df,
        platinum_df,
    )

    medications_df = pd.read_csv(args.medications_csv, usecols=[ID_COL, "NCI_PREFERRED_MED_NM"])
    longitudinal_prediction_df = apply_cohort_filters(
        longitudinal_prediction_df,
        medications_df=medications_df,
    )

    longitudinal_prediction_df.to_csv(args.output_csv, index=False)

    print(f"Wrote unique lab inventory to {args.unique_labs_csv}")
    print(f"Wrote raw longitudinal rows to {args.uncondensed_output_csv}")
    print(f"Wrote consolidated longitudinal rows to {args.consolidated_output_csv}")
    print(f"Wrote survival-ready longitudinal rows to {args.output_csv}")


if __name__ == "__main__":
    main()
