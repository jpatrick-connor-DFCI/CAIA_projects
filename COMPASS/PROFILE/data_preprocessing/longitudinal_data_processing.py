from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .consolidate_dfci_labs import consolidate_dfci_labs
except ImportError:  # pragma: no cover - supports direct script execution
    from consolidate_dfci_labs import consolidate_dfci_labs


DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data")
EMBED_PROJ_PATH = DATA_ROOT / "clinical_text_embedding_project"
NEPC_PROJ_PATH = DATA_ROOT / "CAIA" / "COMPASS"
PROFILE_PATH = Path("/data/gusev/PROFILE/CLINICAL")
SURV_PATH = EMBED_PROJ_PATH / "time-to-event_analysis"

PROFILE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_CSV = PROFILE_DIR / "OMOP_to_DFCI_lab_ids.csv"
DEFAULT_UNIQUE_LABS_CSV = PROFILE_DIR / "unique_lab_ids_w_units.csv"

PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}


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
        ["DFCI_MRN", "START_DT", "HEALTH_HISTORY_TYPE", "RESULTS", "UNITS_CD"],
    ].copy()

    labs_df_col_sub = labs_df[
        [
            "DFCI_MRN",
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
        )[["DFCI_MRN", "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE"]]
    )

    labs_df_col_sub = (
        labs_df_col_sub.rename(
            columns={
                "SPECIMEN_COLLECT_DT": "DATE",
                "NUMERIC_RESULT": "LAB_VALUE",
                "RESULT_UOM_NM": "LAB_UNIT",
                "TEST_NAME": "LAB_NAME",
            }
        )[["DFCI_MRN", "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE"]]
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
    mrns_to_include = icds.loc[~icds["NON_PROSTATE_PRIMARY_ICD10"], "DFCI_MRN"].unique()

    return (
        icds.loc[
            (icds["DIAGNOSIS_ICD10_CD"] == "C61") & icds["DFCI_MRN"].isin(mrns_to_include),
            ["DFCI_MRN", "START_DT"],
        ]
        .groupby("DFCI_MRN", as_index=False)["START_DT"]
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
        ["DFCI_MRN", "DATE", "collapsed_measurement", "numeric_result_standardized"]
    ].dropna().copy()

    pred_df = (
        prediction_df.merge(first_prostate_diagnosis, on="DFCI_MRN", how="inner")
        .merge(death_df, on="DFCI_MRN", how="inner")
        .merge(
            platinum_df[["DFCI_MRN", "medication", "medication_start_time"]],
            on="DFCI_MRN",
            how="left",
        )
        .rename(
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
    pred_df["DIAGNOSIS"] = pred_df["DIAGNOSIS_DATE"].notna().astype(int)
    pred_df["FIRST_TREATMENT"] = pred_df["FIRST_TREATMENT_DATE"].notna().astype(int)
    pred_df["PLATINUM"] = (
        pred_df["PLATINUM_MEDICATION"]
        .astype(str)
        .str.upper()
        .isin(PLATINUM_MEDS)
        .astype(int)
    )
    pred_df["PLATINUM_DATE"] = pred_df["PLATINUM_DATE"].fillna(pred_df["LAST_CONTACT_DATE"])

    pred_df["FIRST_RECORD_DATE"] = pred_df.groupby("DFCI_MRN")["LAB_DATE"].transform("min")
    pred_df["t_lab"] = (pred_df["LAB_DATE"] - pred_df["FIRST_RECORD_DATE"]).dt.days.astype(float)
    pred_df["t_last_contact"] = (
        pred_df["LAST_CONTACT_DATE"] - pred_df["FIRST_RECORD_DATE"]
    ).dt.days.astype(float)
    pred_df["t_death"] = pred_df["t_last_contact"]

    diagnosis_days = (pred_df["DIAGNOSIS_DATE"] - pred_df["FIRST_RECORD_DATE"]).dt.days
    first_treatment_days = (
        pred_df["FIRST_TREATMENT_DATE"] - pred_df["FIRST_RECORD_DATE"]
    ).dt.days
    platinum_days = (pred_df["PLATINUM_DATE"] - pred_df["FIRST_RECORD_DATE"]).dt.days

    pred_df["t_diagnosis"] = np.where(
        pred_df["DIAGNOSIS"].eq(1),
        diagnosis_days,
        pred_df["t_last_contact"],
    ).astype(float)
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
        "DFCI_MRN",
        "AGE_AT_TREATMENTSTART",
        "FIRST_RECORD_DATE",
        "DIAGNOSIS_DATE",
        "DIAGNOSIS",
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
        "--death-csv",
        type=Path,
        default=SURV_PATH / "death_met_surv_df.csv",
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
            "DFCI_MRN",
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
    longitudinal_prediction_df.to_csv(args.output_csv, index=False)

    print(f"Wrote unique lab inventory to {args.unique_labs_csv}")
    print(f"Wrote raw longitudinal rows to {args.uncondensed_output_csv}")
    print(f"Wrote consolidated longitudinal rows to {args.consolidated_output_csv}")
    print(f"Wrote survival-ready longitudinal rows to {args.output_csv}")


if __name__ == "__main__":
    main()
