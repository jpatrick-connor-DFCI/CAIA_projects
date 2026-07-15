from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SURVIVAL_DIR = PROJECT_DIR / "survival_analysis"
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(REPO_ROOT), str(PROJECT_DIR), str(SURVIVAL_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_preprocessing_common.dfci_labs import (  # noqa: E402
    DEFAULT_MAPPING_CSV,
    consolidate_dfci_labs,
)
from data_preprocessing_common.projects.compass_profile import (  # noqa: E402
    UNIQUE_LABS_CSV as DEFAULT_UNIQUE_LABS_CSV,
)

ID_COL = "DFCI_MRN"
AGE_COL = "AGE_AT_TREATMENTSTART"

DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data")
EMBED_PROJ_PATH = DATA_ROOT / "clinical_text_embedding_project"
NEPC_PROJ_PATH = DATA_ROOT / "CAIA" / "COMPASS"
PROFILE_PATH = Path("/data/gusev/PROFILE/CLINICAL")
SURV_PATH = EMBED_PROJ_PATH / "time-to-event_analysis"

# Per-patient survival cohort produced by compile_COMPASS_cohort_data.py
# straight from the raw OncDRS pull. It supplies the outcome/anchor columns
# (age, treatment anchor, death, last-contact, platinum) that used to come from
# death_met_surv_df.csv.gz. See load_death_df_from_survival_cohort.
DEFAULT_SURVIVAL_COHORT_CSV = NEPC_PROJ_PATH / "prostate_arpi_survival_cohort.csv"

PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}
PARPI_MEDS = {"OLAPARIB", "RUCAPARIB", "NIRAPARIB", "TALAZOPARIB", "VELIPARIB"}
MIN_PSA_COUNT = 5

# Highlighted antineoplastic treatments used to anchor the "time to platinum"
# prediction window. The treatment anchor (t_treatment_anchor) is the first date
# a patient received ANY of these drugs (earliest MED_START_DT in the medications
# table, across all treatment lines). Mirrors `drugs_to_filter_for` in
# analyze_prostate_metadata.py: ARPIs/androgen-axis, taxanes, and radium-223.
TREATMENT_ANCHOR_MEDS = {
    "ABIRATERONE ACETATE",
    "ENZALUTAMIDE",
    "APALUTAMIDE",
    "DAROLUTAMIDE",
    "DOCETAXEL",
    "CABAZITAXEL",
    "RADIUM RA 223 DICHLORIDE",
}


def load_death_df_from_survival_cohort(survival_cohort_csv: Path) -> pd.DataFrame:
    """Load the per-patient survival cohort into the ``death_df`` schema.

    ``build_longitudinal_prediction_data`` expects a per-patient frame with
    columns ``first_treatment_date``, ``last_contact_date``, ``death`` and
    ``AGE_AT_TREATMENTSTART``. The cohort file (built by
    compile_COMPASS_cohort_data.py) is ARPI/chemo-anchored, so its
    ``TREATMENT_ANCHOR_DATE`` is the treatment index date used as
    ``first_treatment_date`` here — matching the project's ARPI-anchored-as-primary
    convention.

    Unlike the old death_met_surv_df.csv.gz, this file carries a TRUE death date
    (``DEATH_DATE``). It is passed through as ``death_date`` so downstream code can
    use a real time-to-death duration instead of the last-contact proxy; when a
    patient is flagged dead but has no recorded date, downstream falls back to
    last contact.
    """
    cohort = pd.read_csv(survival_cohort_csv)

    required = {
        ID_COL,
        "TREATMENT_ANCHOR_DATE",
        "LAST_CONTACT_DATE",
        "DEATH",
        "AGE",
    }
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(
            f"{survival_cohort_csv} is missing expected columns: {sorted(missing)}"
        )

    death_df = cohort.rename(
        columns={
            "TREATMENT_ANCHOR_DATE": "first_treatment_date",
            "LAST_CONTACT_DATE": "last_contact_date",
            "DEATH": "death",
            "AGE": "AGE_AT_TREATMENTSTART",
            "DEATH_DATE": "death_date",
        }
    )

    keep_cols = [
        ID_COL,
        "first_treatment_date",
        "last_contact_date",
        "death",
        "AGE_AT_TREATMENTSTART",
    ]
    if "death_date" in death_df.columns:
        keep_cols.append("death_date")
    return death_df[keep_cols].copy()


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
    codes = icds["DIAGNOSIS_ICD10_CD"].astype(str).str.upper().str.strip()

    prostate = icds.loc[
        codes.str.startswith("C61"),
        [ID_COL, "START_DT"],
    ].copy()
    prostate["START_DT"] = pd.to_datetime(prostate["START_DT"], errors="coerce")
    prostate = prostate.dropna(subset=["START_DT"])

    return (
        prostate.groupby(ID_COL, as_index=False)["START_DT"]
        .min()
        .rename(columns={"START_DT": "DIAGNOSIS_DATE"})
    )


def compute_treatment_anchor(medications_df: pd.DataFrame) -> pd.DataFrame:
    """Earliest highlighted-treatment start date per patient.

    Filters the medications table to rows whose NCI_PREFERRED_MED_NM is one of
    ``TREATMENT_ANCHOR_MEDS`` and returns one row per patient with the earliest
    ``MED_START_DT`` (across all treatment lines). Patients who never received a
    highlighted drug are simply absent, so the downstream left-join leaves their
    ``TREATMENT_ANCHOR_DATE`` NaN and the anchor's landmark filter drops them when
    this anchor is selected.
    """
    meds = medications_df.copy()
    meds["NCI_PREFERRED_MED_NM"] = meds["NCI_PREFERRED_MED_NM"].astype(str).str.upper().str.strip()
    meds = meds.loc[meds["NCI_PREFERRED_MED_NM"].isin(TREATMENT_ANCHOR_MEDS)].copy()
    meds["TREATMENT_ANCHOR_DATE"] = pd.to_datetime(meds["MED_START_DT"], errors="coerce")
    meds = meds.dropna(subset=["TREATMENT_ANCHOR_DATE"])
    return (
        meds.groupby(ID_COL, as_index=False)["TREATMENT_ANCHOR_DATE"]
        .min()
    )


def build_longitudinal_prediction_data(
    consolidated_df: pd.DataFrame,
    first_prostate_diagnosis: pd.DataFrame,
    death_df: pd.DataFrame,
    platinum_df: pd.DataFrame,
    treatment_anchor_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
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
    after_dx = prediction_df.merge(first_prostate_diagnosis, on=ID_COL, how="left")
    n_with_c61_dx = after_dx.loc[after_dx["DIAGNOSIS_DATE"].notna(), ID_COL].nunique()
    after_death = after_dx.merge(death_df, on=ID_COL, how="inner")
    n_after_death = after_death[ID_COL].nunique()
    print(
        f"[build_longitudinal_prediction_data] patients with labs={n_lab_mrns}; "
        f"with C61 diagnosis date={n_with_c61_dx}; "
        f"after death-table inner-join={n_after_death} "
        f"(dropped {n_lab_mrns - n_after_death})."
    )
    attrition = {
        "n_with_labs": n_lab_mrns,
        "n_with_c61_diagnosis_date": n_with_c61_dx,
        "n_after_death_table_join": n_after_death,
    }
    pred_df = after_death.merge(platinum_first, on=ID_COL, how="left")
    pred_df = pred_df.merge(treatment_anchor_df, on=ID_COL, how="left")

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
        "TREATMENT_ANCHOR_DATE",
    ]
    if "death_date" in pred_df.columns:
        date_cols.append("death_date")
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
    # The survival cohort (compile_COMPASS_cohort_data.py) carries a TRUE
    # death date, so time-to-death is the real interval from first record to death
    # for dead patients and last-contact time for censored patients. When a patient
    # is flagged dead but has no recorded date, fall back to last contact so the
    # duration stays finite (the DEATH event indicator is still honored).
    if "death_date" in pred_df.columns:
        death_days = (
            pred_df["death_date"] - pred_df["FIRST_RECORD_DATE"]
        ).dt.days.astype(float)
        pred_df["t_death"] = np.where(
            pred_df["DEATH"].eq(1),
            death_days,
            pred_df["t_last_contact"],
        ).astype(float)
        pred_df["t_death"] = pred_df["t_death"].fillna(pred_df["t_last_contact"])
    else:
        # Legacy source with no date-of-death: dead and censored patients share the
        # last-contact duration; only the DEATH event indicator distinguishes them.
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

    # Treatment-anchor index time: days from FIRST_RECORD_DATE to the first
    # highlighted-treatment start. NaN for patients who never received one of the
    # highlighted drugs; the anchor's landmark filter drops them downstream.
    pred_df["t_treatment_anchor"] = (
        (pred_df["TREATMENT_ANCHOR_DATE"] - pred_df["FIRST_RECORD_DATE"]).dt.days.astype(float)
    )

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
        "TREATMENT_ANCHOR_DATE",
        "LAB_DATE",
        "t_lab",
        "t_diagnosis",
        "t_first_treatment",
        "t_platinum",
        "t_treatment_anchor",
        "t_last_contact",
        "t_death",
        "LAB_NAME",
        "LAB_VALUE",
    ]
    return pred_df[ordered_cols].copy(), attrition


def annotate_parpi_exposure(
    pred_df: pd.DataFrame,
    *,
    medications_df: pd.DataFrame,
) -> pd.DataFrame:
    """Carry PARPi exposure as a patient-level flag for downstream cohort builds."""
    out = pred_df.copy()
    parpi_mrns = set(
        medications_df.loc[
            medications_df["NCI_PREFERRED_MED_NM"].astype(str).str.upper().isin(PARPI_MEDS),
            ID_COL,
        ].unique()
    )
    out["PARPI_EXPOSED"] = out[ID_COL].isin(parpi_mrns).astype(int)
    return out


def summarize_default_cohort_filters(
    pred_df: pd.DataFrame,
    *,
    min_psa_count: int = MIN_PSA_COUNT,
) -> dict:
    """Preview the default downstream cohort filters on the broad lab frame.

    These filters used to be applied in this script, which meant every anchor
    inherited the first-treatment cohort before `build_prediction_inputs.py`
    could apply anchor-specific selection. The row-level output is now the broad
    prostate lab frame; the prediction-input builder applies these rules by
    default and can relax them for non-first-treatment anchors.

    Default downstream inclusion:
      - recorded first treatment (FIRST_TREATMENT == 1); never-treated
        patients are excluded so that the "pre-treatment" window used by
        the first-treatment anchor is well-defined.
      - >= ``min_psa_count`` PSA rows in the longitudinal prediction frame
        (LAB_NAME == "PSA", after lab consolidation)
    Default downstream exclusion:
      - any PARPi exposure, identified from the pre-existing prostate
        medications table and carried as PARPI_EXPOSED

    Non-prostate-primary exclusion is already enforced upstream at stage 1.
    A C61 diagnosis date is attached when available, but is not an inclusion
    requirement because the broad source population is the inferred-cancer
    prostate cohort.
    """
    n_before = pred_df[ID_COL].nunique()
    print(f"Broad longitudinal prostate lab frame: {n_before} patients")

    preview_df = pred_df.loc[pred_df["FIRST_TREATMENT"].eq(1)].copy()
    n_after_treated = preview_df[ID_COL].nunique()
    print(f"  Default first-treatment inclusion would keep {n_after_treated}/{n_before}")

    psa_counts = (
        preview_df.loc[preview_df["LAB_NAME"].eq("PSA")]
        .groupby(ID_COL)
        .size()
    )
    keep_psa = psa_counts.loc[psa_counts >= min_psa_count].index
    preview_df = preview_df.loc[preview_df[ID_COL].isin(keep_psa)].copy()
    n_after_psa = preview_df[ID_COL].nunique()
    print(
        f"  Default PSA count filter (>= {min_psa_count}) would keep "
        f"{n_after_psa}/{n_after_treated}"
    )

    if "PARPI_EXPOSED" in preview_df.columns:
        preview_df = preview_df.loc[~preview_df["PARPI_EXPOSED"].eq(1)].copy()
    n_after_parpi = preview_df[ID_COL].nunique()
    print(
        f"  Default PARPi exclusion would drop {n_after_psa - n_after_parpi} "
        f"(remaining: {n_after_parpi})"
    )

    return {
        "filters_applied_to_longitudinal_output": False,
        "n_before_treatment_psa_parpi_filters": n_before,
        "n_after_first_treatment_filter": n_after_treated,
        "n_after_psa_count_filter": n_after_psa,
        "min_psa_count": min_psa_count,
        "n_after_parpi_exclusion": n_after_parpi,
    }


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
        help="Pre-compiled prostate-cohort medications table (used for treatment anchor and PARPi flag).",
    )
    parser.add_argument(
        "--survival-cohort-csv",
        type=Path,
        default=DEFAULT_SURVIVAL_COHORT_CSV,
        help=(
            "Per-patient survival cohort from compile_COMPASS_cohort_data.py. "
            "Supplies the treatment anchor, age, death, and last-contact outcome columns "
            "(replaces the legacy death_met_surv_df.csv.gz)."
        ),
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
    death_df = load_death_df_from_survival_cohort(args.survival_cohort_csv)

    first_prostate_diagnosis = compute_first_prostate_diagnosis(icds)

    # Loaded once and reused for both the treatment anchor (needs MED_START_DT) and
    # the downstream PARPi-exposure flag (needs NCI_PREFERRED_MED_NM only).
    medications_df = pd.read_csv(
        args.medications_csv,
        usecols=[ID_COL, "NCI_PREFERRED_MED_NM", "MED_START_DT"],
    )
    treatment_anchor_df = compute_treatment_anchor(medications_df)
    print(
        f"Treatment anchor: {len(treatment_anchor_df)} patients received a "
        f"highlighted treatment ({', '.join(sorted(TREATMENT_ANCHOR_MEDS))})"
    )

    longitudinal_prediction_df, build_attrition = build_longitudinal_prediction_data(
        consolidated_df,
        first_prostate_diagnosis,
        death_df,
        platinum_df,
        treatment_anchor_df,
    )

    longitudinal_prediction_df = annotate_parpi_exposure(
        longitudinal_prediction_df,
        medications_df=medications_df,
    )
    filter_attrition = summarize_default_cohort_filters(longitudinal_prediction_df)

    longitudinal_prediction_df.to_csv(args.output_csv, index=False)

    # Structured attrition counts for the Figure 1 CONSORT diagram -- these are
    # exactly the counts already printed above, just persisted alongside the
    # other outputs instead of only living in the run log.
    cohort_attrition = {
        **build_attrition,
        **filter_attrition,
        "n_with_highlighted_treatment_anchor": len(treatment_anchor_df),
        "n_output_patients": int(longitudinal_prediction_df[ID_COL].nunique()),
    }
    attrition_path = args.output_csv.parent / "cohort_attrition.json"
    attrition_path.write_text(json.dumps(cohort_attrition, indent=2))

    print(f"Wrote unique lab inventory to {args.unique_labs_csv}")
    print(f"Wrote raw longitudinal rows to {args.uncondensed_output_csv}")
    print(f"Wrote consolidated longitudinal rows to {args.consolidated_output_csv}")
    print(f"Wrote survival-ready longitudinal rows to {args.output_csv}")
    print(f"Wrote cohort attrition counts to {attrition_path}")


if __name__ == "__main__":
    main()
