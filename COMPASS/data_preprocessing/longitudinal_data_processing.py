from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SURVIVAL_DIR = PROJECT_DIR / "survival_analysis"
REPO_ROOT = PROJECT_DIR.parent
for _p in (str(REPO_ROOT), str(PROJECT_DIR), str(SURVIVAL_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_preprocessing_common import fast_io  # noqa: E402
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
ONCDRS_PATH = PROFILE_PATH / "OncDRS" / "ALL_2025_03"
SURV_PATH = EMBED_PROJ_PATH / "time-to-event_analysis"

# Per-patient survival cohort produced by compile_COMPASS_cohort_data.py
# straight from the raw OncDRS pull. It supplies the outcome/anchor columns
# (age, treatment anchor, death, last-contact, platinum) that used to come from
# death_met_surv_df.csv.gz. See load_death_df_from_survival_cohort. Defaults to
# the icd_or_vte UNION cohort (not the icd-only file). main() loads this file
# first and uses its MRN set both to scan-filter the raw HEALTH_HISTORY/
# OUTPT_LAB_RESULTS_LABS/MEDICATIONS reads and to restrict the final output --
# this is the one cohort-membership filter Stage 2 applies. Narrower cohort
# variants (icd-only, vte-only, either ARPI-restricted) remain a Stage 3
# (build_prediction_inputs.py --restrict-to-mrns) concern.
DEFAULT_SURVIVAL_COHORT_CSV = NEPC_PROJ_PATH / "prostate_arpi_survival_cohort_icd_or_vte.csv"

# Cisplatin appears both as a single agent and coded within a combination
# regimen name; both count as platinum exposure. Oxaliplatin is intentionally
# excluded (not a relevant platinum agent for this cohort). Kept in sync with
# PLATINUM_MEDS in compile_COMPASS_cohort_data.py.
PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN", "CISPLATIN/CYCLOPHOSPHAMIDE/ETOPOSIDE"}
# PARPi names include salt forms as coded in OncDRS (RUCAPARIB CAMSYLATE,
# TALAZOPARIB TOSYLATE), which do not match the base names above.
PARPI_MEDS = {
    "OLAPARIB",
    "RUCAPARIB",
    "RUCAPARIB CAMSYLATE",
    "NIRAPARIB",
    "TALAZOPARIB",
    "TALAZOPARIB TOSYLATE",
    "VELIPARIB",
}
MIN_PSA_COUNT = 5
# Broad PSA assay set (raw TEST_TYPE_CD, any assay type: total/free/complexed/
# ultrasensitive/etc.), used only for the downstream PSA-count prevalence gate
# in build_prediction_inputs.py / summarize_default_cohort_filters below. This
# is deliberately NOT the narrow OMOP-collapsed LAB_NAME == "PSA" set, which
# drives prediction features -- see RAW_TEST_CODE passthrough in
# build_raw_longitudinal_data.
BROAD_PSA_CODES = ["PSA", "PSAR", "PSATOTSCRN", "CPSA", "PSAMON", "PSAULT", "PSAT"]

# Highlighted antineoplastic treatments used to anchor the "time to platinum"
# prediction window. The treatment anchor (TREATMENT_ANCHOR_DATE) is the first
# date a patient received ANY of these drugs (earliest MED_START_DT in the
# medications table, across all treatment lines) and defines time 0 for every
# duration in the output. Mirrors `drugs_to_filter_for` in
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
    columns ``treatment_anchor_date``, ``last_contact_date``, ``death`` and
    ``AGE_AT_TREATMENTSTART``. The cohort file (built by
    compile_COMPASS_cohort_data.py) is ARPI/chemo-anchored: its
    ``TREATMENT_ANCHOR_DATE`` (first ARPI/taxane/radium-223 exposure) is the sole
    treatment index date for this pipeline.

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
            "TREATMENT_ANCHOR_DATE": "treatment_anchor_date",
            "LAST_CONTACT_DATE": "last_contact_date",
            "DEATH": "death",
            "AGE": "AGE_AT_TREATMENTSTART",
            "DEATH_DATE": "death_date",
        }
    )

    keep_cols = [
        ID_COL,
        "treatment_anchor_date",
        "last_contact_date",
        "death",
        "AGE_AT_TREATMENTSTART",
    ]
    if "death_date" in death_df.columns:
        keep_cols.append("death_date")
    return death_df[keep_cols].copy()


def generate_new_test_name(code: object, descr: object) -> str:
    """Row-at-a-time reference implementation (kept for documentation /
    parity checking); the production path uses the vectorized polars
    expression `generate_new_test_name_expr` below instead of `.apply(...)`.
    """
    if pd.isna(code):
        return str(descr)
    if code == descr:
        return str(code)
    return f"{code} ({descr})"


def generate_new_test_name_expr(code_col: str, descr_col: str) -> pl.Expr:
    """Vectorized polars equivalent of `generate_new_test_name`.

    - code is null -> str(descr). Faithfully reproduces the original's
      `str(descr)` call even when descr is itself null/NaN: pandas'
      `str(float('nan'))` is the literal string "nan", so a null descr in
      this branch is coalesced to the "nan" literal rather than left null.
    - code == descr -> str(code).
    - otherwise -> "{code} ({descr})".
    """
    code = pl.col(code_col)
    descr = pl.col(descr_col)
    descr_as_str = pl.when(descr.is_null()).then(pl.lit("nan")).otherwise(descr.cast(pl.Utf8))
    code_as_str = code.cast(pl.Utf8)
    return (
        pl.when(code.is_null())
        .then(descr_as_str)
        .when(code == descr)
        .then(code_as_str)
        .otherwise(code_as_str + pl.lit(" (") + descr.cast(pl.Utf8) + pl.lit(")"))
    )


def build_raw_longitudinal_data(
    health_df: pl.DataFrame,
    labs_df: pl.DataFrame,
) -> pl.DataFrame:
    """Reshape raw OncDRS HEALTH_HISTORY/OUTPT_LAB_RESULTS_LABS into one long
    vitals+labs table.

    This function itself applies no MRN restriction -- it reshapes whatever
    `health_df`/`labs_df` it is given. main() passes frames already
    scan-filtered to the icd_or_vte union cohort (see union_cohort_mrns), but
    that filtering happens at the call site, not here. Narrower cohort
    selection (icd / vte / icd_or_vte, each full or ARPI-restricted) is
    applied downstream in build_prediction_inputs.py via --restrict-to-mrns,
    so every narrower cohort variant can be compared from the same
    longitudinal_prediction_data.csv output.
    """
    # Both inputs are scanned all-String (infer_schema_length=0), so DFCI_MRN
    # arrives as Utf8 here. Cast it back to Int64 before it flows into
    # to_pandas()/consolidate_dfci_labs() downstream in main() -- otherwise it
    # stays `object` in pandas and fails to merge against the int64 DFCI_MRN
    # read from icd/platinum/medications CSVs (pandas raises "trying to merge
    # on object and int64 columns").
    health_df = health_df.with_columns(
        pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    )
    labs_df = labs_df.with_columns(
        pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    )

    vital_signs_df = health_df.filter(pl.col("CODE_TYPE") == "Vital Signs").select(
        [ID_COL, "START_DT", "HEALTH_HISTORY_TYPE", "RESULTS", "UNITS_CD"]
    )

    labs_df_col_sub = labs_df.select(
        [
            ID_COL,
            "SPECIMEN_COLLECT_DT",
            "TEST_TYPE_CD",
            "TEST_TYPE_DESCR",
            "NUMERIC_RESULT",
            "RESULT_UOM_NM",
        ]
    )

    labs_df_col_sub = labs_df_col_sub.with_columns(
        generate_new_test_name_expr("TEST_TYPE_CD", "TEST_TYPE_DESCR").alias("TEST_NAME")
    )

    # Carry the raw, un-synthesized TEST_TYPE_CD through as RAW_TEST_CODE so the
    # broad-vs-narrow PSA distinction survives past TEST_NAME synthesis (above)
    # and consolidate_dfci_labs' further canonicalization of LAB_NAME. Vitals
    # have no TEST_TYPE_CD equivalent, so RAW_TEST_CODE is null for those rows.
    vital_signs_df = vital_signs_df.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("RAW_TEST_CODE")
    )
    labs_df_col_sub = labs_df_col_sub.with_columns(
        pl.col("TEST_TYPE_CD").alias("RAW_TEST_CODE")
    )

    vital_signs_df = vital_signs_df.rename(
        {
            "START_DT": "DATE",
            "RESULTS": "LAB_VALUE",
            "HEALTH_HISTORY_TYPE": "LAB_NAME",
            "UNITS_CD": "LAB_UNIT",
        }
    ).select([ID_COL, "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE", "RAW_TEST_CODE"])

    labs_df_col_sub = labs_df_col_sub.rename(
        {
            "SPECIMEN_COLLECT_DT": "DATE",
            "NUMERIC_RESULT": "LAB_VALUE",
            "RESULT_UOM_NM": "LAB_UNIT",
            "TEST_NAME": "LAB_NAME",
        }
    ).select([ID_COL, "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE", "RAW_TEST_CODE"])

    # Both inputs are scanned all-String (infer_schema_length=0), so LAB_VALUE
    # is already Utf8 on both halves here. This explicit cast is defensive --
    # it guarantees a common Utf8 LAB_VALUE dtype (HEALTH_HISTORY.RESULTS is
    # inherently free-text; labs' NUMERIC_RESULT is numeric-as-string) so
    # `pl.concat` never has to reconcile a numeric vs. string LAB_VALUE column
    # if either input is ever read with real schema inference upstream.
    # consolidate_dfci_labs() re-parses NUMERIC_RESULT via pd.to_numeric
    # regardless, so keeping it as a string here loses nothing -- this mirrors
    # the mixed-type `object` column pandas' `pd.concat` produced originally.
    vital_signs_df = vital_signs_df.with_columns(pl.col("LAB_VALUE").cast(pl.Utf8, strict=False))
    labs_df_col_sub = labs_df_col_sub.with_columns(pl.col("LAB_VALUE").cast(pl.Utf8, strict=False))

    return pl.concat([vital_signs_df, labs_df_col_sub], how="vertical_relaxed")


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


def compute_first_platinum(medications_df: pd.DataFrame) -> pd.DataFrame:
    """Earliest platinum MED_START_DT (and drug name) per patient, computed
    in-memory from the raw medications table -- replaces the old
    platinum_chemo_records.csv read (Stage 1 no longer persists that file).
    Mirrors compute_first_platinum in compile_COMPASS_cohort_data.py; output
    schema (``medication``/``medication_start_time``) matches what that file
    used to carry, so build_longitudinal_prediction_data needs no changes.
    """
    meds = medications_df.copy()
    meds["NCI_PREFERRED_MED_NM"] = meds["NCI_PREFERRED_MED_NM"].astype(str).str.upper().str.strip()
    meds = meds.loc[meds["NCI_PREFERRED_MED_NM"].isin(PLATINUM_MEDS)].copy()
    meds["_med_start_dt_parsed"] = pd.to_datetime(meds["MED_START_DT"], errors="coerce")
    meds = meds.dropna(subset=["_med_start_dt_parsed"])
    meds = (
        meds.sort_values("_med_start_dt_parsed")
        .drop_duplicates(subset=ID_COL, keep="first")
        .drop(columns="_med_start_dt_parsed")
    )
    return meds[[ID_COL, "NCI_PREFERRED_MED_NM", "MED_START_DT"]].rename(
        columns={
            "NCI_PREFERRED_MED_NM": "medication",
            "MED_START_DT": "medication_start_time",
        }
    )


def build_longitudinal_prediction_data(
    consolidated_df: pd.DataFrame,
    first_prostate_diagnosis: pd.DataFrame,
    death_df: pd.DataFrame,
    platinum_df: pd.DataFrame,
    treatment_anchor_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Build the survival-ready longitudinal frame.

    No MRN restriction is applied here: this function reshapes whatever
    `consolidated_df` it is given (main() passes data already scan-filtered
    to the icd_or_vte union), and `death_df` is left-joined (not
    inner-joined) so any patient without a death_df row is kept with
    all-null outcome columns rather than silently dropped. The caller
    (main()) applies a belt-and-suspenders union-cohort filter to the
    returned frame afterward; narrower cohort variants remain a
    build_prediction_inputs.py --restrict-to-mrns concern.
    """
    prediction_df = consolidated_df[
        [ID_COL, "DATE", "collapsed_measurement", "numeric_result_standardized", "RAW_TEST_CODE"]
    ].dropna(subset=[ID_COL, "DATE", "collapsed_measurement", "numeric_result_standardized"]).copy()

    n_raw_mrns = prediction_df[ID_COL].nunique()
    print(f"[build_longitudinal_prediction_data] raw longitudinal patients={n_raw_mrns}.")

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
    # Left join: patients outside every compile_COMPASS_cohort_data.py cohort
    # (no row in death_df) are kept with all-null outcome columns rather than
    # dropped -- cohort selection is a Stage 3 (--restrict-to-mrns) concern.
    after_death = after_dx.merge(death_df, on=ID_COL, how="left")
    n_with_outcome_data = after_death.loc[after_death["death"].notna(), ID_COL].nunique()
    print(
        f"[build_longitudinal_prediction_data] patients with labs={n_lab_mrns}; "
        f"with C61 diagnosis date={n_with_c61_dx}; "
        f"with outcome data (any cohort, left-joined)={n_with_outcome_data} "
        f"(no cohort match: {n_lab_mrns - n_with_outcome_data})."
    )
    attrition = {
        "n_raw_longitudinal_patients": n_raw_mrns,
        "n_with_labs": n_lab_mrns,
        "n_with_c61_diagnosis_date": n_with_c61_dx,
        "n_with_outcome_data": n_with_outcome_data,
    }
    pred_df = after_death.merge(platinum_first, on=ID_COL, how="left")
    pred_df = pred_df.merge(treatment_anchor_df, on=ID_COL, how="left")

    # The treatment anchor (first ARPI/taxane/radium-223 exposure) is recomputed
    # from the medications table's ``MED_START_DT`` (see compute_treatment_anchor)
    # and carried as ``TREATMENT_ANCHOR_DATE``. The survival cohort also ships a
    # ``treatment_anchor_date`` (derived from the same raw ``MED_START_DT``
    # calendar dates), but the meds-derived date is authoritative here, so drop the
    # cohort copy to avoid carrying a redundant anchor date.
    pred_df = pred_df.drop(columns=["treatment_anchor_date"], errors="ignore")

    pred_df = (
        pred_df.rename(
            columns={
                "collapsed_measurement": "LAB_NAME",
                "numeric_result_standardized": "LAB_VALUE",
                "DATE": "LAB_DATE",
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
    pred_df["PLATINUM"] = (
        pred_df["PLATINUM_MEDICATION"]
        .astype(str)
        .str.upper()
        .isin(PLATINUM_MEDS)
        .astype(int)
    )
    pred_df["PLATINUM_DATE"] = pred_df["PLATINUM_DATE"].fillna(pred_df["LAST_CONTACT_DATE"])

    # TREATMENT_ANCHOR_DATE (first ARPI/taxane/radium-223 exposure) is the
    # timeline origin for this pipeline: every duration below -- t_lab included --
    # is measured in days FROM the treatment anchor. Positive => after the anchor,
    # negative => before. FIRST_RECORD_DATE is still carried for reference
    # (earliest of first lab, diagnosis, and anchor) but is NOT the clock.
    first_lab_date = pred_df.groupby(ID_COL)["LAB_DATE"].transform("min")
    pred_df["FIRST_RECORD_DATE"] = pd.concat(
        [first_lab_date, pred_df["DIAGNOSIS_DATE"], pred_df["TREATMENT_ANCHOR_DATE"]],
        axis=1,
    ).min(axis=1)

    anchor = pred_df["TREATMENT_ANCHOR_DATE"]
    pred_df["t_lab"] = (pred_df["LAB_DATE"] - anchor).dt.days.astype(float)
    pred_df["t_last_contact"] = (
        pred_df["LAST_CONTACT_DATE"] - anchor
    ).dt.days.astype(float)
    # The survival cohort (compile_COMPASS_cohort_data.py) carries a TRUE
    # death date, so time-to-death is the real interval from the treatment anchor
    # to death for dead patients and the anchor->last-contact time for censored
    # patients. When a patient is flagged dead but has no recorded date, fall back
    # to last contact so the duration stays finite (the DEATH event indicator is
    # still honored).
    if "death_date" in pred_df.columns:
        death_days = (pred_df["death_date"] - anchor).dt.days.astype(float)
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

    pred_df["t_diagnosis"] = (pred_df["DIAGNOSIS_DATE"] - anchor).dt.days.astype(float)

    # Prediction target: days from the treatment anchor to platinum initiation for
    # platinum-positive patients, otherwise days from the anchor to last contact
    # (censored). This is measured directly from the anchor -- it is the interval
    # we want to predict.
    platinum_days = (pred_df["PLATINUM_DATE"] - anchor).dt.days
    pred_df["t_platinum"] = np.where(
        pred_df["PLATINUM"].eq(1),
        platinum_days,
        pred_df["t_last_contact"],
    ).astype(float)

    # Patients who never received a highlighted drug have no anchor date, so every
    # anchor-relative duration is NaN and they are dropped by the outcome builder's
    # notna() validity checks. No separate anchor index column is emitted: the
    # treatment anchor IS time 0, so the durations above already encode it and the
    # downstream landmark is a pure offset (anchor_col=None in make_outcome_df).

    ordered_cols = [
        ID_COL,
        "AGE_AT_TREATMENTSTART",
        "FIRST_RECORD_DATE",
        "DIAGNOSIS_DATE",
        "LAST_CONTACT_DATE",
        "DEATH",
        "PLATINUM_MEDICATION",
        "PLATINUM_DATE",
        "PLATINUM",
        "TREATMENT_ANCHOR_DATE",
        "LAB_DATE",
        "t_lab",
        "t_diagnosis",
        "t_platinum",
        "t_last_contact",
        "t_death",
        "LAB_NAME",
        "LAB_VALUE",
        "RAW_TEST_CODE",
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

    These filters are applied by `build_prediction_inputs.py`, not here; the
    row-level output stays the broad prostate lab frame. This preview reports the
    attrition they would produce so the counts land in cohort_attrition.json.

    Default downstream inclusion:
      - >= ``min_psa_count`` PSA labs counted from the BROAD PSA set
        (RAW_TEST_CODE in BROAD_PSA_CODES: total/free/complexed/ultrasensitive/
        etc.), mirroring build_prediction_inputs.py. This is deliberately NOT
        the narrow OMOP-collapsed LAB_NAME == "PSA" set, which drives
        predictions.
    Default downstream exclusion:
      - any PARPi exposure, identified from the pre-existing prostate
        medications table and carried as PARPI_EXPOSED

    Treated status for the COMPASS pipeline is enforced downstream in
    make_outcome_df: durations are measured from the treatment anchor, so patients
    with no ARPI/taxane/radium-223 anchor have all-NaN durations and fail its
    validity checks. There is no separate first-treatment inclusion step.
    Non-prostate-primary exclusion is already enforced upstream at stage 1. A C61
    diagnosis date is attached when available, but is not an inclusion requirement.
    """
    n_before = pred_df[ID_COL].nunique()
    print(f"Broad longitudinal prostate lab frame: {n_before} patients")

    # Count PSA labs from the broad RAW_TEST_CODE set (all assay types),
    # matching the prevalence gate in build_prediction_inputs.py.
    psa_counts = (
        pred_df.loc[pred_df["RAW_TEST_CODE"].isin(BROAD_PSA_CODES)]
        .groupby(ID_COL)
        .size()
    )
    keep_psa = psa_counts.loc[psa_counts >= min_psa_count].index
    preview_df = pred_df.loc[pred_df[ID_COL].isin(keep_psa)].copy()
    n_after_psa = preview_df[ID_COL].nunique()
    print(
        f"  Default PSA count filter (>= {min_psa_count}) would keep "
        f"{n_after_psa}/{n_before}"
    )

    if "PARPI_EXPOSED" in preview_df.columns:
        preview_df = preview_df.loc[~preview_df["PARPI_EXPOSED"].eq(1)].copy()
    n_after_parpi = preview_df[ID_COL].nunique()
    print(
        f"  Default PARPi exclusion would drop {n_after_psa - n_after_parpi} "
        f"(remaining: {n_after_parpi})"
    )

    # Anchored-patient funnel: how the same PSA/PARPi filters attrite the subset
    # of patients who actually carry a treatment anchor (first ARPI/taxane/
    # radium-223 exposure). The downstream landmark builder keeps only anchored
    # patients (non-anchor patients have all-NaN durations), so this is the funnel
    # that determines the treated model cohort. Without this the large PSA>=5 loss
    # among treated patients is invisible until a manual reconciliation. Mirrors
    # the survival-cohort anchor count in compile_COMPASS_cohort_data.py.
    anchored_mrns = set(
        pred_df.loc[pred_df["TREATMENT_ANCHOR_DATE"].notna(), ID_COL].unique()
    )
    n_anchored_with_labs = len(anchored_mrns)
    n_anchored_after_psa = len(anchored_mrns & set(keep_psa))
    n_anchored_final = preview_df.loc[
        preview_df["TREATMENT_ANCHOR_DATE"].notna(), ID_COL
    ].nunique()
    print(
        f"  Anchored (treated) funnel: {n_anchored_with_labs} anchored patients with "
        f"labs -> {n_anchored_after_psa} after PSA>={min_psa_count} "
        f"(dropped {n_anchored_with_labs - n_anchored_after_psa}) -> "
        f"{n_anchored_final} after PARPi exclusion "
        f"(dropped {n_anchored_after_psa - n_anchored_final})."
    )

    return {
        "filters_applied_to_longitudinal_output": False,
        "n_before_psa_parpi_filters": n_before,
        "n_after_psa_count_filter": n_after_psa,
        "min_psa_count": min_psa_count,
        "n_after_parpi_exclusion": n_after_parpi,
        "n_anchored_with_labs": n_anchored_with_labs,
        "n_anchored_after_psa_count_filter": n_anchored_after_psa,
        "n_anchored_after_parpi_exclusion": n_anchored_final,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build row-level longitudinal survival input from the COMPASS/Profile exports.",
    )
    parser.add_argument(
        "--health-csv",
        type=Path,
        default=ONCDRS_PATH / "HEALTH_HISTORY.csv",
        help="Raw OncDRS HEALTH_HISTORY.csv. Lazily scan-filtered to the "
             "--survival-cohort-csv MRN set (see that flag) before it is "
             "ever fully materialized.",
    )
    parser.add_argument(
        "--labs-csv",
        type=Path,
        default=ONCDRS_PATH / "OUTPT_LAB_RESULTS_LABS.csv",
        help="Raw OncDRS OUTPT_LAB_RESULTS_LABS.csv. Lazily scan-filtered to "
             "the --survival-cohort-csv MRN set (see that flag) before it is "
             "ever fully materialized.",
    )
    parser.add_argument(
        "--icd-csv",
        type=Path,
        default=NEPC_PROJ_PATH / "prostate_icd_data.csv",
        help="ICD-C61 cohort record written by compile_COMPASS_cohort_data.py; "
             "supplies the optional DIAGNOSIS_DATE column (first C61 diagnosis "
             "per patient, left-joined). Does NOT restrict the output cohort -- "
             "patients absent from this file simply get a null DIAGNOSIS_DATE.",
    )
    parser.add_argument(
        "--medications-csv",
        type=Path,
        default=ONCDRS_PATH / "MEDICATIONS.csv",
        help="Raw OncDRS MEDICATIONS.csv, used for treatment anchor, "
             "in-memory platinum computation, and PARPi flag. Lazily "
             "scan-filtered to the --survival-cohort-csv MRN set (see that "
             "flag) before it is ever fully materialized.",
    )
    parser.add_argument(
        "--survival-cohort-csv",
        type=Path,
        default=DEFAULT_SURVIVAL_COHORT_CSV,
        help=(
            "Per-patient survival cohort from compile_COMPASS_cohort_data.py "
            "(defaults to the icd_or_vte UNION cohort). Supplies the treatment "
            "anchor, age, death, and last-contact outcome columns via a LEFT "
            "join, and ALSO defines the output cohort: main() restricts the "
            "final longitudinal_prediction_df to this file's MRN set after the "
            "join. Point this at a narrower/wider cohort file to change which "
            "patients Stage 2 outputs."
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

    # ICD-C61 records, kept for compute_first_prostate_diagnosis below. Small
    # and left-joined later, so it is read in full (unfiltered) rather than
    # gated on union_cohort_mrns.
    icds = pd.read_csv(args.icd_csv)

    # Output cohort = the icd_or_vte union (every MRN in args.survival_cohort_csv,
    # which defaults to prostate_arpi_survival_cohort_icd_or_vte.csv). Loaded
    # first so HEALTH_HISTORY/OUTPT_LAB_RESULTS_LABS/MEDICATIONS -- each tens of
    # millions of rows across the full raw OncDRS universe -- can be filtered to
    # this MRN set during the lazy scan itself, rather than read in full and
    # filtered afterward. Narrower cohort variants (icd-only, vte-only,
    # ARPI-restricted) remain a Stage 3 concern via build_prediction_inputs.py's
    # --restrict-to-mrns.
    death_df = load_death_df_from_survival_cohort(args.survival_cohort_csv)
    union_cohort_mrns = set(int(m) for m in death_df[ID_COL].unique())
    print(f"[main] icd_or_vte union cohort: {len(union_cohort_mrns)} patients.")

    # --- Polars reshape (scope boundary: everything up to consolidate_dfci_labs) ---
    health_df_pl = fast_io.scan_filter(args.health_csv, union_cohort_mrns).collect()
    labs_df_pl = fast_io.scan_filter(args.labs_csv, union_cohort_mrns).collect()
    raw_longitudinal_df_pl = build_raw_longitudinal_data(health_df_pl, labs_df_pl)

    unique_labs_df_pl = (
        raw_longitudinal_df_pl.group_by(["LAB_NAME", "LAB_UNIT"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    unique_labs_df_pl.write_csv(args.unique_labs_csv)
    raw_longitudinal_df_pl.write_csv(args.uncondensed_output_csv)

    # Conversion boundary: consolidate_dfci_labs (data_preprocessing_common/dfci_labs.py)
    # takes and returns pandas DataFrames and is explicitly out of scope for this
    # port (1192 lines of per-row unit math + BP splitting). Convert polars->pandas
    # here; everything from this point on in main() stays pandas, unchanged.
    raw_longitudinal_df = raw_longitudinal_df_pl.to_pandas()

    mapping_df = pd.read_csv(args.mapping_csv)
    consolidated_df = consolidate_dfci_labs(raw_longitudinal_df, mapping_df)
    consolidated_df.to_csv(args.consolidated_output_csv, index=False)

    first_prostate_diagnosis = compute_first_prostate_diagnosis(icds)

    # Loaded once and reused for the treatment anchor, in-memory platinum
    # computation, and the downstream PARPi-exposure flag. Filtered to
    # union_cohort_mrns during the lazy scan for the same reason as
    # health/labs above.
    medications_df = (
        fast_io.scan_filter(
            args.medications_csv,
            union_cohort_mrns,
            cols=[ID_COL, "NCI_PREFERRED_MED_NM", "MED_START_DT"],
        )
        .collect()
        .to_pandas()
    )
    medications_df[ID_COL] = pd.to_numeric(medications_df[ID_COL], errors="coerce")
    medications_df = medications_df.dropna(subset=[ID_COL]).copy()
    medications_df[ID_COL] = medications_df[ID_COL].astype(int)

    treatment_anchor_df = compute_treatment_anchor(medications_df)
    n_anchor = len(treatment_anchor_df)
    print(
        f"Treatment anchor: {n_anchor} patients received a highlighted treatment "
        f"({', '.join(sorted(TREATMENT_ANCHOR_MEDS))})"
    )

    platinum_df = compute_first_platinum(medications_df)
    print(f"Platinum: {len(platinum_df)} patients received a platinum agent (prostate cohort).")

    longitudinal_prediction_df, build_attrition = build_longitudinal_prediction_data(
        consolidated_df,
        first_prostate_diagnosis,
        death_df,
        platinum_df,
        treatment_anchor_df,
    )

    # health/labs/medications were already scan-filtered to union_cohort_mrns
    # above, so this should be a no-op; kept as a cheap belt-and-suspenders
    # check (e.g. against icds/first_prostate_diagnosis, which are read
    # unfiltered) and to record the attrition count explicitly.
    n_before_union_filter = longitudinal_prediction_df[ID_COL].nunique()
    longitudinal_prediction_df = longitudinal_prediction_df.loc[
        longitudinal_prediction_df[ID_COL].isin(union_cohort_mrns)
    ].copy()
    n_after_union_filter = longitudinal_prediction_df[ID_COL].nunique()
    print(
        f"[main] icd_or_vte union cohort restriction: {n_after_union_filter}/"
        f"{n_before_union_filter} patients retained "
        f"(dropped {n_before_union_filter - n_after_union_filter} not in "
        f"{args.survival_cohort_csv.name})."
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
        "n_before_icd_or_vte_union_filter": int(n_before_union_filter),
        "n_after_icd_or_vte_union_filter": int(n_after_union_filter),
        **filter_attrition,
        "n_with_highlighted_treatment_anchor": int(n_anchor),
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
