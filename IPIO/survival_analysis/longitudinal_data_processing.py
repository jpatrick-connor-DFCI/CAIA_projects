from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# COMPASS's helpers.consolidate_dfci_labs pipeline (unit standardization +
# physiologic-range filtering) is dataset-agnostic and reused here unchanged.
sys.path.insert(
    0,
    "/data/gusev/USERS/jpconnor/code/CAIA/COMPASS/survival_analysis",
)
from helpers.consolidate_dfci_labs import consolidate_dfci_labs  # noqa: E402

ID_COL = "DFCI_MRN"

DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/IPIO/")

IPIO_DIR = Path(__file__).resolve().parents[1]
MAPPING_CSV = IPIO_DIR / "OMOP_to_DFCI_lab_ids.csv"
# Regenerated below (mirroring COMPASS's longitudinal_data_processing.py) as an
# inventory/diagnostic artifact of the raw TEST_NAME/unit combinations seen in
# this cohort's labs. NOTE: consolidate_dfci_labs() itself only takes
# (labs_df, mapping_df) -- it does not consume unique_lab_ids_w_units.csv as an
# input. That file only feeds helpers.consolidate_dfci_labs.write_refreshed_mapping
# (a separate, optional "refresh-mapping" CLI step), which COMPASS's own
# longitudinal_data_processing.py does not invoke either. We follow the same
# pattern here: write the inventory file for visibility, but consolidate
# directly against the existing (copied-from-COMPASS) OMOP_to_DFCI_lab_ids.csv.
UNIQUE_LABS_CSV = IPIO_DIR / "unique_lab_ids_w_units.csv"

COHORT_CSV = DATA_PATH / "irae_cohort_data.csv"
LABS_CSV = DATA_PATH / "irae_labs_data.csv"
OUTPUT_CSV = DATA_PATH / "longitudinal_prediction_data.csv"


def generate_new_test_name(code: object, descr: object) -> str:
    if pd.isna(code):
        return str(descr)
    if code == descr:
        return str(code)
    return f"{code} ({descr})"


CANCER_TYPE_PREFIX = "CANCER_TYPE_"


def resolve_cancer_type_columns(cohort_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Discover the cancer-type indicator columns already present in cohort_df.

    No fixed list of cancer types is assumed -- compile_irae_data.py's
    load_cancer_type_df() treats the source cancer_type_df.csv.gz as already
    wide/one-hot (one column per type) and prefixes every such column
    `CANCER_TYPE_<NAME>`. This just finds whichever of those columns are
    present, so a change in how many/which types exist upstream needs no
    code change here.

    If every patient's CANCER_TYPE_* row sums to exactly 1 (a true mutually
    exclusive, exhaustive one-hot partition), the most prevalent column is
    dropped as the implicit reference category -- mirrors COMPASS's
    CANCER_STAGE_II/III/IV pattern (Stage I is never emitted, kept implicit)
    and avoids perfect collinearity in the Cox design matrix. If the columns
    are NOT mutually exclusive/exhaustive (e.g. a patient can have 0 or 2+
    flags set), there's no natural reference category and no collinearity
    risk, so all columns are kept.
    """
    cancer_type_cols = [c for c in cohort_df.columns if c.startswith(CANCER_TYPE_PREFIX)]
    if not cancer_type_cols:
        raise ValueError(
            f"No columns starting with '{CANCER_TYPE_PREFIX}' found in the cohort "
            "table; expected compile_irae_data.py to have already merged one-hot "
            "cancer-type indicator columns."
        )

    cohort_df = cohort_df.copy()
    indicators = cohort_df[cancer_type_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    cohort_df[cancer_type_cols] = indicators
    row_sums = indicators.sum(axis=1)
    is_one_hot_partition = bool((row_sums == 1).all())

    if is_one_hot_partition:
        reference_col = indicators.sum(axis=0).idxmax()
        print(
            f"Cancer-type columns are a mutually exclusive/exhaustive one-hot set "
            f"({len(cancer_type_cols)} columns); dropping '{reference_col}' "
            "(most prevalent) as the implicit reference category."
        )
        cohort_df = cohort_df.drop(columns=[reference_col])
        cancer_type_cols = [c for c in cancer_type_cols if c != reference_col]
    else:
        print(
            f"Cancer-type columns ({len(cancer_type_cols)}) are not a strict one-hot "
            "partition (some patients have 0 or 2+ flags set); keeping all columns "
            "as independent binary covariates, 0-filled where missing."
        )

    return cohort_df, cancer_type_cols


def build_raw_longitudinal_labs(labs_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape raw OncDRS lab rows into the long ID_COL/DATE/LAB_NAME/LAB_UNIT/LAB_VALUE
    schema expected by helpers.consolidate_dfci_labs.consolidate_dfci_labs (mirrors the
    labs half of COMPASS's build_raw_longitudinal_data; IPIO has no HEALTH_HISTORY
    vital-signs table to fold in).
    """
    working = labs_df[
        [
            ID_COL,
            "SPECIMEN_COLLECT_DT",
            "TEST_TYPE_CD",
            "TEST_TYPE_DESCR",
            "NUMERIC_RESULT",
            "RESULT_UOM_NM",
        ]
    ].copy()

    working["TEST_NAME"] = working.apply(
        lambda row: generate_new_test_name(row["TEST_TYPE_CD"], row["TEST_TYPE_DESCR"]),
        axis=1,
    )

    working = working.rename(
        columns={
            "SPECIMEN_COLLECT_DT": "DATE",
            "NUMERIC_RESULT": "LAB_VALUE",
            "RESULT_UOM_NM": "LAB_UNIT",
            "TEST_NAME": "LAB_NAME",
        }
    )[[ID_COL, "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE"]]

    return working


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cohort_df = pd.read_csv(COHORT_CSV)
    labs_df = pd.read_csv(LABS_CSV)

    # --- Static per-patient columns ---
    cohort_df["IO_START"] = pd.to_datetime(cohort_df["IO_START"], errors="coerce")
    cohort_df["LAST_DATE"] = pd.to_datetime(cohort_df["LAST_DATE"], errors="coerce")

    cohort_df["GENDER_MALE"] = (
        cohort_df["GENDER"].astype(str).str.upper().str.strip().eq("MALE").astype(int)
    )
    cohort_df = cohort_df.drop(columns=["GENDER"])

    # Cancer type arrives from compile_irae_data.py already as one-or-more
    # CANCER_TYPE_* indicator columns (no fixed type list assumed -- see
    # resolve_cancer_type_columns's docstring for the one-hot-vs-independent-
    # flags handling).
    cohort_df, cancer_type_cols = resolve_cancer_type_columns(cohort_df)

    # Anchor design choice: FIRST_RECORD_DATE = FIRST_TREATMENT_DATE = IO_START and
    # FIRST_TREATMENT = 1 for everyone, so helpers.cohort.build_feature_matrix's
    # default anchor (t_first_treatment = FIRST_TREATMENT_DATE - FIRST_RECORD_DATE)
    # is exactly 0 for every patient without any code changes downstream.
    cohort_df["FIRST_RECORD_DATE"] = cohort_df["IO_START"]
    cohort_df["FIRST_TREATMENT_DATE"] = cohort_df["IO_START"]
    cohort_df["FIRST_TREATMENT"] = 1
    cohort_df["LAST_CONTACT_DATE"] = cohort_df["LAST_DATE"]
    cohort_df["IRAE"] = cohort_df["event"].astype(str).eq("irAE").astype(int)

    static_cols = (
        [ID_COL, "AGE_AT_TREATMENTSTART", "GENDER_MALE"]
        + list(cancer_type_cols)
        + [
            "pd1pdl1",
            "ctla4",
            "FIRST_RECORD_DATE",
            "FIRST_TREATMENT_DATE",
            "FIRST_TREATMENT",
            "LAST_CONTACT_DATE",
            "IRAE",
        ]
    )
    static_df = cohort_df[static_cols].copy()

    # --- Lab standardization via COMPASS's unchanged consolidate_dfci_labs pipeline ---
    raw_longitudinal_labs = build_raw_longitudinal_labs(labs_df)

    unique_labs_df = (
        raw_longitudinal_labs[["LAB_NAME", "LAB_UNIT"]]
        .value_counts()
        .reset_index(name="count")
    )
    unique_labs_df.to_csv(UNIQUE_LABS_CSV, index=False)

    mapping_df = pd.read_csv(MAPPING_CSV)
    consolidated_df = consolidate_dfci_labs(raw_longitudinal_labs, mapping_df)

    lab_rows = (
        consolidated_df[[ID_COL, "DATE", "collapsed_measurement", "numeric_result_standardized"]]
        .dropna()
        .rename(
            columns={
                "DATE": "LAB_DATE",
                "collapsed_measurement": "LAB_NAME",
                "numeric_result_standardized": "LAB_VALUE",
            }
        )
        .copy()
    )
    lab_rows["LAB_DATE"] = pd.to_datetime(lab_rows["LAB_DATE"], errors="coerce").dt.floor("D")

    # --- Merge lab rows with per-patient static columns; one row per (patient x lab) ---
    longitudinal_df = lab_rows.merge(static_df, on=ID_COL, how="inner")
    longitudinal_df["FIRST_RECORD_DATE"] = pd.to_datetime(
        longitudinal_df["FIRST_RECORD_DATE"], errors="coerce"
    ).dt.floor("D")
    longitudinal_df["t_lab"] = (
        longitudinal_df["LAB_DATE"] - longitudinal_df["FIRST_RECORD_DATE"]
    ).dt.days.astype(float)

    ordered_cols = (
        [ID_COL, "AGE_AT_TREATMENTSTART", "GENDER_MALE"]
        + list(cancer_type_cols)
        + [
            "pd1pdl1",
            "ctla4",
            "FIRST_RECORD_DATE",
            "FIRST_TREATMENT_DATE",
            "FIRST_TREATMENT",
            "LAST_CONTACT_DATE",
            "IRAE",
            "LAB_DATE",
            "t_lab",
            "LAB_NAME",
            "LAB_VALUE",
        ]
    )
    longitudinal_df = longitudinal_df[ordered_cols].copy()

    longitudinal_df.to_csv(OUTPUT_CSV, index=False)

    # --- Sanity-check diagnostics ---
    n_patients = static_df[ID_COL].nunique()
    n_patients_with_labs = longitudinal_df[ID_COL].nunique()
    irae_rate = static_df["IRAE"].mean() if n_patients else float("nan")
    print(f"Cohort patients: {n_patients}")
    print(f"Patients with >=1 standardized lab row: {n_patients_with_labs}")
    print(f"IRAE event rate (cohort-level): {irae_rate:.4f}")
    print(f"Lab row count in output: {len(longitudinal_df)}")
    if len(longitudinal_df):
        print(
            f"LAB_DATE range: {longitudinal_df['LAB_DATE'].min()} to "
            f"{longitudinal_df['LAB_DATE'].max()}"
        )
        print(
            f"t_lab range: {longitudinal_df['t_lab'].min()} to "
            f"{longitudinal_df['t_lab'].max()}"
        )
    print(f"Wrote unique lab inventory to {UNIQUE_LABS_CSV}")
    print(f"Wrote longitudinal prediction data to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
