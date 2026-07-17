from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SURVIVAL_DIR = PROJECT_DIR / "survival_analysis"
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from _paths import ensure_survival_common_on_path  # noqa: E402

ensure_survival_common_on_path()
from data_preprocessing_common.dfci_labs import (  # noqa: E402
    DEFAULT_MAPPING_CSV,
    consolidate_dfci_labs,
)
from data_preprocessing_common.projects.ipio import UNIQUE_LABS_CSV  # noqa: E402

ID_COL = "DFCI_MRN"

DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/IPIO/")

MAPPING_CSV = DEFAULT_MAPPING_CSV
# Regenerated below as an inventory/diagnostic artifact of the raw TEST_NAME/unit
# combinations seen in this cohort's labs. consolidate_dfci_labs() itself only
# takes (labs_df, mapping_df); this inventory file only feeds the optional
# data_preprocessing_common.dfci_labs refresh-mapping CLI.

COHORT_CSV = DATA_PATH / "irae_cohort_data.csv"
LABS_CSV = DATA_PATH / "irae_labs_data.csv"
OUTPUT_CSV = DATA_PATH / "longitudinal_prediction_data.csv"


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


CANCER_TYPE_PREFIX = "CANCER_TYPE_"
CANCER_TYPE_OTHER_COL = f"{CANCER_TYPE_PREFIX}OTHER"
DEFAULT_MIN_CANCER_TYPE_COUNT = 200


def merge_rare_cancer_types(
    cohort_df: pd.DataFrame,
    cancer_type_cols: list[str],
    *,
    min_count: int = DEFAULT_MIN_CANCER_TYPE_COUNT,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge cancer types with fewer than `min_count` patients in the whole
    cohort into a single CANCER_TYPE_OTHER bucket, so rare types don't each
    get their own mostly-empty, unstable covariate. Counted on the full
    compiled cohort (one row per patient), before any lab-based landmark
    eligibility filtering.
    """
    counts = cohort_df[cancer_type_cols].sum(axis=0)
    rare_cols = counts.loc[counts < min_count].index.tolist()
    common_cols = [c for c in cancer_type_cols if c not in rare_cols]
    if not rare_cols:
        print(f"No cancer types below the {min_count}-patient merge threshold.")
        return cohort_df, cancer_type_cols

    cohort_df = cohort_df.copy()
    other_flag = cohort_df[rare_cols].sum(axis=1).gt(0).astype(int)
    if CANCER_TYPE_OTHER_COL in common_cols:
        # An existing (already-common) OTHER-named column would collide; fold
        # the newly-merged rare types into it instead of overwriting.
        other_flag = (cohort_df[CANCER_TYPE_OTHER_COL].fillna(0).astype(int) | other_flag).astype(int)
    cohort_df[CANCER_TYPE_OTHER_COL] = other_flag
    cohort_df = cohort_df.drop(columns=[c for c in rare_cols if c != CANCER_TYPE_OTHER_COL])
    merged_cols = common_cols + (
        [] if CANCER_TYPE_OTHER_COL in common_cols else [CANCER_TYPE_OTHER_COL]
    )
    print(
        f"Merged {len(rare_cols)} cancer type(s) with <{min_count} cohort patients into "
        f"'{CANCER_TYPE_OTHER_COL}' ({int(other_flag.sum())} patients): {', '.join(sorted(rare_cols))}"
    )
    return cohort_df, merged_cols


def resolve_cancer_type_columns(
    cohort_df: pd.DataFrame,
    *,
    min_type_count: int = DEFAULT_MIN_CANCER_TYPE_COUNT,
) -> tuple[pd.DataFrame, list[str]]:
    """Discover the cancer-type indicator columns already present in cohort_df.

    No fixed list of cancer types is assumed -- compile_irae_data.py's
    load_cancer_type_df() treats the source cancer_type_df.csv.gz as already
    wide/one-hot (one column per type) and prefixes every such column
    `CANCER_TYPE_<NAME>`. This just finds whichever of those columns are
    present, so a change in how many/which types exist upstream needs no
    code change here.

    Types with fewer than `min_type_count` patients in the cohort are merged
    into a single CANCER_TYPE_OTHER bucket first (see merge_rare_cancer_types).

    If every patient's (post-merge) CANCER_TYPE_* row sums to exactly 1 (a
    true mutually exclusive, exhaustive one-hot partition), the most
    prevalent column is dropped as the implicit reference category -- mirrors
    COMPASS's CANCER_STAGE_II/III/IV pattern (Stage I is never emitted, kept
    implicit) and avoids perfect collinearity in the Cox design matrix. If the
    columns are NOT mutually exclusive/exhaustive (e.g. a patient can have 0
    or 2+ flags set), there's no natural reference category and no
    collinearity risk, so all columns are kept.
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

    cohort_df, cancer_type_cols = merge_rare_cancer_types(
        cohort_df, cancer_type_cols, min_count=min_type_count
    )

    indicators = cohort_df[cancer_type_cols]
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


def build_raw_longitudinal_labs(labs_df: pl.DataFrame) -> pl.DataFrame:
    """Reshape raw OncDRS lab rows into the long ID_COL/DATE/LAB_NAME/LAB_UNIT/LAB_VALUE
    schema expected by data_preprocessing_common.dfci_labs.consolidate_dfci_labs.
    This mirrors the labs half of COMPASS's build_raw_longitudinal_data; IPIO has
    no HEALTH_HISTORY vital-signs table to fold in.
    """
    working = labs_df.select(
        [
            ID_COL,
            "SPECIMEN_COLLECT_DT",
            "TEST_TYPE_CD",
            "TEST_TYPE_DESCR",
            "NUMERIC_RESULT",
            "RESULT_UOM_NM",
        ]
    )

    working = working.with_columns(
        generate_new_test_name_expr("TEST_TYPE_CD", "TEST_TYPE_DESCR").alias("TEST_NAME")
    )

    working = working.rename(
        {
            "SPECIMEN_COLLECT_DT": "DATE",
            "NUMERIC_RESULT": "LAB_VALUE",
            "RESULT_UOM_NM": "LAB_UNIT",
            "TEST_NAME": "LAB_NAME",
        }
    ).select([ID_COL, "DATE", "LAB_NAME", "LAB_UNIT", "LAB_VALUE"])

    return working


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cohort_df = pd.read_csv(COHORT_CSV)

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
    # FIRST_TREATMENT = 1 for everyone, so survival_common.cohort.build_feature_matrix's
    # default anchor (t_first_treatment = FIRST_TREATMENT_DATE - FIRST_RECORD_DATE)
    # is exactly 0 for every patient without any code changes downstream.
    cohort_df["FIRST_RECORD_DATE"] = cohort_df["IO_START"]
    cohort_df["FIRST_TREATMENT_DATE"] = cohort_df["IO_START"]
    cohort_df["FIRST_TREATMENT"] = 1
    cohort_df["LAST_CONTACT_DATE"] = cohort_df["LAST_DATE"]
    cohort_df["IRAE"] = cohort_df["event"].astype(str).eq("irAE").astype(int)
    # DEATH: derived from the same raw `event` categorical column as IRAE
    # (values irAE / death / censor). IPIO has no separate death-date column --
    # LAST_DATE IS the death date when event=='death' (same source as
    # LAST_CONTACT_DATE), matching how t_irae already uses LAST_CONTACT_DATE
    # for every patient regardless of event type. Used downstream by
    # ipio_cohort.make_irae_outcome_df to build event_type for Fine-Gray
    # competing-risks univariate fitting (survival_common.finegray).
    cohort_df["DEATH"] = cohort_df["event"].astype(str).str.lower().eq("death").astype(int)

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
            "DEATH",
        ]
    )
    static_df = cohort_df[static_cols].copy()

    # --- Shared DFCI lab standardization (polars reshape up to consolidate_dfci_labs) ---
    labs_df_pl = pl.scan_csv(LABS_CSV, infer_schema_length=0).collect()
    raw_longitudinal_labs_pl = build_raw_longitudinal_labs(labs_df_pl)

    unique_labs_df_pl = (
        raw_longitudinal_labs_pl.group_by(["LAB_NAME", "LAB_UNIT"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    unique_labs_df_pl.write_csv(UNIQUE_LABS_CSV)

    # Conversion boundary: consolidate_dfci_labs (data_preprocessing_common/dfci_labs.py)
    # takes and returns pandas DataFrames and is out of scope for this port. Convert
    # polars->pandas here; everything from this point on stays pandas, unchanged.
    raw_longitudinal_labs = raw_longitudinal_labs_pl.to_pandas()

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
            "DEATH",
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
    death_rate = static_df["DEATH"].mean() if n_patients else float("nan")
    print(f"Cohort patients: {n_patients}")
    print(f"Patients with >=1 standardized lab row: {n_patients_with_labs}")
    print(f"IRAE event rate (cohort-level): {irae_rate:.4f}")
    print(f"DEATH event rate (cohort-level, competing event for irAE): {death_rate:.4f}")
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
