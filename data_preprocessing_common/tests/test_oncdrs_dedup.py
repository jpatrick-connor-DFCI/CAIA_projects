import polars as pl
import pytest

from data_preprocessing_common import fast_io
from data_preprocessing_common.oncdrs_dedup import (
    COALESCE_TABLES,
    DATA_COLUMNS,
    DEDUP_COL_MAP,
    apply_dedup,
    dedup_key_columns,
)
from data_preprocessing_common.oncdrs_sources import TABLE_FILES


def test_labs_exact_row_dedup(tmp_path):
    # Two byte-identical rows collapse to 1; a third differing only in
    # TEXT_RESULT is a distinct row and survives alongside them.
    labs = pl.DataFrame(
        {
            "DFCI_MRN": ["1", "1", "1"],
            "SPECIMEN_COLLECT_DT": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "TEST_TYPE_CD": ["PSA", "PSA", "PSA"],
            "TEST_TYPE_DESCR": ["PSA", "PSA", "PSA"],
            "NUMERIC_RESULT": ["1.5", "1.5", "1.5"],
            "TEXT_RESULT": ["1.5", "1.5", "2.5"],
            "RESULT_UOM_NM": ["ng/ml", "ng/ml", "ng/ml"],
        }
    )
    result = apply_dedup(labs.lazy(), "LABS").collect()
    assert len(result) == 2
    assert set(result["TEXT_RESULT"]) == {"1.5", "2.5"}


def test_labs_dedup_ignores_caller_projection(tmp_path):
    """The load-bearing test: dedup must run on the canonical LABS key set,
    not on a caller's narrower `cols=` projection.

    The two rows below are IDENTICAL on the caller's projection
    [DFCI_MRN, TEST_TYPE_CD] but differ on NUMERIC_RESULT/TEXT_RESULT --
    real, distinct lab results drawn on the same day. That is what makes this
    test load-bearing: deduping on the caller's projection (as
    compile_COMPASS_cohort_data.py's PSA gate projects) would collapse them to
    1 row, while deduping on the full canonical LABS column set correctly keeps
    both. Every column outside the projection must stay constant except the
    result values, or the rows would be trivially distinct and the test would
    pass even with dedup disabled.
    """
    csv_path = tmp_path / "OUTPT_LAB_RESULTS_LABS.csv"
    csv_path.write_text(
        "DFCI_MRN,SPECIMEN_COLLECT_DT,TEST_TYPE_CD,TEST_TYPE_DESCR,"
        "NUMERIC_RESULT,TEXT_RESULT,RESULT_UOM_NM\n"
        "1,2024-01-01,PSA,PSA,1.5,1.5,ng/ml\n"
        "1,2024-01-01,PSA,PSA,2.5,2.5,ng/ml\n"
    )

    result = fast_io.scan_filter(
        str(csv_path),
        cohort_mrns={1},
        cols=["DFCI_MRN", "TEST_TYPE_CD"],
        table="LABS",
    ).collect()

    assert len(result) == 2


def test_ehr_diagnoses_dedup_key():
    diagnoses = pl.DataFrame(
        {
            "DFCI_MRN": ["1", "1", "1", "1"],
            "START_DT": ["2024-01-01", "2024-01-01", None, None],
            "END_DT": [None, None, None, None],
            "DIAGNOSIS_ICD10_CD": ["C61", "C61", "C61", "C61"],
            "DIAGNOSIS_ICD10_NM": ["Prostate cancer", "Malignant neoplasm of prostate", "Prostate cancer", "Prostate cancer"],
            "DIAGNOSIS_ICD10_CD2": [None, None, None, "C62"],
            "DIAGNOSIS_ICD10_NM2": [None, None, None, "Testis cancer"],
            "DIAGNOSIS_ICD10_CD3": [None, None, None, None],
            "DIAGNOSIS_ICD10_NM3": [None, None, None, None],
        }
    )
    result = apply_dedup(diagnoses.lazy(), "EHR_DIAGNOSES").collect()

    # Rows 0/1 differ only in DIAGNOSIS_ICD10_NM (not a key column) -> collapse.
    # Row 2 shares DFCI_MRN/START_DT=null(collapsed with itself)/END_DT/CD/CD2/CD3
    # with row... actually rows 2 and 3 differ in _CD2 -> stay distinct.
    # So surviving groups: {row0/row1 key}, {row2 key}, {row3 key} = 3 rows.
    assert len(result) == 3


def test_pt_info_coalesce_fills_from_sparse_rows():
    pt_info = pl.DataFrame(
        {
            "DFCI_MRN": ["1", "1"],
            "BIRTH_DT": ["1950-01-01", None],
            "CLIN_DEATH_DT": [None, None],
            "HYBRID_DEATH_DT": [None, "2024-01-01"],
            "NDI_DEATH_DT": [None, None],
            "DERIVED_LAST_ALIVE_DATE": [None, None],
            "GENDER_NM": ["MALE", None],
            "GENDER_CD": [None, "M"],
            "PT_ONCOPANEL_PROFILED_IND": [None, None],
        }
    )
    result = apply_dedup(pt_info.lazy(), "PT_INFO_STATUS_REGISTRATION").collect()

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["BIRTH_DT"] == "1950-01-01"
    assert row["HYBRID_DEATH_DT"] == "2024-01-01"
    assert row["GENDER_NM"] == "MALE"
    assert row["GENDER_CD"] == "M"


def test_dedup_is_idempotent():
    # Backs the "profile_data dedup is a no-op" claim: running apply_dedup
    # twice on an already-deduped frame must not remove any more rows, for
    # every table this module knows about.
    frames = {
        "EHR_DIAGNOSES": pl.DataFrame(
            {
                "DFCI_MRN": ["1", "2"],
                "START_DT": ["2024-01-01", "2024-02-01"],
                "END_DT": [None, None],
                "DIAGNOSIS_ICD10_CD": ["C61", "C62"],
                "DIAGNOSIS_ICD10_NM": ["A", "B"],
                "DIAGNOSIS_ICD10_CD2": [None, None],
                "DIAGNOSIS_ICD10_NM2": [None, None],
                "DIAGNOSIS_ICD10_CD3": [None, None],
                "DIAGNOSIS_ICD10_NM3": [None, None],
            }
        ),
        "MEDICATIONS": pl.DataFrame(
            {
                "DFCI_MRN": ["1", "2"],
                "NCI_PREFERRED_MED_NM": ["LEUPROLIDE ACETATE", "CARBOPLATIN"],
                "MED_START_DT": ["2024-01-01", "2024-02-01"],
            }
        ),
        "LABS": pl.DataFrame(
            {
                "DFCI_MRN": ["1", "2"],
                "SPECIMEN_COLLECT_DT": ["2024-01-01", "2024-02-01"],
                "TEST_TYPE_CD": ["PSA", "PSA"],
                "TEST_TYPE_DESCR": ["PSA", "PSA"],
                "NUMERIC_RESULT": ["1.5", "2.5"],
                "TEXT_RESULT": ["1.5", "2.5"],
                "RESULT_UOM_NM": ["ng/ml", "ng/ml"],
            }
        ),
        "HEALTH_HISTORY": pl.DataFrame(
            {
                "DFCI_MRN": ["1", "2"],
                "START_DT": ["2024-01-01", "2024-02-01"],
                "CODE": ["A", "B"],
                "HEALTH_HISTORY_TYPE": ["X", "Y"],
                "CODE_TYPE": ["ICD10", "ICD10"],
                "RESULTS": [None, None],
                "UNITS_CD": [None, None],
            }
        ),
        "PT_INFO_STATUS_REGISTRATION": pl.DataFrame(
            {
                "DFCI_MRN": ["1", "2"],
                "BIRTH_DT": ["1950-01-01", "1960-01-01"],
                "CLIN_DEATH_DT": [None, None],
                "HYBRID_DEATH_DT": [None, None],
                "NDI_DEATH_DT": [None, None],
                "DERIVED_LAST_ALIVE_DATE": [None, None],
                "GENDER_NM": ["MALE", "MALE"],
                "GENDER_CD": [None, None],
                "PT_ONCOPANEL_PROFILED_IND": [None, None],
            }
        ),
    }

    for table in TABLE_FILES:
        once = apply_dedup(frames[table].lazy(), table).collect()
        twice = apply_dedup(once.lazy(), table).collect()
        assert len(twice) == len(once), f"{table}: dedup is not idempotent"


def test_unknown_table_and_missing_key_columns():
    frame = pl.DataFrame({"DFCI_MRN": ["1", "1"], "FOO": ["a", "a"]})

    # Unknown table: pass through unchanged (no KeyError, no row removal).
    passthrough = apply_dedup(frame.lazy(), "NOT_A_REAL_TABLE").collect()
    assert len(passthrough) == 2

    # A None-mapped table (LABS) missing all but one of its key columns must
    # raise rather than silently deduping on whatever survives the
    # intersection -- that would collapse the table to near-nothing.
    sparse_labs = pl.DataFrame({"DFCI_MRN": ["1", "1"]})
    with pytest.raises(RuntimeError):
        apply_dedup(sparse_labs.lazy(), "LABS")


def test_dedup_tables_match_table_files():
    assert set(DEDUP_COL_MAP) >= set(TABLE_FILES)
    for table, key_cols in DEDUP_COL_MAP.items():
        if key_cols is None:
            continue
        assert set(key_cols) <= set(DATA_COLUMNS[table]), (
            f"{table}: DEDUP_COL_MAP has a key column not in DATA_COLUMNS"
        )
    assert COALESCE_TABLES <= set(TABLE_FILES)


def test_parquet_input_skips_dedup(tmp_path):
    # profile_data parquets are deduped by construction upstream; scan_filter
    # must not re-run apply_dedup against them (it would be a pure-cost
    # no-op at best -- this test documents the skip is intentional).
    dup_rows = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1],
            "SPECIMEN_COLLECT_DT": ["2024-01-01", "2024-01-01"],
            "TEST_TYPE_CD": ["PSA", "PSA"],
            "TEST_TYPE_DESCR": ["PSA", "PSA"],
            "NUMERIC_RESULT": [1.5, 1.5],
            "TEXT_RESULT": ["1.5", "1.5"],
            "RESULT_UOM_NM": ["ng/ml", "ng/ml"],
        }
    )
    parquet_path = tmp_path / "LABS.parquet"
    dup_rows.write_parquet(parquet_path)

    result = fast_io.scan_filter(
        str(parquet_path), cohort_mrns={1}, table="LABS"
    ).collect()

    assert len(result) == 2
