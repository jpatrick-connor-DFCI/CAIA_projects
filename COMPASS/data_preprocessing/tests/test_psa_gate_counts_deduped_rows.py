import polars as pl

from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (
    MIN_PSA_COUNT,
    build_icd_prostate_mrn_flags,
)


def _write_labs_csv(path):
    # Patient 1: 3 genuinely distinct PSA draws, plus 4 exact-duplicate rows
    # of the third draw (same MRN/date/code/descr/result/text/unit) --
    # e.g. the same OncDRS pull landing in the raw CSV more than once. Raw
    # row count is 7 (>= MIN_PSA_COUNT=5), but only 3 distinct logical draws
    # exist. Deduping on the canonical LABS columns should collapse the 4
    # duplicates to 1, leaving 3 distinct rows -- below MIN_PSA_COUNT -- so
    # patient 1 must be excluded from HAS_5_OR_MORE_PSA_TESTS after the port.
    #
    # Patient 2: 5 genuinely distinct PSA draws (different dates/values), no
    # duplicates -- a real >=5-PSA patient who must stay included, so the
    # test also catches an over-aggressive dedup that wrongly collapses
    # distinct draws.
    rows = [
        # Patient 1: draws 1-2, then draw 3 duplicated x4.
        ("1", "2024-01-01", "PSA", "PSA", "1.0", "1.0", "ng/ml"),
        ("1", "2024-02-01", "PSA", "PSA", "2.0", "2.0", "ng/ml"),
        ("1", "2024-03-01", "PSA", "PSA", "3.0", "3.0", "ng/ml"),
        ("1", "2024-03-01", "PSA", "PSA", "3.0", "3.0", "ng/ml"),
        ("1", "2024-03-01", "PSA", "PSA", "3.0", "3.0", "ng/ml"),
        ("1", "2024-03-01", "PSA", "PSA", "3.0", "3.0", "ng/ml"),
        ("1", "2024-03-01", "PSA", "PSA", "3.0", "3.0", "ng/ml"),
        # Patient 2: 5 genuinely distinct draws.
        ("2", "2024-01-01", "PSA", "PSA", "1.0", "1.0", "ng/ml"),
        ("2", "2024-02-01", "PSA", "PSA", "2.0", "2.0", "ng/ml"),
        ("2", "2024-03-01", "PSA", "PSA", "3.0", "3.0", "ng/ml"),
        ("2", "2024-04-01", "PSA", "PSA", "4.0", "4.0", "ng/ml"),
        ("2", "2024-05-01", "PSA", "PSA", "5.0", "5.0", "ng/ml"),
    ]
    header = (
        "DFCI_MRN,SPECIMEN_COLLECT_DT,TEST_TYPE_CD,TEST_TYPE_DESCR,"
        "NUMERIC_RESULT,TEXT_RESULT,RESULT_UOM_NM\n"
    )
    body = "\n".join(",".join(r) for r in rows) + "\n"
    path.write_text(header + body)


def test_psa_gate_counts_deduped_rows(tmp_path):
    labs_path = tmp_path / "OUTPT_LAB_RESULTS_LABS.csv"
    _write_labs_csv(labs_path)

    assert MIN_PSA_COUNT == 5  # sanity: the fixture is built against this

    meds = pl.DataFrame(
        {"DFCI_MRN": pl.Series([], dtype=pl.Int64), "NCI_PREFERRED_MED_NM": pl.Series([], dtype=pl.Utf8)}
    )

    flags = build_icd_prostate_mrn_flags(
        c61_mrns={1, 2},
        non_prostate_primary_mrns=set(),
        post_adt_exclusion_cancer_mrns=set(),
        meds=meds,
        labs_path=str(labs_path),
    )

    flags_by_mrn = {row["DFCI_MRN"]: row["HAS_5_OR_MORE_PSA_TESTS"] for row in flags.to_dicts()}

    # Patient 1's raw row count (7) clears MIN_PSA_COUNT, but only 3 rows are
    # distinct after intra-release dedup -- excluded.
    assert flags_by_mrn[1] == 0
    # Patient 2 has 5 genuinely distinct draws -- still included.
    assert flags_by_mrn[2] == 1
