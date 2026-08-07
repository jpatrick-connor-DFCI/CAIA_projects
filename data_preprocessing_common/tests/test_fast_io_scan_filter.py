from data_preprocessing_common import fast_io
from data_preprocessing_common.oncdrs_dedup import DATA_COLUMNS


def test_scan_filter_dedup_plan_shape(tmp_path):
    """Verification step (f) from the dedup port plan: assert the lazy plan
    for `scan_filter(..., table="LABS", cols=[...])` shows

      1. the `.unique()` subset is the 7 canonical LABS columns
         (DATA_COLUMNS["LABS"]), not the caller's narrower `cols=`, and
      2. the MRN cohort filter is pushed below the `.unique()` (down into the
         CSV scan itself), rather than hand-optimized -- polars' predicate
         pushdown does this on its own.

    This is a plan-shape assertion (`.explain()`), not a row-count test --
    row-count coverage of the same invariant lives in
    test_oncdrs_dedup.py::test_labs_dedup_ignores_caller_projection.
    """
    csv_path = tmp_path / "OUTPT_LAB_RESULTS_LABS.csv"
    csv_path.write_text(
        "DFCI_MRN,SPECIMEN_COLLECT_DT,TEST_TYPE_CD,TEST_TYPE_DESCR,"
        "NUMERIC_RESULT,TEXT_RESULT,RESULT_UOM_NM\n"
        "1,2024-01-01,PSA,PSA,1.5,1.5,ng/ml\n"
    )

    lf = fast_io.scan_filter(
        str(csv_path),
        cohort_mrns={1},
        cols=["DFCI_MRN", "TEST_TYPE_CD"],
        table="LABS",
    )
    plan = lf.explain()

    assert "UNIQUE" in plan

    # The unique's subset ("BY [...]") must be the full canonical LABS column
    # set, not the caller's 2-column projection.
    for col in DATA_COLUMNS["LABS"]:
        assert col in plan
    assert plan.count("TEXT_RESULT") >= 1  # not in cols=, so only survives if dedup used the canonical set

    # The MRN filter must appear below (after, in top-down .explain() text)
    # the UNIQUE node -- i.e. pushed down into the scan rather than applied
    # only at the top of the plan.
    unique_pos = plan.index("UNIQUE")
    selection_pos = plan.index("SELECTION")
    assert selection_pos > unique_pos, (
        "expected the MRN cohort filter (SELECTION) to be pushed below "
        "UNIQUE in the plan text:\n" + plan
    )
