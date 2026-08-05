import contextlib
import io
import json
import unittest

import pandas as pd

from data_preprocessing_common.dfci_labs import (
    _consolidate_dfci_labs_rowwise,
    consolidate_dfci_labs,
    print_below_detection_report,
)


def _mapping() -> pd.DataFrame:
    rows = []
    for concept_id, measurement, test_name in (
        (1, "PSA", "PSA"),
        (2, "Testosterone", "TESTOSTERONE"),
        (3, "Creatinine", "CREATININE"),
    ):
        rows.append(
            {
                "mapping_status": "mapped",
                "collapsed_measurement": measurement,
                "measurement_concept_id": concept_id,
                "omop_measurement_name": measurement,
                "mapped_test_names_json": json.dumps([test_name]),
                "mapped_test_name_prefixes_json": json.dumps([test_name]),
            }
        )
    return pd.DataFrame(rows)


def _boundary_labs() -> pd.DataFrame:
    """Every side of the imputation rule, one row each.

    row 0  PSA          999999   "<0.1"  -> imputed to 0
    row 1  Testosterone "999999" " < 3"  -> imputed to 0 (numeric arrives as text)
    row 2  PSA          999999   "0.1"   -> sentinel, no '<'
    row 3  PSA          9999999  "<0.1"  -> '<', but a sentinel the rule skips
    row 4  Creatinine   999999   "<0.1"  -> right shape, wrong measurement
    row 5  PSA          -999999  "<0.1"  -> '<', negative sentinel variant
    row 6  PSA          999999   None    -> sentinel, TEXT_RESULT missing
    row 7  PSA          2.5      "<3"    -> ordinary result, untouched
    """
    return pd.DataFrame(
        {
            "row_id": range(8),
            "TEST_NAME": [
                "PSA",
                "TESTOSTERONE",
                "PSA",
                "PSA",
                "CREATININE",
                "PSA",
                "PSA",
                "PSA",
            ],
            "RESULT_UOM_NM": [
                "ng/ml",
                "ng/dl",
                "ng/ml",
                "ng/ml",
                "mg/dl",
                "ng/ml",
                "ng/ml",
                "ng/ml",
            ],
            "NUMERIC_RESULT": [
                999999,
                "999999",
                999999,
                9999999,
                999999,
                -999999,
                999999,
                2.5,
            ],
            "TEXT_RESULT": ["<0.1", " < 3", "0.1", "<0.1", "<0.1", "<0.1", None, "<3"],
        }
    )


class BelowDetectionLabImputationTest(unittest.TestCase):
    def test_only_psa_and_testosterone_999999_with_less_than_are_zero(self) -> None:
        result = consolidate_dfci_labs(_boundary_labs(), _mapping()).set_index("row_id")

        self.assertEqual(result.loc[0, "numeric_result_standardized"], 0.0)
        self.assertEqual(result.loc[1, "numeric_result_standardized"], 0.0)
        for row_id in (2, 3, 4, 5, 6):
            self.assertTrue(pd.isna(result.loc[row_id, "numeric_result_standardized"]))
        self.assertEqual(result.loc[7, "numeric_result_standardized"], 2.5)

    def test_vectorized_and_rowwise_paths_match(self) -> None:
        labs = pd.DataFrame(
            {
                "TEST_NAME": ["PSA", "TESTOSTERONE", "CREATININE"],
                "RESULT_UOM_NM": ["ng/ml", "ng/dl", "mg/dl"],
                "NUMERIC_RESULT": [999999, 999999, 999999],
                "TEXT_RESULT": ["<0.1", "<3", "<0.1"],
            }
        )

        vectorized = consolidate_dfci_labs(labs, _mapping())
        rowwise = _consolidate_dfci_labs_rowwise(labs, _mapping())
        pd.testing.assert_series_equal(
            vectorized["numeric_result_standardized"].reset_index(drop=True),
            pd.to_numeric(
                rowwise["numeric_result_standardized"], errors="coerce"
            ).reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )


class BelowDetectionReportTest(unittest.TestCase):
    def test_report_counts_imputed_rows_and_near_misses(self) -> None:
        report: dict = {}
        consolidate_dfci_labs(_boundary_labs(), _mapping(), report_out=report)

        self.assertEqual(report["n_rows_scanned"], 8)
        self.assertTrue(report["text_result_column_present"])
        self.assertEqual(report["n_text_result_non_null"], 7)  # row 6 is None
        self.assertEqual(report["n_zeroed_total"], 2)

        psa = report["per_measurement"]["PSA"]
        self.assertEqual(psa["n_zeroed"], 1)
        self.assertEqual(psa["n_sentinel_without_lt"], 2)  # rows 2 and 6
        self.assertEqual(psa["n_lt_with_other_sentinel"], 2)  # rows 3 and 5

        testosterone = report["per_measurement"]["Testosterone"]
        self.assertEqual(testosterone["n_zeroed"], 1)
        self.assertEqual(testosterone["n_sentinel_without_lt"], 0)
        self.assertEqual(testosterone["n_lt_with_other_sentinel"], 0)

        # Creatinine row 4 matches the numeric/text shape but is not an imputable
        # measurement, so it must not appear anywhere in the report.
        self.assertNotIn("Creatinine", report["per_measurement"])

        # No DFCI_MRN on this frame, so the patient counts are omitted rather
        # than reported as zero.
        self.assertNotIn("n_patients_zeroed_total", report)
        self.assertNotIn("n_patients_zeroed", psa)

    def test_report_counts_distinct_patients_when_id_column_present(self) -> None:
        labs = pd.DataFrame(
            {
                "DFCI_MRN": [1, 1, 2, 3],
                "TEST_NAME": ["PSA", "PSA", "PSA", "TESTOSTERONE"],
                "RESULT_UOM_NM": ["ng/ml", "ng/ml", "ng/ml", "ng/dl"],
                "NUMERIC_RESULT": [999999, 999999, 999999, 999999],
                "TEXT_RESULT": ["<0.1", "<0.1", "<0.1", "<3"],
            }
        )

        report: dict = {}
        consolidate_dfci_labs(labs, _mapping(), report_out=report)

        self.assertEqual(report["n_zeroed_total"], 4)
        self.assertEqual(report["n_patients_zeroed_total"], 3)
        self.assertEqual(report["per_measurement"]["PSA"]["n_zeroed"], 3)
        self.assertEqual(report["per_measurement"]["PSA"]["n_patients_zeroed"], 2)
        self.assertEqual(report["per_measurement"]["Testosterone"]["n_patients_zeroed"], 1)

    def test_report_flags_absent_text_result_column(self) -> None:
        # The silent-failure case: without TEXT_RESULT the mask is all-False, so
        # undetectables become NaN and n_zeroed alone looks like a clean cohort.
        # The near-miss counts are what reveal it.
        labs = _boundary_labs().drop(columns=["TEXT_RESULT"])

        report: dict = {}
        consolidate_dfci_labs(labs, _mapping(), report_out=report)

        self.assertFalse(report["text_result_column_present"])
        self.assertEqual(report["n_text_result_non_null"], 0)
        self.assertEqual(report["n_zeroed_total"], 0)
        self.assertEqual(report["per_measurement"]["PSA"]["n_sentinel_without_lt"], 3)
        self.assertEqual(report["per_measurement"]["Testosterone"]["n_sentinel_without_lt"], 1)

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_below_detection_report(report)
        self.assertIn("WARNING: TEXT_RESULT column is absent", printed.getvalue())

    def test_report_warns_when_text_result_is_entirely_null(self) -> None:
        labs = _boundary_labs()
        labs["TEXT_RESULT"] = None

        report: dict = {}
        consolidate_dfci_labs(labs, _mapping(), report_out=report)

        self.assertTrue(report["text_result_column_present"])
        self.assertEqual(report["n_text_result_non_null"], 0)
        self.assertEqual(report["n_zeroed_total"], 0)

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_below_detection_report(report)
        self.assertIn("entirely null", printed.getvalue())

    def test_report_prints_even_when_nothing_was_imputed(self) -> None:
        labs = pd.DataFrame(
            {
                "TEST_NAME": ["PSA"],
                "RESULT_UOM_NM": ["ng/ml"],
                "NUMERIC_RESULT": [2.5],
                "TEXT_RESULT": ["2.5"],
            }
        )

        report: dict = {}
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            consolidate_dfci_labs(labs, _mapping(), report_out=report)
        output = printed.getvalue()

        self.assertEqual(report["n_zeroed_total"], 0)
        # A silent zero reads exactly like the rule never having run, so both
        # measurements are always listed.
        self.assertIn("PSA", output)
        self.assertIn("Testosterone", output)
        self.assertIn("TOTAL", output)
        self.assertNotIn("WARNING", output)

    def test_report_matches_between_vectorized_and_rowwise_paths(self) -> None:
        vectorized_report: dict = {}
        rowwise_report: dict = {}
        consolidate_dfci_labs(
            _boundary_labs(), _mapping(), report_out=vectorized_report
        )
        _consolidate_dfci_labs_rowwise(
            _boundary_labs(), _mapping(), report_out=rowwise_report
        )

        self.assertEqual(vectorized_report, rowwise_report)


if __name__ == "__main__":
    unittest.main()
