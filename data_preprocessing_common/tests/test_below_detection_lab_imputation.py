import json
import unittest

import pandas as pd

from data_preprocessing_common.dfci_labs import (
    _consolidate_dfci_labs_rowwise,
    consolidate_dfci_labs,
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


class BelowDetectionLabImputationTest(unittest.TestCase):
    def test_only_psa_and_testosterone_999999_with_less_than_are_zero(self) -> None:
        labs = pd.DataFrame(
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

        result = consolidate_dfci_labs(labs, _mapping()).set_index("row_id")

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


if __name__ == "__main__":
    unittest.main()
