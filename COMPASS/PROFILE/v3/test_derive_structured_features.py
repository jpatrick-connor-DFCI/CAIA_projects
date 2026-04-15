import sys
import unittest
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from derive_structured_features import (  # noqa: E402
    build_lab_feature_table,
    build_somatic_feature_table,
    standardize_labs_input,
    value_is_positive,
)


class StructuredFeaturesTest(unittest.TestCase):
    def test_value_is_positive_handles_common_encodings(self):
        self.assertTrue(value_is_positive(1))
        self.assertTrue(value_is_positive("Pathogenic mutation"))
        self.assertFalse(value_is_positive(0))
        self.assertFalse(value_is_positive("wild-type"))

    def test_standardize_labs_input_creates_single_test_name_column(self):
        labs_df = pd.DataFrame(
            [
                {
                    "DFCI_MRN": 1,
                    "TEST_TYPE_CD": "LDH",
                    "TEST_TYPE_DESCR": "LDH (LACTATE DEHYDROGENASE)",
                    "NUMERIC_RESULT": 100,
                    "RESULT_UOM_NM": "U/L",
                }
            ]
        )

        standardized = standardize_labs_input(labs_df)

        self.assertEqual(1, standardized.columns.tolist().count("TEST_NAME"))
        self.assertEqual("LDH (LACTATE DEHYDROGENASE)", standardized.loc[0, "TEST_NAME"])

    def test_build_lab_feature_table_derives_peak_markers(self):
        labs_df = pd.DataFrame(
            [
                {
                    "DFCI_MRN": 1,
                    "collapsed_measurement": "LDH",
                    "numeric_result_standardized": 600.0,
                    "LAB_DATE": pd.Timestamp("2026-01-05"),
                },
                {
                    "DFCI_MRN": 1,
                    "collapsed_measurement": "CEA",
                    "numeric_result_standardized": 12.0,
                    "LAB_DATE": pd.Timestamp("2026-01-07"),
                },
                {
                    "DFCI_MRN": 1,
                    "collapsed_measurement": "Calcium",
                    "numeric_result_standardized": 10.8,
                    "LAB_DATE": pd.Timestamp("2026-01-08"),
                },
                {
                    "DFCI_MRN": 2,
                    "collapsed_measurement": "LDH",
                    "numeric_result_standardized": 200.0,
                    "LAB_DATE": pd.Timestamp("2026-01-01"),
                },
            ]
        )

        feature_df = build_lab_feature_table(labs_df, [1, 2]).set_index("DFCI_MRN")

        self.assertEqual(600.0, feature_df.loc[1, "peak_ldh_value"])
        self.assertTrue(feature_df.loc[1, "ldh_ge_2x_uln"])
        self.assertTrue(feature_df.loc[1, "cea_ge_2x_uln"])
        self.assertTrue(feature_df.loc[1, "hypercalcemia_present"])
        self.assertTrue(feature_df.loc[1, "c6_supportive_lab_pattern_present"])
        self.assertFalse(feature_df.loc[2, "c6_supportive_lab_pattern_present"])

    def test_build_somatic_feature_table_aggregates_gene_flags(self):
        somatic_df = pd.DataFrame(
            [
                {
                    "DFCI_MRN": 1,
                    "BRCA2_status": "Pathogenic",
                    "ATM_flag": 0,
                    "HRD_signature": 1,
                },
                {
                    "DFCI_MRN": 1,
                    "BRCA2_status": "negative",
                    "ATM_flag": 1,
                    "HRD_signature": 0,
                },
                {
                    "DFCI_MRN": 2,
                    "BRCA2_status": "wild-type",
                    "ATM_flag": 0,
                    "HRD_signature": 0,
                },
            ]
        )

        feature_df = build_somatic_feature_table(somatic_df, [1, 2]).set_index("DFCI_MRN")

        self.assertTrue(feature_df.loc[1, "has_brca2_alteration"])
        self.assertTrue(feature_df.loc[1, "has_atm_alteration"])
        self.assertTrue(feature_df.loc[1, "has_hrd_pathway_alteration"])
        self.assertTrue(feature_df.loc[1, "has_any_dna_repair_alteration"])
        self.assertFalse(feature_df.loc[2, "has_any_dna_repair_alteration"])


if __name__ == "__main__":
    unittest.main()
