import sys
import unittest
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from merge_labels import assign_final_bucket, needs_manual_review  # noqa: E402


class MergeLabelsTest(unittest.TestCase):
    def test_precedence_prefers_nepc_then_avpc_then_biomarker(self):
        self.assertEqual(
            "nepc",
            assign_final_bucket(
                {
                    "nepc_has_nepc_signal": True,
                    "avpc_has_avpc_features": True,
                    "biomarker_has_biomarker_signal": True,
                }
            ),
        )
        self.assertEqual(
            "avpc",
            assign_final_bucket(
                {
                    "nepc_has_nepc_signal": False,
                    "avpc_has_avpc_features": True,
                    "biomarker_has_biomarker_signal": True,
                }
            ),
        )
        self.assertEqual(
            "biomarker",
            assign_final_bucket(
                {
                    "nepc_has_nepc_signal": False,
                    "avpc_has_avpc_features": False,
                    "biomarker_has_biomarker_signal": True,
                }
            ),
        )

    def test_manual_review_only_when_no_positive_bucket_and_null_present(self):
        self.assertTrue(
            needs_manual_review(
                {
                    "nepc_has_nepc_signal": None,
                    "avpc_has_avpc_features": False,
                    "biomarker_has_biomarker_signal": False,
                }
            )
        )
        self.assertFalse(
            needs_manual_review(
                {
                    "nepc_has_nepc_signal": False,
                    "avpc_has_avpc_features": False,
                    "biomarker_has_biomarker_signal": False,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
