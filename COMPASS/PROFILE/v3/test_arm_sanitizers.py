import sys
import unittest
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from arms import avpc, biomarker, nepc  # noqa: E402


class ArmSanitizersTest(unittest.TestCase):
    def test_avpc_sanitizer_tolerates_boolean_mentions(self):
        sanitized = avpc.sanitize_note_extractions(
            [
                {
                    "note_date": "2026-01-01",
                    "note_type": "Clinician",
                    "explicit_avpc_mentions": False,
                    "criteria_mentions": True,
                    "overall_relevance": "high",
                }
            ]
        )

        self.assertEqual([], sanitized[0]["explicit_avpc_mentions"])
        self.assertEqual([], sanitized[0]["criteria_mentions"])
        self.assertEqual("low", sanitized[0]["overall_relevance"])

    def test_nepc_sanitizer_tolerates_singleton_dict(self):
        sanitized = nepc.sanitize_note_extractions(
            [
                {
                    "note_date": "2026-01-01",
                    "note_type": "Pathology",
                    "histology_mentions": {
                        "label": "small_cell",
                        "assertion": "present",
                        "quote": "Small cell carcinoma present.",
                    },
                    "transformation_mentions": None,
                    "overall_relevance": "high",
                }
            ]
        )

        self.assertEqual(1, len(sanitized[0]["histology_mentions"]))
        self.assertEqual([], sanitized[0]["transformation_mentions"])

    def test_biomarker_sanitizer_tolerates_boolean_mentions(self):
        sanitized = biomarker.sanitize_note_extractions(
            [
                {
                    "note_date": "2026-01-01",
                    "note_type": "Clinician",
                    "biomarker_mentions": False,
                    "overall_relevance": "medium",
                }
            ]
        )

        self.assertEqual([], sanitized[0]["biomarker_mentions"])
        self.assertEqual("low", sanitized[0]["overall_relevance"])


if __name__ == "__main__":
    unittest.main()
