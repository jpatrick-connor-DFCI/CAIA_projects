import sys
import unittest
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from evidence_filters import (  # noqa: E402
    has_substantive_v3_evidence,
    is_test_only_quote,
    sanitize_note_extractions,
)


class EvidenceFiltersTest(unittest.TestCase):
    def test_flags_testing_plan_as_test_only(self):
        quote = "Will send synaptophysin/chromogranin stains to test for aggressive variant or NEPC."
        self.assertTrue(is_test_only_quote(quote))

    def test_keeps_confirmed_aggressive_variant_quote(self):
        quote = "Clinical picture is most consistent with aggressive variant prostate cancer."
        self.assertFalse(is_test_only_quote(quote))

    def test_sanitize_removes_test_only_disease_mentions(self):
        note_extractions = [
            {
                "note_date": "2026-01-10",
                "note_type": "Clinician",
                "nepc_scpc_mentions": [
                    {
                        "label": "neuroendocrine",
                        "assertion": "possible",
                        "event_date": None,
                        "quote": "Will send synaptophysin/chromogranin stains to test for NEPC.",
                    }
                ],
                "transformation_mentions": [],
                "aggressive_variant_mentions": [
                    {
                        "label": "aggressive_variant",
                        "assertion": "possible",
                        "event_date": None,
                        "quote": "Biopsy ordered to evaluate for aggressive variant disease.",
                    }
                ],
                "avpc_criteria_mentions": [],
                "biomarker_mentions": [],
                "treatment_resistance_mentions": [],
                "platinum_mentions": [],
                "overall_relevance": "medium",
            }
        ]

        sanitized = sanitize_note_extractions(note_extractions)

        self.assertEqual([], sanitized[0]["nepc_scpc_mentions"])
        self.assertEqual([], sanitized[0]["aggressive_variant_mentions"])
        self.assertEqual("low", sanitized[0]["overall_relevance"])
        self.assertFalse(has_substantive_v3_evidence(sanitized))

    def test_supporting_mentions_still_count_as_evidence(self):
        note_extractions = [
            {
                "note_date": "2026-01-11",
                "note_type": "Clinician",
                "nepc_scpc_mentions": [],
                "transformation_mentions": [],
                "aggressive_variant_mentions": [],
                "avpc_criteria_mentions": [],
                "biomarker_mentions": [
                    {
                        "marker": "BRCA2",
                        "platinum_linked": "true",
                        "quote": "Given BRCA2 loss, carboplatin was selected.",
                    }
                ],
                "treatment_resistance_mentions": [],
                "platinum_mentions": [],
                "overall_relevance": "high",
            }
        ]

        sanitized = sanitize_note_extractions(note_extractions)

        self.assertEqual(1, len(sanitized[0]["biomarker_mentions"]))
        self.assertTrue(has_substantive_v3_evidence(sanitized))


if __name__ == "__main__":
    unittest.main()
