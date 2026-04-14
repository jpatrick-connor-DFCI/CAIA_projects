import sys
import unittest
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from evidence_filters import (  # noqa: E402
    has_substantive_nepc_evidence,
    is_test_only_nepc_quote,
    sanitize_note_extractions,
)


class EvidenceFiltersTest(unittest.TestCase):
    def test_flags_testing_plan_as_test_only(self):
        quote = "Will send synaptophysin/chromogranin stains to test for NEPC."
        self.assertTrue(is_test_only_nepc_quote(quote))

    def test_flags_pending_biopsy_workup_as_test_only(self):
        quote = "Biopsy ordered to evaluate for small cell transformation."
        self.assertTrue(is_test_only_nepc_quote(quote))

    def test_keeps_confirmed_diagnosis_quote(self):
        quote = "Biopsy confirmed small cell carcinoma of prostatic origin."
        self.assertFalse(is_test_only_nepc_quote(quote))

    def test_sanitize_removes_test_only_mentions(self):
        note_extractions = [
            {
                "note_date": "2026-01-10",
                "note_type": "Clinician",
                "histology_mentions": [
                    {
                        "label": "neuroendocrine",
                        "assertion": "possible",
                        "event_date": None,
                        "quote": "Will send synaptophysin/chromogranin stains to test for NEPC.",
                    }
                ],
                "transformation_mentions": [],
                "overall_relevance": "medium",
            }
        ]

        sanitized = sanitize_note_extractions(note_extractions)

        self.assertEqual([], sanitized[0]["histology_mentions"])
        self.assertEqual("low", sanitized[0]["overall_relevance"])
        self.assertFalse(has_substantive_nepc_evidence(sanitized))

    def test_sanitize_preserves_confirmed_histology(self):
        note_extractions = [
            {
                "note_date": "2026-01-11",
                "note_type": "Pathology",
                "histology_mentions": [
                    {
                        "label": "small_cell",
                        "assertion": "present",
                        "event_date": "2026-01-11",
                        "quote": "Biopsy confirmed small cell carcinoma of prostatic origin.",
                    }
                ],
                "transformation_mentions": [],
                "overall_relevance": "high",
            }
        ]

        sanitized = sanitize_note_extractions(note_extractions)

        self.assertEqual(1, len(sanitized[0]["histology_mentions"]))
        self.assertEqual("high", sanitized[0]["overall_relevance"])
        self.assertTrue(has_substantive_nepc_evidence(sanitized))


if __name__ == "__main__":
    unittest.main()
