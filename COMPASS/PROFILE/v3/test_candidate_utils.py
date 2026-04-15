import sys
import unittest
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from candidate_utils import annotate_inventory_notes  # noqa: E402


class DummyBiomarkerArm:
    TRIGGER_REGEX = {
        "biomarker_core": r"\bBRCA2\b",
        "platinum_context": r"\bcarboplatin\b",
    }
    PROSTATE_CONTEXT_REGEX = None
    REQUIRE_PROSTATE_CONTEXT = False
    ALLOW_WITHOUT_CONTEXT_NOTE_TYPES = set()
    ALLOW_WITHOUT_CONTEXT_LABELS = set()
    REQUIRED_TRIGGER_LABELS = {"biomarker_core"}


class CandidateUtilsTest(unittest.TestCase):
    def test_required_trigger_label_blocks_context_only_notes(self):
        note_df = pd.DataFrame(
            [
                {
                    "DFCI_MRN": 1,
                    "EVENT_DATE": "2026-01-01",
                    "NOTE_TYPE": "Clinician",
                    "CLINICAL_TEXT": "Carboplatin started for progressive disease.",
                },
                {
                    "DFCI_MRN": 1,
                    "EVENT_DATE": "2026-01-02",
                    "NOTE_TYPE": "Clinician",
                    "CLINICAL_TEXT": "Given BRCA2 loss, carboplatin was selected.",
                },
            ]
        )

        annotated = annotate_inventory_notes(note_df, DummyBiomarkerArm)

        self.assertEqual(1, len(annotated))
        self.assertIn("BRCA2", annotated.iloc[0]["CLINICAL_TEXT"])
        self.assertIn("carboplatin", annotated.iloc[0]["CLINICAL_TEXT"].lower())


if __name__ == "__main__":
    unittest.main()
