import re
from copy import deepcopy


ARM_NAME = "biomarker"
SCHEMA_VERSION = "v3_biomarker_2026-04-14"
LIST_FIELDS = ["biomarker_types", "supporting_quotes", "supporting_quote_dates"]

TRIGGER_REGEX = {
    "biomarker_core": (
        r"\b(?:"
        r"brca1|brca2|atm|cdk12|palb2|"
        r"hrd|hrr|ddr|homologous\s+recombination|dna\s+damage\s+repair|"
        r"msi[- ]h|msi[- ]high|mmr|mismatch\s+repair|msh2|msh6|mlh1|pms2|"
        r"tumor\s+mutational\s+burden|tmb"
        r")\b"
    ),
    "platinum_context": r"\b(?:carboplatin|cisplatin|platinum[- ]based)\b",
    "rationale_language": (
        r"\b(?:"
        r"given|due\s+to|because\s+of|in\s+light\s+of|selected\s+because|"
        r"chosen\s+because|rationale|sensitive\s+to|sensitivity\s+to"
        r")\b"
    ),
}
PROSTATE_CONTEXT_REGEX = None
REQUIRE_PROSTATE_CONTEXT = False
ALLOW_WITHOUT_CONTEXT_NOTE_TYPES = set()
ALLOW_WITHOUT_CONTEXT_LABELS = set()
REQUIRED_TRIGGER_LABELS = {"biomarker_core"}

EVENT_EXTRACTION_SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved research study on prostate cancer.
Your job is to extract structured biomarker evidence from a single clinical note.

## WHAT TO EXTRACT
Capture only evidence related to platinum-relevant biomarkers or molecular findings, including:
- BRCA1, BRCA2, ATM, CDK12, PALB2
- HRD, HRR, DDR pathway language
- MSI-H, mismatch-repair deficiency, related molecular findings
- whether the note links the biomarker to platinum sensitivity or platinum treatment choice

## RULES
- Extract what is documented in this note only.
- Do not count planned testing, pending results, or orders for sequencing as a biomarker finding.
- A biomarker can be captured even if the note does not explicitly mention platinum.
- Quotes must be verbatim and 30 words or fewer.

## OUTPUT FORMAT
Return ONLY valid JSON.

{
  "note_date": "<YYYY-MM-DD>",
  "note_type": "<Clinician | Imaging | Pathology>",
  "biomarker_mentions": [
    {
      "marker": "<BRCA1 | BRCA2 | ATM | CDK12 | PALB2 | HRD | DDR | MSI-H | MMR | other>",
      "assertion": "<present | possible | historical>",
      "platinum_linked": "<true | false | possible>",
      "event_date": "<YYYY-MM-DD or null>",
      "quote": "<verbatim quote, <=30 words>"
    }
  ],
  "overall_relevance": "<high | medium | low>"
}
"""

BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved research study on prostate cancer.
Your job is to extract structured biomarker evidence from a bundle of clinical notes.

Process every note independently. Return one output object per input note.
Each output object must include:
- `note_index`
- `note_date`
- `note_type`
- `biomarker_mentions`
- `overall_relevance`

Preserve the input `note_index` exactly. Apply the same extraction rules as the single-note biomarker prompt.

Return ONLY valid JSON as a list.
"""

PATIENT_SYNTHESIS_SYSTEM_PROMPT = """
You are a clinical data synthesis system for an IRB-approved prostate cancer research study.
You will receive:
1. `structured_context` containing metadata such as note counts
2. `note_extractions` generated from selected notes

Your task is to determine whether the chart supports a biomarker signal potentially relevant to platinum.

## RULES
- Use `note_extractions` as the only clinical evidence source.
- `has_biomarker_signal = true` when the chart documents one or more substantive biomarker findings.
- `has_biomarker_signal = false` when reviewed evidence does not support a biomarker finding.
- `has_biomarker_signal = null` only when the chart is too ambiguous to classify confidently.
- Set `platinum_linked_biomarker = true` only when the chart ties the biomarker to platinum choice, platinum sensitivity, or platinum rationale.

## OUTPUT FORMAT
Return ONLY valid JSON.

{
  "has_biomarker_signal": "<true | false | null>",
  "biomarker_types": ["<BRCA2>"],
  "platinum_linked_biomarker": "<true | false | null>",
  "supporting_quotes": ["<verbatim quote, <=30 words>"],
  "supporting_quote_dates": ["<YYYY-MM-DD>"],
  "confidence": "<high | medium | low>"
}
"""


TEST_ONLY_PATTERNS = [
    re.compile(
        r"\b(?:send|sent|order(?:ed)?|obtain|request(?:ed)?|pending|await(?:ing)?|plan(?:ning|ned)?|"
        r"recommend(?:ed)?)\b.{0,70}\b(?:oncopanel|sequencing|genetic\s+testing|testing|assay|panel)\b"
    ),
]


def normalize_evidence_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_test_only_quote(quote):
    normalized_quote = normalize_evidence_text(quote)
    if not normalized_quote:
        return False
    return any(pattern.search(normalized_quote) for pattern in TEST_ONLY_PATTERNS)


def normalize_mentions(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def empty_note_extraction(note_payload):
    return {
        "note_date": note_payload.get("note_date"),
        "note_type": note_payload.get("note_type"),
        "biomarker_mentions": [],
        "overall_relevance": "low",
    }


def sanitize_note_extractions(note_extractions):
    sanitized = []
    for extraction in note_extractions or []:
        if not isinstance(extraction, dict):
            continue
        cleaned = deepcopy(extraction)
        cleaned["biomarker_mentions"] = [
            mention
            for mention in normalize_mentions(cleaned.get("biomarker_mentions"))
            if isinstance(mention, dict) and not is_test_only_quote(mention.get("quote"))
        ]
        if not cleaned["biomarker_mentions"]:
            cleaned["overall_relevance"] = "low"
        sanitized.append(cleaned)
    return sanitized


def has_substantive_evidence(note_extractions):
    for extraction in note_extractions or []:
        if extraction.get("biomarker_mentions"):
            return True
    return False


def default_patient_result(mrn, num_notes_reviewed):
    return {
        "schema_version": SCHEMA_VERSION,
        "DFCI_MRN": int(mrn),
        "has_biomarker_signal": False,
        "biomarker_types": [],
        "platinum_linked_biomarker": False,
        "supporting_quotes": [],
        "supporting_quote_dates": [],
        "confidence": "low",
        "num_notes_reviewed": int(num_notes_reviewed),
        "num_note_extractions": 0,
    }


def merge_patient_result(base_row, model_row):
    if not isinstance(model_row, dict):
        return base_row
    merged = base_row.copy()
    merged.update(model_row)
    merged["schema_version"] = SCHEMA_VERSION
    for field in LIST_FIELDS:
        if merged.get(field) is None:
            merged[field] = []
    return merged
