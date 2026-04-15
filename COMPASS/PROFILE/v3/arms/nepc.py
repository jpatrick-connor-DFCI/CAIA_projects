import re
from copy import deepcopy

from settings import PROSTATE_CONTEXT_REGEX


ARM_NAME = "nepc"
SCHEMA_VERSION = "v3_nepc_2026-04-14"
LIST_FIELDS = ["supporting_quotes", "supporting_quote_dates"]

TRIGGER_REGEX = {
    "nepc_core": (
        r"\b(?:"
        r"neuroendocrine|neuro-endocrine|nepc|t-nepc|"
        r"small[\s-]?cell|small[\s-]?cell\s+carcinoma|scpc|scnc|oat[\s-]?cell|"
        r"neuroendocrine\s+carcinoma|small[- ]cell\s+neuroendocrine\s+carcinoma"
        r")\b"
    ),
    "transformation": (
        r"\b(?:"
        r"histolog(?:ic|ical)\s+transform(?:ation|ed|ing)|"
        r"transform(?:ation|ed|ing)(?:\s+(?:to|into))?|"
        r"transdifferentiat(?:e|ed|ion|ing)|"
        r"dedifferentiat(?:e|ed|ion|ing)|"
        r"lineage\s+plasticity|"
        r"treatment[\s-]?emergent\s+neuroendocrine|"
        r"evolved\s+to|converted\s+to"
        r")\b"
    ),
    "ne_markers": (
        r"\b(?:"
        r"synaptophysin|chromogranin(?:\s+a)?|cd56|"
        r"neuron[- ]specific\s+enolase|nse"
        r")\b"
    ),
}
PROSTATE_CONTEXT_REGEX = PROSTATE_CONTEXT_REGEX
REQUIRE_PROSTATE_CONTEXT = True
ALLOW_WITHOUT_CONTEXT_NOTE_TYPES = {"Pathology"}
ALLOW_WITHOUT_CONTEXT_LABELS = {"transformation"}
REQUIRED_TRIGGER_LABELS = set()

EVENT_EXTRACTION_SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved research study on prostate cancer.
Your job is to extract structured evidence from a single clinical note.

## WHAT TO EXTRACT
Capture only evidence related to:
- Neuroendocrine prostate cancer
- Small cell prostate cancer
- Histologic transformation from adenocarcinoma or other prostate histology into neuroendocrine or small cell disease

## RULES
- Extract what is documented in this note only.
- Only capture neuroendocrine or small-cell language when it clearly refers to the patient's prostate cancer.
- Do not treat workup alone as disease evidence. Testing plans, pending stains, planned biopsies, and pathology review requests without resulting diagnosis should not populate disease arrays.
- Pathology is most authoritative for histology.
- Quotes must be verbatim and 30 words or fewer.
- Return empty arrays when the note contains no evidence for an event family.

## OUTPUT FORMAT
Return ONLY valid JSON.

{
  "note_date": "<YYYY-MM-DD>",
  "note_type": "<Clinician | Imaging | Pathology>",
  "histology_mentions": [
    {
      "label": "<neuroendocrine | small_cell | both>",
      "assertion": "<present | possible | historical>",
      "event_date": "<YYYY-MM-DD or null>",
      "quote": "<verbatim quote, <=30 words>"
    }
  ],
  "transformation_mentions": [
    {
      "from_histology": "<adenocarcinoma | other | unknown | null>",
      "to_histology": "<neuroendocrine | small_cell | both | unknown | null>",
      "assertion": "<documented | possible | historical>",
      "event_date": "<YYYY-MM-DD or null>",
      "quote": "<verbatim quote, <=30 words>"
    }
  ],
  "overall_relevance": "<high | medium | low>"
}
"""

BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved research study on prostate cancer.
Your job is to extract structured evidence from a bundle of clinical notes.

Process every note independently. Return one output object per input note.
Each output object must include:
- `note_index`
- `note_date`
- `note_type`
- `histology_mentions`
- `transformation_mentions`
- `overall_relevance`

Preserve the input `note_index` exactly. Apply the same extraction rules as the single-note NEPC prompt.

Return ONLY valid JSON as a list.
"""

PATIENT_SYNTHESIS_SYSTEM_PROMPT = """
You are a clinical data synthesis system for an IRB-approved prostate cancer research study.
You will receive:
1. `structured_context` containing metadata such as note counts
2. `note_extractions` generated from selected notes

Your task is to determine whether the patient has neuroendocrine or small-cell prostate cancer,
and whether transformation is documented or suspected.

## RULES
- Use `note_extractions` as the only clinical evidence source.
- Pathology is most authoritative for histology.
- Do not call NEPC/SCPC present from workup alone.
- Use `has_nepc_signal = true` when the chart supports neuroendocrine or small-cell prostate cancer.
- Use `has_nepc_signal = false` when reviewed evidence does not support NEPC/SCPC.
- Use `has_nepc_signal = null` only when the chart is too ambiguous to classify confidently.

## OUTPUT FORMAT
Return ONLY valid JSON.

{
  "has_nepc_signal": "<true | false | null>",
  "nepc_subtype": "<neuroendocrine | small_cell | both | null>",
  "has_transformation_signal": "<true | false | null>",
  "transformation_status": "<documented | suspected | not_documented | indeterminate>",
  "transformation_date": "<YYYY-MM-DD or null>",
  "supporting_quotes": ["<verbatim quote, <=30 words>"],
  "supporting_quote_dates": ["<YYYY-MM-DD>"],
  "confidence": "<high | medium | low>"
}
"""


TEST_ONLY_PATTERNS = [
    re.compile(
        r"\b(?:test(?:ing)?|work[\s-]?up|evaluate|evaluation|assess|screen|check|rule\s*out|r\/o|look\s+for)\b"
        r".{0,50}\b(?:neuroendocrine|nepc|small[\s-]?cell|scpc|transform(?:ation|ed|ing)|histology)\b"
    ),
    re.compile(
        r"\b(?:will|would|plan(?:ning|ned)?|recommend(?:ed)?|request(?:ed)?|order(?:ed)?|send|sent|obtain|pending|await(?:ing)?)\b"
        r".{0,70}\b(?:test(?:ing)?|work[\s-]?up|stain(?:s|ing)?|ihc|immunostain(?:s|ing)?|marker(?:s)?|biopsy|pathology\s+review)\b"
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
        "histology_mentions": [],
        "transformation_mentions": [],
        "overall_relevance": "low",
    }


def sanitize_note_extractions(note_extractions):
    sanitized = []
    for extraction in note_extractions or []:
        if not isinstance(extraction, dict):
            continue
        cleaned = deepcopy(extraction)
        cleaned["histology_mentions"] = [
            mention
            for mention in normalize_mentions(cleaned.get("histology_mentions"))
            if isinstance(mention, dict) and not is_test_only_quote(mention.get("quote"))
        ]
        cleaned["transformation_mentions"] = [
            mention
            for mention in normalize_mentions(cleaned.get("transformation_mentions"))
            if isinstance(mention, dict) and not is_test_only_quote(mention.get("quote"))
        ]
        if not cleaned["histology_mentions"] and not cleaned["transformation_mentions"]:
            cleaned["overall_relevance"] = "low"
        sanitized.append(cleaned)
    return sanitized


def has_substantive_evidence(note_extractions):
    for extraction in note_extractions or []:
        if extraction.get("histology_mentions") or extraction.get("transformation_mentions"):
            return True
    return False


def default_patient_result(mrn, num_notes_reviewed):
    return {
        "schema_version": SCHEMA_VERSION,
        "DFCI_MRN": int(mrn),
        "has_nepc_signal": False,
        "nepc_subtype": None,
        "has_transformation_signal": False,
        "transformation_status": "not_documented",
        "transformation_date": None,
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
