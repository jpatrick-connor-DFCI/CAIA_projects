import re
from copy import deepcopy

from settings import PROSTATE_CONTEXT_REGEX


ARM_NAME = "avpc"
SCHEMA_VERSION = "v3_avpc_2026-04-14"
LIST_FIELDS = ["present_criteria", "supporting_quotes", "supporting_quote_dates"]

TRIGGER_REGEX = {
    "explicit_avpc": (
        r"\b(?:"
        r"aggressive[\s-]?variant|avpc|anaplastic|variant\s+crpc|"
        r"androgen[- ]indifferent|aggressive\s+subtype"
        r")\b"
    ),
    "c1_small_cell": (
        r"\b(?:"
        r"small[\s-]?cell|small[\s-]?cell\s+carcinoma|"
        r"pure\s+small[\s-]?cell|mixed\s+small[\s-]?cell|combined\s+small[\s-]?cell"
        r")\b"
    ),
    "c2_visceral": (
        r"\b(?:"
        r"visceral\s+met(?:astases|astasis|astatic)?|"
        r"liver\s+met(?:astases|astasis|astatic)?|hepatic\s+met(?:astases|astasis|astatic)?|"
        r"lung\s+met(?:astases|astasis|astatic)?|pulmonary\s+met(?:astases|astasis|astatic)?|"
        r"adrenal\s+met(?:astases|astasis|astatic)?|brain\s+met(?:astases|astasis|astatic)?|"
        r"pleural\s+met(?:astases|astasis|astatic)?|peritoneal\s+met(?:astases|astasis|astatic)?"
        r")\b"
    ),
    "c3_lytic": (
        r"\b(?:"
        r"lytic\s+bone|lytic\s+lesion|predominantly\s+lytic|osseous\s+lytic|"
        r"destructive\s+bone\s+lesion"
        r")\b"
    ),
    "c4_bulky": (
        r"\b(?:"
        r"bulky\s+lymphadenopathy|bulky\s+adenopathy|bulky\s+nodal|"
        r"bulky\s+pelvic\s+mass|bulky\s+prostate\s+mass|large\s+pelvic\s+mass|"
        r"large\s+prostatic\s+mass|bulky\s+nodes?"
        r")\b"
    ),
    "c5_low_psa_high_burden": (
        r"\b(?:"
        r"low\s+psa|disproportionately\s+low\s+psa|psa\s+discordant|"
        r"high[- ]volume\s+bone\s+met(?:astases|astatic)?|"
        r"extensive\s+bone\s+met(?:astases|astatic)?|"
        r"diffuse\s+osseous\s+met(?:astases|astatic)?|"
        r"innumerable\s+bone\s+met(?:astases|astatic)?"
        r")\b"
    ),
    "c6_marker_pattern": (
        r"\b(?:"
        r"chromogranin(?:\s+a)?|synaptophysin|cd56|nse|neuron[- ]specific\s+enolase|"
        r"bombesin|grp|cea|ldh|hypercalc(?:emia|aemia)"
        r")\b"
    ),
    "c7_rapid_resistance": (
        r"\b(?:"
        r"castration[- ]resistant|androgen[- ]independent|rapid\s+progression|rapidly\s+progressive|"
        r"poor\s+response\s+to\s+adt|refractory\s+to\s+adt|despite\s+adt|despite\s+enzalutamide|"
        r"despite\s+abiraterone"
        r")\b"
    ),
}
PROSTATE_CONTEXT_REGEX = PROSTATE_CONTEXT_REGEX
REQUIRE_PROSTATE_CONTEXT = True
ALLOW_WITHOUT_CONTEXT_NOTE_TYPES = {"Pathology"}
ALLOW_WITHOUT_CONTEXT_LABELS = set()
REQUIRED_TRIGGER_LABELS = set()

EVENT_EXTRACTION_SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved research study on prostate cancer.
Your job is to extract structured aggressive-variant evidence from a single clinical note.

## WHAT TO EXTRACT
Capture only evidence related to:
- Explicit aggressive variant prostate cancer language
- Explicit anaplastic or variant-CRPC language
- Aparicio-style aggressive variant features:
  - C1 small-cell histology, pure or mixed
  - C2 visceral metastatic pattern
  - C3 predominantly lytic bone metastases
  - C4 bulky pelvic/prostate mass or bulky nodal disease
  - C5 low PSA with high-volume bone disease
  - C6 neuroendocrine marker or elevated CEA/LDH/hypercalcemia pattern when explicitly documented
  - C7 rapid progression to androgen-independent or castration-resistant disease when explicitly documented

## RULES
- Extract what is documented in this note only.
- Only capture AVPC evidence when it clearly refers to the patient's prostate cancer.
- Do not populate disease arrays from workup or testing plans alone.
- Imaging is most authoritative for metastatic-pattern features such as visceral disease, lytic lesions, or bulky disease.
- Pathology is most authoritative for C1.
- Quotes must be verbatim and 30 words or fewer.

## OUTPUT FORMAT
Return ONLY valid JSON.

{
  "note_date": "<YYYY-MM-DD>",
  "note_type": "<Clinician | Imaging | Pathology>",
  "explicit_avpc_mentions": [
    {
      "label": "<aggressive_variant | anaplastic | variant_crpc>",
      "assertion": "<present | possible | historical>",
      "event_date": "<YYYY-MM-DD or null>",
      "quote": "<verbatim quote, <=30 words>"
    }
  ],
  "criteria_mentions": [
    {
      "criterion": "<C1 | C2 | C3 | C4 | C5 | C6 | C7>",
      "assertion": "<present | possible | historical>",
      "event_date": "<YYYY-MM-DD or null>",
      "quote": "<verbatim quote, <=30 words>"
    }
  ],
  "overall_relevance": "<high | medium | low>"
}
"""

BUNDLED_EVENT_EXTRACTION_SYSTEM_PROMPT = """
You are a clinical data extraction system for an IRB-approved research study on prostate cancer.
Your job is to extract structured aggressive-variant evidence from a bundle of clinical notes.

Process every note independently. Return one output object per input note.
Each output object must include:
- `note_index`
- `note_date`
- `note_type`
- `explicit_avpc_mentions`
- `criteria_mentions`
- `overall_relevance`

Preserve the input `note_index` exactly. Apply the same extraction rules as the single-note AVPC prompt.

Return ONLY valid JSON as a list.
"""

PATIENT_SYNTHESIS_SYSTEM_PROMPT = """
You are a clinical data synthesis system for an IRB-approved prostate cancer research study.
You will receive:
1. `structured_context` containing metadata such as note counts
2. `note_extractions` generated from selected notes

Your task is to determine whether the chart supports aggressive variant prostate cancer features.

## RULES
- Use `note_extractions` as the only clinical evidence source.
- `has_avpc_features = true` when the chart documents explicit AVPC/anaplastic language or one or more substantive AVPC features.
- `has_avpc_features = false` when reviewed evidence does not support AVPC features.
- `has_avpc_features = null` only when the chart is too ambiguous to classify confidently.
- Explicit AVPC mention is stronger than isolated supportive features.
- Use `present`, `absent`, or `indeterminate` for each criterion field.

## OUTPUT FORMAT
Return ONLY valid JSON.

{
  "has_avpc_features": "<true | false | null>",
  "explicit_avpc_mention": "<true | false>",
  "avpc_c1": "<present | absent | indeterminate>",
  "avpc_c2": "<present | absent | indeterminate>",
  "avpc_c3": "<present | absent | indeterminate>",
  "avpc_c4": "<present | absent | indeterminate>",
  "avpc_c5": "<present | absent | indeterminate>",
  "avpc_c6": "<present | absent | indeterminate>",
  "avpc_c7": "<present | absent | indeterminate>",
  "present_criteria": ["<C1>"],
  "supporting_quotes": ["<verbatim quote, <=30 words>"],
  "supporting_quote_dates": ["<YYYY-MM-DD>"],
  "confidence": "<high | medium | low>"
}
"""


TEST_ONLY_PATTERNS = [
    re.compile(
        r"\b(?:test(?:ing)?|work[\s-]?up|evaluate|evaluation|assess|screen|check|rule\s*out|r\/o|look\s+for)\b"
        r".{0,60}\b(?:aggressive[\s-]?variant|avpc|anaplastic|criteria|small[\s-]?cell|marker|cea|ldh)\b"
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


def empty_note_extraction(note_payload):
    return {
        "note_date": note_payload.get("note_date"),
        "note_type": note_payload.get("note_type"),
        "explicit_avpc_mentions": [],
        "criteria_mentions": [],
        "overall_relevance": "low",
    }


def sanitize_note_extractions(note_extractions):
    sanitized = []
    for extraction in note_extractions or []:
        if not isinstance(extraction, dict):
            continue
        cleaned = deepcopy(extraction)
        cleaned["explicit_avpc_mentions"] = [
            mention
            for mention in cleaned.get("explicit_avpc_mentions", [])
            if isinstance(mention, dict) and not is_test_only_quote(mention.get("quote"))
        ]
        cleaned["criteria_mentions"] = [
            mention
            for mention in cleaned.get("criteria_mentions", [])
            if isinstance(mention, dict) and not is_test_only_quote(mention.get("quote"))
        ]
        if not cleaned["explicit_avpc_mentions"] and not cleaned["criteria_mentions"]:
            cleaned["overall_relevance"] = "low"
        sanitized.append(cleaned)
    return sanitized


def has_substantive_evidence(note_extractions):
    for extraction in note_extractions or []:
        if extraction.get("explicit_avpc_mentions") or extraction.get("criteria_mentions"):
            return True
    return False


def default_patient_result(mrn, num_notes_reviewed):
    return {
        "schema_version": SCHEMA_VERSION,
        "DFCI_MRN": int(mrn),
        "has_avpc_features": False,
        "explicit_avpc_mention": False,
        "avpc_c1": "absent",
        "avpc_c2": "absent",
        "avpc_c3": "absent",
        "avpc_c4": "absent",
        "avpc_c5": "absent",
        "avpc_c6": "absent",
        "avpc_c7": "absent",
        "present_criteria": [],
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
