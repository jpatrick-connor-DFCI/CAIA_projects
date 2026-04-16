import re
from copy import deepcopy

from .evidence_utils import TRIAL_ONLY_PATTERNS, collect_partitioned_mentions, normalize_label, sanitize_mentions
from helpers import PROSTATE_CONTEXT_REGEX


ARM_NAME = "avpc"
SCHEMA_VERSION = "v3_avpc_2026-04-15"
LIST_FIELDS = ["present_criteria", "supporting_quotes", "supporting_quote_dates"]
REQUIRED_SYNTHESIS_FIELDS = [
    "has_avpc_features",
    "explicit_avpc_mention",
    "avpc_c1",
    "avpc_c2",
    "avpc_c3",
    "avpc_c4",
    "avpc_c5",
    "avpc_c6",
    "avpc_c7",
]

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
- Clinical trial screening, eligibility, enrollment, or protocol language does not establish AVPC by itself.
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
- Do not call AVPC present from suspicion, clinical trial screening, or eligibility language alone.
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
EXCLUSION_PATTERNS = TEST_ONLY_PATTERNS + TRIAL_ONLY_PATTERNS


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
        cleaned["explicit_avpc_mentions"] = sanitize_mentions(
            cleaned.get("explicit_avpc_mentions"),
            EXCLUSION_PATTERNS,
        )
        cleaned["criteria_mentions"] = sanitize_mentions(cleaned.get("criteria_mentions"), EXCLUSION_PATTERNS)
        if not cleaned["explicit_avpc_mentions"] and not cleaned["criteria_mentions"]:
            cleaned["overall_relevance"] = "low"
        sanitized.append(cleaned)
    return sanitized


def has_substantive_evidence(note_extractions):
    explicit_affirmative, explicit_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["explicit_avpc_mentions"],
        affirmative_assertions={"present", "historical"},
    )
    criteria_affirmative, criteria_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["criteria_mentions"],
        affirmative_assertions={"present", "historical"},
    )
    return bool(explicit_affirmative or explicit_equivocal or criteria_affirmative or criteria_equivocal)


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


def merge_patient_result(base_row, model_row, note_extractions=None):
    if not isinstance(model_row, dict):
        model_row = {}
    merged = base_row.copy()
    merged.update(model_row)
    merged["schema_version"] = SCHEMA_VERSION
    for field in LIST_FIELDS:
        if merged.get(field) is None:
            merged[field] = []

    if note_extractions is None:
        return merged

    explicit_affirmative, explicit_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["explicit_avpc_mentions"],
        affirmative_assertions={"present", "historical"},
    )
    criteria_affirmative, criteria_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["criteria_mentions"],
        affirmative_assertions={"present", "historical"},
    )

    has_affirmative_signal = bool(explicit_affirmative or criteria_affirmative)
    has_equivocal_signal = bool(explicit_equivocal or criteria_equivocal)
    merged["explicit_avpc_mention"] = bool(explicit_affirmative)

    if has_affirmative_signal:
        merged["has_avpc_features"] = True
    elif has_equivocal_signal:
        merged["has_avpc_features"] = None
        merged["confidence"] = "low"
    else:
        merged["has_avpc_features"] = False

    affirmative_criteria = {
        normalize_label(mention.get("criterion")).upper()
        for mention in criteria_affirmative
        if normalize_label(mention.get("criterion")) in {f"c{idx}" for idx in range(1, 8)}
    }
    equivocal_criteria = {
        normalize_label(mention.get("criterion")).upper()
        for mention in criteria_equivocal
        if normalize_label(mention.get("criterion")) in {f"c{idx}" for idx in range(1, 8)}
    }

    merged["present_criteria"] = sorted(affirmative_criteria)
    for idx in range(1, 8):
        criterion = f"C{idx}"
        field_name = f"avpc_c{idx}"
        if criterion in affirmative_criteria:
            merged[field_name] = "present"
        elif criterion in equivocal_criteria:
            merged[field_name] = "indeterminate"
        else:
            merged[field_name] = "absent"

    return merged
