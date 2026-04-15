import re
from copy import deepcopy

from .evidence_utils import (
    TRIAL_ONLY_PATTERNS,
    collect_partitioned_mentions,
    normalize_label,
    sanitize_mentions,
)
from helpers import PROSTATE_CONTEXT_REGEX


ARM_NAME = "nepc"
SCHEMA_VERSION = "v3_nepc_2026-04-15"
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
- Clinical trial screening, eligibility, enrollment, or protocol language does not establish NEPC or transformation by itself.
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
- Do not call NEPC/SCPC present from suspicion, clinical trial screening, or eligibility language alone.
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
EXCLUSION_PATTERNS = TEST_ONLY_PATTERNS + TRIAL_ONLY_PATTERNS


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
        cleaned["histology_mentions"] = sanitize_mentions(cleaned.get("histology_mentions"), EXCLUSION_PATTERNS)
        cleaned["transformation_mentions"] = sanitize_mentions(
            cleaned.get("transformation_mentions"),
            EXCLUSION_PATTERNS,
        )
        if not cleaned["histology_mentions"] and not cleaned["transformation_mentions"]:
            cleaned["overall_relevance"] = "low"
        sanitized.append(cleaned)
    return sanitized


def has_substantive_evidence(note_extractions):
    histology_affirmative, histology_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["histology_mentions"],
        affirmative_assertions={"present", "historical"},
    )
    transformation_affirmative, transformation_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["transformation_mentions"],
        affirmative_assertions={"documented", "historical"},
    )
    return bool(
        histology_affirmative
        or histology_equivocal
        or transformation_affirmative
        or transformation_equivocal
    )


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


def infer_nepc_subtype(histology_mentions, transformation_mentions):
    subtype_labels = set()
    for mention in histology_mentions or []:
        label = normalize_label(mention.get("label"))
        if label in {"neuroendocrine", "small_cell", "both"}:
            subtype_labels.add(label)
    for mention in transformation_mentions or []:
        label = normalize_label(mention.get("to_histology"))
        if label in {"neuroendocrine", "small_cell", "both"}:
            subtype_labels.add(label)

    if "both" in subtype_labels or {"neuroendocrine", "small_cell"}.issubset(subtype_labels):
        return "both"
    if "small_cell" in subtype_labels:
        return "small_cell"
    if "neuroendocrine" in subtype_labels:
        return "neuroendocrine"
    return None


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

    histology_affirmative, histology_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["histology_mentions"],
        affirmative_assertions={"present", "historical"},
    )
    transformation_affirmative, transformation_equivocal = collect_partitioned_mentions(
        note_extractions,
        field_names=["transformation_mentions"],
        affirmative_assertions={"documented", "historical"},
    )

    has_affirmative_signal = bool(histology_affirmative or transformation_affirmative)
    has_equivocal_signal = bool(histology_equivocal or transformation_equivocal)
    inferred_subtype = infer_nepc_subtype(histology_affirmative, transformation_affirmative)

    if has_affirmative_signal:
        merged["has_nepc_signal"] = True
        merged["nepc_subtype"] = inferred_subtype or merged.get("nepc_subtype")
    elif has_equivocal_signal:
        merged["has_nepc_signal"] = None
        merged["nepc_subtype"] = None
        merged["confidence"] = "low"
    else:
        merged["has_nepc_signal"] = False
        merged["nepc_subtype"] = None

    if transformation_affirmative:
        merged["has_transformation_signal"] = True
        merged["transformation_status"] = "documented"
    elif transformation_equivocal:
        merged["has_transformation_signal"] = None
        merged["transformation_status"] = "suspected"
        merged["confidence"] = "low"
    else:
        merged["has_transformation_signal"] = False
        merged["transformation_status"] = "not_documented"
        merged["transformation_date"] = None

    return merged
