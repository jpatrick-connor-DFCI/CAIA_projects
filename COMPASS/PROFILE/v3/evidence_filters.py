import re
from copy import deepcopy


DISEASE_MENTION_FIELDS = (
    "nepc_scpc_mentions",
    "transformation_mentions",
    "aggressive_variant_mentions",
    "avpc_criteria_mentions",
)
SUPPORTING_MENTION_FIELDS = (
    "biomarker_mentions",
    "treatment_resistance_mentions",
    "platinum_mentions",
)
ALL_MENTION_FIELDS = DISEASE_MENTION_FIELDS + SUPPORTING_MENTION_FIELDS


def normalize_evidence_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


TEST_ONLY_PATTERNS = [
    re.compile(
        r"\b(?:test(?:ing)?|work[\s-]?up|evaluate|evaluation|assess|screen|check|rule\s*out|r\/o|look\s+for)\b"
        r".{0,50}\b(?:neuroendocrine|nepc|small[\s-]?cell|scpc|transform(?:ation|ed|ing)|"
        r"aggressive[\s-]?variant|avpc|anaplastic|criteria|differentiation)\b"
    ),
    re.compile(
        r"\b(?:will|would|plan(?:ning|ned)?|recommend(?:ed)?|request(?:ed)?|order(?:ed)?|send|sent|"
        r"obtain|pursu(?:e|ing)|arrang(?:e|ed)|pending|await(?:ing)?)\b"
        r".{0,70}\b(?:test(?:ing)?|work[\s-]?up|stain(?:s|ing)?|ihc|immunostain(?:s|ing)?|"
        r"marker(?:s)?|biopsy|pathology\s+review)\b"
    ),
    re.compile(
        r"\b(?:stain(?:s|ing)?|ihc|immunostain(?:s|ing)?|marker(?:s)?|biopsy|pathology\s+review|"
        r"test(?:ing)?|work[\s-]?up)\b"
        r".{0,70}\b(?:to|for)\s+(?:evaluate|assessment|assess|rule\s*out|r\/o|check|look\s+for)\b"
    ),
]


def is_test_only_quote(quote):
    normalized_quote = normalize_evidence_text(quote)
    if not normalized_quote:
        return False
    return any(pattern.search(normalized_quote) for pattern in TEST_ONLY_PATTERNS)


def _filter_mentions(mentions, *, allow_test_only=False):
    kept_mentions = []
    for mention in mentions or []:
        if not isinstance(mention, dict):
            continue
        if not allow_test_only and is_test_only_quote(mention.get("quote")):
            continue
        kept_mentions.append(mention)
    return kept_mentions


def _has_any_mentions(extraction):
    for field in ALL_MENTION_FIELDS:
        if extraction.get(field):
            return True
    return False


def sanitize_note_extractions(note_extractions):
    sanitized = []
    for extraction in note_extractions or []:
        if not isinstance(extraction, dict):
            continue
        cleaned = deepcopy(extraction)
        for field in DISEASE_MENTION_FIELDS:
            cleaned[field] = _filter_mentions(cleaned.get(field))
        for field in SUPPORTING_MENTION_FIELDS:
            cleaned[field] = _filter_mentions(cleaned.get(field), allow_test_only=True)
        if not _has_any_mentions(cleaned):
            cleaned["overall_relevance"] = "low"
        sanitized.append(cleaned)
    return sanitized


def has_substantive_v3_evidence(note_extractions):
    for extraction in note_extractions or []:
        if _has_any_mentions(extraction):
            return True
    return False
