import re
from copy import deepcopy


def normalize_evidence_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


TEST_ONLY_NEPC_PATTERNS = [
    re.compile(
        r"\b(?:test(?:ing)?|work[\s-]?up|evaluate|evaluation|assess|screen|check|rule\s*out|r\/o|look\s+for)\b"
        r".{0,40}\b(?:neuroendocrine|nepc|small[\s-]?cell|scpc|transform(?:ation|ed|ing)|histology|differentiation)\b"
    ),
    re.compile(
        r"\b(?:will|would|plan(?:ning|ned)?|recommend(?:ed)?|request(?:ed)?|order(?:ed)?|send|sent|obtain|pursu(?:e|ing)|arrang(?:e|ed)|pending|await(?:ing)?)\b"
        r".{0,60}\b(?:test(?:ing)?|work[\s-]?up|stain(?:s|ing)?|ihc|immunostain(?:s|ing)?|marker(?:s)?|biopsy|pathology\s+review)\b"
    ),
    re.compile(
        r"\b(?:stain(?:s|ing)?|ihc|immunostain(?:s|ing)?|marker(?:s)?|biopsy|pathology\s+review|test(?:ing)?|work[\s-]?up)\b"
        r".{0,60}\b(?:to|for)\s+(?:evaluate|assessment|assess|rule\s*out|r\/o|check|look\s+for)\b"
    ),
    re.compile(
        r"\b(?:stain(?:s)?|ihc|immunostain(?:s)?|marker(?:s)?)\b"
        r".{0,30}\b(?:pending|send(?:ing)?|sent|order(?:ed)?|request(?:ed)?)\b"
    ),
]


def is_test_only_nepc_quote(quote):
    normalized_quote = normalize_evidence_text(quote)
    if not normalized_quote:
        return False
    return any(pattern.search(normalized_quote) for pattern in TEST_ONLY_NEPC_PATTERNS)


def _filter_mentions(mentions):
    kept_mentions = []
    for mention in mentions or []:
        if not isinstance(mention, dict):
            continue
        if is_test_only_nepc_quote(mention.get("quote")):
            continue
        kept_mentions.append(mention)
    return kept_mentions


def sanitize_note_extractions(note_extractions):
    sanitized = []
    for extraction in note_extractions or []:
        if not isinstance(extraction, dict):
            continue
        cleaned = deepcopy(extraction)
        cleaned["histology_mentions"] = _filter_mentions(cleaned.get("histology_mentions"))
        cleaned["transformation_mentions"] = _filter_mentions(cleaned.get("transformation_mentions"))
        if not cleaned["histology_mentions"] and not cleaned["transformation_mentions"]:
            cleaned["overall_relevance"] = "low"
        sanitized.append(cleaned)
    return sanitized


def has_substantive_nepc_evidence(note_extractions):
    for extraction in note_extractions or []:
        histology_mentions = extraction.get("histology_mentions") or []
        transformation_mentions = extraction.get("transformation_mentions") or []
        if histology_mentions or transformation_mentions:
            return True
    return False
