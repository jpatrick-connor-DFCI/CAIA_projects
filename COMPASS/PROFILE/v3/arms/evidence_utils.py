import re


TRIAL_ONLY_PATTERNS = [
    re.compile(
        r"\b(?:screen(?:ing|ed)?|eligib(?:le|ility)|enroll(?:ed|ment|ing)?|consent(?:ed)?|candidate\s+for)\b"
        r".{0,60}\b(?:clinical\s+trial|trial|study|protocol|cohort|investigational)\b"
    ),
    re.compile(
        r"\b(?:clinical\s+trial|trial|study|protocol|cohort|investigational)\b"
        r".{0,60}\b(?:screen(?:ing|ed)?|eligib(?:le|ility)|enroll(?:ed|ment|ing)?|consent(?:ed)?|candidate)\b"
    ),
]

UNCERTAINTY_PATTERNS = [
    re.compile(
        r"\b(?:possible|possibly|potential(?:ly)?|maybe|perhaps|"
        r"concern(?:ed)?\s+for|concerning\s+for|suspicious\s+for|suspect(?:ed|ing)?|"
        r"question\s+of|may\s+represent|could\s+represent|cannot\s+exclude|can't\s+exclude|"
        r"rule\s*out|r\/o|evaluate\s+for|screen\s+for|work[\s-]?up\s+for|"
        r"pending|await(?:ing)?|send(?:ing)?\s+for|sent\s+for)\b"
    ),
]

EQUIVOCAL_ASSERTIONS = {"possible", "suspected", "indeterminate", "unclear", "unknown"}


def normalize_evidence_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def quote_matches_patterns(quote, patterns):
    normalized_quote = normalize_evidence_text(quote)
    if not normalized_quote:
        return False
    return any(pattern.search(normalized_quote) for pattern in patterns)


def normalize_mentions(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def sanitize_mentions(value, exclusion_patterns):
    return [
        mention
        for mention in normalize_mentions(value)
        if isinstance(mention, dict) and not quote_matches_patterns(mention.get("quote"), exclusion_patterns)
    ]


def normalize_label(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def partition_mentions(mentions, *, affirmative_assertions, uncertainty_patterns=None):
    affirmative = []
    equivocal = []
    uncertainty_patterns = list(uncertainty_patterns or UNCERTAINTY_PATTERNS)

    for mention in normalize_mentions(mentions):
        if not isinstance(mention, dict):
            continue

        assertion = normalize_label(mention.get("assertion"))
        quote = mention.get("quote")
        if not normalize_evidence_text(quote):
            equivocal.append(mention)
            continue
        if quote_matches_patterns(quote, uncertainty_patterns) or assertion in EQUIVOCAL_ASSERTIONS:
            equivocal.append(mention)
            continue
        if assertion in affirmative_assertions:
            affirmative.append(mention)
            continue
        if assertion is None:
            equivocal.append(mention)

    return affirmative, equivocal


def collect_partitioned_mentions(
    note_extractions,
    *,
    field_names,
    affirmative_assertions,
    uncertainty_patterns=None,
):
    affirmative = []
    equivocal = []
    for extraction in note_extractions or []:
        if not isinstance(extraction, dict):
            continue
        for field_name in field_names:
            field_affirmative, field_equivocal = partition_mentions(
                extraction.get(field_name),
                affirmative_assertions=affirmative_assertions,
                uncertainty_patterns=uncertainty_patterns,
            )
            affirmative.extend(field_affirmative)
            equivocal.extend(field_equivocal)
    return affirmative, equivocal
