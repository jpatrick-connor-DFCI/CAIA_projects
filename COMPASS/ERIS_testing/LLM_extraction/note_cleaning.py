"""
NOTE_TYPE-aware text cleaning for clinical notes.

Usage:
    from note_cleaning import clean_note, extract_structured_elements

    cleaned = clean_note(text, note_type='Clinician')
    extractions = extract_structured_elements(text, note_type='Pathology')

Rules are organized into:
  - UNIVERSAL_RULES: applied to all notes regardless of type
  - TYPE_SPECIFIC_RULES: keyed by NOTE_TYPE, applied only to matching notes
  - EXTRACTION_PATTERNS: regex patterns for pre-extracting structured clinical info

To populate rules: run sample_notes_for_regex_generation.py, send the output
with regex_generation_prompt.txt to the enterprise GPT, then paste the generated
rules into the dictionaries below.
"""

import re

# ---------------------------------------------------------------------------
# UNIVERSAL RULES — applied to all note types
# ---------------------------------------------------------------------------
UNIVERSAL_RULES = [
    {
        'name': 'collapse_blank_lines',
        'pattern': r'\n\s*\n+',
        'replacement': '\n\n',
        'flags': 0,
    },
    {
        'name': 'collapse_whitespace',
        'pattern': r'[ \t]+',
        'replacement': ' ',
        'flags': 0,
    },
    {
        'name': 'decorative_lines',
        'pattern': r'^[\s]*[-=_*]{3,}[\s]*$',
        'replacement': '',
        'flags': re.MULTILINE,
    },
    {
        'name': 'confidential_line',
        'pattern': r'^.*confidential.*$',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
    },
    {
        'name': 'electronically_signed',
        'pattern': r'^.*electronically signed.*$',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
    },
    {
        'name': 'printed_by',
        'pattern': r'^.*printed by.*$',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
    },
    {
        'name': 'page_numbers',
        'pattern': r'^.*page \d+ of \d+.*$',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
    },
    # ---- ADD MORE UNIVERSAL RULES FROM GPT OUTPUT BELOW ----
]

# ---------------------------------------------------------------------------
# NOTE_TYPE-SPECIFIC RULES
# ---------------------------------------------------------------------------
CLINICIAN_RULES = [
    {
        'name': 'vitals_block',
        'pattern': r'^.*\b(vitals?|bp|blood pressure|heart rate|temp|spo2|pulse|weight|height|bmi)\b[:\s]+[\d/.]+.*$',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
    },
    # ---- ADD MORE CLINICIAN RULES FROM GPT OUTPUT BELOW ----
]

IMAGING_RULES = [
    # ---- ADD IMAGING-SPECIFIC RULES FROM GPT OUTPUT BELOW ----
    # Examples: technique/protocol lines, accession numbers, contrast details
]

PATHOLOGY_RULES = [
    # ---- ADD PATHOLOGY-SPECIFIC RULES FROM GPT OUTPUT BELOW ----
    # Examples: specimen labeling boilerplate, staining method boilerplate
]

TYPE_SPECIFIC_RULES = {
    'Clinician': CLINICIAN_RULES,
    'Imaging': IMAGING_RULES,
    'Pathology': PATHOLOGY_RULES,
}

# ---------------------------------------------------------------------------
# EXTRACTION PATTERNS — for pre-extracting structured clinical elements
# ---------------------------------------------------------------------------
EXTRACTION_PATTERNS = [
    {
        'name': 'histology_neuroendocrine',
        'pattern': r'(?i)\b(neuroendocrine\s+(?:differentiation|carcinoma|features|component|transformation)|small\s+cell\s+(?:carcinoma|component|features|transformation))\b',
        'note_types': ['Clinician', 'Imaging', 'Pathology'],
    },
    {
        'name': 'platinum_mention',
        'pattern': r'(?i)\b((?:carbo|cis)platin\b.{0,80})',
        'note_types': ['Clinician'],
    },
    {
        'name': 'psa_value',
        'pattern': r'(?i)\bPSA[:\s]+([><=]?\s*[\d,.]+)',
        'note_types': ['Clinician'],
    },
    # ---- ADD MORE EXTRACTION PATTERNS FROM GPT OUTPUT BELOW ----
]


def clean_note(text, note_type=None):
    """Clean a clinical note by applying universal and type-specific regex rules.

    Args:
        text: Raw clinical note text.
        note_type: One of 'Clinician', 'Imaging', 'Pathology', or None.
            If None, only universal rules are applied.

    Returns:
        Cleaned text string.
    """
    text = str(text)

    # Apply universal rules
    for rule in UNIVERSAL_RULES:
        text = re.sub(rule['pattern'], rule['replacement'], text, flags=rule['flags'])

    # Apply type-specific rules
    if note_type and note_type in TYPE_SPECIFIC_RULES:
        for rule in TYPE_SPECIFIC_RULES[note_type]:
            text = re.sub(rule['pattern'], rule['replacement'], text, flags=rule['flags'])

    # Final whitespace cleanup
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def extract_structured_elements(text, note_type=None):
    """Pre-extract structured clinical elements from note text via regex.

    Args:
        text: Raw or cleaned clinical note text.
        note_type: One of 'Clinician', 'Imaging', 'Pathology', or None.
            If None, all patterns are applied regardless of note_type filter.

    Returns:
        Dict mapping pattern name to list of matched strings.
    """
    text = str(text)
    results = {}

    for pat in EXTRACTION_PATTERNS:
        if note_type and note_type not in pat['note_types']:
            continue
        matches = re.findall(pat['pattern'], text)
        if matches:
            results[pat['name']] = matches

    return results
