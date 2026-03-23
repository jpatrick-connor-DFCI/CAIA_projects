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

```python
import re

# Universal rules: These apply across all note types and are deduplicated from the analyses.
UNIVERSAL_RULES = [
    {
        'name': 'signature_block',
        'pattern': r'(?:Staff Surgeon|MD|PhD|NP|RN|DMD|MSN|Instructor in Medicine|Physician).*?(?:Dana Farber Cancer Institute|DFCI|Harvard Medical School|450 Brookline Avenue).*|(?:Signed by|This report was electronically signed by).*|By his/her signature.*?Electronically signed.*?\d{2}:\d{2}:\d{2}(?:AM|PM)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes signature blocks and institutional affiliations.'
    },
    {
        'name': 'confidentiality_disclaimer',
        'pattern': r'(?:DANA FARBER CANCER INSTITUTE|LANK CENTER FOR GENITOURINARY ONCOLOGY).*?(?:Boston, MA|Brookline Avenue).*|This report is limited to the body part and modality requested.*|Massachusetts General Physicians Organization.*?www\.mydermpath\.org',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes confidentiality disclaimers and institutional headers.'
    },
    {
        'name': 'pagination_marker',
        'pattern': r'Page \d+ of \d+|\[Length: \d+ chars\]',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes pagination markers.'
    },
    {
        'name': 'decorative_separator',
        'pattern': r'[=*-]{3,}',
        'replacement': '',
        'flags': re.MULTILINE,
        'confidence': 'high',
        'description': 'Removes decorative separators.'
    },
    {
        'name': 'repeated_whitespace',
        'pattern': r'\s{2,}',
        'replacement': ' ',
        'flags': re.MULTILINE,
        'confidence': 'high',
        'description': 'Removes repeated whitespace and formatting artifacts.'
    }
]

# Clinician-specific rules: These apply only to clinician notes.
CLINICIAN_RULES = [
    {
        'name': 'vitals_block',
        'pattern': r'(?:BP|Pulse|Temp|Resp|Ht|Wt|BMI).*?(?:kg|lb|cm|m|C|F)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes vitals blocks.'
    },
    {
        'name': 'medication_list',
        'pattern': r'(?:Current Outpatient Prescriptions|Medications Reviewed).*?(?:tablet|capsule|injection|spray).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes medication list dumps.'
    },
    {
        'name': 'allergy_list',
        'pattern': r'(?:Allergies).*?(?:No Known Allergies|NKDA).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes allergy lists.'
    },
    {
        'name': 'review_of_systems',
        'pattern': r'(?:Review of Systems).*?(?:negative|denies).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes review of systems template text.'
    },
    {
        'name': 'problem_list',
        'pattern': r'(?:Problem List Items Addressed This Visit|Active Problem List).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes problem list headers.'
    }
]

# Imaging-specific rules: These apply only to imaging notes.
IMAGING_RULES = [
    {
        'name': 'system_generated_headers',
        'pattern': r'(?:Exam Number|Report Status|Type|Date/Time|Ordering Provider|Accession number).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes system-generated headers with metadata.'
    },
    {
        'name': 'technical_parameters',
        'pattern': r'(?:TECHNIQUE|CTDIvol|DLP|Dose|MRI COIL CHARGE).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes technical parameters related to imaging techniques.'
    },
    {
        'name': 'standardized_report_headers',
        'pattern': r'(?:INDICATION|COMPARISON|FINDINGS|IMPRESSION).*',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'low',
        'description': 'Removes standardized report headers. Borderline rule flagged for review.'
    }
]

# Pathology-specific rules: These apply only to pathology notes.
PATHOLOGY_RULES = [
    {
        'name': 'specimen_labeling_boilerplate',
        'pattern': r'(Received in formalin.*?submitted in toto.*?cassette.*?pieces)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes specimen labeling and accessioning boilerplate.'
    },
    {
        'name': 'gross_description_boilerplate',
        'pattern': r'(GROSS DESCRIPTION.*?submitted in toto.*?Dictated by.*?Physician)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes gross description templates.'
    },
    {
        'name': 'staining_protocol_boilerplate',
        'pattern': r'(Immunohistochemistry performed.*?FDA has determined.*?not necessary)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes immunohistochemistry method boilerplate.'
    },
    {
        'name': 'addendum_boilerplate',
        'pattern': r'(Addendum.*?Electronically signed.*?\d{2}:\d{2}:\d{2}(?:AM|PM))',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes addendum/amendment blocks.'
    }
]

# Extraction patterns: Deduplicated and merged across note types.
EXTRACTION_PATTERNS = [
    {
        'name': 'histology_type',
        'pattern': r'\b(small cell carcinoma|neuroendocrine differentiation|prostatic adenocarcinoma|ductal type|acinar carcinoma)\b',
        'note_types': ['clinician', 'imaging', 'pathology'],
        'flags': re.IGNORECASE,
        'description': 'Extracts histology type mentions.'
    },
    {
        'name': 'platinum_drug_mentions',
        'pattern': r'\b(carboplatin|cisplatin|oxaliplatin)(?:-based)?\b.*?(?:started|initiated|administered|used)',
        'note_types': ['clinician', 'imaging', 'pathology'],
        'flags': re.IGNORECASE,
        'description': 'Extracts mentions of platinum drugs with context.'
    },
    {
        'name': 'psa_values',
        'pattern': r'\bPSA[:\s]*(?:>|<)?\d+(\.\d+)?\b',
        'note_types': ['clinician', 'imaging', 'pathology'],
        'flags': re.IGNORECASE,
        'description': 'Extracts PSA values.'
    },
    {
        'name': 'dates',
        'pattern': r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b',
        'note_types': ['clinician', 'imaging', 'pathology'],
        'flags': re.IGNORECASE,
        'description': 'Extracts dates associated with diagnoses or treatment changes.'
    }
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
