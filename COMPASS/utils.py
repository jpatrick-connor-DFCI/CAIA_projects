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
        'name': 'collapse_blank_lines',
        'pattern': r'\n\s*\n+',
        'replacement': '\n\n',
        'flags': 0,
        'confidence': 'high',
        'description': 'Collapses multiple blank lines into one, preserving paragraph structure.'
    },
    {
        'name': 'collapse_horizontal_whitespace',
        'pattern': r'[ \t]{2,}',
        'replacement': ' ',
        'flags': 0,
        'confidence': 'high',
        'description': 'Collapses repeated spaces/tabs on the same line.'
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
        'pattern': r'(?:Exam Number|Report Status|Ordering Provider|Accession number)[:\s].*',
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
        'name': 'standardized_report_header_labels',
        'pattern': r'^[ \t]*(INDICATION|COMPARISON|FINDINGS|IMPRESSION|TECHNIQUE|EXAM)[ \t]*:[ \t]*(?=\S)',
        'replacement': '',
        'flags': re.MULTILINE | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes section header labels when inline with content (e.g. "FINDINGS: ..."), preserving the content itself.'
    }
]

# Pathology-specific rules: These apply only to pathology notes.
PATHOLOGY_RULES = [
    {
        'name': 'gross_description_boilerplate',
        'pattern': r'(GROSS DESCRIPTION.*?submitted in toto.*?Dictated by.*?Physician)',
        'replacement': '',
        'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes gross description templates.'
    },
    {
        'name': 'staining_protocol_boilerplate',
        'pattern': r'(Immunohistochemistry performed.*?FDA has determined.*?not necessary)',
        'replacement': '',
        'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE,
        'confidence': 'high',
        'description': 'Removes immunohistochemistry method boilerplate.'
    },
]

TYPE_SPECIFIC_RULES = {
    'Clinician': CLINICIAN_RULES,
    'Imaging': IMAGING_RULES,
    'Pathology': PATHOLOGY_RULES,
}

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

"""
Per-Note and Synthesis Prompts for DFCI-enterprise GPT-4o instance.

For each patient, apply the prompt_note_extraction prompt for each cleaned note to generate a json file for that note.
Then, collate the per-note json files into a json array to pass into a synthesis API call using prompt_platinum_classification.

"""

### Per-Note Extraction Prompt
prompt_note_extraction = """
You are a clinical data extraction system for an IRB-approved research study on prostate cancer patients who received platinum-based chemotherapy. Your job is to extract ALL clinically relevant evidence from a single note that could help explain why platinum was used.

## INPUT
- **Note Type** — Clinician, Imaging, or Pathology.
- **Note Date** — YYYY-MM-DD.
- **Note Text** — full text of one clinical note.

## WHAT TO EXTRACT
Surface any evidence related to:
- Histology (adenocarcinoma, neuroendocrine, small cell, mixed, ductal, etc.)
- Histologic transformation (e.g., adenocarcinoma transforming to neuroendocrine/small cell)
- Metastatic disease (bone lesions, visceral metastases, lymph node involvement, pathology stating "metastatic")
- Castration resistance (rising PSA on ADT, progression on hormonal therapy)
- Platinum chemotherapy rationale (why it was chosen, what it was combined with)
- Clinical trial enrollment (protocol names, study references)
- Biomarker/genomic findings (BRCA2, HRD, MSI-H, etc.)
- Disease status (progression, response, stable)
- Other cancer diagnoses (non-prostate primaries that might explain platinum use)

## RULES
- Extract what is documented. When something is implied but not explicitly stated (e.g., a pathology report saying "consistent with metastatic prostate adenocarcinoma"), extract it and note the source.
- Use null when the note does not contain relevant information for a field.
- Quotes must be verbatim, ≤30 words each.

## OUTPUT FORMAT
Return ONLY valid JSON. No markdown fencing, no commentary.

{
  "note_date": "<YYYY-MM-DD>",
  "note_type": "<Clinician | Imaging | Pathology>",
  "histology": "<'adenocarcinoma' | 'neuroendocrine' | 'small_cell' | 'mixed' | 'other' | null>",
  "metastatic_disease": <true | false | null>,
  "transformation_mentioned": <true | false>,
  "transformation_detail": "<e.g. 'adenocarcinoma to neuroendocrine' or null>",
  "castration_resistant": <true | false | null>,
  "platinum_mentioned": <true | false>,
  "platinum_context": "<brief description of how platinum is discussed, or null>",
  "clinical_trial_mentioned": <true | false>,
  "biomarkers": "<any genomic/molecular findings mentioned, or null>",
  "other_cancer": "<non-prostate cancer mentioned, or null>",
  "key_quotes": ["<verbatim quote, ≤30 words>"]
}

Now extract from the following note:
"""

### Patient-Level Synthesis Prompt
prompt_platinum_classification = """
You are a clinical data synthesis system for an IRB-approved research study. You receive per-note extractions for a prostate cancer patient who received platinum-based chemotherapy (carboplatin or cisplatin). Your task is to synthesize the evidence across all notes and determine WHY platinum was used.

## CLINICAL CONTEXT
Platinum chemotherapy is NOT standard treatment for prostate cancer. When a prostate cancer patient receives platinum, there is always a specific clinical reason. We are studying whether platinum initiation can serve as a proxy for aggressive disease phenotypes. Common reasons include:
- **Neuroendocrine or small cell transformation** — the tumor changed from adenocarcinoma to a neuroendocrine or small cell phenotype, treated with platinum-based regimens.
- **De novo small cell / neuroendocrine prostate cancer** — rare cases where the initial diagnosis is small cell or neuroendocrine, not a transformation from adenocarcinoma.
- **Clinical trial enrollment** — platinum given as part of a trial protocol.
- **Castration-resistant prostate cancer (CRPC)** — platinum used in the setting of documented castration resistance (rising PSA or progression despite ADT/hormonal therapy), without histologic transformation.
- **Disease progression on standard therapies** — platinum used empirically after exhausting standard options, without documented castration resistance or histologic transformation.
- **Non-prostate primary** — the platinum is for a different cancer in a patient who also has prostate cancer.
- **Biomarker-driven** — platinum selected based on genomic findings (e.g., BRCA2, HRD, MSI-H).

## INPUT
A JSON array of per-note extractions in chronological order by `note_date`.

## RULES
- Base your answer on the evidence across all notes. When the clinical reasoning for platinum is not expressly stated, draw conclusions from the available evidence. For example:
  - A pathology extraction showing histology "small cell" or "neuroendocrine" supports transformation even if no note explicitly says "transformation occurred."
  - Extractions showing `metastatic_disease: true` document metastatic disease even if no clinician note explicitly says "metastatic."
  - Extractions showing `castration_resistant: true` support CRPC.
  - Extractions showing `clinical_trial_mentioned: true` with platinum context support clinical trial.
- When drawing conclusions from indirect evidence, set `confidence` to "medium" rather than "high."
- Weigh evidence by note type: Pathology extractions are most authoritative for histology; Clinician extractions for treatment rationale; Imaging extractions for disease burden and metastatic status.
- If extractions contradict on histology, prefer the most recent Pathology note. If none, use the most recent Clinician note.
- If multiple reasons apply, report the PRIMARY reason that drove the platinum decision.
- Use null only when the extractions genuinely do not contain enough information to determine the answer.

## OUTPUT FORMAT
Return ONLY valid JSON. No markdown fencing, no commentary.

{
  "platinum_reason": "<enum or null>",
  "platinum_reason_detail": "<one-sentence explanation grounded in the note extractions, or null>",
  "histology_at_platinum_start": "<'adenocarcinoma' | 'neuroendocrine' | 'small_cell' | 'mixed' | 'other' | null>",
  "metastatic_disease": <true | false | null>,
  "transformation_documented": <true | false | null>,
  "transformation_detail": "<e.g. 'adenocarcinoma to small cell carcinoma' or null>",
  "supporting_quotes": ["<verbatim quote from extractions, ≤30 words>"],
  "supporting_quote_dates": ["<YYYY-MM-DD of note containing each quote>"],
  "confidence": "<'high' | 'medium' | 'low'>"
}

Field `platinum_reason` must be one of:
  - "neuroendocrine_transformation" — documented transformation to neuroendocrine or small cell
  - "de_novo_neuroendocrine" — initial diagnosis was neuroendocrine or small cell, no prior adenocarcinoma
  - "clinical_trial" — platinum given as part of a clinical trial
  - "crpc" — platinum in the setting of castration-resistant prostate cancer, no histologic transformation documented
  - "disease_progression" — empiric platinum after progression on standard therapies, without documented castration resistance or histologic transformation
  - "non_prostate_primary" — platinum is for a different cancer, not the prostate cancer
  - "biomarker_driven" — platinum selected based on genomic/molecular findings
  - "other" — documented reason that does not fit the above categories
  - null — notes do not contain enough information to determine the reason

Field `confidence`:
  - "high" — the reason is explicitly stated or clearly documented
  - "medium" — the reason is strongly implied by the clinical context but not explicitly stated
  - "low" — ambiguous or insufficient documentation; best guess from available evidence

Now synthesize the following per-note extractions:
"""