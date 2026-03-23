prompt_regex_creation = """
You are helping me build a regex-based text preprocessing pipeline for clinical oncology notes. These notes come from an EHR system and contain useful clinical information mixed with boilerplate formatting, system-generated text, and structural noise. My goal is to clean these notes BEFORE sending them to an LLM for structured data extraction about platinum chemotherapy use in prostate cancer patients.

I have three NOTE_TYPEs: Clinician, Imaging, and Pathology. I am providing you with a sample of notes from each type.

## Your Task

For EACH NOTE_TYPE, analyze the sample notes and produce:

### 1. UNIVERSAL RULES (apply to all note types)
Identify boilerplate patterns that appear across all note types, such as:
- Signature blocks, attestation lines, electronic signature footers
- Confidentiality/disclaimer banners
- Pagination markers (e.g., "Page X of Y")
- System-generated headers with timestamps, encounter IDs, etc.
- Decorative separators (dashes, equals signs, asterisks)
- Repeated whitespace / formatting artifacts

For each pattern, provide:
- A Python-compatible regex (using `re` module syntax with appropriate flags)
- An example of what it matches
- What to replace it with (empty string, newline, etc.)

### 2. NOTE_TYPE-SPECIFIC RULES
For each of Clinician, Imaging, and Pathology, identify type-specific boilerplate:

**Clinician notes** — look for:
- Vitals blocks (BP, HR, Temp, SpO2, weight, height, BMI)
- Medication list dumps that are not narrative discussion
- Allergy lists
- Problem list / diagnosis list boilerplate headers
- Review of systems (ROS) template text with blanks or checkboxes
- After-visit summary / patient instruction boilerplate

**Imaging notes** — look for:
- Standardized report headers (EXAM, CLINICAL INDICATION, COMPARISON, TECHNIQUE)
- Technical parameters (contrast type, dose, scanner info)
- Standardized footer/addendum blocks
- Accession numbers, order IDs

**Pathology notes** — look for:
- Specimen labeling/accessioning boilerplate
- Gross description templates
- Synoptic report formatting artifacts
- Addendum/amendment blocks
- Staining protocol details (immunohistochemistry method boilerplate, NOT results)

### 3. EXTRACTABLE STRUCTURED ELEMENTS
Identify regex patterns to PRE-EXTRACT clinically relevant structured information from the text, such as:
- Histology type mentions (e.g., "small cell carcinoma", "neuroendocrine differentiation")
- Platinum drug mentions with context (e.g., "started on carboplatin", "cisplatin-based")
- PSA values (e.g., "PSA 45.2", "PSA: >100")
- Dates associated with diagnoses or treatment changes

For each, provide a Python regex and note which NOTE_TYPE(s) it is most relevant to.

## Output Format

Return your response as a Python dictionary structure that I can directly paste into a module, like this:

```python
UNIVERSAL_RULES = [
    {
        'name': 'rule_name',
        'pattern': r'regex_pattern_here',
        'replacement': '',
        'flags': 're.MULTILINE | re.IGNORECASE',
        'description': 'What this removes'
    },
    ...
]

CLINICIAN_RULES = [...]
IMAGING_RULES = [...]
PATHOLOGY_RULES = [...]

EXTRACTION_PATTERNS = [
    {
        'name': 'histology_neuroendocrine',
        'pattern': r'regex_pattern_here',
        'flags': 're.IGNORECASE',
        'note_types': ['Clinician', 'Imaging', 'Pathology'],
        'description': 'Extracts neuroendocrine/small cell mentions'
    },
    ...
]
```

## Important Notes
- All regexes must be valid Python `re` module syntax
- Use raw strings (r'...') for all patterns
- Specify flags explicitly (re.MULTILINE, re.IGNORECASE, etc.)
- Be conservative: when in doubt, do NOT remove text. It is better to leave noise in than to accidentally strip clinical content relevant to platinum chemotherapy rationale, neuroendocrine/small cell transformation, or histologic findings.
- Focus on HIGH-CONFIDENCE patterns that are clearly boilerplate, not clinical narrative.

## Sample Notes

The notes below are separated by NOTE_TYPE. Each note is delimited by `===NOTE START===` and `===NOTE END===`.

[PASTE YOUR SAMPLED NOTES HERE — the output from sample_notes_for_regex_generation.py]
"""

prompt_note_extraction = """
You are a clinical data extraction system reading oncology notes for prostate cancer patients who received platinum-based chemotherapy. Extract features that explain WHY platinum was used, as this is not standard first-line treatment for prostate cancer.

## INPUT
1. **Note Date** — YYYY-MM-DD.
2. **Note Text** — full text of one clinical note.

## RULES
- Extract ONLY what is explicitly stated. Never infer or speculate.
- Use null when information is absent.
- Quote must be verbatim, ≤30 words.

## OUTPUT FORMAT
Return ONLY valid JSON. No markdown fencing, no commentary.

{
  "note_date": "<YYYY-MM-DD>",
  "histology": "<'adenocarcinoma' | 'neuroendocrine' | 'small_cell' | 'mixed' | 'other' | null>",
  "transformation_mentioned": <true | false>,
  "diagnosis_date": "<YYYY-MM-DD if explicitly mentioned, or null>",
  "aggressive_variant_date": "<YYYY-MM-DD if explicitly mentioned, or null>",
  "platinum_reason": "<brief stated rationale for platinum use, or null>",
  "quote": "<most relevant verbatim quote or null>"
}

Now extract from the following note:
"""


prompt_platinum_classification = """
You are a clinical data synthesis system. You receive per-note extractions for a prostate cancer patient who received platinum-based chemotherapy. All patients in this cohort received carboplatin or cisplatin. Your task is to determine WHY platinum was used, as this is not standard first-line treatment for prostate cancer.

## INPUT
A JSON array of per-note extractions in chronological order by `note_date`.

## RULES
- Base answers ONLY on what is explicitly documented. Never infer or speculate.
- For histopathology, use the most recent note that mentions histology.
- For dates, use the earliest explicit mention across notes.
- If the histology is neuroendocrine, small_cell, or mixed, OR if transformation from adenocarcinoma to any of those subtypes occurred, then `platinum_reason` MUST be "aggressive_variant". The aggressive subtype itself is the reason for platinum use.
- Use null when information is not available.

## OUTPUT FORMAT
Return ONLY valid JSON. No markdown fencing, no commentary.

{
  "histology": "<'adenocarcinoma' | 'neuroendocrine' | 'small_cell' | 'mixed' | 'other' | null>",
  "transformation_occurred": <true | false | null>,
  "transformation_detail": "<e.g. 'adenocarcinoma to neuroendocrine' or null>",
  "diagnosis_date": "<YYYY-MM-DD or null>",
  "aggressive_variant_date": "<YYYY-MM-DD or null>",
  "platinum_reason": "<enum or null>",
  "platinum_reason_detail": "<one-sentence explanation drawn from the notes, or null>",
  "supporting_quote": "<verbatim quote from notes, ≤30 words, or null>"
}

Field `platinum_reason` must be one of:
  - "aggressive_variant"
  - "disease_progression"
  - "biomarker_driven"
  - "clinical_trial"
  - "physician_judgment"
  - "other"
  - null

Now synthesize the following note-level extractions:
"""
