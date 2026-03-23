### Regex Generation Prompts

# Type-specific boilerplate hints — inserted into the template per NOTE_TYPE
_TYPE_SPECIFIC_HINTS = {
    'Clinician': """
- Vitals blocks (BP, HR, Temp, SpO2, weight, height, BMI)
- Medication list dumps that are not narrative discussion
- Allergy lists
- Problem list / diagnosis list boilerplate headers
- Review of systems (ROS) template text with blanks or checkboxes
- After-visit summary / patient instruction boilerplate""",

    'Imaging': """
- Standardized report headers (EXAM, CLINICAL INDICATION, COMPARISON, TECHNIQUE)
- Technical parameters (contrast type, dose, scanner info)
- Standardized footer/addendum blocks
- Accession numbers, order IDs""",

    'Pathology': """
- Specimen labeling/accessioning boilerplate
- Gross description templates
- Synoptic report formatting artifacts
- Addendum/amendment blocks
- Staining protocol details (immunohistochemistry method boilerplate, NOT results)""",
}

prompt_regex_per_type = """
You are helping me build a regex-based text preprocessing pipeline for clinical oncology notes from an EHR system. These notes contain useful clinical information mixed with boilerplate formatting, system-generated text, and structural noise. My goal is to clean these notes BEFORE sending them to an LLM for structured data extraction about platinum chemotherapy use in prostate cancer patients.

You are analyzing **{note_type}** notes only.

## Your Task

Analyze the sample notes below and produce:

### 1. BOILERPLATE REMOVAL RULES
Identify patterns that are clearly non-clinical boilerplate.

**Universal patterns** (likely shared across note types):
- Signature blocks, attestation lines, electronic signature footers
- Confidentiality/disclaimer banners
- Pagination markers (e.g., "Page X of Y")
- System-generated headers with timestamps, encounter IDs, etc.
- Decorative separators (dashes, equals signs, asterisks)
- Repeated whitespace / formatting artifacts

**{note_type}-specific patterns** — look especially for:
{type_specific_hints}

For each pattern, provide:
- A Python-compatible regex (using `re` module syntax with appropriate flags)
- An example of what it matches from the sample notes
- What to replace it with (empty string, newline, etc.)

### 2. EXTRACTABLE STRUCTURED ELEMENTS
Identify regex patterns to PRE-EXTRACT clinically relevant structured information, such as:
- Histology type mentions (e.g., "small cell carcinoma", "neuroendocrine differentiation")
- Platinum drug mentions with context (e.g., "started on carboplatin", "cisplatin-based")
- PSA values (e.g., "PSA 45.2", "PSA: >100")
- Dates associated with diagnoses or treatment changes

## Output Format

Return your response as a Python dictionary structure:
```python
BOILERPLATE_RULES = [
    {{
        'name': 'rule_name',
        'pattern': r'regex_pattern_here',
        'replacement': '',
        'flags': 're.MULTILINE | re.IGNORECASE',
        'description': 'What this removes',
        'example_match': 'Example text this would match',
        'source': 'universal' | '{note_type_lower}'
    }},
    ...
]

EXTRACTION_PATTERNS = [
    {{
        'name': 'pattern_name',
        'pattern': r'regex_pattern_here',
        'flags': 're.IGNORECASE',
        'description': 'What this extracts',
        'example_match': 'Example text this would match'
    }},
    ...
]
```

## Important Notes
- All regexes must be valid Python `re` module syntax with raw strings (r'...')
- Specify flags explicitly (re.MULTILINE, re.IGNORECASE, etc.)
- Be conservative: do NOT remove text that could be relevant to platinum chemotherapy rationale, neuroendocrine/small cell transformation, or histologic findings.
- Focus on HIGH-CONFIDENCE patterns that are clearly boilerplate, not clinical narrative.
- Ground every rule in an actual example from the sample notes below.

## Sample {note_type} Notes

{notes}
"""


def build_regex_prompt(note_type, notes):
    """Format the per-type regex generation prompt."""
    return prompt_regex_per_type.format(
        note_type=note_type,
        note_type_lower=note_type.lower(),
        type_specific_hints=_TYPE_SPECIFIC_HINTS[note_type],
        notes=notes,
    )


prompt_regex_synthesis = """
You are finalizing a regex-based text preprocessing pipeline for clinical oncology notes. I ran a per-note-type analysis on Clinician, Imaging, and Pathology notes and got candidate regex rules from each. Your job is to deduplicate, reconcile, and organize them into a single production-ready Python module.

## Inputs

### Clinician Analysis
{clinician_output}

### Imaging Analysis
{imaging_output}

### Pathology Analysis
{pathology_output}

## Your Task

### 1. Deduplicate & Merge Boilerplate Rules
- Rules tagged `'source': 'universal'` or appearing in multiple outputs → merge into UNIVERSAL_RULES, picking the most general regex.
- Rules unique to one type or harmful to apply broadly → place in CLINICIAN_RULES, IMAGING_RULES, or PATHOLOGY_RULES.

### 2. Deduplicate & Merge Extraction Patterns
- Merge overlapping extraction regexes across types. For each, note which note_types it applies to.

### 3. Validate
- Ensure no rule strips clinical narrative about platinum chemotherapy, histology, neuroendocrine transformation, or treatment rationale.
- Flag borderline rules with `'confidence': 'low'`; mark the rest `'confidence': 'high'`.
- Ensure all regexes are valid Python `re` syntax.

## Output Format

Return a single Python code block I can paste directly into note_cleaning.py:
```python
import re

UNIVERSAL_RULES = [
    {{'name': '...', 'pattern': r'...', 'replacement': '', 'flags': 0, 'confidence': 'high'}},
    ...
]
CLINICIAN_RULES = [...]
IMAGING_RULES = [...]
PATHOLOGY_RULES = [...]
EXTRACTION_PATTERNS = [
    {{'name': '...', 'pattern': r'...', 'note_types': [...], 'description': '...'}},
    ...
]
```

## Important Notes
- Do NOT invent new rules — only merge, deduplicate, and validate what the three analyses provided.
- Be conservative: if analyses disagreed on whether something is boilerplate, do NOT remove it.
- Pipeline order: UNIVERSAL_RULES → type-specific rules → EXTRACTION_PATTERNS on cleaned text.
"""

### NEPC Extraction Prompts
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
