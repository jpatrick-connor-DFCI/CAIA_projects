### Regex Generation Prompts
prompt_regex_clinician = """
You are helping me build a regex-based text preprocessing pipeline for clinical oncology notes from an EHR system. These notes contain useful clinical information mixed with boilerplate formatting, system-generated text, and structural noise. My goal is to clean these notes BEFORE sending them to an LLM for structured data extraction about platinum chemotherapy use in prostate cancer patients.

You are analyzing **Clinician** notes only.

## Your Task

Analyze the sample notes below and produce:

### 1. BOILERPLATE REMOVAL RULES
Identify patterns that are clearly non-clinical boilerplate in Clinician notes.

**Universal patterns** (likely shared across note types):
- Signature blocks, attestation lines, electronic signature footers
- Confidentiality/disclaimer banners
- Pagination markers (e.g., "Page X of Y")
- System-generated headers with timestamps, encounter IDs, etc.
- Decorative separators (dashes, equals signs, asterisks)
- Repeated whitespace / formatting artifacts

**Clinician-specific patterns** — look especially for:
- Vitals blocks (BP, HR, Temp, SpO2, weight, height, BMI)
- Medication list dumps that are not narrative discussion
- Allergy lists
- Problem list / diagnosis list boilerplate headers
- Review of systems (ROS) template text with blanks or checkboxes
- After-visit summary / patient instruction boilerplate

For each pattern, provide:
- A Python-compatible regex (using `re` module syntax with appropriate flags)
- An example of what it matches from the sample notes
- What to replace it with (empty string, newline, etc.)

### 2. EXTRACTABLE STRUCTURED ELEMENTS
Identify regex patterns to PRE-EXTRACT clinically relevant structured information from Clinician notes, such as:
- Histology type mentions (e.g., "small cell carcinoma", "neuroendocrine differentiation")
- Platinum drug mentions with context (e.g., "started on carboplatin", "cisplatin-based")
- PSA values (e.g., "PSA 45.2", "PSA: >100")
- Dates associated with diagnoses or treatment changes
- Any other structured clinical data points you can reliably extract from these notes

## Output Format

Return your response as a Python dictionary structure:
```python
BOILERPLATE_RULES = [
    {
        'name': 'rule_name',
        'pattern': r'regex_pattern_here',
        'replacement': '',
        'flags': 're.MULTILINE | re.IGNORECASE',
        'description': 'What this removes',
        'example_match': 'Example text this would match',
        'source': 'universal' | 'clinician'
    },
    ...
]

EXTRACTION_PATTERNS = [
    {
        'name': 'pattern_name',
        'pattern': r'regex_pattern_here',
        'flags': 're.IGNORECASE',
        'description': 'What this extracts',
        'example_match': 'Example text this would match'
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
- Ground every rule in an actual example from the sample notes below.

## Sample Clinician Notes

{notes}
"""

prompt_regex_imaging = """
You are helping me build a regex-based text preprocessing pipeline for clinical oncology notes from an EHR system. These notes contain useful clinical information mixed with boilerplate formatting, system-generated text, and structural noise. My goal is to clean these notes BEFORE sending them to an LLM for structured data extraction about platinum chemotherapy use in prostate cancer patients.

You are analyzing **Imaging** notes only.

## Your Task

Analyze the sample notes below and produce:

### 1. BOILERPLATE REMOVAL RULES
Identify patterns that are clearly non-clinical boilerplate in Imaging notes.

**Universal patterns** (likely shared across note types):
- Signature blocks, attestation lines, electronic signature footers
- Confidentiality/disclaimer banners
- Pagination markers (e.g., "Page X of Y")
- System-generated headers with timestamps, encounter IDs, etc.
- Decorative separators (dashes, equals signs, asterisks)
- Repeated whitespace / formatting artifacts

**Imaging-specific patterns** — look especially for:
- Standardized report headers (EXAM, CLINICAL INDICATION, COMPARISON, TECHNIQUE)
- Technical parameters (contrast type, dose, scanner info)
- Standardized footer/addendum blocks
- Accession numbers, order IDs

For each pattern, provide:
- A Python-compatible regex (using `re` module syntax with appropriate flags)
- An example of what it matches from the sample notes
- What to replace it with (empty string, newline, etc.)

### 2. EXTRACTABLE STRUCTURED ELEMENTS
Identify regex patterns to PRE-EXTRACT clinically relevant structured information from Imaging notes, such as:
- Histology type mentions (e.g., "small cell carcinoma", "neuroendocrine differentiation")
- Platinum drug mentions with context (e.g., "started on carboplatin", "cisplatin-based")
- PSA values (e.g., "PSA 45.2", "PSA: >100")
- Dates associated with diagnoses or treatment changes
- Any other structured clinical data points you can reliably extract from these notes

## Output Format

Return your response as a Python dictionary structure:
```python
BOILERPLATE_RULES = [
    {
        'name': 'rule_name',
        'pattern': r'regex_pattern_here',
        'replacement': '',
        'flags': 're.MULTILINE | re.IGNORECASE',
        'description': 'What this removes',
        'example_match': 'Example text this would match',
        'source': 'universal' | 'imaging'
    },
    ...
]

EXTRACTION_PATTERNS = [
    {
        'name': 'pattern_name',
        'pattern': r'regex_pattern_here',
        'flags': 're.IGNORECASE',
        'description': 'What this extracts',
        'example_match': 'Example text this would match'
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
- Ground every rule in an actual example from the sample notes below.

## Sample Imaging Notes

{notes}
"""

prompt_regex_pathology = """
You are helping me build a regex-based text preprocessing pipeline for clinical oncology notes from an EHR system. These notes contain useful clinical information mixed with boilerplate formatting, system-generated text, and structural noise. My goal is to clean these notes BEFORE sending them to an LLM for structured data extraction about platinum chemotherapy use in prostate cancer patients.

You are analyzing **Pathology** notes only.

## Your Task

Analyze the sample notes below and produce:

### 1. BOILERPLATE REMOVAL RULES
Identify patterns that are clearly non-clinical boilerplate in Pathology notes.

**Universal patterns** (likely shared across note types):
- Signature blocks, attestation lines, electronic signature footers
- Confidentiality/disclaimer banners
- Pagination markers (e.g., "Page X of Y")
- System-generated headers with timestamps, encounter IDs, etc.
- Decorative separators (dashes, equals signs, asterisks)
- Repeated whitespace / formatting artifacts

**Pathology-specific patterns** — look especially for:
- Specimen labeling/accessioning boilerplate
- Gross description templates
- Synoptic report formatting artifacts
- Addendum/amendment blocks
- Staining protocol details (immunohistochemistry method boilerplate, NOT results)

For each pattern, provide:
- A Python-compatible regex (using `re` module syntax with appropriate flags)
- An example of what it matches from the sample notes
- What to replace it with (empty string, newline, etc.)

### 2. EXTRACTABLE STRUCTURED ELEMENTS
Identify regex patterns to PRE-EXTRACT clinically relevant structured information from Pathology notes, such as:
- Histology type mentions (e.g., "small cell carcinoma", "neuroendocrine differentiation")
- Platinum drug mentions with context (e.g., "started on carboplatin", "cisplatin-based")
- PSA values (e.g., "PSA 45.2", "PSA: >100")
- Gleason scores / Grade Groups
- Dates associated with diagnoses or treatment changes
- Any other structured clinical data points you can reliably extract from these notes

## Output Format

Return your response as a Python dictionary structure:
```python
BOILERPLATE_RULES = [
    {
        'name': 'rule_name',
        'pattern': r'regex_pattern_here',
        'replacement': '',
        'flags': 're.MULTILINE | re.IGNORECASE',
        'description': 'What this removes',
        'example_match': 'Example text this would match',
        'source': 'universal' | 'pathology'
    },
    ...
]

EXTRACTION_PATTERNS = [
    {
        'name': 'pattern_name',
        'pattern': r'regex_pattern_here',
        'flags': 're.IGNORECASE',
        'description': 'What this extracts',
        'example_match': 'Example text this would match'
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
- Ground every rule in an actual example from the sample notes below.

## Sample Pathology Notes

{notes}
"""

prompt_regex_synthesis = """
You are finalizing a regex-based text preprocessing pipeline for clinical oncology notes. I ran a per-note-type analysis on three note types (Clinician, Imaging, Pathology) and got candidate regex rules from each. Your job is to deduplicate, reconcile, and organize them into a single production-ready Python module.

## Inputs

Below are the raw outputs from each per-note-type analysis:

### Clinician Analysis
{clinician_output}

### Imaging Analysis
{imaging_output}

### Pathology Analysis
{pathology_output}

## Your Task

### 1. Deduplicate & Merge Boilerplate Rules
- Identify rules tagged as `'source': 'universal'` or that appear in multiple note-type outputs (e.g., signature blocks, disclaimers, separators). Merge these into a single UNIVERSAL_RULES list, picking the most general regex that covers all observed variants.
- If two note types produced slightly different regexes for the same pattern, unify into one regex that handles both, or keep the broader one.
- Place note-type-specific rules (those that only appeared in one type, or that would be harmful to apply to other types) into CLINICIAN_RULES, IMAGING_RULES, or PATHOLOGY_RULES.

### 2. Deduplicate & Merge Extraction Patterns
- Merge extraction patterns across note types. For each pattern, note which note_types it applies to.
- If two analyses produced overlapping extraction regexes (e.g., both found PSA patterns), pick the most robust one or merge them.

### 3. Validate & Tighten
- Check that no rule would accidentally strip clinical narrative about platinum chemotherapy, histology, neuroendocrine transformation, treatment rationale, Gleason scores, or pathologic findings.
- Flag any rules you consider borderline risky and mark them with `'confidence': 'low'` so I can review them manually. All other rules should be marked `'confidence': 'high'`.
- Ensure all regexes are valid Python `re` syntax.

## Output Format

Return the final module as a single Python code block I can paste directly into a .py file:
```python
import re

# =============================================================================
# UNIVERSAL BOILERPLATE RULES
# Applied to ALL note types before type-specific rules.
# N rules covering: [brief category summary]
# =============================================================================
UNIVERSAL_RULES = [
    {
        'name': 'rule_name',
        'pattern': r'regex_pattern_here',
        'replacement': '',
        'flags': 're.MULTILINE | re.IGNORECASE',
        'description': 'What this removes',
        'confidence': 'high' | 'low'
    },
    ...
]

# =============================================================================
# CLINICIAN-SPECIFIC BOILERPLATE RULES
# Applied only to Clinician notes, after universal rules.
# N rules covering: [brief category summary]
# =============================================================================
CLINICIAN_RULES = [...]

# =============================================================================
# IMAGING-SPECIFIC BOILERPLATE RULES
# Applied only to Imaging notes, after universal rules.
# N rules covering: [brief category summary]
# =============================================================================
IMAGING_RULES = [...]

# =============================================================================
# PATHOLOGY-SPECIFIC BOILERPLATE RULES
# Applied only to Pathology notes, after universal rules.
# N rules covering: [brief category summary]
# =============================================================================
PATHOLOGY_RULES = [...]

# =============================================================================
# EXTRACTION PATTERNS
# Run on cleaned text to pre-extract structured clinical data points.
# N patterns covering: [brief category summary]
# =============================================================================
EXTRACTION_PATTERNS = [
    {
        'name': 'pattern_name',
        'pattern': r'regex_pattern_here',
        'flags': 're.IGNORECASE',
        'note_types': ['Clinician', 'Imaging', 'Pathology'],
        'description': 'What this extracts'
    },
    ...
]
```

## Important Notes
- All regexes must be valid Python `re` module syntax with raw strings (r'...')
- Be conservative: if the per-note analyses disagreed on whether something is boilerplate, err on the side of NOT removing it
- The pipeline will apply UNIVERSAL_RULES first, then the appropriate note-type-specific rules, then run EXTRACTION_PATTERNS on the cleaned text
- Do NOT invent new rules beyond what the three analyses provided — your job is to merge, deduplicate, and validate, not to add new patterns
- Fill in the comment headers with actual counts and category summaries based on the final merged lists
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
