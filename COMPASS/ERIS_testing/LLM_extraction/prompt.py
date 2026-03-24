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
