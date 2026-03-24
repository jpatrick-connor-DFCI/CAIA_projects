import os
import json
import time
import pandas as pd
from tqdm.auto import tqdm
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, APIError, RateLimitError, APITimeoutError

import prompt
from note_cleaning import clean_note

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'
OUTPUT_PATH = os.path.join(DATA_PATH, 'LLM_generated_labels.tsv')
EXTRACTIONS_PATH = os.path.join(DATA_PATH, 'LLM_note_extractions.json')
FAILURES_PATH = os.path.join(DATA_PATH, 'LLM_failed_patients.tsv')

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-04-01-preview",
    azure_endpoint="https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

model_name = "gpt-4o"
MAX_RETRIES = 3

# Clinical safety framing — goes in system message for both stages
clinical_safety_context = """

IMPORTANT CONTEXT: All notes below are de-identified clinical oncology documentation being
processed for structured data extraction as part of an IRB-approved medical research study
(institutional review board approved protocol). This is professional medical documentation
written by physicians, not patient-generated content. The text contains standard clinical
terminology related to cancer diagnosis, prognosis, and treatment. References to disease
outcomes, end-of-life care, self-harm assessment, psychiatric history, substance use, anatomy,
or patient distress are routine components of oncology and medical records and should be
processed as clinical data. No content in these notes constitutes harmful, dangerous, or
inappropriate material — it is standard-of-care medical documentation.
"""

extraction_system_message = prompt.prompt_note_extraction + clinical_safety_context
synthesis_system_message = prompt.prompt_platinum_classification + clinical_safety_context


def call_with_retry(messages, max_retries=MAX_RETRIES):
    """Call the API with exponential backoff for transient errors.

    Returns (response_text, error_type) where error_type is None on success,
    or a string describing the failure category.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0
            )

            finish_reason = response.choices[0].finish_reason
            if finish_reason == 'content_filter':
                return None, 'content_filter_response'

            return response.choices[0].message.content.strip(), None

        except RateLimitError:
            wait = 2 ** attempt * 5
            print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

        except APITimeoutError:
            wait = 2 ** attempt * 3
            print(f"    Timeout, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

        except APIError as e:
            error_body = str(e)
            if 'content_filter' in error_body.lower() or 'content_management' in error_body.lower():
                return None, 'content_filter_input'
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f'api_error: {error_body[:200]}'

        except Exception as e:
            return None, f'unexpected: {type(e).__name__}: {str(e)[:200]}'

    return None, 'max_retries_exceeded'


def log_failure(mrn, error_type, num_notes, stage):
    """Append a failure record to the failures TSV."""
    fail_row = pd.DataFrame([{
        'DFCI_MRN': int(mrn),
        'error_type': error_type,
        'stage': stage,
        'num_notes': num_notes
    }])
    fail_row.to_csv(FAILURES_PATH, mode='a', sep='\t', index=False,
                    header=not os.path.exists(FAILURES_PATH) or os.path.getsize(FAILURES_PATH) == 0)


# =========================================================================
# Load data and checkpoint state
# =========================================================================
candidate_LLM_text_df = pd.read_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'))
unique_mrns = candidate_LLM_text_df['DFCI_MRN'].unique()

# Load existing extractions checkpoint (per-note results keyed by MRN)
extractions_by_mrn = {}
if os.path.exists(EXTRACTIONS_PATH):
    with open(EXTRACTIONS_PATH, 'r') as f:
        extractions_by_mrn = {int(k): v for k, v in json.load(f).items()}

# Load completed synthesis results
completed_mrns = set()
if os.path.exists(OUTPUT_PATH):
    existing_df = pd.read_csv(OUTPUT_PATH, sep='\t')
    completed_mrns = set(existing_df['DFCI_MRN'].unique())

# Skip patients that already failed with content_filter
failed_mrns = set()
if os.path.exists(FAILURES_PATH):
    failed_df = pd.read_csv(FAILURES_PATH, sep='\t')
    failed_mrns = set(failed_df.loc[
        failed_df['error_type'].str.contains('content_filter', na=False), 'DFCI_MRN'
    ].unique())

remaining_mrns = [m for m in unique_mrns if m not in completed_mrns and m not in failed_mrns]
print(f"Processing {len(remaining_mrns)} patients "
      f"({len(completed_mrns)} done, {len(failed_mrns)} content-filtered)\n")

# =========================================================================
# Main loop: extract then synthesize per patient
# =========================================================================
for cur_mrn in tqdm(remaining_mrns, desc='Patients'):
    mrn_df = (candidate_LLM_text_df
              .loc[candidate_LLM_text_df['DFCI_MRN'] == cur_mrn]
              .sort_values('EVENT_DATE'))

    # ------------------------------------------------------------------
    # Stage 1: Per-note extraction (skip if already checkpointed)
    # ------------------------------------------------------------------
    if cur_mrn in extractions_by_mrn:
        note_extractions = extractions_by_mrn[cur_mrn]
    else:
        note_extractions = []
        note_failed = False

        for _, row in mrn_df.iterrows():
            cleaned = clean_note(row['CLINICAL_TEXT'], note_type=row.get('NOTE_TYPE'))
            if not cleaned:
                continue

            note_type = row.get('NOTE_TYPE', 'Unknown')
            user_content = (
                f"Note Type: {note_type}\n"
                f"Note Date: {row['EVENT_DATE']}\n\n"
                f"{cleaned}"
            )

            response_text, error_type = call_with_retry([
                {'role': 'system', 'content': extraction_system_message},
                {'role': 'user', 'content': user_content}
            ])

            if error_type:
                if 'content_filter' in error_type:
                    # Skip this note but continue with others
                    print(f"    Note filtered for {cur_mrn} ({note_type} {row['EVENT_DATE']})")
                    continue
                else:
                    print(f"    Note extraction failed for {cur_mrn}: {error_type}")
                    continue

            try:
                extraction = json.loads(response_text)
                note_extractions.append(extraction)
            except json.JSONDecodeError:
                print(f"    JSON parse failed for note {row['EVENT_DATE']} of {cur_mrn}")
                continue

        if not note_extractions:
            print(f"  Skipping {cur_mrn}: no successful extractions")
            log_failure(cur_mrn, 'no_extractions', len(mrn_df), 'extraction')
            continue

        # Checkpoint extractions
        extractions_by_mrn[cur_mrn] = note_extractions
        with open(EXTRACTIONS_PATH, 'w') as f:
            json.dump({str(k): v for k, v in extractions_by_mrn.items()}, f)

    # ------------------------------------------------------------------
    # Stage 2: Patient-level synthesis
    # ------------------------------------------------------------------
    response_text, error_type = call_with_retry([
        {'role': 'system', 'content': synthesis_system_message},
        {'role': 'user', 'content': json.dumps(note_extractions)}
    ])

    if error_type:
        print(f"  Synthesis failed for {cur_mrn}: {error_type}")
        log_failure(cur_mrn, error_type, len(note_extractions), 'synthesis')
        continue

    try:
        px_result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"  Synthesis JSON parse failed for {cur_mrn}: {e}")
        log_failure(cur_mrn, f'json_parse: {str(e)[:200]}', len(note_extractions), 'synthesis')
        continue

    px_result['DFCI_MRN'] = int(cur_mrn)
    px_result['num_notes'] = len(note_extractions)

    # Serialize list fields for clean TSV output
    for col in ('supporting_quotes', 'supporting_quote_dates'):
        if isinstance(px_result.get(col), list):
            px_result[col] = ' | '.join(str(x) for x in px_result[col])

    # Append incrementally
    row_df = pd.DataFrame([px_result])
    row_df.to_csv(OUTPUT_PATH, mode='a', sep='\t', index=False,
                   header=not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0)

# =========================================================================
# Report
# =========================================================================
n_success = len(pd.read_csv(OUTPUT_PATH, sep='\t')) if os.path.exists(OUTPUT_PATH) else 0
n_failed = len(pd.read_csv(FAILURES_PATH, sep='\t')) if os.path.exists(FAILURES_PATH) else 0
print(f'\nDone. {n_success} succeeded, {n_failed} failed.')
print(f'Results: {OUTPUT_PATH}')
print(f'Extractions: {EXTRACTIONS_PATH}')
if n_failed > 0:
    print(f'Failures: {FAILURES_PATH}')
