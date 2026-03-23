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
FAILURES_PATH = os.path.join(DATA_PATH, 'LLM_failed_patients.tsv')
NOTE_DELIMITER = '\n\n---\n\n'

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-04-01-preview",
    azure_endpoint="https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

model_name = "gpt-4o"
MAX_CONTEXT_CHARS = 400_000  # conservative limit for gpt-4o 128k context (~3.5 chars/token)
MAX_RETRIES = 3

# Clinical context goes in system message so Azure content filter weighs it
# when evaluating the user message containing clinical notes
system_message = prompt.prompt_platinum_classification + """

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


def format_patient_notes(mrn_df):
    """Format all notes for a patient into a list of note blocks."""
    note_blocks = []
    for _, row in mrn_df.iterrows():
        cleaned = clean_note(row['CLINICAL_TEXT'], note_type=row.get('NOTE_TYPE'))
        if not cleaned:
            continue
        note_type = row.get('NOTE_TYPE', 'Unknown')
        note_blocks.append(
            f"[{note_type} — {row['EVENT_DATE']}]\n{cleaned}"
        )
    return note_blocks


def truncate_notes(note_blocks, max_chars):
    """Join note blocks up to max_chars, truncating at note boundaries."""
    joined = []
    total = 0
    for block in note_blocks:
        block_len = len(block) + len(NOTE_DELIMITER)
        if total + block_len > max_chars:
            break
        joined.append(block)
        total += block_len
    return NOTE_DELIMITER.join(joined), len(note_blocks) - len(joined)


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

            # Check for content filter on the response
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
            # Other API errors — retry once, then fail
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f'api_error: {error_body[:200]}'

        except Exception as e:
            return None, f'unexpected: {type(e).__name__}: {str(e)[:200]}'

    return None, 'max_retries_exceeded'


# Load data and check for existing checkpoint
candidate_LLM_text_df = pd.read_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'))
unique_mrns = candidate_LLM_text_df['DFCI_MRN'].unique()

completed_mrns = set()
if os.path.exists(OUTPUT_PATH):
    existing_df = pd.read_csv(OUTPUT_PATH, sep='\t')
    completed_mrns = set(existing_df['DFCI_MRN'].unique())

# Also skip patients that already failed with content_filter (no point retrying)
failed_mrns = set()
if os.path.exists(FAILURES_PATH):
    failed_df = pd.read_csv(FAILURES_PATH, sep='\t')
    failed_mrns = set(failed_df.loc[
        failed_df['error_type'].str.contains('content_filter', na=False), 'DFCI_MRN'
    ].unique())

remaining_mrns = [m for m in unique_mrns if m not in completed_mrns and m not in failed_mrns]
print(f"Processing {len(remaining_mrns)} patients "
      f"({len(completed_mrns)} done, {len(failed_mrns)} content-filtered)\n")

for cur_mrn in tqdm(remaining_mrns, desc='Patients'):
    mrn_df = (candidate_LLM_text_df
              .loc[candidate_LLM_text_df['DFCI_MRN'] == cur_mrn]
              .sort_values('EVENT_DATE'))

    note_blocks = format_patient_notes(mrn_df)
    if not note_blocks:
        print(f"  Skipping {cur_mrn}: no notes after cleaning")
        continue

    patient_notes, n_dropped = truncate_notes(note_blocks, MAX_CONTEXT_CHARS)
    if n_dropped > 0:
        print(f"  Warning: dropped {n_dropped} notes for {cur_mrn} to fit context limit")

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': patient_notes}
    ]

    response_text, error_type = call_with_retry(messages)

    if error_type:
        print(f"  Failed for {cur_mrn}: {error_type}")
        fail_row = pd.DataFrame([{
            'DFCI_MRN': int(cur_mrn),
            'error_type': error_type,
            'num_notes': len(note_blocks)
        }])
        fail_row.to_csv(FAILURES_PATH, mode='a', sep='\t', index=False,
                        header=not os.path.exists(FAILURES_PATH) or os.path.getsize(FAILURES_PATH) == 0)
        continue

    try:
        px_result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed for {cur_mrn}: {e}")
        fail_row = pd.DataFrame([{
            'DFCI_MRN': int(cur_mrn),
            'error_type': f'json_parse: {str(e)[:200]}',
            'num_notes': len(note_blocks)
        }])
        fail_row.to_csv(FAILURES_PATH, mode='a', sep='\t', index=False,
                        header=not os.path.exists(FAILURES_PATH) or os.path.getsize(FAILURES_PATH) == 0)
        continue

    px_result['DFCI_MRN'] = int(cur_mrn)
    px_result['num_notes'] = len(note_blocks)

    # Serialize list fields for clean TSV output
    if isinstance(px_result.get('supporting_quotes'), list):
        px_result['supporting_quotes'] = ' | '.join(px_result['supporting_quotes'])

    # Append incrementally
    row_df = pd.DataFrame([px_result])
    row_df.to_csv(OUTPUT_PATH, mode='a', sep='\t', index=False,
                   header=not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0)

# Report final counts
n_success = len(pd.read_csv(OUTPUT_PATH, sep='\t')) if os.path.exists(OUTPUT_PATH) else 0
n_failed = len(pd.read_csv(FAILURES_PATH, sep='\t')) if os.path.exists(FAILURES_PATH) else 0
print(f'\nDone. {n_success} succeeded, {n_failed} failed.')
print(f'Results: {OUTPUT_PATH}')
if n_failed > 0:
    print(f'Failures: {FAILURES_PATH}')
