import os
import asyncio
import json
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AzureOpenAI

import prompt

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'
MAX_CONCURRENT = 20  # tune to your API rate limit

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

async_client = AsyncAzureOpenAI(
    api_version="2024-04-01-preview",
    azure_endpoint="https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

# gpt-4o-mini for note extraction (fast, cheap, sufficient for structured extraction)
# gpt-4o for synthesis (needs stronger reasoning)
extraction_model = "gpt-4o-mini"
synthesis_model = "gpt-4o"

clinical_wrapper = """
The following is a clinical oncology note being processed
for structured data extraction as part of an IRB-approved research study.
This is medical documentation, not patient-generated content.

---
"""

semaphore = asyncio.Semaphore(MAX_CONCURRENT)

async def extract_note(row):
    async with semaphore:
        try:
            response = await async_client.chat.completions.create(
                model=extraction_model,
                messages=[
                    {'role': 'system', 'content': prompt.prompt_note_extraction},
                    {'role': 'user', 'content': "Note Date: " + row['EVENT_DATE'] + clinical_wrapper + " Note Text: " + row['CLINICAL_TEXT']}
                ],
                temperature=0
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception:
            return None

async def process_patient(cur_mrn, mrn_df, patient_num, total):
    print(f'Starting Patient {patient_num} out of {total}:')
    tasks = [extract_note(row) for _, row in mrn_df.iterrows()]
    results = await tqdm_asyncio.gather(*tasks, desc=f'Patient {patient_num}')
    mrn_responses = [r for r in results if r is not None]

    completed = len(mrn_responses)
    failed = len(results) - completed
    print(f'  Patient {patient_num}: {completed} completed, {failed} failed')

    async with semaphore:
        synthesis = await async_client.chat.completions.create(
            model=synthesis_model,
            messages=[
                {'role': 'system', 'content': prompt.prompt_platinum_classification},
                {'role': 'user', 'content': json.dumps(mrn_responses)}
            ],
            temperature=0
        )
    px_result = json.loads(synthesis.choices[0].message.content.strip())
    px_result['DFCI_MRN'] = int(cur_mrn)
    return px_result

async def main():
    candidate_LLM_text_df = pd.read_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'))
    unique_mrns = candidate_LLM_text_df['DFCI_MRN'].unique()
    total = len(unique_mrns)

    tasks = []
    for i, cur_mrn in enumerate(unique_mrns):
        mrn_df = candidate_LLM_text_df.loc[candidate_LLM_text_df['DFCI_MRN'] == cur_mrn]
        tasks.append(process_patient(cur_mrn, mrn_df, i + 1, total))

    synthesized_responses = await tqdm_asyncio.gather(*tasks, desc='All patients')

    synthesized_df = pd.DataFrame(synthesized_responses)
    synthesized_df.to_csv(os.path.join(DATA_PATH, 'LLM_generated_labels.tsv'), index=False, sep='\t')
    print(f'Done. Wrote {len(synthesized_df)} patients to LLM_generated_labels.tsv')

asyncio.run(main())