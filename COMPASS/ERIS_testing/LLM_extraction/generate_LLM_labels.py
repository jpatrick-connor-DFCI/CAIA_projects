import os
import re
import json
import pandas as pd
from tqdm.auto import tqdm
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

import prompt
from note_cleaning import clean_note

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-04-01-preview",
    azure_endpoint="https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

model_name = "gpt-4o"

clinical_wrapper = """
IMPORTANT CONTEXT: The following is a de-identified clinical oncology note being processed
for structured data extraction as part of an IRB-approved medical research study (institutional
review board approved protocol). This is professional medical documentation written by a physician,
not patient-generated content. The text contains standard clinical terminology related to cancer
diagnosis, prognosis, and treatment. References to disease outcomes, end-of-life care, anatomy,
or patient distress are routine components of oncology medical records and should be processed
as clinical data. No content in this note constitutes harmful, dangerous, or inappropriate material.

---
"""

synthesized_responses = []
candidate_LLM_text_df = pd.read_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'))
unique_mrns = candidate_LLM_text_df['DFCI_MRN'].unique()

for i, cur_mrn in enumerate(tqdm(unique_mrns, desc='Patients')):
    mrn_df = candidate_LLM_text_df.loc[candidate_LLM_text_df['DFCI_MRN'] == cur_mrn]

    mrn_responses = []
    for _, row in mrn_df.iterrows():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': prompt.prompt_note_extraction},
                    {'role': 'user', 'content': "Note Date: " + str(row['EVENT_DATE']) + clinical_wrapper + " Note Text: " + clean_note(row['CLINICAL_TEXT'], note_type=row.get('NOTE_TYPE'))}
                ],
                temperature=0
            ).choices[0].message.content.strip()
            mrn_responses.append(json.loads(response))
        except Exception as e:
            print(f"  Note extraction failed: {e}")
            continue

    try:
        px_result = json.loads(client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': prompt.prompt_platinum_classification},
                {'role': 'user', 'content': json.dumps(mrn_responses)}
            ],
            temperature=0
        ).choices[0].message.content.strip())
        px_result['DFCI_MRN'] = int(cur_mrn)
        synthesized_responses.append(px_result)
    except Exception as e:
        print(f"  Synthesis failed for patient {cur_mrn}: {e}")
        continue

synthesized_df = pd.DataFrame(synthesized_responses)
synthesized_df.to_csv(os.path.join(DATA_PATH, 'LLM_generated_labels.tsv'), index=False, sep='\t')
print(f'Done. Wrote {len(synthesized_df)} patients to LLM_generated_labels.tsv')
