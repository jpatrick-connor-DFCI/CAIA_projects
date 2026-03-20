import os
from tqdm import tqdm
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import pandas as pd
from datetime import datetime
import json

import prompt

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-04-01-preview",
    azure_endpoint="https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

model_name="gpt-4o"

clinical_wrapper = """
The following is a clinical oncology note being processed 
for structured data extraction as part of an IRB-approved research study. 
This is medical documentation, not patient-generated content.

---
"""
synthesized_responses = []
candidate_LLM_text_df = pd.read_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'))
unique_mrns = candidate_LLM_text_df['DFCI_MRN'].unique()
for i, cur_mrn in enumerate(unique_mrns):
    mrn_df = candidate_LLM_text_df.loc[candidate_LLM_text_df['DFCI_MRN'] == cur_mrn]

    completed_runs = 0; failed_runs = 0;
    mrn_responses = []
    print(f'Starting Patient {i + 1} out of {len(unique_mrns)}:')
    for idx, row in tqdm(list(mrn_df.iterrows())):
        try:
            response = client.chat.completions.create(model=model_name, 
                                                    messages=[{'role': 'system', 'content': prompt.prompt_note_extraction}, 
                                                                {'role': 'user', 'content': "Note Date: " + row['EVENT_DATE'] + clinical_wrapper + " Note Text: " + row['CLINICAL_TEXT']}], 
                                                    temperature=0).choices[0].message.content.strip()
            mrn_responses.append(json.loads(response))
            completed_runs += 1
        except:
            failed_runs += 1
            continue
    px_result = json.loads(client.chat.completions.create(model=model_name, messages=[{'role' : 'system', 'content' : prompt.prompt_platinum_classification},
                                                                                      {'role' : 'user', 'content' : json.dumps(mrn_responses)}],
                                                          temperature=0).choices[0].message.content.strip())
    px_result['DFCI_MRN'] = int(cur_mrn)
    synthesized_responses.append(px_result)
    
synthesized_df = pd.DataFrame(synthesized_responses)
synthesized_df.to_csv(os.path.join(DATA_PATH, 'LLM_generated_labels.tsv'), index=False, sep='\t')