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
for cur_mrn in candidate_LLM_text_df['DFCI_MRN'].unique():
    mrn_df = candidate_LLM_text_df.loc[candidate_LLM_text_df['DFCI_MRN'] == cur_mrn]

    mrn_responses = []
    for idx, row in tqdm(list(mrn_df.iterrows())):
        try:
            response = client.chat.completions.create(model=model_name, 
                                                    messages=[{'role': 'system', 'content': prompt.prompt_note_extraction}, 
                                                                {'role': 'user', 'content': "Note Date: " + row['EVENT_DATE'] + clinical_wrapper + " Note Text: " + row['CLINICAL_TEXT']}], 
                                                    temperature=0).choices[0].message.content.strip()
            mrn_responses.append(json.loads(response))
        except:
            continue
    synthesized_responses.append(json.loads(client.chat.completions.create(model=model_name, messages=[{'role' : 'system', 'content' : prompt.prompt_patient_synthesis}, 
                                                                                    {'role' : 'user', 'content' : mrn_responses}]).choices[0].message.content.strip()))
    
synthesized_df = pd.DataFrame([item for sublist in synthesized_responses for item in sublist])
synthesized_df.to_csv(os.path.join(DATA_PATH, 'LLM_generated_labels.csv'), index=False)