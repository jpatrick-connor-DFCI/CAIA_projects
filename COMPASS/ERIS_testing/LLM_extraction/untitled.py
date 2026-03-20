import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import ijson
import pandas as pd
from datetime import datetime
import json

import prompt

df_mrn = pd.read_csv("/Users/zeyunlu/Documents/profile_notes/test/pts_list_irae.tsv", sep="\t")
df_rpt = pd.read_csv("~/Documents/profile_notes/test/selected_notes_irae.tsv", sep="\t")[["RPT_ID"]]

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-04-01-preview",
    azure_endpoint="https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

model_name="gpt-4o"

for idx in range(df_mrn.shape[0]):
    # if idx <= 6500:
    #     continue
    
    # if df_mrn.DFCI_MRN[idx] != 928289:
    #     continue
    
    print(f"Processing idx: {idx}, MRN: {df_mrn.DFCI_MRN[idx]}")
    
    tmp_folder=f"/Users/zeyunlu/Documents/profile_notes/test/gpt_tmp_results/{df_mrn.DFCI_MRN[idx]}"
    final_folder=f"/Users/zeyunlu/Documents/profile_notes/test/gpt_results/{df_mrn.DFCI_MRN[idx]}"

    os.makedirs(tmp_folder, exist_ok=True)
    os.makedirs(final_folder, exist_ok=True)
    
    tmp_json=f"/Users/zeyunlu/Documents/profile_notes/final/pts/by_pts/pt_{df_mrn.DFCI_MRN[idx]}.json"
    jdx = 0 
    with open(tmp_json, 'r') as f:
    # Parse the 'docs' array in the JSON file
        for item in ijson.items(f, 'item.item'):
            print(f"Processing idx: {idx}, jdx: {jdx}, MRN: {df_mrn.DFCI_MRN[idx]}")
            jdx += 1
            if not int(item["RPT_ID"]) in df_rpt.RPT_ID.values:
                continue

            event_date = datetime.strptime(item['EVENT_DATE'], '%Y-%m-%d').date().strftime("%Y-%m-%d")
            start_date = datetime.strptime(df_mrn.IO_START[idx], '%Y-%m-%d').date().strftime("%Y-%m-%d")
            rpt_ID = item["RPT_ID"]
            
            tmp_file_path1=f"{tmp_folder}/{rpt_ID}_tmp_res_v1.txt"
            tmp_file_path2=f"{tmp_folder}/{rpt_ID}_tmp_res_v2.txt"
            
            if not os.path.exists(tmp_file_path1):
                try:
                    response1 = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {'role': 'system', 'content': prompt.system_content1},
                            
                            {'role': 'user', 'content': "Note created date: " + event_date + " Clinical note: " + item["RPT_TEXT"]}
                            ],
                            temperature=0
                        )
                    
                    response2 = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {'role': 'system', 'content': prompt.system_content1},
                            
                            {'role': 'user', 'content': "Note created date: " + event_date + " Clinical note: " + item["RPT_TEXT"]}
                            ],
                            temperature=0
                        )
                except Exception as e:
                    print(f"Error processing rpt_ID {rpt_ID}: {e}")
                    continue
                
                first_response_1 = response1.choices[0].message.content.strip()
                first_response_2 = response2.choices[0].message.content.strip()
                
                with open(tmp_file_path1, "a", encoding="utf-8") as f:
                    f.write(first_response_1)
                with open(tmp_file_path2, "a", encoding="utf-8") as f:
                    f.write(first_response_2)
            else:
                with open(tmp_file_path1, "r", encoding="utf-8") as f:
                    first_response_1 = f.read()
                with open(tmp_file_path2, "r", encoding="utf-8") as f:
                    first_response_2 = f.read()
                    
            final_file_path=f"{final_folder}/{rpt_ID}_final_res_v2.tsv"

            if not os.path.exists(final_file_path):
                try:
                    user_content = (
                        f"Immunotherapy start date: {start_date}\n"
                        f"Note created date: {event_date}\n"
                        "-----BEGIN NOTES-----\n"
                        f"{first_response_1}\n\n"
                        "-----END NOTES-----"
                    )
                
                    response2 = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {'role': 'system', 'content': prompt.system_content2_one},
                            #  get results from first response
                            {"role": "user", "content": user_content},
                            ],
                        temperature=0,
                        functions = [{
                            "name": "return_answer",
                            "parameters": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "records": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "Adverse Event": {
                                                    "type": "string",
                                                    "description": "Specific immune-related adverse event (irAE) explicitly attributed to immunotherapy. One irAE per JSON object. Exclude events attributable to non-IO therapies (e.g., chemotherapy, targeted therapy, radiation)."
                                                },
                                                "Severity": {
                                                    "type": "string",
                                                    "enum": ["Mild","Moderate","Severe","Life-threatening","Death","Not sure","Not applicable"],
                                                    "description": "Mild (Grade 1) / Moderate (Grade 2) / Severe (Grade 3) / Life-threatening (Grade 4) / Death (Grade 5). If ambiguous, 'Not sure'. If no irAE, 'Not applicable'."
                                                },
                                                "irAE Date": {
                                                    "type": "string",
                                                    "description": "Onset date for the irAE (YYYY-MM-DD). May be inferred given the note created date and therapy start date; if missing, use the note created date; if ambiguous, 'Not sure'."
                                                },
                                                "IV Steroids": {
                                                    "type": "string",
                                                    "enum": ["Yes","No","Not sure","Not applicable"],
                                                    "description": "Whether IV corticosteroids were given for the irAE."
                                                },
                                                "Treatment": {
                                                    "type": "string",
                                                    "enum": ["Immunotherapy","Chemotherapy","Targeted Therapy","Radiation Therapy","Other","Not sure"],
                                                    "description": "Extract the patient's CURRENT therapy. Exclude previous therapies. If multiple treatments are mentioned, separate them with commas. If none are mentioned, use 'Not sure'."
                                                },
                                                "IO Medications": {
                                                    "type": "string",
                                                    "description": "IO medications used for current immunotherapy (IO) only. Do not include medications for other therapies. Use commas to separate  multiple IO medications."
                                                }
                                            },
                                            "required": ["Adverse Event","Severity","irAE Date","IV Steroids", "Treatment", "IO Medications"]
                                        }
                                    }
                                },
                                "required": ["records"]
                            }
                        }],
                        function_call={"name": "return_answer"}
                        )  
                except Exception as e:
                    print(f"Error processing rpt_ID {rpt_ID} for final response: {e}")
                    continue
                            
                second_response = json.loads(response2.choices[0].message.function_call.arguments)
                final_df = pd.DataFrame(second_response['records'])
                final_df["DFCI_MRN"] = item['DFCI_MRN']
                final_df["RPT_ID"] = item["RPT_ID"]
                final_df["IO_START"] = df_mrn.IO_START[idx]
                final_df['EVENT_DATE'] = event_date
                final_df["MODEL"] = "ChatGPT-4o"
                final_df = final_df[["DFCI_MRN", "RPT_ID", "IO_START", "EVENT_DATE", "MODEL", "Treatment", "Adverse Event", "Severity", "irAE Date", "IV Steroids", "IO Medications"]]

                final_df.to_csv(final_file_path, sep='\t', index=False)
