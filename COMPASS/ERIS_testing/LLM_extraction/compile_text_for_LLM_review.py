import os
import numpy as np
import pandas as pd

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'
baca_df = pd.read_csv(
    os.path.join(DATA_PATH, 'baca_lab_patient_annotations.tsv'),
    sep="\t",
    encoding="utf-8",
    encoding_errors="replace")
text_df = pd.read_csv(os.path.join(DATA_PATH, 'prostate_text_data.csv'))

just_NEPC_baca_df = baca_df.loc[~baca_df['NEPC_DIAG_DATE'].isna()]
just_NEPC_baca_df['TIME_BETWEEN_MED_AND_NEPC_DIAG'] = np.abs(pd.to_datetime(just_NEPC_baca_df['MEDICATION_START_TIME']) - pd.to_datetime(just_NEPC_baca_df['NEPC_DIAG_DATE']))
just_NEPC_baca_df['TIME_BETWEEN_MED_AND_NEPC_DIAG'] = just_NEPC_baca_df['TIME_BETWEEN_MED_AND_NEPC_DIAG'].apply(lambda x : x.days)

candidate_patients = baca_df.loc[baca_df['Inidcation of Platinum Therapy'].isna(), 
                                 ['DFCI_MRN', 'PLATINUM_CHEMO_MED', 'MEDICATION_START_TIME']]
LLM_text_df = text_df.merge(candidate_patients, on='DFCI_MRN', how='inner')
event_dt = pd.to_datetime(LLM_text_df['EVENT_DATE'], errors='coerce', utc=True)
med_dt = pd.to_datetime(LLM_text_df['MEDICATION_START_TIME'], errors='coerce', utc=True)

candidate_LLM_text_df = LLM_text_df[['EVENT_DATE', 'DFCI_MRN', 'NOTE_TYPE', 'CLINICAL_TEXT']].sort_values(by=['DFCI_MRN', 'EVENT_DATE'])
candidate_LLM_text_df.to_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'), index=False)