import os
import numpy as np
import pandas as pd

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'
baca_df = pd.read_csv(os.path.join(DATA_PATH, 'LLM_NEPC_labels/baca_lab_annotations.csv') )
text_df = pd.read_csv(os.path.join(DATA_PATH, 'prostate_text_data.csv'))

text_med_df = text_df.merge(baca_df, on='DFCI_MRN', how='inner')

event_dt = pd.to_datetime(text_med_df['EVENT_DATE'], errors='coerce', utc=True)
med_dt = pd.to_datetime(text_med_df['MEDICATION_START_TIME'], errors='coerce', utc=True)

text_med_df['NOTE_DAYS_REL_PLATINUM'] = (event_dt - med_dt).dt.days

train_mrns = text_med_df.loc[text_med_df['Inidcation of Platinum Therapy'].isna(), 'DFCI_MRN'].unique()
test_mrns = text_med_df.loc[~text_med_df['Inidcation of Platinum Therapy'].isna(), 'DFCI_MRN'].unique()

LLM_text_df = (text_med_df
               .loc[np.abs(text_med_df['NOTE_DAYS_REL_PLATINUM']) <= 90,
                    ['EVENT_DATE', 'DFCI_MRN', 'NOTE_TYPE', 'CLINICAL_TEXT']]
               .sort_values(by=['DFCI_MRN', 'EVENT_DATE']))
train_LLM_df = LLM_text_df.loc[LLM_text_df['DFCI_MRN'].isin(train_mrns)]
test_LLM_df = LLM_text_df.loc[LLM_text_df['DFCI_MRN'].isin(test_mrns)]

train_LLM_df.to_csv(os.path.join(DATA_PATH, 'LLM_NEPC_labels/extraction_train_data.csv'), index=False)
test_LLM_df.to_csv(os.path.join(DATA_PATH, 'LLM_NEPC_labels/extraction_test_data.csv'), index=False)