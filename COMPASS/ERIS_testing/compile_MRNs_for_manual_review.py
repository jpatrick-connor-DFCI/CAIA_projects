import os
import re
import pandas as pd
from tqdm import tqdm

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/'
EMBED_PROJ_PATH = os.path.join(DATA_PATH, 'clinical_text_embedding_project/')
NEPC_PROJ_PATH = os.path.join(DATA_PATH, 'COMPASS_CAIA/')

PROC_PATH = os.path.join(EMBED_PROJ_PATH, 'batched_datasets/processed_datasets/')
TEXT_PATH = os.path.join(EMBED_PROJ_PATH, 'batched_datasets/batched_text/')

PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
INTAE_DATA_PATH = os.path.join(PROFILE_PATH, 'robust_VTE_pred_project_2025_03_cohort/data/')
ONCDRS_PATH = os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/')

# # Load metadata
cancer_types = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'first_treatments_dfci_w_inferred_cancers.csv'))[['DFCI_MRN', 'med_genomics_merged_cancer_group']]
prostate_mrns = cancer_types.loc[cancer_types['med_genomics_merged_cancer_group'] == 'PROSTATE', 'DFCI_MRN'].unique()

prostate_text_df = pd.read_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_text_data.csv'))
meds = pd.read_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_medications_data.csv'))
total_psa = pd.read_csv(os.path.join(NEPC_PROJ_PATH, 'total_psa_records.csv'))

platinum_meds = pd.read_csv(os.path.join(NEPC_PROJ_PATH, 'platinum_chemo_records.csv'))

mrns_with_note_counts = platinum_meds.merge(prostate_text_df['DFCI_MRN'].value_counts().reset_index().rename({'count' : 'number_of_clinical_notes'}), on='DFCI_MRN', how='inner')

mrns_with_note_count = (prostate_text_df
                        .pivot_table(index="DFCI_MRN", 
                                     columns="NOTE_TYPE", 
                                     aggfunc="size", 
                                     fill_value=0)
                        .reset_index()
                        .merge(platinum_meds, on='DFCI_MRN', how='inner')
                        .rename(columns={'Clinician' : 'NUM_CLINICIAN_NOTES', 
                                         'Imaging' : 'NUM_IMAGING_NOTES',
                                         'Pathology' : 'NUM_PATHOLOGY_NOTES'}))
mrns_with_note_count['TOTAL_NUM_NOTES'] = mrns_with_note_count[['NUM_CLINICIAN_NOTES', 'NUM_IMAGING_NOTES', 'NUM_PATHOLOGY_NOTES']].sum(axis=1)

platinum_mentions = []
for mrn in tqdm(mrns_with_note_count['DFCI_MRN'].unique()):
    num_notes_with_med_mention = (prostate_text_df.loc[prostate_text_df['DFCI_MRN'] == mrn]['CLINICAL_TEXT']
                                  .apply(lambda x : ('carboplatin' in str(x).lower()) or ('cisplatin' in str(x).lower())).sum())
    platinum_mentions.append([mrn, num_notes_with_med_mention > 0, num_notes_with_med_mention])
    
med_mention_df = pd.DataFrame(platinum_mentions, columns = ['DFCI_MRN', 'MED_NAME_IN_NOTES', 'NUM_NOTES_WITH_MED_NAME'])
med_mention_df = med_mention_df.loc[med_mention_df['MED_NAME_IN_NOTES']]

df_to_send = (med_mention_df
              .merge(mrns_with_note_count, on='DFCI_MRN')
              .drop(columns=['MED_NAME_IN_NOTES'])
              .rename(columns={'medication_start_time' : 'MEDICATION_START_TIME', 
                               'medication' : 'PLATINUM_CHEMO_MED', 
                               'count' : 'NUMBER_OF_TOTAL_CLINICAL_NOTES'})
             .sort_values(by='NUM_NOTES_WITH_MED_NAME', ascending=False))

df_to_send[['DFCI_MRN', 'PLATINUM_CHEMO_MED', 'MEDICATION_START_TIME', 'NUM_NOTES_WITH_MED_NAME',
            'NUM_CLINICIAN_NOTES', 'NUM_IMAGING_NOTES', 'NUM_PATHOLOGY_NOTES', 'TOTAL_NUM_NOTES']].to_csv(os.path.join(NEPC_PATH, 'prostate_patients_for_manual_review.csv'), index=False)