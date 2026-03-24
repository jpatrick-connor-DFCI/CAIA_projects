import os
import re
import pandas as pd
from tqdm import tqdm

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/'
EMBED_PROJ_PATH = os.path.join(DATA_PATH, 'clinical_text_embedding_project/')
NEPC_PROJ_PATH = os.path.join(DATA_PATH, 'CAIA/COMPASS/')

PROC_PATH = os.path.join(EMBED_PROJ_PATH, 'batched_datasets/processed_datasets/')
TEXT_PATH = os.path.join(EMBED_PROJ_PATH, 'batched_datasets/batched_text/')

PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
INTAE_DATA_PATH = os.path.join(PROFILE_PATH, 'robust_VTE_pred_project_2025_03_cohort/data/')
ONCDRS_PATH = os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/')

# Load metadata
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
              .rename(columns={'medication_start_time' : 'PLATINUM_START_TIME', 
                               'medication' : 'PLATINUM_NM', 
                               'count' : 'NUMBER_OF_TOTAL_CLINICAL_NOTES'})
             .sort_values(by='NUM_NOTES_WITH_MED_NAME', ascending=False))

unfiltered_df = df_to_send[['DFCI_MRN', 'PLATINUM_NM', 'PLATINUM_START_TIME', 'NUM_NOTES_WITH_MED_NAME', 
                           'NUM_CLINICIAN_NOTES', 'NUM_IMAGING_NOTES', 'NUM_PATHOLOGY_NOTES', 'TOTAL_NUM_NOTES']]

## ADDING INFO FOR NON-PROSTATE PRIMARY MALIGNANCIES
prostate_icds = pd.read_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_icd_data.csv'))
codes = prostate_icds['DIAGNOSIS_ICD10_CD'].astype(str).str.upper().str.strip()
letter = codes.str.extract(r'^([A-Z])', expand=False)
number = pd.to_numeric(codes.str.extract(r'^[A-Z](\d{1,3})', expand=False), errors='coerce')
prostate_icds['PRIMARY_NON_PROSTATE'] = ((letter == 'C') & (number >= 0) & (number <= 75) & (codes != 'C61'))
df_clean = (
    prostate_icds.loc[prostate_icds['PRIMARY_NON_PROSTATE'],
                      ['DFCI_MRN', 'DIAGNOSIS_ICD10_CD', 'START_DT']]
    .sort_values(['DFCI_MRN', 'DIAGNOSIS_ICD10_CD', 'START_DT'])
    .drop_duplicates(subset=['DFCI_MRN', 'DIAGNOSIS_ICD10_CD'], keep='first')
)
df_out = (
    prostate_icds.loc[
        prostate_icds['PRIMARY_NON_PROSTATE'],
        ['DFCI_MRN', 'DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM', 'START_DT']
    ]
    .sort_values(['DFCI_MRN', 'DIAGNOSIS_ICD10_CD', 'START_DT'])
    .drop_duplicates(['DFCI_MRN', 'DIAGNOSIS_ICD10_CD'], keep='first')
    .sort_values(['DFCI_MRN', 'START_DT'])
    .groupby('DFCI_MRN', as_index=False)
    .agg({
        'DIAGNOSIS_ICD10_CD': list,
        'DIAGNOSIS_ICD10_NM': list,
        'START_DT': list
    })
)

## ADDING INFO FOR PARP INHIBITORS
parp_df = (meds.loc[meds['NCI_PREFERRED_MED_NM'].isin(['OLAPARIB', 'RUCAPARIB', 'NIRAPARIB', 'TALAZOPARIB'])]
           .sort_values(by='MED_START_DT').drop_duplicates(subset='DFCI_MRN')
           .rename(columns={'NCI_PREFERRED_MED_NM' : 'PARPi_NM',
                            'MED_START_DT' : 'PARPi_START_TIME'}))[['DFCI_MRN', 'PARPi_NM', 'PARPi_START_TIME']]

prostate_somatic_df = pd.read_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_somatic_data.csv'))
brca_cols = [col for col in prostate_somatic_df.columns if 'BRCA2' in col]
brca_df = prostate_somatic_df[['DFCI_MRN'] + brca_cols]
brca_df['ANY_BRCA2_MUTATION'] = brca_df[brca_cols].sum(axis=1) > 0
brca_cols.append('ANY_BRCA2_MUTATION')

filtered_df = (unfiltered_df
               .merge(df_out, on='DFCI_MRN', how='left')
               .merge(parp_df, on='DFCI_MRN', how='left')
               .merge(brca_df, on='DFCI_MRN', how='left'))

filtered_df = (filtered_df.rename(columns={'DIAGNOSIS_ICD10_CD' : 'NON_PROSTATE_PRIMARY_ICD10_CD',
                                           'DIAGNOSIS_ICD10_NM' : 'NON_PROSTATE_PRIMARY_ICD10_NM', 
                                           'START_DT' : 'NON_PROSTATE_PRIMARY_ICD10_START_TIME'})
               [['DFCI_MRN', 'PLATINUM_NM', 'PLATINUM_START_TIME', 'PARPi_NM', 'PARPi_START_TIME',
                 'NON_PROSTATE_PRIMARY_ICD10_CD', 'NON_PROSTATE_PRIMARY_ICD10_NM', 
                 'NON_PROSTATE_PRIMARY_ICD10_START_TIME'] + brca_cols])

filtered_df.to_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_pxs_for_review_v2.csv'), index=False)

filtered_df['HAS_NON_PROSTATE'] = (~filtered_df['NON_PROSTATE_PRIMARY_ICD10_CD'].isna())
filtered_df['HAS_PARPi'] = (~filtered_df['PARPi_NM'].isna())
filtered_df['HAS_OncoPanel'] = (~filtered_df['ANY_BRCA2_MUTATION'].isna())

print(f'Num px with non_prostate = {filtered_df['HAS_NON_PROSTATE'].sum()}')
print(f'Num px with PARPi = {filtered_df['HAS_PARPi'].sum()}')
print(f'Num px with non_prostate AND PARPi = {(filtered_df['HAS_PARPi'] & filtered_df['HAS_NON_PROSTATE']).sum()}')
print(f'Num px with prostate AND PARPi = {(filtered_df['HAS_PARPi'] & ~filtered_df['HAS_NON_PROSTATE']).sum()} \n')

print(f'Num px with BRCA2 = {(filtered_df['ANY_BRCA2_MUTATION'].sum())}')
print(f'Num px with OncoPanel and PARPi = {(filtered_df['HAS_OncoPanel'] & filtered_df['HAS_PARPi']).sum()}')
print(f'Num px with OncoPanel + BRCA2 AND PARPi = {(filtered_df['HAS_OncoPanel'] & filtered_df['ANY_BRCA2_MUTATION'] & filtered_df['HAS_PARPi']).sum()}')