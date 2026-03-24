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

SURV_PATH = os.path.join(EMBED_PROJ_PATH, 'time-to-event_analysis/')

def filter_and_save(filename, outname, cols=None):
    df = pd.read_csv(filename)
    filtered = df.loc[df['DFCI_MRN'].isin(complete_df['DFCI_MRN'].unique())]
    if cols:
        filtered = filtered[cols]
    filtered.to_csv(outname, index=False)
    return filtered

# Load metadata
full_meta = pd.read_csv(os.path.join(PROC_PATH, 'full_VTE_embeddings_metadata.csv'))

cancer_types = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'first_treatments_dfci_w_inferred_cancers.csv'))[['DFCI_MRN', 'med_genomics_merged_cancer_group']]
prostate_mrns = cancer_types.loc[cancer_types['med_genomics_merged_cancer_group'] == 'PROSTATE', 'DFCI_MRN'].unique()

# Collect prostate text notes
prostate_meta = full_meta.loc[full_meta['DFCI_MRN'].isin(prostate_mrns)]
batch_ids = [re.split('_', sid)[1] for sid in prostate_meta['SUB_BATCH_FILE_ID'].unique()]

all_notes = []
for batch_id in tqdm(batch_ids, desc='Loading prostate text batches'):
    file_path = os.path.join(TEXT_PATH, f'VTE_notes_with_full_metadata_batch_{batch_id}.json')
    batch = pd.read_json(file_path)
    prostate_batch = batch.loc[batch['DFCI_MRN'].isin(prostate_mrns)]
    all_notes.append(prostate_batch)

complete_df = pd.concat(all_notes, ignore_index=True)
complete_df.to_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_text_data.csv'), index=False)

# Filter related datasets by prostate patients
icds = filter_and_save(os.path.join(EMBED_PROJ_PATH, 'time-to-event_analysis/timestamped_icd_info.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_icd_data.csv'))
health = filter_and_save(os.path.join(ONCDRS_PATH, 'HEALTH_HISTORY.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_health_history_data.csv'))
meds = filter_and_save(os.path.join(ONCDRS_PATH, 'MEDICATIONS.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_medications_data.csv'))
labs = filter_and_save(os.path.join(ONCDRS_PATH, 'OUTPT_LAB_RESULTS_LABS.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_labs_data.csv'), 
                       cols=['DFCI_MRN', 'SPECIMEN_COLLECT_DT', 'D_SPECIMEN_COLLECT_DT', 'TEST_TYPE_CD', 'TEST_TYPE_DESCR', 'RESULT_NBR', 
                             'RESULT_TYPE_CD', 'RESULT_TYPE_DESCR', 'NUMERIC_RESULT', 'TEXT_RESULT', 'SPECIMEN_SRC_CD', 'SPECIMEN_SRC_DESCR'])
somatic = filter_and_save(os.path.join(EMBED_PROJ_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'), 
                          os.path.join(NEPC_PROJ_PATH, 'prostate_somatic_data.csv'))

# PSA lab filtering
total_psa_labels = ['PSA', 'PSAR', 'PSATOTSCRN', 'CPSA', 'PSAMON', 'PSAULT', 'PSAT']
total_psa = labs.loc[(labs['TEST_TYPE_CD'].isin(total_psa_labels)) &
                     (labs['NUMERIC_RESULT'].notna()) &
                     (labs['NUMERIC_RESULT'] != 9999999.0)]
total_psa.to_csv(os.path.join(NEPC_PROJ_PATH, 'total_psa_records.csv'), index=False)

# Platinum meds filtering
platinum_df = (meds.loc[(meds['NCI_PREFERRED_MED_NM'] == 'CISPLATIN') |
                        (meds['NCI_PREFERRED_MED_NM'] == 'CARBOPLATIN')]
               .sort_values(by='MED_START_DT').drop_duplicates(subset='DFCI_MRN')
               .rename(columns={'NCI_PREFERRED_MED_NM' : 'medication', 
                                'MED_START_DT' : 'medication_start_time'}))
platinum_df.to_csv(os.path.join(NEPC_PROJ_PATH, 'platinum_chemo_records.csv'), index=False)