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
ONCDRS_PATH = os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/')

SURV_PATH = os.path.join(EMBED_PROJ_PATH, 'time-to-event_analysis/')

def mark_non_prostate_primary_icd(icds):
    icds = icds.copy()
    codes = icds['DIAGNOSIS_ICD10_CD'].astype(str).str.upper().str.strip()

    letter = codes.str.extract(r'^([A-Z])', expand=False)
    number = pd.to_numeric(codes.str.extract(r'^[A-Z](\d{2,3})', expand=False), errors='coerce')

    is_c00_c76 = (letter == 'C') & (number >= 0) & (number <= 76)
    is_c81_c96 = (letter == 'C') & (number >= 81) & (number <= 96)
    is_c97 = codes.str.startswith('C97')
    is_c7a = codes.str.startswith('C7A')
    is_c801 = codes.str.startswith('C801') | codes.str.startswith('C80.1')

    is_primary = is_c00_c76 | is_c81_c96 | is_c97 | is_c7a | is_c801
    is_prostate = codes.str.startswith('C61')
    is_secondary = ((letter == 'C') & (number >= 77) & (number <= 79)) | codes.str.startswith('C7B')
    is_nmsc = codes.str.startswith('C44')
    is_nos = codes.str.startswith('C80.9') | codes.str.startswith('C809')

    icds['NON_PROSTATE_PRIMARY_ICD10'] = (
        is_primary
        & ~is_prostate
        & ~is_secondary
        & ~is_nmsc
        & ~is_nos
    )
    return icds


def compute_prostate_mrns_from_icd(icds):
    icds = mark_non_prostate_primary_icd(icds)
    non_prostate_primary_mrns = set(
        icds.loc[icds['NON_PROSTATE_PRIMARY_ICD10'], 'DFCI_MRN'].unique()
    )
    codes = icds['DIAGNOSIS_ICD10_CD'].astype(str).str.upper().str.strip()
    mrns = icds.loc[
        codes.str.startswith('C61') & ~icds['DFCI_MRN'].isin(non_prostate_primary_mrns),
        'DFCI_MRN',
    ]
    return pd.to_numeric(mrns, errors='coerce').dropna().astype(int).unique()


def filter_and_save(filename, outname, cohort_mrns, cols=None):
    cohort_mrns = set(pd.to_numeric(pd.Series(cohort_mrns), errors='coerce').dropna().astype(int))
    df = pd.read_csv(filename)
    ids = pd.to_numeric(df['DFCI_MRN'], errors='coerce')
    filtered = df.loc[ids.isin(cohort_mrns)].copy()
    if cols:
        filtered = filtered[cols]
    filtered.to_csv(outname, index=False)
    return filtered

# Load metadata
full_meta = pd.read_csv(os.path.join(PROC_PATH, 'full_VTE_embeddings_metadata.csv'))

icd_path = os.path.join(EMBED_PROJ_PATH, 'time-to-event_analysis/timestamped_icd_info.csv.gz')
icd_source = pd.read_csv(icd_path)
prostate_mrns = compute_prostate_mrns_from_icd(icd_source)
prostate_mrn_set = set(prostate_mrns)
print(f"ICD-defined prostate cohort: {len(prostate_mrns)} MRNs")

# Collect prostate text notes when available. Related structured tables are
# filtered against the full ICD-defined prostate set below, not this note subset.
prostate_meta = full_meta.loc[
    pd.to_numeric(full_meta['DFCI_MRN'], errors='coerce').isin(prostate_mrn_set)
]
batch_ids = [re.split('_', sid)[1] for sid in prostate_meta['SUB_BATCH_FILE_ID'].unique()]

all_notes = []
for batch_id in tqdm(batch_ids, desc='Loading prostate text batches'):
    file_path = os.path.join(TEXT_PATH, f'VTE_notes_with_full_metadata_batch_{batch_id}.json')
    batch = pd.read_json(file_path)
    prostate_batch = batch.loc[
        pd.to_numeric(batch['DFCI_MRN'], errors='coerce').isin(prostate_mrn_set)
    ]
    all_notes.append(prostate_batch)

complete_df = pd.concat(all_notes, ignore_index=True) if all_notes else prostate_meta.iloc[0:0].copy()
complete_df.to_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_text_data.csv'), index=False)

# Filter related datasets by prostate patients
icds = filter_and_save(icd_path, os.path.join(NEPC_PROJ_PATH, 'prostate_icd_data.csv'), prostate_mrns)
health = filter_and_save(os.path.join(ONCDRS_PATH, 'HEALTH_HISTORY.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_health_history_data.csv'), prostate_mrns)
meds = filter_and_save(os.path.join(ONCDRS_PATH, 'MEDICATIONS.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_medications_data.csv'), prostate_mrns)
labs = filter_and_save(os.path.join(ONCDRS_PATH, 'OUTPT_LAB_RESULTS_LABS.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_labs_data.csv'), prostate_mrns,
                       cols=['DFCI_MRN', 'SPECIMEN_COLLECT_DT', 'D_SPECIMEN_COLLECT_DT', 'TEST_TYPE_CD', 'TEST_TYPE_DESCR', 'RESULT_NBR', 
                             'RESULT_TYPE_CD', 'RESULT_TYPE_DESCR', 'NUMERIC_RESULT', 'TEXT_RESULT', 'RESULT_UOM_NM', 'SPECIMEN_SRC_CD', 'SPECIMEN_SRC_DESCR'])
somatic = filter_and_save(os.path.join(EMBED_PROJ_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'), 
                          os.path.join(NEPC_PROJ_PATH, 'prostate_somatic_data.csv'), prostate_mrns)

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
