import os
import numpy as np
import pandas as pd 
from pathlib import Path

DATA_PATH = Path('/data/gusev/USERS/jpconnor/data/')
CLINICAL_FEATURE_PATH = DATA_PATH / 'clinical_text_embedding_project/clinical_and_genomic_features/'
COMPASS_DATA_PATH = DATA_PATH / 'CAIA/COMPASS/'
GLEASON_PATH = COMPASS_DATA_PATH / 'LLM_gleason_timeline/'
LONG_NEPC_PATH = COMPASS_DATA_PATH / 'LLM_avpc_nepc_timeline/'

stage_df = pd.read_csv(CLINICAL_FEATURE_PATH / 'cancer_stage_df.csv.gz')
treatment_df = pd.read_csv(CLINICAL_FEATURE_PATH / 'categorical_treatment_data_by_line.csv.gz')
meds_df = pd.read_csv(COMPASS_DATA_PATH / 'prostate_medications_data.csv')

ICD_df = pd.read_csv(COMPASS_DATA_PATH / 'prostate_icd_data.csv')
prostate_mrns = pd.read_csv(COMPASS_DATA_PATH / 'mrn_lists/longitudinal_mrns.csv')['DFCI_MRN']
treatment_mapping = pd.read_csv(COMPASS_DATA_PATH / 'prostate_medication_categories.csv')

gleason_timeline = pd.read_csv(GLEASON_PATH / 'gleason_timeline.tsv', sep='\t')
prostate_stages = stage_df.loc[stage_df['DFCI_MRN'].isin(prostate_mrns)]
prostate_treatments = treatment_df.loc[treatment_df['DFCI_MRN'].isin(prostate_mrns)]

first_line_treatments = (prostate_treatments.loc[prostate_treatments['treatment_line'] == 1, 
                            [col for col in prostate_treatments.columns if col.startswith('PX_')]]
                        .sum(axis=0).sort_values(ascending=False))

meds_columns = ['DFCI_MRN', 'NCI_PREFERRED_MED_NM', 'MED_START_DT']
meds_df = meds_df.loc[meds_df['DFCI_MRN'].isin(prostate_mrns), meds_columns]
meds_df['MED_START_DT'] = meds_df['MED_START_DT'].apply(lambda x : x.split()[0])

antineo_meds_df = (meds_df.merge(
                    (prostate_treatments[['DFCI_MRN', 'treatment_start_date', 'treatment_line']]
                     .rename(columns={'treatment_start_date' : 'MED_START_DT'})), 
                     on=['DFCI_MRN', 'MED_START_DT'])
                     .rename(columns={'NCI_PREFERRED_MED_NM' : 'drug_name'})
                     .sort_values(by=['DFCI_MRN', 'treatment_line'])
                     .merge(treatment_mapping, on=['drug_name']))

first_line_antineo_meds_df = (antineo_meds_df.loc[
                              (antineo_meds_df['treatment_line'] == 1) &
                              (antineo_meds_df['pca_relevance']
                                  .isin(['directed', 'other_malignancy']))]
                             .drop(columns=['notes', 'pca_relevance', 'category', 'treatment_line']))

meds_per_px = (first_line_antineo_meds_df[['DFCI_MRN', 'drug_name']].drop_duplicates()['DFCI_MRN']
                .value_counts()
                .reset_index()
                .rename(columns={'count' : 'num_first_line_meds'}))

pxs_on_AR = first_line_antineo_meds_df.loc[
                first_line_antineo_meds_df['androgen_axis_target'] == 'androgen_axis', 
                'DFCI_MRN'].unique().tolist()
pxs_not_on_AR = list(set(first_line_antineo_meds_df['DFCI_MRN']).difference(set(pxs_on_AR)))
px_AR_status = pd.DataFrame({'DFCI_MRN' : pxs_on_AR + pxs_not_on_AR,
                             'PX_on_AR' : [True for _ in pxs_on_AR] + [False for _ in pxs_not_on_AR]})

first_line_antineo_meds_df = (first_line_antineo_meds_df
                              .merge(meds_per_px, on=['DFCI_MRN'])
                              .merge(px_AR_status, on=['DFCI_MRN']))

non_AR_px_first_line_df = first_line_antineo_meds_df.loc[~first_line_antineo_meds_df['PX_on_AR']]
AR_px_first_line_df = first_line_antineo_meds_df.loc[first_line_antineo_meds_df['PX_on_AR']]

combination_first_line_df = first_line_antineo_meds_df.loc[first_line_antineo_meds_df['num_first_line_meds'] > 1, 
    ['DFCI_MRN', 'drug_name', 'subclass', 'androgen_axis_target']]

from collections import Counter
import pandas as pd

combo_counts = (
    combination_first_line_df.groupby("DFCI_MRN")["drug_name"]
    .apply(lambda x: tuple(sorted(set(x))))
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"drug_name": "combination"})
)

first_line_antineo_meds_df['drug_name'].unique()

drugs_to_filter_for = ['ABIRATERONE ACETATE', 'ENZALUTAMIDE', 'APALUTAMIDE', 'DAROLUTAMIDE',
                       'DOCETAXEL', 'CABAZITAXEL', 'RADIUM RA 223 DICHLORIDE']
first_line_antineo_meds_df.loc[first_line_antineo_meds_df['drug_name'].isin(drugs_to_filter_for)].to_csv(COMPASS_DATA_PATH / 'mrn_lists/ARPI_mrn_subset.csv', index=False)