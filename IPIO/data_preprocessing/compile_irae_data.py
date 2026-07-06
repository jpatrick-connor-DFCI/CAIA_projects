import re
from pathlib import Path

import pandas as pd

# Paths
IRAE_PATH = Path('/data/gusev/PROFILE/CLINICAL/irAE/PATRICK/cleaned_data/')
EMBED_PROJ_PATH = Path('/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/')
PROFILE_PATH = Path('/data/gusev/PROFILE/CLINICAL/')
ONCDRS_PATH = PROFILE_PATH / 'OncDRS/ALL_2025_03/'

DATA_PATH = Path('/data/gusev/USERS/jpconnor/data/CAIA/IPIO/')

CANCER_TYPE_PREFIX = 'CANCER_TYPE_'


def filter_and_save(filename, outname, cohort_mrns, cols=None):
    df = pd.read_csv(filename)
    filtered = df.loc[df['DFCI_MRN'].isin(cohort_mrns)]
    if cols:
        filtered = filtered[cols]
    filtered.to_csv(outname, index=False)
    return filtered


def sanitize_column_name(value: object) -> str:
    """Make a raw column/type name safe for use as a pandas dummy-column suffix."""
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return text.strip("_").upper()


def load_cancer_type_df(filename):
    """Load the cancer-type dataframe with NO assumption about which/how many
    cancer types exist: every non-`DFCI_MRN` column is treated as its own
    cancer-type indicator (the file is assumed to already be in wide/one-hot
    form, one column per type -- mirrors how genomic features are discovered
    dynamically from `build_genomic_inputs.py` rather than a fixed gene list).

    Columns are renamed to `CANCER_TYPE_<SANITIZED_NAME>` (if not already
    prefixed that way) so `cox_aggregated.baseline_covariate_columns`'s
    `CANCER_TYPE_` prefix-detection picks them up unchanged downstream.
    """
    cancer_type_df = pd.read_csv(filename)

    if 'DFCI_MRN' not in cancer_type_df.columns:
        raise ValueError(
            f"{filename} is missing a 'DFCI_MRN' column; found columns: "
            f"{list(cancer_type_df.columns)}"
        )

    type_cols = [c for c in cancer_type_df.columns if c != 'DFCI_MRN']
    if not type_cols:
        raise ValueError(
            f"{filename} has no non-DFCI_MRN columns to use as cancer-type "
            "indicators; cannot proceed."
        )

    rename_map = {}
    for col in type_cols:
        if col.upper().startswith(CANCER_TYPE_PREFIX):
            rename_map[col] = col.upper()
        else:
            rename_map[col] = f"{CANCER_TYPE_PREFIX}{sanitize_column_name(col)}"
    cancer_type_df = cancer_type_df.rename(columns=rename_map)

    renamed_cols = list(rename_map.values())
    print(
        f"Cancer-type columns discovered in {filename}: {len(renamed_cols)} "
        f"({', '.join(renamed_cols)})"
    )
    return cancer_type_df[['DFCI_MRN', *renamed_cols]]


DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load irAE outcome cohort
cohort_df = pd.read_csv(IRAE_PATH / 'OS_IO_2021_irAE_df.csv')
cohort_df = cohort_df.rename(columns={'PATIENT_ID': 'DFCI_MRN'})
cohort_df['IO_START'] = pd.to_datetime(cohort_df['IO_START'], errors='coerce')
cohort_df['LAST_DATE'] = pd.to_datetime(cohort_df['LAST_DATE'], errors='coerce')

# Merge cancer type (wide/one-hot indicator columns, discovered dynamically)
cancer_type_df = load_cancer_type_df(
    EMBED_PROJ_PATH / 'clinical_and_genomic_features/cancer_type_df.csv.gz'
)
cancer_type_cols = [c for c in cancer_type_df.columns if c != 'DFCI_MRN']
cohort_df = cohort_df.merge(cancer_type_df, on='DFCI_MRN', how='left')

n_missing_cancer_type = cohort_df[cancer_type_cols].isna().all(axis=1).sum()
print(
    f"WARNING: {n_missing_cancer_type}/{len(cohort_df)} IPIO patients have no matching "
    "cancer-type row after the left-merge (all CANCER_TYPE_* columns are NaN for them). "
    "These patients are retained here; longitudinal_data_processing.py fills unmatched "
    "indicator columns with 0 rather than dropping the patient."
)

# Drop unused column
cohort_df = cohort_df.drop(columns=['combination'])

# Filter raw OncDRS labs to this cohort
cohort_mrns = cohort_df['DFCI_MRN'].unique()
labs = filter_and_save(
    ONCDRS_PATH / 'OUTPT_LAB_RESULTS_LABS.csv',
    DATA_PATH / 'irae_labs_data.csv',
    cohort_mrns,
    cols=['DFCI_MRN', 'SPECIMEN_COLLECT_DT', 'D_SPECIMEN_COLLECT_DT', 'TEST_TYPE_CD', 'TEST_TYPE_DESCR', 'RESULT_NBR',
          'RESULT_TYPE_CD', 'RESULT_TYPE_DESCR', 'NUMERIC_RESULT', 'TEXT_RESULT', 'RESULT_UOM_NM', 'SPECIMEN_SRC_CD', 'SPECIMEN_SRC_DESCR']
)

# Write merged cohort+cancer-type table
cohort_df.to_csv(DATA_PATH / 'irae_cohort_data.csv', index=False)

print(f"Wrote {len(cohort_df)} cohort rows to {DATA_PATH / 'irae_cohort_data.csv'}")
print(f"Wrote {len(labs)} lab rows (cohort={len(cohort_mrns)} DFCI_MRNs) to {DATA_PATH / 'irae_labs_data.csv'}")
