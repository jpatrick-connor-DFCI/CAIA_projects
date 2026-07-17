import re
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data_preprocessing_common import fast_io  # noqa: E402

# Paths
IRAE_PATH = Path('/data/gusev/PROFILE/CLINICAL/irAE/PATRICK/cleaned_data/')
EMBED_PROJ_PATH = Path('/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/')
PROFILE_PATH = Path('/data/gusev/PROFILE/CLINICAL/')
ONCDRS_PATH = PROFILE_PATH / 'OncDRS/ALL_2025_03/'

DATA_PATH = Path('/data/gusev/USERS/jpconnor/data/CAIA/IPIO/')

CANCER_TYPE_PREFIX = 'CANCER_TYPE_'


def filter_and_save(filename, outname, cohort_mrns, cols=None) -> pl.DataFrame:
    """Cohort-filter `filename` (whole-file pandas read today = biggest single
    win) via the shared polars scan_filter/recover_numeric helpers, write the
    result to `outname`, and return it.
    """
    lf = fast_io.scan_filter(filename, cohort_mrns, cols=cols)
    filtered = lf.collect()
    # Exclude ID_COL from recover_numeric's generic Utf8->Float64 cast: MRNs
    # are all-digit, so they'd otherwise become Float64, inconsistent with the
    # Int64 DFCI_MRN used elsewhere in this pipeline (cohort_df, cancer_type_df,
    # etc.) and prone to join dtype mismatches. Cast explicitly to Int64 instead.
    filtered = fast_io.recover_numeric(filtered, exclude=(fast_io.ID_COL,))
    filtered = filtered.with_columns(
        pl.col(fast_io.ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    )
    if cols:
        filtered = filtered.select(list(cols))
    filtered.write_csv(outname)
    return filtered


def sanitize_column_name(value: object) -> str:
    """Make a raw column/type name safe for use as a dummy-column suffix."""
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return text.strip("_").upper()


def load_cancer_type_df(filename) -> pl.DataFrame:
    """Load the cancer-type dataframe with NO assumption about which/how many
    cancer types exist: every non-`DFCI_MRN` column is treated as its own
    cancer-type indicator (the file is assumed to already be in wide/one-hot
    form, one column per type -- mirrors how genomic features are discovered
    dynamically from `build_genomic_inputs.py` rather than a fixed gene list).

    Columns are renamed to `CANCER_TYPE_<SANITIZED_NAME>` (if not already
    prefixed that way) so `cox_aggregated.baseline_covariate_columns`'s
    `CANCER_TYPE_` prefix-detection picks them up unchanged downstream.
    """
    cancer_type_df = pl.scan_csv(filename, infer_schema_length=0).collect()

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
    cancer_type_df = cancer_type_df.rename(rename_map)

    renamed_cols = list(rename_map.values())
    print(
        f"Cancer-type columns discovered in {filename}: {len(renamed_cols)} "
        f"({', '.join(renamed_cols)})"
    )
    return cancer_type_df.select(['DFCI_MRN', *renamed_cols])


DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load irAE outcome cohort
cohort_df = pl.scan_csv(IRAE_PATH / 'OS_IO_2021_irAE_df.csv', infer_schema_length=0).collect()
cohort_df = cohort_df.rename({'PATIENT_ID': 'DFCI_MRN'})
cohort_df = cohort_df.with_columns(
    pl.col('IO_START').str.to_datetime(strict=False).alias('IO_START'),
    pl.col('LAST_DATE').str.to_datetime(strict=False).alias('LAST_DATE'),
)

# Merge cancer type (wide/one-hot indicator columns, discovered dynamically)
cancer_type_df = load_cancer_type_df(
    EMBED_PROJ_PATH / 'clinical_and_genomic_features/cancer_type_df.csv.gz'
)
cancer_type_cols = [c for c in cancer_type_df.columns if c != 'DFCI_MRN']
cohort_df = cohort_df.join(cancer_type_df, on='DFCI_MRN', how='left')

all_null_mask = pl.all_horizontal([pl.col(c).is_null() for c in cancer_type_cols])
n_missing_cancer_type = cohort_df.select(all_null_mask.sum()).item()
print(
    f"WARNING: {n_missing_cancer_type}/{len(cohort_df)} IPIO patients have no matching "
    "cancer-type row after the left-merge (all CANCER_TYPE_* columns are NaN for them). "
    "These patients are retained here; longitudinal_data_processing.py fills unmatched "
    "indicator columns with 0 rather than dropping the patient."
)

# Drop unused column
cohort_df = cohort_df.drop('combination')

# Filter raw OncDRS labs to this cohort
cohort_mrns_series = cohort_df['DFCI_MRN'].cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
cohort_mrns = set(cohort_mrns_series.drop_nulls().unique().to_list())
# Read/write only the labs columns consumed downstream: the longitudinal
# reshape (IPIO longitudinal_data_processing.build_raw_longitudinal_labs)
# selects DFCI_MRN, SPECIMEN_COLLECT_DT, TEST_TYPE_CD, TEST_TYPE_DESCR,
# NUMERIC_RESULT, RESULT_UOM_NM. IPIO has no PSA/somatic path off this frame,
# so the other 7 previously-carried columns (D_SPECIMEN_COLLECT_DT, RESULT_NBR,
# RESULT_TYPE_CD, RESULT_TYPE_DESCR, TEXT_RESULT, SPECIMEN_SRC_CD,
# SPECIMEN_SRC_DESCR) are unused and dropped from scan + emitted CSV.
labs = filter_and_save(
    ONCDRS_PATH / 'OUTPT_LAB_RESULTS_LABS.csv',
    DATA_PATH / 'irae_labs_data.csv',
    cohort_mrns,
    cols=['DFCI_MRN', 'SPECIMEN_COLLECT_DT', 'TEST_TYPE_CD', 'TEST_TYPE_DESCR',
          'NUMERIC_RESULT', 'RESULT_UOM_NM']
)

# Write merged cohort+cancer-type table
cohort_df.write_csv(DATA_PATH / 'irae_cohort_data.csv')

print(f"Wrote {len(cohort_df)} cohort rows to {DATA_PATH / 'irae_cohort_data.csv'}")
print(f"Wrote {len(labs)} lab rows (cohort={len(cohort_mrns)} DFCI_MRNs) to {DATA_PATH / 'irae_labs_data.csv'}")
