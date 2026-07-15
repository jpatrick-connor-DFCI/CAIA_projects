"""
Script Name: compile_COMPASS_cohort_data.py

Description:
Single entry point that builds the full COMPASS prostate-cancer cohort data
directly from the raw OncDRS 2025-03 pull. Merges what used to be two
separate scripts:

  * compile_prostate_data.py    -- cohort-filtered text/ICD/health/meds/labs/
                                    somatic tables, PSA records, platinum
                                    chemo records.
  * prostate_arpi_survival_preprocessing.py -- per-patient ARPI/chemo-anchored
                                    survival cohort (age, treatment anchor,
                                    death, platinum time-to-event).

Cohort definition (unified):
All patients are defined by ICD-10 code C61 (prostate primary), minus
patients whose ICD history shows a competing non-prostate primary
malignancy (see `mark_non_prostate_primary_icd`). This single ICD-C61
cohort now drives every output below -- the text/ICD/health/meds/labs/
somatic/PSA/platinum tables AND the survival cohort file all share it
(previously the former used an inferred-cancer cohort from
first_treatments_dfci_w_inferred_cancers.csv; that source is no longer
read).

Raw date handling:
OncDRS de-identifies dates by shifting them; the raw files expose integer
"days since reference" columns (D_MED_START_DT, D_START_DT, D_BIRTH_DT, ...).
A single reference_date reconstructs calendar dates as
    real_date = reference_date + Timedelta(days=D_*).
Reference-date sanity checks are asserted against several files.

Inputs (OncDRS raw pull + auxiliary project files):
  * EHR_DIAGNOSIS.csv                        (ICD-10 -> cohort + exclusion)
  * MEDICATIONS.csv                          (anchor, PARPi + platinum drugs)
  * PT_INFO_STATUS_REGISTRATION.csv          (birth date, sex, death/last-alive)
  * HEALTH_HISTORY.csv
  * OUTPT_LAB_RESULTS_LABS.csv
  * complete_somatic_data_df.csv.gz

Outputs (in NEPC_PROJ_PATH):
  * prostate_icd_data.csv
  * prostate_health_history_data.csv
  * prostate_medications_data.csv
  * prostate_labs_data.csv
  * prostate_somatic_data.csv
  * total_psa_records.csv
  * platinum_chemo_records.csv
  * prostate_arpi_survival_cohort.csv

Author: J. Patrick Connor
Date: 2026-07-14
"""

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

ID_COL = "DFCI_MRN"

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/'
EMBED_PROJ_PATH = os.path.join(DATA_PATH, 'clinical_text_embedding_project/')
NEPC_PROJ_PATH = os.path.join(DATA_PATH, 'CAIA/COMPASS/')

PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
ONCDRS_PATH = os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/')

SURV_PATH = os.path.join(EMBED_PROJ_PATH, 'time-to-event_analysis/')

# ARPI / defined-chemo anchor drugs (matches TREATMENT_ANCHOR_MEDS in
# longitudinal_data_processing.py: ARPIs/androgen-axis, taxanes, radium-223).
TREATMENT_ANCHOR_MEDS = {
    "ABIRATERONE ACETATE",
    "ENZALUTAMIDE",
    "APALUTAMIDE",
    "DAROLUTAMIDE",
    "DOCETAXEL",
    "CABAZITAXEL",
    "RADIUM RA 223 DICHLORIDE",
}

PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}

# Reference date for the OncDRS de-identified day offsets. [IMPORTANT: update
# for each new data pull.]
REFERENCE_DATE = pd.to_datetime("2021-02-22") - pd.Timedelta(days=44158)


def assert_reference_date(anchor_real_date, offset_days, label, tol_days=2):
    """Reconstruct the reference date implied by (anchor_real_date, offset_days)
    from a given file and assert it matches REFERENCE_DATE within tol_days."""
    implied = pd.to_datetime(anchor_real_date) - pd.Timedelta(days=offset_days)
    delta = abs((REFERENCE_DATE - implied).days)
    print(f"Reference date check [{label}]: implied {implied.date()} (delta {delta}d)")
    assert delta <= tol_days, f"Reference date mismatch for {label}!"


# ---------------------------------------------------------------------------
# ICD cohort definition (shared by every output below)
# ---------------------------------------------------------------------------

def mark_non_prostate_primary_icd(icds):
    """Flag ICD rows that indicate a non-prostate PRIMARY malignancy."""
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


def compute_prostate_cohort(icds):
    """Return (prostate_mrns, excluded_mrns) from an exploded ICD dataframe.

    prostate_mrns : patients with any C61 code, minus those with a
                    non-prostate primary ICD.
    """
    codes = icds['DIAGNOSIS_ICD10_CD'].astype(str).str.upper().str.strip()
    c61_mrns = set(pd.to_numeric(icds.loc[codes.str.startswith('C61'), ID_COL], errors='coerce').dropna().astype(int))

    marked = mark_non_prostate_primary_icd(icds)
    non_prostate_primary_mrns = set(
        pd.to_numeric(marked.loc[marked['NON_PROSTATE_PRIMARY_ICD10'], ID_COL], errors='coerce').dropna().astype(int)
    )

    excluded = c61_mrns & non_prostate_primary_mrns
    prostate_mrns = c61_mrns - excluded
    print(
        f"ICD-C61 prostate cohort: {len(c61_mrns)} patients; "
        f"excluded {len(excluded)} with a non-prostate-primary ICD; "
        f"retained {len(prostate_mrns)}."
    )
    return prostate_mrns, excluded


def load_and_explode_icd(icd_path):
    """Load the ICD source (timestamped_icd_info.csv.gz) and explode any
    comma-separated ICD list column into one code per row, matching the
    raw-OncDRS EHR_DIAGNOSIS convention used elsewhere in this pipeline."""
    icds = pd.read_csv(icd_path)
    if 'DIAGNOSIS_ICD10_LIST' in icds.columns and 'DIAGNOSIS_ICD10_CD' not in icds.columns:
        icds['DIAGNOSIS_ICD10_CD'] = icds['DIAGNOSIS_ICD10_LIST'].astype(str).str.split(',')
        icds = icds.explode('DIAGNOSIS_ICD10_CD')
        icds['DIAGNOSIS_ICD10_CD'] = icds['DIAGNOSIS_ICD10_CD'].astype(str).str.strip().str.upper()
        icds = icds.loc[icds['DIAGNOSIS_ICD10_CD'] != '']
    return icds


# ---------------------------------------------------------------------------
# compile_prostate_data.py-style cohort-filtered table dumps
# ---------------------------------------------------------------------------

def filter_and_save(filename, outname, cohort_mrns, cols=None, chunksize=250_000):
    """Stream `filename` in batches via pyarrow, keep only cohort_mrns rows
    (and `cols` columns, if given), and write the concatenated result to
    `outname`.

    pyarrow's CSV reader is used instead of pandas because it parses each
    batch in C++ (no chunk-by-chunk Python/regex dtype sniffing) and
    transparently decompresses `.gz` inputs. Unlike pandas' `low_memory=False`
    (which settles on a single permissive dtype after seeing the whole file),
    pyarrow infers each column's type from only the first block and then
    raises on later rows that don't fit (e.g. OncDRS free-text fields like
    RESULT_NBR that are numeric for most rows but occasionally hold values
    such as '05055/D'). To sidestep that, every column is parsed as a plain
    string; after filtering, columns that are fully numeric-parseable across
    the *filtered* (small) result are cast back to numeric, mirroring what
    pandas' whole-file dtype inference would have produced. `chunksize` sets
    the pyarrow read block size (rows per batch) to bound peak memory.
    """
    cohort_mrns = set(pd.to_numeric(pd.Series(list(cohort_mrns)), errors='coerce').dropna().astype(int))
    usecols = None
    if cols:
        usecols = list(cols) if ID_COL in cols else [ID_COL] + list(cols)

    read_options = pa_csv.ReadOptions(block_size=chunksize * 200)
    header = pa_csv.open_csv(filename, read_options=read_options).schema.names
    col_names = usecols if usecols else header
    column_types = {c: pa.string() for c in col_names}
    convert_options = pa_csv.ConvertOptions(include_columns=usecols, column_types=column_types) \
        if usecols else pa_csv.ConvertOptions(column_types=column_types)

    reader = pa_csv.open_csv(filename, read_options=read_options, convert_options=convert_options)
    filtered_batches = []
    schema = reader.schema
    for batch in reader:
        table = pa.Table.from_batches([batch])
        ids = pd.to_numeric(table.column(ID_COL).to_pandas(), errors='coerce')
        mask = pa.array(ids.isin(cohort_mrns).to_numpy())
        filtered_batches.append(table.filter(mask))

    filtered_table = pa.concat_tables(filtered_batches) if filtered_batches else pa.table({c: [] for c in schema.names})
    filtered = filtered_table.to_pandas()
    # Forcing every pyarrow column to string() turns empty/NA CSV cells into
    # "" rather than NaN -- restore real NaNs before any dtype recovery below.
    filtered = filtered.mask(filtered == '', np.nan)
    for col in filtered.columns:
        as_num = pd.to_numeric(filtered[col], errors='coerce')
        non_null = filtered[col].notna()
        if non_null.any() and as_num[non_null].notna().all():
            filtered[col] = as_num
    if cols:
        filtered = filtered[cols]
    filtered.to_csv(outname, index=False)
    return filtered


def compile_cohort_tables(prostate_mrns, icds, icd_path):
    """Build the cohort-filtered ICD/health/meds/labs/somatic/PSA/
    platinum tables. Returns (meds_df, platinum_df) for reuse by the
    survival cohort builder below."""
    prostate_mrn_set = set(prostate_mrns)

    # Filter related datasets by the prostate cohort.
    icds_filtered = icds.loc[
        pd.to_numeric(icds[ID_COL], errors='coerce').isin(prostate_mrn_set)
    ].copy()
    icds_filtered.to_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_icd_data.csv'), index=False)

    health = filter_and_save(os.path.join(ONCDRS_PATH, 'HEALTH_HISTORY.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_health_history_data.csv'), prostate_mrn_set)
    meds = filter_and_save(os.path.join(ONCDRS_PATH, 'MEDICATIONS.csv'), os.path.join(NEPC_PROJ_PATH, 'prostate_medications_data.csv'), prostate_mrn_set)
    labs = filter_and_save(
        os.path.join(ONCDRS_PATH, 'OUTPT_LAB_RESULTS_LABS.csv'),
        os.path.join(NEPC_PROJ_PATH, 'prostate_labs_data.csv'),
        prostate_mrn_set,
        cols=['DFCI_MRN', 'SPECIMEN_COLLECT_DT', 'D_SPECIMEN_COLLECT_DT', 'TEST_TYPE_CD', 'TEST_TYPE_DESCR', 'RESULT_NBR',
              'RESULT_TYPE_CD', 'RESULT_TYPE_DESCR', 'NUMERIC_RESULT', 'TEXT_RESULT', 'RESULT_UOM_NM', 'SPECIMEN_SRC_CD', 'SPECIMEN_SRC_DESCR'],
    )
    filter_and_save(
        os.path.join(EMBED_PROJ_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv.gz'),
        os.path.join(NEPC_PROJ_PATH, 'prostate_somatic_data.csv'),
        prostate_mrn_set,
    )

    # PSA lab filtering.
    total_psa_labels = ['PSA', 'PSAR', 'PSATOTSCRN', 'CPSA', 'PSAMON', 'PSAULT', 'PSAT']
    total_psa = labs.loc[(labs['TEST_TYPE_CD'].isin(total_psa_labels)) &
                         (labs['NUMERIC_RESULT'].notna()) &
                         (labs['NUMERIC_RESULT'] != 9999999.0)]
    total_psa.to_csv(os.path.join(NEPC_PROJ_PATH, 'total_psa_records.csv'), index=False)

    # Platinum meds filtering. Normalize NCI_PREFERRED_MED_NM (upper/strip)
    # before matching -- an un-normalized `==` comparison here previously
    # risked silently missing case/whitespace-variant platinum records that
    # the ARPI-anchored survival cohort (below) would still catch, since that
    # code path normalizes. Both now agree.
    med_name = meds['NCI_PREFERRED_MED_NM'].astype(str).str.upper().str.strip()
    platinum_df = (meds.loc[med_name.isin(PLATINUM_MEDS)]
                   .assign(NCI_PREFERRED_MED_NM=med_name.loc[med_name.isin(PLATINUM_MEDS)])
                   .sort_values(by='MED_START_DT').drop_duplicates(subset=ID_COL)
                   .rename(columns={'NCI_PREFERRED_MED_NM': 'medication',
                                    'MED_START_DT': 'medication_start_time'}))
    platinum_df.to_csv(os.path.join(NEPC_PROJ_PATH, 'platinum_chemo_records.csv'), index=False)

    return meds, platinum_df


# ---------------------------------------------------------------------------
# ARPI/chemo-anchored survival cohort (prostate_arpi_survival_preprocessing.py)
# ---------------------------------------------------------------------------

def load_medications_for_survival(meds):
    """Reconstruct MED_START_DT from D_MED_START_DT for the anchor + platinum
    drug rows already filtered to the prostate cohort by compile_cohort_tables.
    """
    keep_meds = {m.upper() for m in TREATMENT_ANCHOR_MEDS | PLATINUM_MEDS}
    out = meds.copy()
    out['NCI_PREFERRED_MED_NM'] = out['NCI_PREFERRED_MED_NM'].astype(str).str.upper().str.strip()
    out = out.loc[out['NCI_PREFERRED_MED_NM'].isin(keep_meds)]
    out = out.dropna(subset=['D_MED_START_DT'])
    out['MED_START_DT'] = REFERENCE_DATE + pd.to_timedelta(out['D_MED_START_DT'], unit='D')

    row0 = out.iloc[0]
    assert_reference_date(row0['MED_START_DT'], row0['D_MED_START_DT'], 'MEDICATIONS')
    return out


def compute_treatment_anchor(meds):
    """Earliest anchor-drug MED_START_DT per patient -> TREATMENT_ANCHOR_DATE."""
    anchor = meds.loc[meds['NCI_PREFERRED_MED_NM'].isin(TREATMENT_ANCHOR_MEDS)].copy()
    anchor = anchor.dropna(subset=['MED_START_DT'])
    return (
        anchor.groupby(ID_COL, as_index=False)['MED_START_DT']
        .min()
        .rename(columns={'MED_START_DT': 'TREATMENT_ANCHOR_DATE'})
    )


def compute_first_platinum(meds):
    """Earliest platinum MED_START_DT (and drug name) per patient."""
    plat = meds.loc[meds['NCI_PREFERRED_MED_NM'].isin(PLATINUM_MEDS)].copy()
    plat = plat.dropna(subset=['MED_START_DT'])
    plat = plat.sort_values('MED_START_DT')
    plat = plat.drop_duplicates(subset=ID_COL, keep='first')
    return plat[[ID_COL, 'NCI_PREFERRED_MED_NM', 'MED_START_DT']].rename(
        columns={
            'NCI_PREFERRED_MED_NM': 'PLATINUM_MED',
            'MED_START_DT': 'PLATINUM_DATE',
        }
    )


def load_patient_status(path):
    """Load birth date, sex, and death / last-alive info from
    PT_INFO_STATUS_REGISTRATION.csv, reconstructing calendar dates.

    Returns a dataframe with:
        DFCI_MRN, BIRTH_DATE, GENDER, DEATH_DATE, LAST_CONTACT_DATE
    """
    pt = pd.read_csv(os.path.join(path, 'PT_INFO_STATUS_REGISTRATION.csv'))

    pt['BIRTH_DATE'] = REFERENCE_DATE + pd.to_timedelta(pt['D_BIRTH_DT'], unit='D')

    row0 = pt.dropna(subset=['D_BIRTH_DT']).iloc[0]
    assert_reference_date(row0['BIRTH_DATE'], row0['D_BIRTH_DT'], 'PT_INFO_BIRTH')

    # Death and last-alive dates are provided as calendar strings.
    pt['DEATH_DATE'] = pd.to_datetime(pt['HYBRID_DEATH_DT'], errors='coerce')
    pt['LAST_CONTACT_DATE'] = pd.to_datetime(pt['DERIVED_LAST_ALIVE_DATE'], errors='coerce')

    return pt[
        [ID_COL, 'BIRTH_DATE', 'GENDER_NM', 'DEATH_DATE', 'LAST_CONTACT_DATE']
    ].rename(columns={'GENDER_NM': 'GENDER'})


def build_survival_cohort(prostate_mrns, anchor_df, platinum_df, status_df):
    """Assemble the per-patient ARPI/chemo-anchored survival table."""
    cohort = pd.DataFrame({ID_COL: sorted(prostate_mrns)})

    cohort = cohort.merge(status_df, on=ID_COL, how='left')
    cohort = cohort.merge(anchor_df, on=ID_COL, how='left')
    cohort = cohort.merge(platinum_df, on=ID_COL, how='left')

    cohort['AGE'] = (
        (cohort['TREATMENT_ANCHOR_DATE'] - cohort['BIRTH_DATE'])
        / pd.Timedelta(days=365.2425)
    ).astype(float)

    cohort['FOLLOW_UP_END_DATE'] = cohort['DEATH_DATE'].fillna(cohort['LAST_CONTACT_DATE'])

    cohort['DEATH'] = cohort['DEATH_DATE'].notna().astype(int)
    cohort['TT_DEATH'] = (
        (cohort['FOLLOW_UP_END_DATE'] - cohort['TREATMENT_ANCHOR_DATE']).dt.days
    )

    has_platinum = cohort['PLATINUM_DATE'].notna()
    cohort['PLATINUM'] = has_platinum.astype(int)
    platinum_end = cohort['PLATINUM_DATE'].where(has_platinum, cohort['FOLLOW_UP_END_DATE'])
    cohort['TT_PLATINUM'] = (
        (platinum_end - cohort['TREATMENT_ANCHOR_DATE']).dt.days
    )

    no_anchor = cohort['TREATMENT_ANCHOR_DATE'].isna()
    cohort.loc[no_anchor, ['AGE', 'TT_DEATH', 'TT_PLATINUM']] = np.nan

    return cohort[
        [
            ID_COL,
            'GENDER',
            'AGE',
            'BIRTH_DATE',
            'TREATMENT_ANCHOR_DATE',
            'FOLLOW_UP_END_DATE',
            'DEATH_DATE',
            'LAST_CONTACT_DATE',
            'TT_DEATH',
            'DEATH',
            'PLATINUM_MED',
            'PLATINUM_DATE',
            'TT_PLATINUM',
            'PLATINUM',
        ]
    ]


def summarize_survival_cohort(cohort):
    n = len(cohort)
    n_anchor = cohort['TREATMENT_ANCHOR_DATE'].notna().sum()
    print("\n=== Survival cohort summary ===")
    print(f"Total ICD-C61 patients (post-exclusion): {n}")
    print(f"With an ARPI/chemo anchor drug: {n_anchor}")
    print(f"Deaths: {int(cohort['DEATH'].sum())}")
    print(f"Received platinum: {int(cohort['PLATINUM'].sum())}")
    with_times = cohort.dropna(subset=['TT_DEATH'])
    if len(with_times):
        print(
            f"Median TT_DEATH (days): {with_times['TT_DEATH'].median():.0f}; "
            f"median TT_PLATINUM (days): {with_times['TT_PLATINUM'].median():.0f}"
        )
    neg = cohort.loc[cohort['TT_DEATH'] < 0]
    if len(neg):
        print(
            f"WARNING: {len(neg)} patients have negative TT_DEATH "
            f"(follow-up end before anchor); inspect before analysis."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compile the full COMPASS prostate cohort data (ICD/health/"
        "meds/labs/somatic/PSA/platinum tables + ARPI/chemo-anchored survival cohort) "
        "from the raw OncDRS pull, all sharing one ICD-C61 cohort definition.",
    )
    parser.add_argument(
        "--icd-source",
        type=str,
        default=os.path.join(SURV_PATH, 'timestamped_icd_info.csv.gz'),
        help="Pre-compiled ICD source (timestamped_icd_info.csv.gz) used to define the cohort.",
    )
    parser.add_argument(
        "--oncdrs-path",
        type=str,
        default=ONCDRS_PATH,
        help="OncDRS raw data pull directory (for PT_INFO_STATUS_REGISTRATION.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=NEPC_PROJ_PATH,
        help="Directory to write all cohort CSVs.",
    )
    args = parser.parse_args()

    print(f"Reference date: {REFERENCE_DATE.date()}")

    # 1. Shared ICD-C61 cohort (drives every output below).
    icds = load_and_explode_icd(args.icd_source)
    prostate_mrns, _ = compute_prostate_cohort(icds)

    # 2. compile_prostate_data.py-style cohort-filtered table dumps.
    meds, platinum_df_compat = compile_cohort_tables(prostate_mrns, icds, args.icd_source)

    # 3. ARPI/chemo-anchored survival cohort, reusing the meds already
    #    filtered to the prostate cohort above.
    meds_for_survival = load_medications_for_survival(meds)
    anchor_df = compute_treatment_anchor(meds_for_survival)
    platinum_df = compute_first_platinum(meds_for_survival)
    print(
        f"Anchor drug recipients: {len(anchor_df)}; "
        f"platinum recipients: {len(platinum_df)} (prostate cohort)."
    )

    status_df = load_patient_status(args.oncdrs_path)

    survival_cohort = build_survival_cohort(prostate_mrns, anchor_df, platinum_df, status_df)
    summarize_survival_cohort(survival_cohort)

    out_path = os.path.join(args.out_dir, 'prostate_arpi_survival_cohort.csv')
    survival_cohort.to_csv(out_path, index=False)
    print(f"\nSaved survival cohort to {out_path}")


if __name__ == "__main__":
    main()
