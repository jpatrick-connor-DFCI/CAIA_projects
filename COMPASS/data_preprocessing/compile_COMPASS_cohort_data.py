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

Implementation note:
This script is fully polars (zero pandas) up through writing the cohort
CSVs and the survival cohort. `data_preprocessing_common/fast_io.py`
supplies the shared cohort-filtering (`scan_filter`) and dirty-numeric
recovery (`recover_numeric`) helpers.
"""

import argparse
import datetime as dt
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_preprocessing_common import fast_io  # noqa: E402

ID_COL = "DFCI_MRN"

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/'
EMBED_PROJ_PATH = os.path.join(DATA_PATH, 'clinical_text_embedding_project/')
NEPC_PROJ_PATH = os.path.join(DATA_PATH, 'CAIA/COMPASS/')

PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
ONCDRS_PATH = os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/')

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

# Cisplatin appears both as a single agent and coded within a combination
# regimen name; both count as platinum exposure. Oxaliplatin is intentionally
# excluded (not a relevant platinum agent for this cohort).
PLATINUM_MEDS = {
    "CARBOPLATIN",
    "CISPLATIN",
    "CISPLATIN/CYCLOPHOSPHAMIDE/ETOPOSIDE",
}

# Reference date for the OncDRS de-identified day offsets. [IMPORTANT: update
# for each new data pull.] Kept as a datetime (not date) so it composes as a
# polars Datetime literal, matching the Datetime dtype produced by
# str.to_datetime() elsewhere (DEATH_DATE, LAST_CONTACT_DATE) -- mixing Date
# and Datetime dtypes in later subtractions would otherwise be an easy trap.
REFERENCE_DATE = dt.datetime(2021, 2, 22) - dt.timedelta(days=44158)


def assert_reference_date(anchor_real_date, offset_days, label, tol_days=2):
    """Reconstruct the reference date implied by (anchor_real_date, offset_days)
    from a given file and assert it matches REFERENCE_DATE within tol_days."""
    if isinstance(anchor_real_date, dt.datetime):
        anchor_real_date = anchor_real_date.date()
    implied = anchor_real_date - dt.timedelta(days=int(offset_days))
    delta = abs((REFERENCE_DATE.date() - implied).days)
    print(f"Reference date check [{label}]: implied {implied} (delta {delta}d)")
    assert delta <= tol_days, f"Reference date mismatch for {label}!"


# ---------------------------------------------------------------------------
# ICD cohort definition (shared by every output below)
# ---------------------------------------------------------------------------

def mark_non_prostate_primary_icd(icds: pl.DataFrame) -> pl.DataFrame:
    """Flag ICD rows that indicate a non-prostate PRIMARY malignancy."""
    codes = pl.col("DIAGNOSIS_ICD10_CD").cast(pl.Utf8).str.to_uppercase().str.strip_chars()

    letter = codes.str.extract(r'^([A-Z])', 1)
    number = codes.str.extract(r'^[A-Z](\d{2,3})', 1).cast(pl.Float64, strict=False)

    is_c00_c76 = (letter == 'C') & (number >= 0) & (number <= 76)
    is_c81_c96 = (letter == 'C') & (number >= 81) & (number <= 96)
    is_c97 = codes.str.starts_with('C97')
    is_c7a = codes.str.starts_with('C7A')
    is_c801 = codes.str.starts_with('C801') | codes.str.starts_with('C80.1')

    is_primary = is_c00_c76 | is_c81_c96 | is_c97 | is_c7a | is_c801
    is_prostate = codes.str.starts_with('C61')
    is_secondary = ((letter == 'C') & (number >= 77) & (number <= 79)) | codes.str.starts_with('C7B')
    is_nmsc = codes.str.starts_with('C44')
    is_nos = codes.str.starts_with('C80.9') | codes.str.starts_with('C809')

    non_prostate_primary = (
        is_primary
        & ~is_prostate
        & ~is_secondary
        & ~is_nmsc
        & ~is_nos
    )
    return icds.with_columns(non_prostate_primary.alias("NON_PROSTATE_PRIMARY_ICD10"))


def compute_prostate_cohort(icds: pl.DataFrame):
    """Return (prostate_mrns, excluded_mrns) from an exploded ICD dataframe.

    prostate_mrns : patients with any C61 code, minus those with a
                    non-prostate primary ICD.
    """
    codes = icds["DIAGNOSIS_ICD10_CD"].cast(pl.Utf8).str.to_uppercase().str.strip_chars()
    is_c61 = codes.str.starts_with('C61')
    c61_ids = (
        icds.filter(is_c61)[ID_COL].cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    )
    c61_mrns = set(c61_ids.drop_nulls().to_list())

    marked = mark_non_prostate_primary_icd(icds)
    non_prostate_ids = (
        marked.filter(pl.col("NON_PROSTATE_PRIMARY_ICD10"))[ID_COL]
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
    )
    non_prostate_primary_mrns = set(non_prostate_ids.drop_nulls().to_list())

    excluded = c61_mrns & non_prostate_primary_mrns
    prostate_mrns = c61_mrns - excluded
    print(
        f"ICD-C61 prostate cohort: {len(c61_mrns)} patients; "
        f"excluded {len(excluded)} with a non-prostate-primary ICD; "
        f"retained {len(prostate_mrns)}."
    )
    return prostate_mrns, excluded


def load_and_explode_icd(icd_path) -> pl.DataFrame:
    """Load the raw OncDRS EHR_DIAGNOSIS.csv and normalize it to one ICD-10
    code per row (columns DFCI_MRN, START_DT, DIAGNOSIS_ICD10_CD,
    DIAGNOSIS_ICD10_NM) -- the shape every downstream consumer here expects.

    Raw EHR_DIAGNOSIS packs up to THREE codes per diagnosis row across paired
    columns DIAGNOSIS_ICD10_CD/_NM, _CD2/_NM2, _CD3/_NM3. This mirrors the
    unpacking in the embedding project's extract_ICD_times.py: melt the three
    code/name pairs into one long table so a C61 sitting in _CD2 or _CD3 is not
    silently missed (the old single-column read only saw _CD).

    A pre-derived source that already has one code per row (i.e. only
    DIAGNOSIS_ICD10_CD, no _CD2/_CD3) is passed through unchanged so this
    remains compatible with timestamped_icd_info.csv.gz-style inputs.
    """
    icds = pl.scan_csv(icd_path, infer_schema_length=0).collect()

    pair_cols = [
        ('DIAGNOSIS_ICD10_CD', 'DIAGNOSIS_ICD10_NM'),
        ('DIAGNOSIS_ICD10_CD2', 'DIAGNOSIS_ICD10_NM2'),
        ('DIAGNOSIS_ICD10_CD3', 'DIAGNOSIS_ICD10_NM3'),
    ]
    extra_pairs = [(c, n) for c, n in pair_cols[1:] if c in icds.columns]

    # Legacy / already-flat source: only the primary code column present.
    if not extra_pairs:
        return icds

    # Raw EHR_DIAGNOSIS: melt CD/CD2/CD3 (+ names) into one code per row,
    # carrying every other (non-code) column along on each melted row.
    all_pair_cols = {col for pair in pair_cols for col in pair}
    carry_cols = [c for c in icds.columns if c not in all_pair_cols]

    parts = []
    for cd_col, nm_col in pair_cols:
        if cd_col not in icds.columns:
            continue
        select_exprs = [pl.col(c) for c in carry_cols]
        select_exprs.append(pl.col(cd_col).alias("DIAGNOSIS_ICD10_CD"))
        if nm_col in icds.columns:
            select_exprs.append(pl.col(nm_col).alias("DIAGNOSIS_ICD10_NM"))
        else:
            select_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("DIAGNOSIS_ICD10_NM"))
        parts.append(icds.select(select_exprs))

    exploded = pl.concat(parts, how="vertical_relaxed")
    # Drop rows whose (secondary/tertiary) code slot was empty.
    exploded = exploded.with_columns(
        pl.col("DIAGNOSIS_ICD10_CD").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    )
    exploded = exploded.filter(~pl.col("DIAGNOSIS_ICD10_CD").is_in(['', 'NAN', 'NONE']))
    return exploded


# ---------------------------------------------------------------------------
# compile_prostate_data.py-style cohort-filtered table dumps
# ---------------------------------------------------------------------------

def filter_and_save(filename, outname, cohort_mrns, cols=None) -> pl.DataFrame:
    """Stream `filename` lazily via polars, keep only cohort_mrns rows
    (and `cols` columns, if given), recover dirty-numeric columns, and write
    the result to `outname`. See `data_preprocessing_common/fast_io.py` for
    the shared scan/filter/recover implementation; this wrapper only adds the
    column re-projection (to preserve the exact requested column order) and
    the `.write_csv()` so output paths/signatures stay stable.
    """
    lf = fast_io.scan_filter(filename, cohort_mrns, cols=cols)
    filtered = lf.collect()
    filtered = fast_io.recover_numeric(filtered)
    if cols:
        filtered = filtered.select(list(cols))
    filtered.write_csv(outname)
    return filtered


def compile_cohort_tables(prostate_mrns, icds: pl.DataFrame, icd_path):
    """Build the cohort-filtered ICD/health/meds/labs/somatic/PSA/
    platinum tables. Returns (meds_df, platinum_df) for reuse by the
    survival cohort builder below."""
    prostate_mrn_set = set(int(m) for m in prostate_mrns)

    # Filter related datasets by the prostate cohort.
    mrn_num = icds[ID_COL].cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    icds_filtered = icds.filter(mrn_num.is_in(list(prostate_mrn_set)))
    icds_filtered.write_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_icd_data.csv'))

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
    total_psa = labs.filter(
        pl.col('TEST_TYPE_CD').is_in(total_psa_labels)
        & pl.col('NUMERIC_RESULT').is_not_null()
        & (pl.col('NUMERIC_RESULT') != 9999999.0)
    )
    total_psa.write_csv(os.path.join(NEPC_PROJ_PATH, 'total_psa_records.csv'))

    # Platinum meds filtering. Normalize NCI_PREFERRED_MED_NM (upper/strip)
    # before matching -- an un-normalized `==` comparison here previously
    # risked silently missing case/whitespace-variant platinum records that
    # the ARPI-anchored survival cohort (below) would still catch, since that
    # code path normalizes. Both now agree.
    meds_norm = meds.with_columns(
        pl.col('NCI_PREFERRED_MED_NM').cast(pl.Utf8).str.to_uppercase().str.strip_chars().alias('NCI_PREFERRED_MED_NM')
    )
    platinum_df = (
        meds_norm.filter(pl.col('NCI_PREFERRED_MED_NM').is_in(list(PLATINUM_MEDS)))
        .sort('MED_START_DT')
        .unique(subset=[ID_COL], keep='first')
        .rename({'NCI_PREFERRED_MED_NM': 'medication', 'MED_START_DT': 'medication_start_time'})
    )
    platinum_df.write_csv(os.path.join(NEPC_PROJ_PATH, 'platinum_chemo_records.csv'))

    return meds, platinum_df


# ---------------------------------------------------------------------------
# ARPI/chemo-anchored survival cohort (prostate_arpi_survival_preprocessing.py)
# ---------------------------------------------------------------------------

def load_medications_for_survival(meds: pl.DataFrame) -> pl.DataFrame:
    """Reconstruct MED_START_DT from D_MED_START_DT for the anchor + platinum
    drug rows already filtered to the prostate cohort by compile_cohort_tables.
    """
    keep_meds = {m.upper() for m in TREATMENT_ANCHOR_MEDS | PLATINUM_MEDS}
    out = meds.with_columns(
        pl.col('NCI_PREFERRED_MED_NM').cast(pl.Utf8).str.to_uppercase().str.strip_chars().alias('NCI_PREFERRED_MED_NM')
    )
    out = out.filter(pl.col('NCI_PREFERRED_MED_NM').is_in(list(keep_meds)))
    out = out.filter(pl.col('D_MED_START_DT').is_not_null())
    d_offset = pl.col('D_MED_START_DT').cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    out = out.with_columns(
        (pl.lit(REFERENCE_DATE, dtype=pl.Datetime) + pl.duration(days=d_offset)).alias('MED_START_DT')
    )

    row0 = out.row(0, named=True)
    assert_reference_date(row0['MED_START_DT'], row0['D_MED_START_DT'], 'MEDICATIONS')
    return out


def compute_treatment_anchor(meds: pl.DataFrame) -> pl.DataFrame:
    """Earliest anchor-drug MED_START_DT per patient -> TREATMENT_ANCHOR_DATE."""
    anchor = meds.filter(pl.col('NCI_PREFERRED_MED_NM').is_in(list(TREATMENT_ANCHOR_MEDS)))
    anchor = anchor.filter(pl.col('MED_START_DT').is_not_null())
    return (
        anchor.group_by(ID_COL)
        .agg(pl.col('MED_START_DT').min())
        .rename({'MED_START_DT': 'TREATMENT_ANCHOR_DATE'})
    )


def compute_first_platinum(meds: pl.DataFrame) -> pl.DataFrame:
    """Earliest platinum MED_START_DT (and drug name) per patient."""
    plat = meds.filter(pl.col('NCI_PREFERRED_MED_NM').is_in(list(PLATINUM_MEDS)))
    plat = plat.filter(pl.col('MED_START_DT').is_not_null())
    plat = plat.sort('MED_START_DT').unique(subset=[ID_COL], keep='first')
    return plat.select([ID_COL, 'NCI_PREFERRED_MED_NM', 'MED_START_DT']).rename(
        {
            'NCI_PREFERRED_MED_NM': 'PLATINUM_MED',
            'MED_START_DT': 'PLATINUM_DATE',
        }
    )


def load_patient_status(path) -> pl.DataFrame:
    """Load birth date, sex, and death / last-alive info from
    PT_INFO_STATUS_REGISTRATION.csv, reconstructing calendar dates.

    Returns a dataframe with:
        DFCI_MRN, BIRTH_DATE, GENDER, DEATH_DATE, LAST_CONTACT_DATE
    """
    pt = pl.scan_csv(os.path.join(path, 'PT_INFO_STATUS_REGISTRATION.csv'), infer_schema_length=0).collect()

    d_birth = pl.col('D_BIRTH_DT').cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    pt = pt.with_columns(
        (pl.lit(REFERENCE_DATE, dtype=pl.Datetime) + pl.duration(days=d_birth)).alias('BIRTH_DATE')
    )

    row0 = pt.filter(pl.col('D_BIRTH_DT').is_not_null()).row(0, named=True)
    assert_reference_date(row0['BIRTH_DATE'], row0['D_BIRTH_DT'], 'PT_INFO_BIRTH')

    # Death and last-alive dates are provided as calendar strings.
    pt = pt.with_columns(
        pl.col('HYBRID_DEATH_DT').str.to_datetime(strict=False).alias('DEATH_DATE'),
        pl.col('DERIVED_LAST_ALIVE_DATE').str.to_datetime(strict=False).alias('LAST_CONTACT_DATE'),
    )

    return pt.select(
        [ID_COL, 'BIRTH_DATE', 'GENDER_NM', 'DEATH_DATE', 'LAST_CONTACT_DATE']
    ).rename({'GENDER_NM': 'GENDER'})


def build_survival_cohort(prostate_mrns, anchor_df: pl.DataFrame, platinum_df: pl.DataFrame, status_df: pl.DataFrame) -> pl.DataFrame:
    """Assemble the per-patient ARPI/chemo-anchored survival table."""
    cohort = pl.DataFrame({ID_COL: sorted(int(m) for m in prostate_mrns)})

    cohort = cohort.join(status_df, on=ID_COL, how='left')
    cohort = cohort.join(anchor_df, on=ID_COL, how='left')
    cohort = cohort.join(platinum_df, on=ID_COL, how='left')

    cohort = cohort.with_columns(
        (
            (pl.col('TREATMENT_ANCHOR_DATE') - pl.col('BIRTH_DATE')).dt.total_days()
            / 365.2425
        ).alias('AGE')
    )

    cohort = cohort.with_columns(
        pl.col('DEATH_DATE').fill_null(pl.col('LAST_CONTACT_DATE')).alias('FOLLOW_UP_END_DATE')
    )

    cohort = cohort.with_columns(
        pl.col('DEATH_DATE').is_not_null().cast(pl.Int64).alias('DEATH')
    )
    cohort = cohort.with_columns(
        (pl.col('FOLLOW_UP_END_DATE') - pl.col('TREATMENT_ANCHOR_DATE')).dt.total_days().alias('TT_DEATH')
    )

    has_platinum = pl.col('PLATINUM_DATE').is_not_null()
    cohort = cohort.with_columns(has_platinum.cast(pl.Int64).alias('PLATINUM'))
    platinum_end = pl.when(has_platinum).then(pl.col('PLATINUM_DATE')).otherwise(pl.col('FOLLOW_UP_END_DATE'))
    cohort = cohort.with_columns(
        (platinum_end - pl.col('TREATMENT_ANCHOR_DATE')).dt.total_days().alias('TT_PLATINUM')
    )

    no_anchor = pl.col('TREATMENT_ANCHOR_DATE').is_null()
    cohort = cohort.with_columns(
        pl.when(no_anchor).then(None).otherwise(pl.col('AGE')).alias('AGE'),
        pl.when(no_anchor).then(None).otherwise(pl.col('TT_DEATH')).alias('TT_DEATH'),
        pl.when(no_anchor).then(None).otherwise(pl.col('TT_PLATINUM')).alias('TT_PLATINUM'),
    )

    return cohort.select(
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
    )


def summarize_survival_cohort(cohort: pl.DataFrame):
    n = len(cohort)
    n_anchor = cohort['TREATMENT_ANCHOR_DATE'].is_not_null().sum()
    print("\n=== Survival cohort summary ===")
    print(f"Total ICD-C61 patients (post-exclusion): {n}")
    print(f"With an ARPI/chemo anchor drug: {n_anchor}")
    print(f"Deaths: {int(cohort['DEATH'].sum())}")
    print(f"Received platinum: {int(cohort['PLATINUM'].sum())}")
    with_times = cohort.filter(pl.col('TT_DEATH').is_not_null())
    if len(with_times):
        print(
            f"Median TT_DEATH (days): {with_times['TT_DEATH'].median():.0f}; "
            f"median TT_PLATINUM (days): {with_times['TT_PLATINUM'].median():.0f}"
        )
    neg = cohort.filter(pl.col('TT_DEATH') < 0)
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
        default=os.path.join(ONCDRS_PATH, 'EHR_DIAGNOSIS.csv'),
        help="Raw OncDRS ICD source (EHR_DIAGNOSIS.csv) used to define the C61 cohort "
             "over the full patient universe.",
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
    survival_cohort.write_csv(out_path)
    print(f"\nSaved survival cohort to {out_path}")


if __name__ == "__main__":
    main()
