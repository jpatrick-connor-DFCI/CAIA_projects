"""
Script Name: compile_COMPASS_cohort_data.py

Description:
Single entry point that builds the COMPASS prostate-cancer cohort's ICD
inclusion/exclusion record and ARPI/chemo-anchored survival (outcomes)
cohorts directly from the raw OncDRS 2025-03 pull.

  * compile_prostate_data.py    -- ICD cohort definition (the rest of that
                                    script's cohort-filtered health/meds/
                                    labs/somatic/PSA/platinum table dumps
                                    have been removed; downstream consumers
                                    now read the raw OncDRS tables directly --
                                    see longitudinal_data_processing.py).
  * prostate_arpi_survival_preprocessing.py -- per-patient ARPI/chemo-anchored
                                    survival cohort (age, treatment anchor,
                                    death, platinum time-to-event).
Cohort definitions:
Two ICD-C61 MRN sets are built, each driving a full outcomes cohort plus an
ARPI/chemo-exposure-restricted subset (non-null TREATMENT_ANCHOR_DATE):

  * without_other_primaries -- ICD-10 C61 patients minus patients whose ICD
                              history shows a competing non-prostate primary.
  * with_other_primaries    -- every ICD-10 C61 patient, retaining patients
                              with a competing non-prostate primary.

The former VTE-derived arms are omitted because the VTE prostate MRNs are
fully contained within the ICD-C61 universe.

Raw date handling:
All dates use the raw calendar columns directly (MED_START_DT, BIRTH_DT,
HYBRID_DEATH_DT, DERIVED_LAST_ALIVE_DATE), parsed with polars
str.to_datetime. The de-identified "days since reference" offset columns
(D_MED_START_DT, D_BIRTH_DT, ...) are NOT used.

Inputs:
  * EHR_DIAGNOSIS.csv (OncDRS raw)            ICD-10 -> icd cohort + exclusion
  * MEDICATIONS.csv (OncDRS raw)              anchor + platinum drugs, read
                                               in-memory only -- not persisted
  * PT_INFO_STATUS_REGISTRATION.csv (OncDRS raw)  birth date, sex, death/last-alive

Outputs (in NEPC_PROJ_PATH):
  * prostate_icd_data.csv                        (ICD inclusion/exclusion record)
  * prostate_arpi_survival_cohort_without_other_primaries.csv
  * prostate_arpi_survival_cohort_without_other_primaries_arpi.csv
  * prostate_arpi_survival_cohort_with_other_primaries.csv
  * prostate_arpi_survival_cohort_with_other_primaries_arpi.csv

Also writes four matching bare DFCI_MRN-only CSVs to ``mrn_lists_dir``
(default ``NEPC_PROJ_PATH/mrn_lists/``).

Finally, writes ``icd_prostate_mrn_flags.csv`` to ``mrn_lists_dir``. This
patient-level audit table includes every MRN with an ICD-10 C61 diagnosis
(before the non-prostate-primary exclusion) and binary indicators for a
competing non-prostate primary, PARPi exposure, ARPI/docetaxel exposure, and
at least five broad PSA tests.

Author: J. Patrick Connor
Date: 2026-07-18

Implementation note:
This script is fully polars (zero pandas) throughout. `data_preprocessing_common/fast_io.py`
supplies the shared cohort-filtering (`scan_filter`) and dirty-numeric
recovery (`recover_numeric`) helpers used to read MEDICATIONS.csv in memory.
"""

from __future__ import annotations

import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_preprocessing_common import fast_io  # noqa: E402

ID_COL = "DFCI_MRN"

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/'
NEPC_PROJ_PATH = os.path.join(DATA_PATH, 'CAIA/COMPASS/')

PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
ONCDRS_PATH = os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/')

# ARPI / defined-chemo anchor drugs (matches TREATMENT_ANCHOR_MEDS in
# longitudinal_data_processing.py: ARPIs/androgen-axis, taxanes, radium-223).
ARPI_ANCHOR_MEDS = {
    "ABIRATERONE ACETATE",
    "ENZALUTAMIDE",
    "APALUTAMIDE",
    "DAROLUTAMIDE",
    "DOCETAXEL",
    "CABAZITAXEL",
    "RADIUM RA 223 DICHLORIDE",
}
TREATMENT_ANCHOR_MEDS = ARPI_ANCHOR_MEDS

# ADT anchor drugs: GnRH agonists, GnRH antagonists, and first-generation
# antiandrogens (matches ADT_ANCHOR_MEDS in longitudinal_data_processing.py).
ADT_ANCHOR_MEDS = {
    # GnRH agonists
    "LEUPROLIDE ACETATE",
    "GOSERELIN ACETATE",
    "TRIPTORELIN",
    "TRIPTORELIN PAMOATE",
    "HISTRELIN ACETATE",
    # GnRH antagonists
    "DEGARELIX",
    "RELUGOLIX",
    # First-generation antiandrogens
    "BICALUTAMIDE",
    "FLUTAMIDE",
    "NILUTAMIDE",
}

ANCHOR_MED_SETS = {"arpi": ARPI_ANCHOR_MEDS, "adt": ADT_ANCHOR_MEDS}

# The MRN audit table requested below distinguishes ARPI/docetaxel exposure
# from the pipeline's broader treatment anchor, which additionally includes
# cabazitaxel and radium-223.
ARPI_DOCETAXEL_MEDS = {
    "ABIRATERONE ACETATE",
    "ENZALUTAMIDE",
    "APALUTAMIDE",
    "DAROLUTAMIDE",
    "DOCETAXEL",
}

PARPI_MEDS = {
    "OLAPARIB",
    "RUCAPARIB",
    "NIRAPARIB",
    "TALAZOPARIB",
    "VELIPARIB",
}

# Broad PSA assay definition used by the longitudinal preprocessing gate.
BROAD_PSA_CODES = {
    "PSA",
    "PSAR",
    "PSATOTSCRN",
    "CPSA",
    "PSAMON",
    "PSAULT",
    "PSAT",
}
MIN_PSA_COUNT = 5

# Cisplatin appears both as a single agent and coded within a combination
# regimen name; both count as platinum exposure. Oxaliplatin is intentionally
# excluded (not a relevant platinum agent for this cohort).
PLATINUM_MEDS = {
    "CARBOPLATIN",
    "CISPLATIN",
    "CISPLATIN/CYCLOPHOSPHAMIDE/ETOPOSIDE",
}



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


def compute_non_prostate_primary_mrns(icds: pl.DataFrame) -> set:
    """MRNs with ANY ICD row indicating a competing non-prostate primary
    malignancy. Computed once from the full (cohort-independent) ICD history
    so it can be applied uniformly to every cohort MRN set below -- not just
    the ICD-derived one."""
    marked = mark_non_prostate_primary_icd(icds)
    non_prostate_ids = (
        marked.filter(pl.col("NON_PROSTATE_PRIMARY_ICD10"))[ID_COL]
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
    )
    return set(non_prostate_ids.drop_nulls().to_list())


def compute_prostate_cohort(icds: pl.DataFrame, non_prostate_primary_mrns: set):
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

    excluded = c61_mrns & non_prostate_primary_mrns
    prostate_mrns = c61_mrns - excluded
    print(
        f"ICD-C61 prostate cohort: {len(c61_mrns)} patients; "
        f"excluded {len(excluded)} with a non-prostate-primary ICD; "
        f"retained {len(prostate_mrns)}."
    )
    return prostate_mrns, excluded


def assert_cohort_set_invariants(cohorts: dict[str, set]) -> None:
    """Fail early unless the excluded ICD cohort is a subset of the full one."""
    if not (
        cohorts["without_other_primaries"]
        <= cohorts["with_other_primaries"]
    ):
        raise AssertionError(
            "without_other_primaries must be a subset of with_other_primaries."
        )


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
# ICD inclusion/exclusion output + in-memory medications read
# ---------------------------------------------------------------------------

def filter_and_save(filename, outname, cohort_mrns, cols=None) -> pl.DataFrame:
    """Stream `filename` lazily via polars, keep only cohort_mrns rows
    (and `cols` columns, if given), recover dirty-numeric columns, and write
    the result to `outname`. See `data_preprocessing_common/fast_io.py` for
    the shared scan/filter/recover implementation; this wrapper only adds the
    column re-projection (to preserve the exact requested column order) and
    the `.write_csv()` so output paths/signatures stay stable.
    """
    filtered = filter_cohort(filename, cohort_mrns, cols=cols)
    filtered.write_csv(outname)
    return filtered


def filter_cohort(filename, cohort_mrns, cols=None) -> pl.DataFrame:
    """Same as `filter_and_save` but in-memory only -- no `.write_csv()`.
    Used for tables that feed the outcomes cohort but are not themselves
    persisted (e.g. MEDICATIONS.csv)."""
    lf = fast_io.scan_filter(filename, cohort_mrns, cols=cols)
    filtered = lf.collect()
    # Exclude ID_COL from recover_numeric's generic Utf8->Float64 cast: MRNs
    # are all-digit, so they'd otherwise become Float64, which doesn't match
    # the Int64 DFCI_MRN used everywhere else in this file (cohort, icds_filtered,
    # etc.) and fails polars joins on dtype mismatch. Cast explicitly to Int64
    # instead, consistent with the rest of the pipeline.
    filtered = fast_io.recover_numeric(filtered, exclude=(fast_io.ID_COL,))
    filtered = filtered.with_columns(
        pl.col(fast_io.ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    )
    if cols:
        filtered = filtered.select(list(cols))
    return filtered


def compile_cohort_tables(icd_mrns, all_cohort_mrns, icds: pl.DataFrame):
    """Write the ICD record (scoped to the widest C61 cohort) and return a
    medications table in memory (not persisted), scoped to the union of every
    cohort (`all_cohort_mrns`) so it can feed the outcomes
    cohort builder for both cohort definitions below. No other
    cohort-filtered raw-table dumps are written here -- longitudinal_data_processing.py
    now reads+scopes the raw OncDRS health/labs tables itself, and the
    somatic table is read directly by compile_MRNs_for_manual_review.py when
    needed."""
    icd_mrn_set = set(int(m) for m in icd_mrns)

    mrn_num = icds[ID_COL].cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    icds_filtered = icds.filter(mrn_num.is_in(list(icd_mrn_set)))
    icds_filtered.write_csv(os.path.join(NEPC_PROJ_PATH, 'prostate_icd_data.csv'))

    meds = filter_cohort(os.path.join(ONCDRS_PATH, 'MEDICATIONS.csv'), set(int(m) for m in all_cohort_mrns))

    return meds


def build_icd_prostate_mrn_flags(
    c61_mrns: set,
    non_prostate_primary_mrns: set,
    meds: pl.DataFrame,
    labs_path: str,
) -> pl.DataFrame:
    """Build one audit row for every ICD-C61 patient.

    Medication exposure is based on any matching medication row. PSA
    eligibility counts rows from the same broad raw TEST_TYPE_CD set used by
    longitudinal preprocessing.
    """
    c61_mrns = set(int(m) for m in c61_mrns)
    cohort = pl.DataFrame({ID_COL: sorted(c61_mrns)}, schema={ID_COL: pl.Int64})

    normalized_meds = meds.with_columns(
        pl.col("NCI_PREFERRED_MED_NM")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.strip_chars()
        .alias("_MED_NAME")
    )
    parpi_mrns = set(
        normalized_meds.filter(pl.col("_MED_NAME").is_in(sorted(PARPI_MEDS)))[ID_COL]
        .drop_nulls()
        .to_list()
    )
    arpi_docetaxel_mrns = set(
        normalized_meds.filter(
            pl.col("_MED_NAME").is_in(sorted(ARPI_DOCETAXEL_MEDS))
        )[ID_COL]
        .drop_nulls()
        .to_list()
    )
    adt_mrns = set(
        normalized_meds.filter(pl.col("_MED_NAME").is_in(sorted(ADT_ANCHOR_MEDS)))[ID_COL]
        .drop_nulls()
        .to_list()
    )

    labs = filter_cohort(
        labs_path,
        c61_mrns,
        cols=[ID_COL, "TEST_TYPE_CD"],
    ).with_columns(
        pl.col("TEST_TYPE_CD")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.strip_chars()
        .alias("TEST_TYPE_CD")
    )
    psa_eligible_mrns = set(
        labs.filter(pl.col("TEST_TYPE_CD").is_in(sorted(BROAD_PSA_CODES)))
        .group_by(ID_COL)
        .agg(pl.len().alias("_PSA_COUNT"))
        .filter(pl.col("_PSA_COUNT") >= MIN_PSA_COUNT)[ID_COL]
        .drop_nulls()
        .to_list()
    )

    return cohort.with_columns(
        pl.col(ID_COL)
        .is_in(sorted(c61_mrns & non_prostate_primary_mrns))
        .cast(pl.Int8)
        .alias("HAS_NON_PROSTATE_PRIMARY"),
        pl.col(ID_COL)
        .is_in(sorted(parpi_mrns))
        .cast(pl.Int8)
        .alias("PARPI_EXPOSED"),
        pl.col(ID_COL)
        .is_in(sorted(arpi_docetaxel_mrns))
        .cast(pl.Int8)
        .alias("ARPI_DOCETAXEL_EXPOSED"),
        pl.col(ID_COL)
        .is_in(sorted(adt_mrns))
        .cast(pl.Int8)
        .alias("ADT_EXPOSED"),
        pl.col(ID_COL)
        .is_in(sorted(psa_eligible_mrns))
        .cast(pl.Int8)
        .alias("HAS_5_OR_MORE_PSA_TESTS"),
    )


# ---------------------------------------------------------------------------
# ARPI/chemo-anchored survival cohort (prostate_arpi_survival_preprocessing.py)
# ---------------------------------------------------------------------------

def load_medications_for_survival(meds: pl.DataFrame) -> pl.DataFrame:
    """Parse the raw calendar MED_START_DT for the anchor + platinum drug rows
    already filtered to the prostate cohort by compile_cohort_tables. Uses the
    original date column directly (NOT the de-identified D_MED_START_DT offset),
    matching how longitudinal_data_processing.py reads MED_START_DT downstream.
    """
    keep_meds = {m.upper() for m in ARPI_ANCHOR_MEDS | ADT_ANCHOR_MEDS | PLATINUM_MEDS}
    out = meds.with_columns(
        pl.col('NCI_PREFERRED_MED_NM').cast(pl.Utf8).str.to_uppercase().str.strip_chars().alias('NCI_PREFERRED_MED_NM')
    )
    out = out.filter(pl.col('NCI_PREFERRED_MED_NM').is_in(list(keep_meds)))
    out = out.with_columns(
        pl.col('MED_START_DT').str.to_datetime(strict=False).alias('MED_START_DT')
    )
    out = out.filter(pl.col('MED_START_DT').is_not_null())
    return out


def compute_treatment_anchor(meds: pl.DataFrame, meds_set: set = ARPI_ANCHOR_MEDS) -> pl.DataFrame:
    """Earliest anchor-drug MED_START_DT per patient -> TREATMENT_ANCHOR_DATE."""
    anchor = meds.filter(pl.col('NCI_PREFERRED_MED_NM').is_in(list(meds_set)))
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
    PT_INFO_STATUS_REGISTRATION.csv. All dates use the raw calendar columns
    (BIRTH_DT, HYBRID_DEATH_DT, DERIVED_LAST_ALIVE_DATE) directly -- NOT the
    de-identified D_BIRTH_DT offset.

    Returns a dataframe with:
        DFCI_MRN, BIRTH_DATE, GENDER, DEATH_DATE, LAST_CONTACT_DATE
    """
    pt = pl.scan_csv(os.path.join(path, 'PT_INFO_STATUS_REGISTRATION.csv'), infer_schema_length=0).collect()

    # All dates are raw calendar strings.
    pt = pt.with_columns(
        pl.col('BIRTH_DT').str.to_datetime(strict=False).alias('BIRTH_DATE'),
        pl.col('HYBRID_DEATH_DT').str.to_datetime(strict=False).alias('DEATH_DATE'),
        pl.col('DERIVED_LAST_ALIVE_DATE').str.to_datetime(strict=False).alias('LAST_CONTACT_DATE'),
        # infer_schema_length=0 reads every column (including DFCI_MRN) as
        # Utf8; cast back to Int64 here so this frame's join key matches the
        # Int64 DFCI_MRN used everywhere else (e.g. build_survival_cohort's
        # `cohort` frame), instead of failing the join on dtype mismatch.
        pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).alias(ID_COL),
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


def summarize_survival_cohort(cohort: pl.DataFrame, label="cohort"):
    n = len(cohort)
    n_anchor = cohort['TREATMENT_ANCHOR_DATE'].is_not_null().sum()
    print(f"\n=== Survival cohort summary ({label}) ===")
    print(f"Total patients (post-exclusion): {n}")
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
        description="Compile the COMPASS prostate cohort's ICD inclusion/"
        "exclusion record and ARPI/chemo-anchored survival cohorts from the "
        "raw OncDRS pull, with and without the non-prostate-primary exclusion.",
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
        "--labs-csv",
        type=str,
        default=os.path.join(ONCDRS_PATH, "OUTPT_LAB_RESULTS_LABS.csv"),
        help="Raw OncDRS outpatient labs file used for the broad PSA-count flag.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=NEPC_PROJ_PATH,
        help="Directory to write all cohort CSVs.",
    )
    parser.add_argument(
        "--mrn-lists-dir",
        type=str,
        default=os.path.join(NEPC_PROJ_PATH, "mrn_lists"),
        help="Directory for cohort MRN lists and icd_prostate_mrn_flags.csv.",
    )
    args = parser.parse_args()

    # 1. The two ICD-C61 cohort MRN sets. The non-prostate-primary exclusion
    #    is computed once from the full ICD history.
    icds = load_and_explode_icd(args.icd_source)
    non_prostate_primary_mrns = compute_non_prostate_primary_mrns(icds)
    without_other_primaries_mrns, icd_excluded_mrns = compute_prostate_cohort(
        icds,
        non_prostate_primary_mrns,
    )
    with_other_primaries_mrns = (
        without_other_primaries_mrns | icd_excluded_mrns
    )
    print(
        f"ICD-C61 cohorts: {len(without_other_primaries_mrns)} without other "
        f"primaries; {len(with_other_primaries_mrns)} with other primaries "
        f"allowed ({len(icd_excluded_mrns)} additional patients)."
    )

    cohorts = {
        "without_other_primaries": without_other_primaries_mrns,
        "with_other_primaries": with_other_primaries_mrns,
    }
    assert_cohort_set_invariants(cohorts)

    # 2. ICD output + in-memory medications scoped to every ICD-C61 patient.
    meds = compile_cohort_tables(
        with_other_primaries_mrns,
        with_other_primaries_mrns,
        icds,
    )

    # 3. ARPI/chemo-anchored survival cohort, reusing the meds already
    #    scoped to every cohort above -- shared across both cohort
    #    definitions since anchor/platinum dates don't depend on cohort.
    meds_for_survival = load_medications_for_survival(meds)
    anchor_df = compute_treatment_anchor(meds_for_survival, meds_set=ARPI_ANCHOR_MEDS)
    adt_anchor_df = compute_treatment_anchor(meds_for_survival, meds_set=ADT_ANCHOR_MEDS)
    platinum_df = compute_first_platinum(meds_for_survival)
    print(
        f"ARPI anchor drug recipients: {len(anchor_df)}; "
        f"ADT anchor drug recipients: {len(adt_anchor_df)}; "
        f"platinum recipients: {len(platinum_df)} (across all cohorts)."
    )

    status_df = load_patient_status(args.oncdrs_path)

    # Stable, explicit filenames mirror the two public cohort labels.
    out_names = {
        "without_other_primaries": (
            "prostate_arpi_survival_cohort_without_other_primaries.csv"
        ),
        "with_other_primaries": (
            "prostate_arpi_survival_cohort_with_other_primaries.csv"
        ),
    }

    os.makedirs(args.mrn_lists_dir, exist_ok=True)

    icd_prostate_flags = build_icd_prostate_mrn_flags(
        with_other_primaries_mrns,
        non_prostate_primary_mrns,
        meds,
        args.labs_csv,
    )
    icd_prostate_flags_path = os.path.join(
        args.mrn_lists_dir,
        "icd_prostate_mrn_flags.csv",
    )
    icd_prostate_flags.write_csv(icd_prostate_flags_path)
    print(
        f"Saved ICD-C61 MRN flags ({len(icd_prostate_flags)} patients) to "
        f"{icd_prostate_flags_path}"
    )

    for cohort_key, cohort_mrns in cohorts.items():
        survival_cohort = build_survival_cohort(cohort_mrns, anchor_df, platinum_df, status_df)
        summarize_survival_cohort(survival_cohort, label=cohort_key)

        out_path = os.path.join(args.out_dir, out_names[cohort_key])
        survival_cohort.write_csv(out_path)
        print(f"Saved {cohort_key} survival cohort to {out_path}")

        mrn_list_path = os.path.join(args.mrn_lists_dir, f"{cohort_key}_mrns.csv")
        survival_cohort.select(ID_COL).unique().sort(ID_COL).write_csv(mrn_list_path)
        print(f"Saved {cohort_key} MRN list to {mrn_list_path}")

        arpi_cohort = survival_cohort.filter(pl.col('TREATMENT_ANCHOR_DATE').is_not_null())
        summarize_survival_cohort(arpi_cohort, label=f"{cohort_key}_arpi")

        arpi_out_path = os.path.join(args.out_dir, f"prostate_arpi_survival_cohort_{cohort_key}_arpi.csv")
        arpi_cohort.write_csv(arpi_out_path)
        print(f"Saved {cohort_key}_arpi survival cohort to {arpi_out_path}")

        arpi_mrn_list_path = os.path.join(args.mrn_lists_dir, f"{cohort_key}_arpi_mrns.csv")
        arpi_cohort.select(ID_COL).unique().sort(ID_COL).write_csv(arpi_mrn_list_path)
        print(f"Saved {cohort_key}_arpi MRN list to {arpi_mrn_list_path}")

        # ADT-anchored variant of this same cohort, additive alongside the
        # ARPI arm above -- reuses the same unanchored survival_cohort rows,
        # just re-derived against adt_anchor_df instead of anchor_df.
        adt_survival_cohort = build_survival_cohort(cohort_mrns, adt_anchor_df, platinum_df, status_df)
        adt_cohort = adt_survival_cohort.filter(pl.col('TREATMENT_ANCHOR_DATE').is_not_null())
        summarize_survival_cohort(adt_cohort, label=f"{cohort_key}_adt")

        adt_out_path = os.path.join(args.out_dir, f"prostate_adt_survival_cohort_{cohort_key}_adt.csv")
        adt_cohort.write_csv(adt_out_path)
        print(f"Saved {cohort_key}_adt survival cohort to {adt_out_path}")

        adt_mrn_list_path = os.path.join(args.mrn_lists_dir, f"{cohort_key}_adt_mrns.csv")
        adt_cohort.select(ID_COL).unique().sort(ID_COL).write_csv(adt_mrn_list_path)
        print(f"Saved {cohort_key}_adt MRN list to {adt_mrn_list_path}")


if __name__ == "__main__":
    main()
