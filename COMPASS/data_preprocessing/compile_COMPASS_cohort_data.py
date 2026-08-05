"""
Script Name: compile_COMPASS_cohort_data.py

Description:
Single entry point that builds the ADT-entry COMPASS prostate-cancer cohort's
ICD inclusion/exclusion record and ARPI/chemo-anchored survival (outcomes)
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
The primary entry requirement is a dated ADT exposure among ICD-10 C61
patients. Patients are then excluded if bladder, lung, head-and-neck, or
testicular cancer is diagnosed strictly after ADT start. This common eligible
MRN set drives two survival cohorts:

  * arpi -- eligible ADT-entry patients anchored on first ARPI/chemo exposure.
  * adt  -- eligible ADT-entry patients anchored on first ADT (GnRH
            agonist/antagonist or 1st-gen antiandrogen) exposure.

Other competing non-prostate primaries, and the four specified cancers when
diagnosed on or before ADT start, do not trigger this exclusion.

The former VTE-derived arms are omitted because the VTE prostate MRNs are
fully contained within the ICD-C61 universe.

Raw date handling:
All dates use the raw calendar columns directly (MED_START_DT, BIRTH_DT,
HYBRID_DEATH_DT, DERIVED_LAST_ALIVE_DATE), parsed with polars
str.to_datetime. The de-identified "days since reference" offset columns
(D_MED_START_DT, D_BIRTH_DT, ...) are NOT used.

Inputs (each overridable by flag; a raw OncDRS CSV or a merged Parquet from
PROFILE_data_processing -- see data_preprocessing_common/oncdrs_sources.py):
  * EHR_DIAGNOSIS.csv (--icd-source)          ICD-10 -> icd cohort + exclusion
  * MEDICATIONS.csv (--medications-source)    anchor + platinum drugs, read
                                               in-memory only -- not persisted
  * PT_INFO_STATUS_REGISTRATION.csv (--patient-status-source)
                                              birth date, sex, death/last-alive
  * OUTPT_LAB_RESULTS_LABS.csv (--labs-csv)   broad PSA-count flag

Outputs (in --out-dir, default NEPC_PROJ_PATH):
  * prostate_icd_data.csv                        (ICD inclusion/exclusion record)
  * prostate_arpi_survival_cohort_arpi.csv
  * prostate_adt_survival_cohort_adt.csv

Also writes matching bare DFCI_MRN-only CSVs to ``mrn_lists_dir``
(default ``NEPC_PROJ_PATH/mrn_lists/``): ``arpi_mrns.csv``, ``adt_mrns.csv``,
and ``platinum_MRN_list.csv``. The platinum list contains dated platinum
recipients from the eligible ADT-entry cohort only.

Finally, writes ``icd_prostate_mrn_flags.csv`` to ``mrn_lists_dir``. This
patient-level audit table includes every MRN with an ICD-10 C61 diagnosis and
binary indicators for a competing non-prostate primary, a specified
post-ADT exclusion cancer, PARPi exposure, ARPI/docetaxel exposure, ADT
exposure, and at least five broad PSA tests.

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
from data_preprocessing_common.oncdrs_sources import scan_source  # noqa: E402

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

# ICD-10-CM primary malignancies requested for exclusion when their diagnosis
# date is strictly after first ADT. Head and neck follows the standard
# topographic groupings C00-C14 and C30-C32.
POST_ADT_EXCLUSION_CANCER_GROUPS = {
    "bladder": {"C67"},
    "lung": {"C34"},
    "head_and_neck": {
        *(f"C{i:02d}" for i in range(0, 15)),
        "C30",
        "C31",
        "C32",
    },
    "testicular": {"C62"},
}
POST_ADT_EXCLUSION_CANCER_PREFIXES = set().union(
    *POST_ADT_EXCLUSION_CANCER_GROUPS.values()
)



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


def compute_post_adt_exclusion_cancer_mrns(
    icds: pl.DataFrame,
    adt_anchor_df: pl.DataFrame,
) -> set:
    """Return MRNs with a requested exclusion cancer diagnosed after ADT.

    Diagnosis on the ADT start date is retained: "post ADT start" is
    implemented as the strict comparison START_DT > TREATMENT_ANCHOR_DATE.
    Rows without a parseable diagnosis date cannot establish a post-ADT
    diagnosis and therefore do not trigger exclusion.
    """
    diagnosis = icds.select(
        pl.col(ID_COL)
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
        .alias(ID_COL),
        pl.col("DIAGNOSIS_ICD10_CD")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.strip_chars()
        .str.extract(r"^(C\d{2})", 1)
        .alias("_ICD_PREFIX"),
        pl.col("START_DT")
        .cast(pl.Utf8)
        .str.to_datetime(strict=False)
        .alias("_DIAGNOSIS_DATE"),
    )
    post_adt = (
        diagnosis.filter(
            pl.col("_ICD_PREFIX").is_in(sorted(POST_ADT_EXCLUSION_CANCER_PREFIXES))
        )
        .join(adt_anchor_df, on=ID_COL, how="inner")
        .filter(pl.col("_DIAGNOSIS_DATE") > pl.col("TREATMENT_ANCHOR_DATE"))
    )
    return set(post_adt[ID_COL].drop_nulls().to_list())


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
    icds = scan_source(icd_path).collect()

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


def compile_cohort_tables(
    icd_mrns,
    all_cohort_mrns,
    icds: pl.DataFrame,
    medications_path,
    out_dir,
):
    """Write the ICD record (scoped to the widest C61 cohort) and return a
    medications table in memory (not persisted), scoped to the union of every
    cohort (`all_cohort_mrns`) so it can feed the outcomes
    cohort builder for both cohort definitions below. No other
    cohort-filtered raw-table dumps are written here -- longitudinal_data_processing.py
    now reads+scopes the raw OncDRS health/labs tables itself, and the
    somatic table is read directly by compile_MRNs_for_manual_review.py when
    needed.

    `medications_path` and `out_dir` are passed in (rather than read from the
    module-level constants) so a run against a different OncDRS source writes
    to its own output root instead of clobbering the default one.
    """
    icd_mrn_set = set(int(m) for m in icd_mrns)

    mrn_num = icds[ID_COL].cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    icds_filtered = icds.filter(mrn_num.is_in(list(icd_mrn_set)))
    icds_filtered.write_csv(os.path.join(out_dir, 'prostate_icd_data.csv'))

    meds = filter_cohort(medications_path, set(int(m) for m in all_cohort_mrns))

    return meds


def build_icd_prostate_mrn_flags(
    c61_mrns: set,
    non_prostate_primary_mrns: set,
    post_adt_exclusion_cancer_mrns: set,
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
        .is_in(sorted(c61_mrns & post_adt_exclusion_cancer_mrns))
        .cast(pl.Int8)
        .alias("HAS_POST_ADT_EXCLUSION_CANCER"),
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
    """Load birth date, sex, and death / last-alive info from the
    PT_INFO_STATUS_REGISTRATION table (`path` is the file itself -- raw
    OncDRS CSV or merged Parquet). All dates use the raw calendar columns
    (BIRTH_DT, HYBRID_DEATH_DT, DERIVED_LAST_ALIVE_DATE) directly -- NOT the
    de-identified D_BIRTH_DT offset.

    Returns a dataframe with:
        DFCI_MRN, BIRTH_DATE, GENDER, DEATH_DATE, LAST_CONTACT_DATE
    """
    pt = scan_source(path).collect()

    # All dates are raw calendar strings.
    pt = pt.with_columns(
        pl.col('BIRTH_DT').str.to_datetime(strict=False).alias('BIRTH_DATE'),
        pl.col('HYBRID_DEATH_DT').str.to_datetime(strict=False).alias('DEATH_DATE'),
        pl.col('DERIVED_LAST_ALIVE_DATE').str.to_datetime(strict=False).alias('LAST_CONTACT_DATE'),
        pl.when(
            pl.col('GENDER_NM')
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_uppercase()
            .is_in(['', 'NAN', 'NONE', 'NULL'])
        )
        .then(None)
        .otherwise(pl.col('GENDER_NM').cast(pl.Utf8).str.strip_chars())
        .alias('GENDER'),
        # infer_schema_length=0 reads every column (including DFCI_MRN) as
        # Utf8; cast back to Int64 here so this frame's join key matches the
        # Int64 DFCI_MRN used everywhere else (e.g. build_survival_cohort's
        # `cohort` frame), instead of failing the join on dtype mismatch.
        pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False).alias(ID_COL),
    )

    # Merged PROFILE releases can contribute multiple status rows for one MRN,
    # including sparse rows from a newer pull. Collapse field-by-field before
    # joining to the cohort so an arbitrary null row cannot erase demographics.
    # Birth date and gender are stable patient attributes; death and last-contact
    # dates use the latest available date.
    status = (
        pt.filter(pl.col(ID_COL).is_not_null())
        .group_by(ID_COL)
        .agg(
            pl.col('BIRTH_DATE').drop_nulls().first().alias('BIRTH_DATE'),
            pl.col('GENDER').drop_nulls().first().alias('GENDER'),
            pl.col('DEATH_DATE').max().alias('DEATH_DATE'),
            pl.col('LAST_CONTACT_DATE').max().alias('LAST_CONTACT_DATE'),
        )
    )
    n_birth = int(status['BIRTH_DATE'].is_not_null().sum())
    n_gender = int(status['GENDER'].is_not_null().sum())
    print(
        "[patient-status] "
        f"{len(pt):,} source rows -> {len(status):,} unique MRNs; "
        f"birth date present={n_birth:,}; gender present={n_gender:,}"
    )
    if len(status) and (n_birth == 0 or n_gender == 0):
        missing_fields = [
            name
            for name, count in (("parsed BIRTH_DT", n_birth), ("GENDER_NM", n_gender))
            if count == 0
        ]
        raise ValueError(
            "PT_INFO_STATUS_REGISTRATION has no usable values for "
            f"{', '.join(missing_fields)} after parsing."
        )
    return status


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
    print(f"Total eligible patients: {n}")
    print(f"With the {label.upper()} analysis anchor: {n_anchor}")
    anchored = cohort.filter(pl.col('TREATMENT_ANCHOR_DATE').is_not_null())
    print(
        "Anchored demographics present: "
        f"age={anchored['AGE'].is_not_null().sum():,}/{len(anchored):,}; "
        f"gender={anchored['GENDER'].is_not_null().sum():,}/{len(anchored):,}"
    )
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
        description="Compile the COMPASS ADT-entry prostate cohort and its "
        "ARPI- and ADT-anchored survival outputs from the raw OncDRS pull.",
    )
    parser.add_argument(
        "--icd-source",
        type=str,
        default=os.path.join(ONCDRS_PATH, 'EHR_DIAGNOSIS.csv'),
        help="ICD source (CSV or Parquet, e.g. raw OncDRS EHR_DIAGNOSIS.csv) "
             "used to define the C61 cohort over the full patient universe.",
    )
    parser.add_argument(
        "--oncdrs-path",
        type=str,
        default=ONCDRS_PATH,
        help="OncDRS raw data pull directory. Only used to build the defaults "
             "for --medications-source and --patient-status-source.",
    )
    parser.add_argument(
        "--medications-source",
        type=str,
        default=None,
        help="Medications table (CSV or Parquet) supplying the anchor and "
             "platinum drugs. Defaults to <--oncdrs-path>/MEDICATIONS.csv.",
    )
    parser.add_argument(
        "--patient-status-source",
        type=str,
        default=None,
        help="Patient status table (CSV or Parquet) supplying birth date, sex, "
             "and death/last-alive dates. Defaults to "
             "<--oncdrs-path>/PT_INFO_STATUS_REGISTRATION.csv.",
    )
    parser.add_argument(
        "--labs-csv",
        type=str,
        default=os.path.join(ONCDRS_PATH, "OUTPT_LAB_RESULTS_LABS.csv"),
        help="Outpatient labs file (CSV or Parquet) used for the broad "
             "PSA-count flag.",
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

    # Table defaults are derived from --oncdrs-path so the historical
    # single-release invocation keeps working with no new flags.
    if args.medications_source is None:
        args.medications_source = os.path.join(args.oncdrs_path, "MEDICATIONS.csv")
    if args.patient_status_source is None:
        args.patient_status_source = os.path.join(
            args.oncdrs_path, "PT_INFO_STATUS_REGISTRATION.csv"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. The single ICD-C61 cohort MRN set (every ICD-C61 patient, including
    #    those with a competing non-prostate primary). The non-prostate-
    #    primary exclusion is still computed, but only feeds the descriptive
    #    HAS_NON_PROSTATE_PRIMARY flag below -- it no longer gates cohort
    #    membership.
    icds = load_and_explode_icd(args.icd_source)
    non_prostate_primary_mrns = compute_non_prostate_primary_mrns(icds)
    prostate_excl_mrns, icd_excluded_mrns = compute_prostate_cohort(
        icds,
        non_prostate_primary_mrns,
    )
    all_cohort_mrns = prostate_excl_mrns | icd_excluded_mrns
    print(
        f"ICD-C61 cohort: {len(all_cohort_mrns)} patients "
        f"({len(icd_excluded_mrns)} with a competing non-prostate primary)."
    )

    # 2. ICD output + in-memory medications scoped to every ICD-C61 patient.
    meds = compile_cohort_tables(
        all_cohort_mrns,
        all_cohort_mrns,
        icds,
        args.medications_source,
        args.out_dir,
    )

    # 3. Establish first ADT (the primary entry requirement), then exclude
    #    the requested cancers only when diagnosed strictly after ADT start.
    meds_for_survival = load_medications_for_survival(meds)
    anchor_df = compute_treatment_anchor(meds_for_survival, meds_set=ARPI_ANCHOR_MEDS)
    adt_anchor_df = compute_treatment_anchor(meds_for_survival, meds_set=ADT_ANCHOR_MEDS)
    platinum_df = compute_first_platinum(meds_for_survival)
    post_adt_exclusion_cancer_mrns = compute_post_adt_exclusion_cancer_mrns(
        icds,
        adt_anchor_df,
    )
    adt_entry_mrns = set(adt_anchor_df[ID_COL].drop_nulls().to_list())
    eligible_mrns = (
        all_cohort_mrns & adt_entry_mrns
    ) - post_adt_exclusion_cancer_mrns
    eligible_platinum_df = platinum_df.filter(
        pl.col(ID_COL).is_in(sorted(eligible_mrns))
    )
    print(
        f"ARPI anchor drug recipients: {len(anchor_df)}; "
        f"ADT anchor drug recipients: {len(adt_anchor_df)}; "
        f"post-ADT specified-cancer exclusions: "
        f"{len(all_cohort_mrns & post_adt_exclusion_cancer_mrns)}; "
        f"eligible after ADT entry/exclusion: {len(eligible_mrns)}; "
        f"eligible platinum recipients: {len(eligible_platinum_df)} "
        f"(of {len(platinum_df)} across ICD-C61)."
    )

    status_df = load_patient_status(args.patient_status_source)

    os.makedirs(args.mrn_lists_dir, exist_ok=True)

    platinum_mrn_list_path = os.path.join(
        args.mrn_lists_dir,
        "platinum_MRN_list.csv",
    )
    eligible_platinum_df.select(ID_COL).unique().sort(ID_COL).write_csv(
        platinum_mrn_list_path
    )
    print(
        f"Saved ADT-cohort platinum MRN list "
        f"({len(eligible_platinum_df)} patients) to "
        f"{platinum_mrn_list_path}"
    )

    icd_prostate_flags = build_icd_prostate_mrn_flags(
        all_cohort_mrns,
        non_prostate_primary_mrns,
        post_adt_exclusion_cancer_mrns,
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

    # Both arms inherit the same ADT-entry eligibility universe. The ARPI arm
    # additionally requires its own non-null analysis anchor.
    survival_cohort = build_survival_cohort(eligible_mrns, anchor_df, platinum_df, status_df)
    arpi_cohort = survival_cohort.filter(pl.col('TREATMENT_ANCHOR_DATE').is_not_null())
    summarize_survival_cohort(arpi_cohort, label="arpi")

    arpi_out_path = os.path.join(args.out_dir, "prostate_arpi_survival_cohort_arpi.csv")
    arpi_cohort.write_csv(arpi_out_path)
    print(f"Saved arpi survival cohort to {arpi_out_path}")

    arpi_mrn_list_path = os.path.join(args.mrn_lists_dir, "arpi_mrns.csv")
    arpi_cohort.select(ID_COL).unique().sort(ID_COL).write_csv(arpi_mrn_list_path)
    print(f"Saved arpi MRN list to {arpi_mrn_list_path}")

    adt_survival_cohort = build_survival_cohort(eligible_mrns, adt_anchor_df, platinum_df, status_df)
    adt_cohort = adt_survival_cohort.filter(pl.col('TREATMENT_ANCHOR_DATE').is_not_null())
    summarize_survival_cohort(adt_cohort, label="adt")

    adt_out_path = os.path.join(args.out_dir, "prostate_adt_survival_cohort_adt.csv")
    adt_cohort.write_csv(adt_out_path)
    print(f"Saved adt survival cohort to {adt_out_path}")

    adt_mrn_list_path = os.path.join(args.mrn_lists_dir, "adt_mrns.csv")
    adt_cohort.select(ID_COL).unique().sort(ID_COL).write_csv(adt_mrn_list_path)
    print(f"Saved adt MRN list to {adt_mrn_list_path}")


if __name__ == "__main__":
    main()
