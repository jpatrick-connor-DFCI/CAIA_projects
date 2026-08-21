"""
Script Name: validate_adt_intent.py

Description:
Measures the medication-only ADT intent label from `classify_adt_intent.py`
against an independent reference standard.

The label deliberately uses medications alone so it transfers to sites without
curated staging. That portability is worth nothing if the misclassification
rate is unknown, so this harness builds two signals the classifier never sees
and reports agreement:

  * secondary-malignancy ICD codes (C77-C79, C7B) -- a coded metastasis. Its
    absence is weak evidence (coding is inconsistent), but its presence in a
    patient labelled LOCALIZED_ADJUVANT is a hard contradiction.
  * PSA trajectory -- a deep, sustained nadir with no subsequent rise is the
    adjuvant pattern; a nadir followed by a rise is biochemical progression.

Two further reference standards are optional, cross-referenced from
`clinical_text_embedding_project` (its `generate_all_non_text_covariates.py`
writes both into that project's FEATURE_PATH, keyed on the same DFCI_MRN):

  * curated stage at diagnosis (`cancer_stage_df.csv.gz`) -- the most direct
    external read of localized vs. metastatic, but it adjudicates in one
    direction only. Stage IV in a patient labelled LOCALIZED_ADJUVANT is a
    hard contradiction; stage I-III in a METASTATIC patient is the expected
    progression pathway and is not scored as an error.
  * metastatic burden (`met_burden_df.csv.gz`) -- distinct metastatic organ
    groups from C77-C79, as a count plus per-group indicators. This upgrades
    the presence/absence ICD signal above to a burden and pattern signal,
    which is what distinguishes oligometastatic from disseminated disease.

Both are read as flat CSVs rather than by importing that project, so this
module gains no cross-project or cluster-only dependency, and both reports
degrade to silence when the files are not supplied.

Nothing computed here is ever fed back into the label. Reports are printed and
optionally written as CSVs.

Interpreting the output
-----------------------
The primary go/no-go signal is the survival contrast: LOCALIZED_ADJUVANT should
show a markedly lower death rate and lower metastasis-code rate than
METASTATIC. If it does not, the label is not working and downstream use should
stop.

Two reports exist to expose known failure modes rather than assume them away:
the calendar-year stratification surfaces ARPI-era drift (apalutamide and
darolutamide are approved in non-metastatic disease, and ARPIs are increasingly
used in high-risk localized disease), and the gap-threshold sensitivity shows
how much the episode reconstruction depends on a threshold that no end-date
column exists to validate.
"""

from __future__ import annotations

import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from COMPASS.data_preprocessing.classify_adt_intent import (  # noqa: E402
    GAP_THRESHOLD_DAYS,
    INTENT_LOCALIZED,
    INTENT_METASTATIC,
    classify_adt_intent,
)
from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (  # noqa: E402
    ID_COL,
    parse_mixed_datetime_expr,
)

# Secondary/metastatic neoplasm codes. C77 lymph nodes, C78 respiratory and
# digestive, C79 other sites, C7B secondary neuroendocrine.
METASTASIS_ICD_PREFIXES = ("C77", "C78", "C79", "C7B")

# PSA below this is a deep response; castrate-level nadir.
PSA_DEEP_NADIR = 0.1
# A rise of this much above nadir is conventionally biochemical progression.
PSA_PROGRESSION_RISE = 2.0


def compute_first_metastasis_icd_date(icds: pl.DataFrame) -> pl.DataFrame:
    """Earliest dated secondary-malignancy ICD code per patient.

    Mirrors `compute_first_prostate_diagnosis`, but matches on normalized code
    prefixes rather than the numeric-range regex used by
    `mark_non_prostate_primary_icd`. That regex captures `^[A-Z](\\d{2,3})`,
    which yields 781 for a dotless `C7810` and silently fails its `77 <= n <=
    79` test; prefix matching is correct for both `C78.01` and `C7801`.
    """
    return (
        icds.select(
            pl.col(ID_COL)
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64, strict=False)
            .alias(ID_COL),
            pl.col("DIAGNOSIS_ICD10_CD")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .str.strip_chars()
            .alias("_ICD_CODE"),
            parse_mixed_datetime_expr("START_DT").alias("_ICD_DATE"),
        )
        .filter(
            pl.any_horizontal(
                [
                    pl.col("_ICD_CODE").str.starts_with(p)
                    for p in METASTASIS_ICD_PREFIXES
                ]
            )
            & pl.col(ID_COL).is_not_null()
            & pl.col("_ICD_DATE").is_not_null()
        )
        .group_by(ID_COL)
        .agg(pl.col("_ICD_DATE").min().alias("FIRST_METASTASIS_ICD_DATE"))
    )


def compute_psa_nadir_features(
    labs: pl.DataFrame,
    psa_lab_names: tuple[str, ...] = ("PSA",),
) -> pl.DataFrame:
    """Post-ADT PSA nadir and subsequent rise, per patient.

    Expects the long prediction frame: `LAB_NAME`, `LAB_VALUE`, and `t_lab`
    (days relative to the treatment anchor, so `t_lab >= 0` is post-ADT).
    Below-detection PSA is genuinely imputed to 0.0 upstream in `dfci_labs`, so
    zeros are real values rather than missing.
    """
    psa = labs.filter(
        pl.col("LAB_NAME").is_in(list(psa_lab_names))
        & pl.col("t_lab").is_not_null()
        & (pl.col("t_lab") >= 0)
        & pl.col("LAB_VALUE").is_not_null()
    ).select(
        pl.col(ID_COL)
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
        .alias(ID_COL),
        pl.col("LAB_VALUE").cast(pl.Float64, strict=False),
        pl.col("t_lab").cast(pl.Float64, strict=False),
    )

    if psa.height == 0:
        return pl.DataFrame(
            schema={
                ID_COL: pl.Int64,
                "PSA_NADIR": pl.Float64,
                "PSA_RISE_FROM_NADIR": pl.Float64,
                "PSA_DEEP_NADIR_NO_RISE": pl.Boolean,
                "PSA_N_POST_ADT": pl.UInt32,
            }
        )

    nadir = psa.group_by(ID_COL).agg(
        pl.col("LAB_VALUE").min().alias("PSA_NADIR"),
        pl.len().cast(pl.UInt32).alias("PSA_N_POST_ADT"),
        # Time of the nadir; only values measured after it can count as a rise.
        pl.col("t_lab").get(pl.col("LAB_VALUE").arg_min()).alias("_t_nadir"),
    )

    post_nadir_max = (
        psa.join(nadir.select(ID_COL, "_t_nadir"), on=ID_COL, how="inner")
        .filter(pl.col("t_lab") > pl.col("_t_nadir"))
        .group_by(ID_COL)
        .agg(pl.col("LAB_VALUE").max().alias("_post_nadir_max"))
    )

    return (
        nadir.join(post_nadir_max, on=ID_COL, how="left")
        .with_columns(
            (pl.col("_post_nadir_max") - pl.col("PSA_NADIR"))
            .fill_null(0.0)
            .alias("PSA_RISE_FROM_NADIR")
        )
        .with_columns(
            (
                (pl.col("PSA_NADIR") < PSA_DEEP_NADIR)
                & (pl.col("PSA_RISE_FROM_NADIR") < PSA_PROGRESSION_RISE)
            ).alias("PSA_DEEP_NADIR_NO_RISE")
        )
        .select(
            ID_COL,
            "PSA_NADIR",
            "PSA_RISE_FROM_NADIR",
            "PSA_DEEP_NADIR_NO_RISE",
            "PSA_N_POST_ADT",
        )
    )


def _rate_report(df: pl.DataFrame, flag_cols: list[str]) -> pl.DataFrame:
    """Per-class counts plus the mean of each boolean flag, as a percentage."""
    aggs = [pl.len().alias("n_patients")]
    for col in flag_cols:
        if col in df.columns:
            aggs.append(
                (pl.col(col).cast(pl.Float64).mean() * 100)
                .round(1)
                .alias(f"pct_{col.lower()}")
            )
    return df.group_by("ADT_INTENT").agg(aggs).sort("n_patients", descending=True)


def report_against_metastasis_icd(labelled: pl.DataFrame) -> pl.DataFrame:
    """Cross-tab of the label against a coded secondary malignancy."""
    return _rate_report(
        labelled.with_columns(
            pl.col("FIRST_METASTASIS_ICD_DATE").is_not_null().alias("HAS_MET_ICD")
        ),
        ["HAS_MET_ICD"],
    )


def report_survival(labelled: pl.DataFrame) -> pl.DataFrame:
    """Death rate and median follow-up by class -- the primary sanity check.

    Returns an empty frame carrying the full schema when no `DEATH` column is
    present, so callers can filter on `ADT_INTENT` and get no rows rather than
    a ColumnNotFoundError.
    """
    if "DEATH" not in labelled.columns:
        return pl.DataFrame(
            schema={
                "ADT_INTENT": pl.Utf8,
                "n_patients": pl.UInt32,
                "pct_died": pl.Float64,
                "median_followup_days": pl.Float64,
                "median_adt_span_days": pl.Float64,
            }
        )
    return labelled.group_by("ADT_INTENT").agg(
        pl.len().alias("n_patients"),
        (pl.col("DEATH").cast(pl.Float64).mean() * 100).round(1).alias("pct_died"),
        pl.col("FOLLOWUP_DAYS_FROM_ADT").median().round(0).alias("median_followup_days"),
        pl.col("ADT_SPAN_DAYS").median().round(0).alias("median_adt_span_days"),
    ).sort("n_patients", descending=True)


def report_by_adt_start_year(labelled: pl.DataFrame) -> pl.DataFrame:
    """Class mix by calendar year of ADT start, to expose ARPI-era drift."""
    return (
        labelled.with_columns(pl.col("ADT_FIRST_DATE").dt.year().alias("adt_start_year"))
        .group_by(["adt_start_year", "ADT_INTENT"])
        .agg(pl.len().alias("n_patients"))
        .sort(["adt_start_year", "ADT_INTENT"])
    )


def report_gap_sensitivity(
    meds: pl.DataFrame,
    follow_up: pl.DataFrame | None,
    thresholds: tuple[int, ...] = (180, 270, 365),
) -> pl.DataFrame:
    """Class counts under several episode-gap thresholds.

    No end-date column exists to validate the threshold against, so its
    influence on the label is reported rather than assumed negligible.
    """
    frames = []
    for threshold in thresholds:
        labelled = classify_adt_intent(
            meds, follow_up=follow_up, gap_threshold_days=threshold
        )
        frames.append(
            labelled.group_by("ADT_INTENT")
            .agg(pl.len().alias("n_patients"))
            .with_columns(pl.lit(threshold).alias("gap_threshold_days"))
        )
    return pl.concat(frames).sort(["gap_threshold_days", "ADT_INTENT"])


def report_contradictions(labelled: pl.DataFrame) -> pl.DataFrame:
    """Excluded patients who nonetheless carry a metastasis code.

    Under an exclusion framing these are the errors that actually cost
    something: a patient dropped from the cohort who had coded metastatic
    disease. Retained patients without a code are not errors -- retention is
    the default and needs no evidence -- so only this direction is reported.
    These are the right sample for manual review.
    """
    if "FIRST_METASTASIS_ICD_DATE" not in labelled.columns:
        return pl.DataFrame()
    cols = [
        c
        for c in (
            ID_COL,
            "ADT_INTENT",
            "IS_LOCALIZED_ADJUVANT",
            "ADT_EXCLUSION_REASON",
            "ADT_SPAN_DAYS",
            "ADT_N_EPISODES",
            "FOLLOWUP_DAYS_FROM_ADT",
            "FIRST_METASTASIS_ICD_DATE",
        )
        if c in labelled.columns
    ]
    return labelled.filter(
        pl.col("IS_LOCALIZED_ADJUVANT")
        & pl.col("FIRST_METASTASIS_ICD_DATE").is_not_null()
    ).select(cols)


# ---------------------------------------------------------------------------
# Cross-reference: stage / metastatic burden from clinical_text_embedding_project
# ---------------------------------------------------------------------------
# `cancer_stage_df.csv.gz` and `met_burden_df.csv.gz` are written by that
# project's generate_all_non_text_covariates.py (build_cancer_stage_df /
# build_met_burden_df) into its FEATURE_PATH. Both key on DFCI_MRN, the same
# ID this module uses, so they join directly. They are read here as flat CSVs
# rather than by importing that project, which keeps this repo free of a
# cross-project dependency and a cluster-only config import.
#
# What each one is, and what it can and cannot adjudicate:
#
#   CANCER_STAGE  -- curated major stage {I, II, III, IV} from the CANCER
#     ANNOTATIONS parquets, normalized by shared.stages.normalize_stage
#     (substages collapsed, in-situ dropped to null). This is stage AT
#     DIAGNOSIS. It is the most direct external read of the thing the label
#     infers, but the mapping to intent is ASYMMETRIC:
#       * stage IV + LOCALIZED_ADJUVANT is a genuine contradiction -- de novo
#         metastatic disease is not treated with a completed adjuvant course.
#       * stage I-III + METASTATIC is NOT an error. Most metastatic ADT in
#         this cohort follows progression from initially localized disease,
#         so that cell is the expected majority, not a misclassification.
#     Only the first direction is scored as a contradiction below.
#
#   N_MET_SITES / MET_SITE_{group} -- count and indicators of distinct
#     metastatic organ groups from C77-C79 ICD codes, pre-index relative to
#     that project's anchor. This overlaps the C77-C79 signal already built by
#     compute_first_metastasis_icd_date, but adds BURDEN (how many distinct
#     organ systems) rather than mere presence, which is what separates
#     oligometastatic from widely disseminated disease.
#
# Anchor caveat: met burden is pre-index on the embedding project's treatment
# anchor, which is derived from its own cohort definition. It is not
# guaranteed to be the same date as this repo's TREATMENT_ANCHOR_DATE. Treat
# N_MET_SITES as "burden coded around the start of systemic therapy", not as a
# quantity measured at exactly this module's anchor.

STAGE_ORDER = ("I", "II", "III", "IV")


def load_stage_reference(path: str) -> pl.DataFrame:
    """Per-patient major stage from the embedding project's cancer_stage_df.

    Reads the raw `CANCER_STAGE` column that build_cancer_stage_df now always
    emits. Falls back to reconstructing from the drop-first one-hot columns
    (`CANCER_STAGE_II/III/IV`, all-zero meaning the omitted stage I reference
    category) so older files still resolve, mirroring
    shared.stages.load_stage_map.
    """
    stage_df = pl.read_csv(path)
    ids = (
        pl.col("DFCI_MRN")
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
        .alias(ID_COL)
    )

    if "CANCER_STAGE" in stage_df.columns:
        stage = stage_df.select(ids, pl.col("CANCER_STAGE").cast(pl.Utf8))
    else:
        dummies = [c for c in stage_df.columns if c.startswith("CANCER_STAGE_")]
        if not dummies:
            return pl.DataFrame(schema={ID_COL: pl.Int64, "CANCER_STAGE": pl.Utf8})
        expr = pl.lit("I")
        for col in dummies:
            expr = (
                pl.when(pl.col(col).cast(pl.Int8, strict=False).fill_null(0) == 1)
                .then(pl.lit(col.removeprefix("CANCER_STAGE_")))
                .otherwise(expr)
            )
        stage = stage_df.select(ids, expr.alias("CANCER_STAGE"))

    return (
        stage.filter(pl.col("CANCER_STAGE").is_in(list(STAGE_ORDER)))
        .unique(subset=ID_COL, keep="first")
        .with_columns((pl.col("CANCER_STAGE") == "IV").alias("IS_STAGE_IV"))
    )


def load_met_burden_reference(path: str) -> pl.DataFrame:
    """Per-patient metastatic burden from the embedding project's met_burden_df.

    Keeps `N_MET_SITES` and every `MET_SITE_*` indicator present in the file.
    That file is zero-filled to its whole cohort by construction, so a 0 there
    means "no pre-index metastatic ICD code", not "unobserved" -- but only for
    patients that cohort contains. Patients absent from the file join to null
    and are excluded from the rate denominators below rather than counted as
    zero-burden.
    """
    burden = pl.read_csv(path)
    site_cols = [c for c in burden.columns if c.startswith("MET_SITE_")]
    keep = (["N_MET_SITES"] if "N_MET_SITES" in burden.columns else []) + site_cols
    if not keep:
        return pl.DataFrame(schema={ID_COL: pl.Int64})
    return burden.select(
        pl.col("DFCI_MRN")
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
        .alias(ID_COL),
        *[pl.col(c).cast(pl.Int32, strict=False) for c in keep],
    ).unique(subset=ID_COL, keep="first")


def report_against_stage(labelled: pl.DataFrame) -> pl.DataFrame:
    """Label x curated stage-at-diagnosis cross-tab, with coverage.

    Restricted to patients the stage file actually covers, so the percentages
    are not diluted by absent rows.
    """
    if "CANCER_STAGE" not in labelled.columns:
        return pl.DataFrame()
    return (
        labelled.filter(pl.col("CANCER_STAGE").is_not_null())
        .group_by(["ADT_INTENT", "CANCER_STAGE"])
        .agg(pl.len().alias("n_patients"))
        .sort(["ADT_INTENT", "CANCER_STAGE"])
    )


def report_against_met_burden(labelled: pl.DataFrame) -> pl.DataFrame:
    """Metastatic organ-group burden by label.

    The expected pattern is a higher mean/median site count in METASTATIC and
    a much larger zero-burden share in LOCALIZED_ADJUVANT. A LOCALIZED_ADJUVANT
    group with substantial multi-site burden means the exclusion is dropping
    patients with documented disseminated disease.
    """
    if "N_MET_SITES" not in labelled.columns:
        return pl.DataFrame()
    covered = labelled.filter(pl.col("N_MET_SITES").is_not_null())
    if covered.height == 0:
        return pl.DataFrame()
    return (
        covered.group_by("ADT_INTENT")
        .agg(
            pl.len().alias("n_covered"),
            pl.col("N_MET_SITES").mean().round(2).alias("mean_met_sites"),
            pl.col("N_MET_SITES").median().alias("median_met_sites"),
            ((pl.col("N_MET_SITES") == 0).cast(pl.Float64).mean() * 100)
            .round(1)
            .alias("pct_zero_sites"),
            ((pl.col("N_MET_SITES") >= 2).cast(pl.Float64).mean() * 100)
            .round(1)
            .alias("pct_multisite"),
        )
        .sort("n_covered", descending=True)
    )


def report_met_site_pattern(labelled: pl.DataFrame) -> pl.DataFrame:
    """Per-organ-group involvement rate by label, long-form.

    Bone-predominant disease with little visceral involvement is the classic
    metastatic-prostate pattern; a liver/brain-heavy signal in a group is worth
    inspecting, since visceral spread skews toward aggressive-variant disease.
    """
    site_cols = [c for c in labelled.columns if c.startswith("MET_SITE_")]
    if not site_cols or "N_MET_SITES" not in labelled.columns:
        return pl.DataFrame()
    covered = labelled.filter(pl.col("N_MET_SITES").is_not_null())
    if covered.height == 0:
        return pl.DataFrame()
    frames = []
    for col in site_cols:
        frames.append(
            covered.group_by("ADT_INTENT")
            .agg(
                (pl.col(col).cast(pl.Float64).mean() * 100)
                .round(1)
                .alias("pct_involved")
            )
            .with_columns(pl.lit(col.removeprefix("MET_SITE_")).alias("met_site"))
        )
    return pl.concat(frames).select("met_site", "ADT_INTENT", "pct_involved").sort(
        ["met_site", "ADT_INTENT"]
    )


def report_stage_contradictions(labelled: pl.DataFrame) -> pl.DataFrame:
    """Excluded patients carrying curated stage IV at diagnosis.

    The sharpest external contradiction available: de novo stage IV disease is
    not treated with a completed adjuvant course, so any patient in this frame
    is a probable false exclusion. Only this direction is scored -- stage I-III
    patients labelled METASTATIC are the expected progression pathway, not
    errors (see the asymmetry note above).
    """
    if "IS_STAGE_IV" not in labelled.columns:
        return pl.DataFrame()
    cols = [
        c
        for c in (
            ID_COL,
            "ADT_INTENT",
            "IS_LOCALIZED_ADJUVANT",
            "ADT_EXCLUSION_REASON",
            "CANCER_STAGE",
            "N_MET_SITES",
            "ADT_SPAN_DAYS",
            "ADT_N_EPISODES",
            "FOLLOWUP_DAYS_FROM_ADT",
        )
        if c in labelled.columns
    ]
    return labelled.filter(
        pl.col("IS_LOCALIZED_ADJUVANT") & pl.col("IS_STAGE_IV").fill_null(False)
    ).select(cols)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate medication-only ADT intent labels."
    )
    parser.add_argument("--medications-path", required=True)
    parser.add_argument(
        "--icd-path",
        default=None,
        help="prostate_icd_data.csv (exploded, one ICD code per row).",
    )
    parser.add_argument(
        "--labs-path",
        default=None,
        help="longitudinal_prediction_data.csv (LAB_NAME/LAB_VALUE/t_lab).",
    )
    parser.add_argument(
        "--patient-status-path",
        default=None,
        help="CSV/Parquet with DFCI_MRN, FOLLOW_UP_END_DATE and optionally DEATH.",
    )
    parser.add_argument(
        "--cancer-stage-path",
        default=None,
        help=(
            "cancer_stage_df.csv.gz from clinical_text_embedding_project's "
            "generate_all_non_text_covariates.py (its FEATURE_PATH). Adds "
            "curated stage-at-diagnosis as a cross-reference."
        ),
    )
    parser.add_argument(
        "--met-burden-path",
        default=None,
        help=(
            "met_burden_df.csv.gz from the same project. Adds N_MET_SITES and "
            "per-organ-group metastatic indicators as a cross-reference."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gap-threshold-days", type=int, default=GAP_THRESHOLD_DAYS)
    args = parser.parse_args()

    from data_preprocessing_common.oncdrs_sources import scan_source

    meds = scan_source(args.medications_path).collect()

    follow_up = None
    if args.patient_status_path:
        follow_up = scan_source(args.patient_status_path).collect().with_columns(
            parse_mixed_datetime_expr("FOLLOW_UP_END_DATE").alias("FOLLOW_UP_END_DATE")
        )

    labelled = classify_adt_intent(
        meds, follow_up=follow_up, gap_threshold_days=args.gap_threshold_days
    )

    if follow_up is not None and "DEATH" in follow_up.columns:
        labelled = labelled.join(
            follow_up.select(
                pl.col(ID_COL)
                .cast(pl.Float64, strict=False)
                .cast(pl.Int64, strict=False)
                .alias(ID_COL),
                pl.col("DEATH").cast(pl.Int64, strict=False),
            ),
            on=ID_COL,
            how="left",
        )

    if args.icd_path:
        met = compute_first_metastasis_icd_date(pl.read_csv(args.icd_path))
        labelled = labelled.join(met, on=ID_COL, how="left")

    if args.labs_path:
        psa = compute_psa_nadir_features(pl.read_csv(args.labs_path))
        labelled = labelled.join(psa, on=ID_COL, how="left")

    if args.cancer_stage_path:
        stage_ref = load_stage_reference(args.cancer_stage_path)
        labelled = labelled.join(stage_ref, on=ID_COL, how="left")

    if args.met_burden_path:
        burden_ref = load_met_burden_reference(args.met_burden_path)
        labelled = labelled.join(burden_ref, on=ID_COL, how="left")

    print("\n=== Class counts ===")
    print(
        labelled.group_by("ADT_INTENT")
        .agg(pl.len().alias("n_patients"))
        .sort("n_patients", descending=True)
    )

    if "HAS_POSITIVE_METASTATIC_EVIDENCE" in labelled.columns:
        retained = labelled.filter(~pl.col("IS_LOCALIZED_ADJUVANT"))
        if retained.height:
            n_ev = retained.filter(
                pl.col("HAS_POSITIVE_METASTATIC_EVIDENCE")
            ).height
            print("\n=== Metastatic-by-default audit ===")
            print(
                f"retained (assumed METASTATIC): {retained.height:,}\n"
                f"  affirmatively evidenced    : {n_ev:,} "
                f"({100 * n_ev / retained.height:.1f}%)\n"
                f"  by assumption only         : {retained.height - n_ev:,} "
                f"({100 * (retained.height - n_ev) / retained.height:.1f}%)"
            )

    print("\n=== Exclusion decision ===")
    print(
        labelled.group_by("IS_LOCALIZED_ADJUVANT")
        .agg(pl.len().alias("n_patients"))
        .sort("IS_LOCALIZED_ADJUVANT")
    )

    print("\n=== Reason breakdown (why each patient was kept or excluded) ===")
    print(
        labelled.group_by(["ADT_EXCLUSION_REASON", "ADT_INTENT"])
        .agg(pl.len().alias("n_patients"))
        .sort("n_patients", descending=True)
    )

    if "FIRST_METASTASIS_ICD_DATE" in labelled.columns:
        print("\n=== vs. metastasis ICD (C77-C79, C7B) ===")
        print(report_against_metastasis_icd(labelled))
        contradictions = report_contradictions(labelled)
        print(
            f"\n=== Contradictions: {contradictions.height} adjuvant-labelled "
            f"patients carry a metastasis code ==="
        )
        print(contradictions.head(20))

    if "PSA_DEEP_NADIR_NO_RISE" in labelled.columns:
        print("\n=== vs. PSA trajectory ===")
        print(_rate_report(labelled, ["PSA_DEEP_NADIR_NO_RISE"]))

    if "CANCER_STAGE" in labelled.columns:
        n_cov = labelled.filter(pl.col("CANCER_STAGE").is_not_null()).height
        print(
            f"\n=== vs. curated stage at diagnosis "
            f"({n_cov:,}/{labelled.height:,} covered) ==="
        )
        print(report_against_stage(labelled))
        print(
            "  NOTE: stage I-III + METASTATIC is the expected progression\n"
            "  pathway, not an error. Only stage IV + LOCALIZED_ADJUVANT is\n"
            "  scored as a contradiction."
        )
        stage_bad = report_stage_contradictions(labelled)
        print(
            f"\n=== Contradictions: {stage_bad.height} excluded patients "
            f"carry curated stage IV ==="
        )
        if stage_bad.height:
            print(stage_bad.head(20))

    if "N_MET_SITES" in labelled.columns:
        n_cov = labelled.filter(pl.col("N_MET_SITES").is_not_null()).height
        print(
            f"\n=== vs. metastatic burden "
            f"({n_cov:,}/{labelled.height:,} covered) ==="
        )
        print(report_against_met_burden(labelled))
        pattern = report_met_site_pattern(labelled)
        if pattern.height:
            print("\n=== Metastatic site pattern by class (% involved) ===")
            print(pattern)

    survival = report_survival(labelled)
    if survival.height:
        # The label is binary, so this and the excluded/retained split below
        # are the same contrast; both are printed because the class names are
        # what previously written files carry.
        print("\n=== Survival by class (descriptive) ===")
        print(survival)
        met = survival.filter(pl.col("ADT_INTENT") == INTENT_METASTATIC)
        loc = survival.filter(pl.col("ADT_INTENT") == INTENT_LOCALIZED)
        if met.height and loc.height:
            if loc["pct_died"][0] >= met["pct_died"][0]:
                print(
                    "\nWARNING: LOCALIZED_ADJUVANT does not show better survival "
                    "than METASTATIC. The label is not separating the "
                    "populations -- do not use it downstream until resolved."
                )

    # The primary go/no-go under an exclusion framing: the contrast that
    # matters is between the patients the filter drops and the ones it keeps,
    # not between the two descriptive sub-classes of the retained group.
    if "DEATH" in labelled.columns:
        split = (
            labelled.group_by("IS_LOCALIZED_ADJUVANT")
            .agg(
                pl.len().alias("n_patients"),
                (pl.col("DEATH").cast(pl.Float64).mean() * 100)
                .round(1)
                .alias("pct_died"),
            )
            .sort("IS_LOCALIZED_ADJUVANT")
        )
        print("\n=== Survival: excluded vs retained (primary go/no-go) ===")
        print(split)
        excl = split.filter(pl.col("IS_LOCALIZED_ADJUVANT"))
        keep = split.filter(~pl.col("IS_LOCALIZED_ADJUVANT"))
        if excl.height and keep.height:
            if excl["pct_died"][0] >= keep["pct_died"][0]:
                print(
                    "\nWARNING: the excluded (adjuvant) group does not show "
                    "better survival than the retained group. The exclusion is "
                    "not removing the population it claims to -- do not use it "
                    "downstream until resolved."
                )

    print("\n=== Class mix by ADT start year (ARPI-era drift) ===")
    print(report_by_adt_start_year(labelled))

    print("\n=== Gap-threshold sensitivity ===")
    print(report_gap_sensitivity(meds, follow_up))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        labelled.write_csv(os.path.join(args.output_dir, "adt_intent_validation.csv"))
        print(f"\nWrote per-patient validation table to {args.output_dir}")


if __name__ == "__main__":
    main()
