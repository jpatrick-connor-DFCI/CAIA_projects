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
    """Patients labelled adjuvant who nonetheless carry a metastasis code.

    These are the label's hard errors and the right sample for manual review.
    """
    if "FIRST_METASTASIS_ICD_DATE" not in labelled.columns:
        return pl.DataFrame()
    cols = [
        c
        for c in (
            ID_COL,
            "ADT_INTENT",
            "ADT_INTENT_REASON",
            "ADT_SPAN_DAYS",
            "ADT_N_EPISODES",
            "FOLLOWUP_DAYS_FROM_ADT",
            "FIRST_METASTASIS_ICD_DATE",
        )
        if c in labelled.columns
    ]
    return labelled.filter(
        (pl.col("ADT_INTENT") == INTENT_LOCALIZED)
        & pl.col("FIRST_METASTASIS_ICD_DATE").is_not_null()
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

    print("\n=== Class counts ===")
    print(
        labelled.group_by("ADT_INTENT")
        .agg(pl.len().alias("n_patients"))
        .sort("n_patients", descending=True)
    )

    print("\n=== Reason breakdown ===")
    print(
        labelled.group_by(["ADT_INTENT", "ADT_INTENT_REASON"])
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

    survival = report_survival(labelled)
    if survival.height:
        print("\n=== Survival by class (primary go/no-go) ===")
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
