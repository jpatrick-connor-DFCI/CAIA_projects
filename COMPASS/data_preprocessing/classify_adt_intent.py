"""
Script Name: classify_adt_intent.py

Description:
Per-patient ADT treatment-intent label derived from medication history alone.

The COMPASS pipeline reduces each patient's whole ADT history to a single date
(`TREATMENT_ANCHOR_DATE`, the earliest `MED_START_DT` in `ADT_ANCHOR_MEDS`).
That collapses two clinically opposite populations into one cohort:

  * localized/adjuvant ADT -- 6-24 months of a GnRH agonist alongside definitive
    local therapy, then a permanent stop. Most such patients are cured.
  * metastatic ADT -- indefinite, continuous castration, escalated with an ARPI
    or a taxane, ending in castration resistance and death.

Mixing them biases every downstream survival model, because the adjuvant group
contributes long event-free follow-up unrelated to the mCRPC/NEPC biology the
models target. This module emits `ADT_INTENT` in {METASTATIC,
LOCALIZED_ADJUVANT, INDETERMINATE} plus the features and the rule that produced
it, so any label can be audited back to its evidence.

Medication-only by design
-------------------------
The label uses nothing but MEDICATIONS, so it transfers to sites without
curated staging -- and in this repo there is no stage/TNM, radiation, or
surgery table anyway. ICD C77-C79 and PSA are deliberately NOT inputs; they are
held out as an independent reference standard by `validate_adt_intent.py`.

Two signals carry the label. Escalation is the most specific: taxanes,
radium-223, sipuleucel-T and estramustine are not given for localized disease,
so exposure is near-definitive evidence of advanced disease. Duration is the
one with the best face validity: adjuvant ADT is protocol-limited, metastatic
ADT is lifelong.

Why duration is reconstructed rather than read
----------------------------------------------
`MEDICATIONS` is projected to exactly three columns -- `DFCI_MRN`,
`NCI_PREFERRED_MED_NM`, `MED_START_DT` (see `oncdrs_dedup.DATA_COLUMNS`). There
is no end date, no dose, no route, no days-supply and no formulation, so
exposure has to be inferred from the *sequence* of start dates: consecutive
refills closer together than `GAP_THRESHOLD_DAYS` belong to one on-treatment
episode. The threshold exceeds the longest depot formulation (6 months) plus a
realistic refill delay, so a late refill does not fracture continuous therapy,
while a genuine years-later re-challenge still splits.

Two guards keep that reconstruction honest, and both are load-bearing:

  1. `apply_dedup` collapses exact duplicate (MRN, drug, date) rows, so record
     counts are NOT administration counts. Duration therefore rests on span and
     gaps; `ADT_N_RECORDS` is reported for audit but never classifies.
  2. A patient whose first ADT falls shortly before the data cutoff looks
     exactly like one who stopped after a short course. Anyone with less than
     `MIN_FOLLOWUP_DAYS` of observation after ADT start is forced to
     INDETERMINATE and can never be labelled LOCALIZED_ADJUVANT. Without this
     the adjuvant class silently fills with recently-treated metastatic
     patients.

This module is purely additive: it changes no existing constant, cohort rule,
or pipeline output.
"""

from __future__ import annotations

import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (  # noqa: E402
    ADT_ANCHOR_MEDS,
    ID_COL,
    parse_mixed_datetime_expr,
)

# ---------------------------------------------------------------------------
# Escalation tiers.
#
# Defined independently of ARPI_ANCHOR_MEDS on purpose: that set bundles ARPIs
# with taxanes for anchoring, and two tests assert its exact contents and its
# cross-file identity. Intent classification needs the two tiers separated,
# because they carry very different evidential weight.
# ---------------------------------------------------------------------------

# Not given for localized prostate cancer under any current practice pattern.
# Exposure at any time is treated as definitive evidence of advanced disease.
#
# Note on PARPis and platinums: both are cohort *exclusions* upstream
# (PARPi exposure at all; platinum before prostate diagnosis), so within the
# COMPASS cohort PARPi patients are absent and platinum is the modelled
# endpoint rather than a predictor. They are deliberately omitted here -- see
# module docstring of compile_COMPASS_cohort_data.py. Classifying on the
# endpoint would leak it into the label.
DEFINITIVE_METASTATIC_MEDS = {
    "DOCETAXEL",
    "CABAZITAXEL",
    "MITOXANTRONE HYDROCHLORIDE",
    "RADIUM RA 223 DICHLORIDE",
    "SIPULEUCEL-T",
    "ESTRAMUSTINE",
    "ESTRAMUSTINE PHOSPHATE SODIUM",
}

# Strong but not definitive. Apalutamide (SPARTAN/TITAN) and darolutamide
# (ARAMIS) are approved for NON-metastatic CRPC, and since roughly 2023 ARPIs
# are increasingly used in high-risk localized disease. Because that drift makes
# the drug uninformative about metastatic status, ARPI exposure does not
# classify at all: it is recorded as HAS_ARPI / FIRST_ARPI_DATE for audit and
# never reads on the label. The validation harness stratifies by calendar year
# so the era drift is measured rather than assumed away.
ARPI_METASTATIC_MEDS = {
    "ABIRATERONE ACETATE",
    "ENZALUTAMIDE",
    "APALUTAMIDE",
    "DAROLUTAMIDE",
}

# Suggestive of bone metastases, but also standard treatment for ADT-induced
# osteoporosis in localized disease. Recorded as a feature for audit and never
# used to classify on its own.
BONE_AGENT_MEDS = {
    "DENOSUMAB",
    "ZOLEDRONIC ACID",
    "PAMIDRONATE DISODIUM",
}

# First-generation antiandrogens. A short course starting at or just before a
# GnRH agonist is flare prophylaxis and is intent-neutral.
ANTIANDROGEN_MEDS = {"BICALUTAMIDE", "FLUTAMIDE", "NILUTAMIDE"}

# GnRH antagonists. RELUGOLIX is intentionally absent from ADT_ANCHOR_MEDS
# upstream (a test asserts this), but it is real ADT, so intent features count
# it as exposure. ADT_ANCHOR_MEDS itself is left untouched.
GNRH_ANTAGONIST_MEDS = {"DEGARELIX", "RELUGOLIX"}

# Every drug that counts as androgen deprivation for duration purposes.
ADT_EXPOSURE_MEDS = set(ADT_ANCHOR_MEDS) | GNRH_ANTAGONIST_MEDS

# Injectable/implant depot agents. Duration is measured on these alone: oral
# antiandrogens have very different refill cadence and are frequently dispensed
# as short flare-prophylaxis courses.
DEPOT_ADT_MEDS = ADT_EXPOSURE_MEDS - ANTIANDROGEN_MEDS

# Drugs that may set ADT_FIRST_DATE. Deliberately identical to ADT_ANCHOR_MEDS
# -- and NOT to DEPOT_ADT_MEDS -- because ADT_FIRST_DATE is cross-checked
# against TREATMENT_ANCHOR_DATE, which the longitudinal builder derives from
# ADT_ANCHOR_MEDS. Any drug in one set and not the other shifts the two dates
# apart for real patients and trips that check: oral antiandrogens (in the
# anchor set, out of the depot set) move it earlier via the standard
# flare-prophylaxis lead-in, relugolix (in the depot set, out of the anchor
# set) moves it earlier for antagonist-first patients.
#
# Duration features stay on DEPOT_ADT_MEDS. That is a separate question --
# refill cadence, not treatment origin -- and relugolix must keep counting
# toward span even though it can never set the anchor.
ADT_FIRST_DATE_MEDS = set(ADT_ANCHOR_MEDS)

ALL_INTENT_MEDS = (
    ADT_EXPOSURE_MEDS
    | DEFINITIVE_METASTATIC_MEDS
    | ARPI_METASTATIC_MEDS
    | BONE_AGENT_MEDS
)

# ---------------------------------------------------------------------------
# Thresholds. All are CLI-overridable; the validation harness reports
# sensitivity of the label to GAP_THRESHOLD_DAYS at 180/270/365.
# ---------------------------------------------------------------------------

# Exceeds the longest depot formulation (6 months) plus refill slack.
GAP_THRESHOLD_DAYS = 270
# Minimum observation after ADT start before a short course may be read as a
# completed adjuvant course rather than administrative censoring.
MIN_FOLLOWUP_DAYS = 730
# Upper bound on a plausible adjuvant course (36 months; long-course adjuvant
# ADT in high-risk disease runs 24-36 months).
ADJUVANT_MAX_SPAN_DAYS = 1095
# A GnRH-agonist course sustained past this is not an adjuvant course.
SUSTAINED_ADT_DAYS = 1095
# ADT still being filled within this window of last contact counts as ongoing.
ONGOING_WINDOW_DAYS = 270
# Antiandrogen courses at or below this, starting near a depot start, are
# flare prophylaxis.
FLARE_PROPHYLAXIS_MAX_DAYS = 90
# How long before/after a depot start an antiandrogen may begin and still read
# as flare prophylaxis.
FLARE_PROPHYLAXIS_WINDOW_DAYS = 30

INTENT_METASTATIC = "METASTATIC"
INTENT_LOCALIZED = "LOCALIZED_ADJUVANT"
INTENT_INDETERMINATE = "INDETERMINATE"


def _normalized_med_name() -> pl.Expr:
    """Uppercase/strip `NCI_PREFERRED_MED_NM`, matching the pipeline idiom."""
    return (
        pl.col("NCI_PREFERRED_MED_NM")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.strip_chars()
    )


def load_medications_for_intent(meds: pl.DataFrame) -> pl.DataFrame:
    """Normalize names/dates and keep only medications intent classification uses.

    Deliberately NOT `load_medications_for_survival`: that helper filters to
    ADT + ARPI + platinum, which would drop mitoxantrone, sipuleucel-T,
    estramustine, the bone agents and relugolix, making the definitive tier
    look artificially empty.
    """
    out = meds.with_columns(
        _normalized_med_name().alias("NCI_PREFERRED_MED_NM"),
        parse_mixed_datetime_expr("MED_START_DT").alias("MED_START_DT"),
        pl.col(ID_COL)
        .cast(pl.Float64, strict=False)
        .cast(pl.Int64, strict=False)
        .alias(ID_COL),
    )
    return out.filter(
        pl.col("NCI_PREFERRED_MED_NM").is_in(sorted(ALL_INTENT_MEDS))
        & pl.col("MED_START_DT").is_not_null()
        & pl.col(ID_COL).is_not_null()
    )


def build_adt_episodes(
    meds: pl.DataFrame,
    gap_threshold_days: int = GAP_THRESHOLD_DAYS,
) -> pl.DataFrame:
    """Per-patient ADT exposure features from the sequence of start dates.

    `ADT_FIRST_DATE` is the earliest start in `ADT_FIRST_DATE_MEDS` (identical
    to the upstream `ADT_ANCHOR_MEDS`) so it agrees with `TREATMENT_ANCHOR_DATE`;
    every other feature here is measured over depot agents only.

    Episodes are maximal runs of depot starts separated by at most
    `gap_threshold_days`. Because same-day duplicate rows are collapsed
    upstream by `apply_dedup`, `ADT_N_RECORDS` is reported for audit only and
    never drives the label -- span and gaps do.
    """
    depot = meds.filter(
        pl.col("NCI_PREFERRED_MED_NM").is_in(sorted(DEPOT_ADT_MEDS))
    ).select(ID_COL, "NCI_PREFERRED_MED_NM", "MED_START_DT")

    if depot.height == 0:
        return pl.DataFrame(
            schema={
                ID_COL: pl.Int64,
                "ADT_FIRST_DATE": pl.Datetime,
                "ADT_LAST_DATE": pl.Datetime,
                "ADT_SPAN_DAYS": pl.Float64,
                "ADT_N_RECORDS": pl.UInt32,
                "ADT_N_EPISODES": pl.UInt32,
                "ADT_LONGEST_EPISODE_DAYS": pl.Float64,
                "ADT_N_DISTINCT_AGENTS": pl.UInt32,
                "ADT_USED_ANTAGONIST": pl.Boolean,
            }
        )

    # One row per (patient, day): duplicate same-day fills of different agents
    # must not read as a shorter gap than truly elapsed.
    daily = depot.unique(subset=[ID_COL, "MED_START_DT"]).sort(
        [ID_COL, "MED_START_DT"]
    )

    # An episode starts at the first record and at every gap over threshold.
    # The cumulative sum of those breaks numbers the episodes within a patient.
    daily = daily.with_columns(
        (
            pl.col("MED_START_DT").diff().over(ID_COL).dt.total_days()
            > gap_threshold_days
        )
        .fill_null(False)
        .alias("_is_new_episode")
    ).with_columns(
        pl.col("_is_new_episode").cum_sum().over(ID_COL).alias("_episode_idx")
    )

    episode_spans = (
        daily.group_by([ID_COL, "_episode_idx"])
        .agg(
            pl.col("MED_START_DT").min().alias("_ep_start"),
            pl.col("MED_START_DT").max().alias("_ep_end"),
        )
        .with_columns(
            (pl.col("_ep_end") - pl.col("_ep_start"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("_ep_days")
        )
    )

    per_patient_episodes = episode_spans.group_by(ID_COL).agg(
        pl.col("_episode_idx").n_unique().cast(pl.UInt32).alias("ADT_N_EPISODES"),
        pl.col("_ep_days").max().alias("ADT_LONGEST_EPISODE_DAYS"),
    )

    # ADT_FIRST_DATE comes from the anchor set, not the depot set, so it lines
    # up with TREATMENT_ANCHOR_DATE downstream. Everything else here is a
    # duration feature and stays on depot agents.
    first_date = (
        meds.filter(
            pl.col("NCI_PREFERRED_MED_NM").is_in(sorted(ADT_FIRST_DATE_MEDS))
        )
        .group_by(ID_COL)
        .agg(pl.col("MED_START_DT").min().alias("ADT_FIRST_DATE"))
    )

    per_patient_records = depot.group_by(ID_COL).agg(
        pl.col("MED_START_DT").min().alias("_depot_first"),
        pl.col("MED_START_DT").max().alias("ADT_LAST_DATE"),
        pl.len().cast(pl.UInt32).alias("ADT_N_RECORDS"),
        pl.col("NCI_PREFERRED_MED_NM")
        .n_unique()
        .cast(pl.UInt32)
        .alias("ADT_N_DISTINCT_AGENTS"),
        pl.col("NCI_PREFERRED_MED_NM")
        .is_in(sorted(GNRH_ANTAGONIST_MEDS))
        .any()
        .alias("ADT_USED_ANTAGONIST"),
    )

    return (
        per_patient_records.join(per_patient_episodes, on=ID_COL, how="left")
        .join(first_date, on=ID_COL, how="left")
        # A patient whose only ADT is relugolix has depot records but nothing in
        # the anchor set. Falling back to their first depot date keeps span
        # measured over real exposure; such a patient has no TREATMENT_ANCHOR_DATE
        # upstream either, so no anchor comparison is affected.
        .with_columns(
            pl.coalesce("ADT_FIRST_DATE", "_depot_first").alias("ADT_FIRST_DATE")
        )
        .with_columns(
            (pl.col("ADT_LAST_DATE") - pl.col("ADT_FIRST_DATE"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("ADT_SPAN_DAYS")
        )
        .select(
            ID_COL,
            "ADT_FIRST_DATE",
            "ADT_LAST_DATE",
            "ADT_SPAN_DAYS",
            "ADT_N_RECORDS",
            "ADT_N_EPISODES",
            "ADT_LONGEST_EPISODE_DAYS",
            "ADT_N_DISTINCT_AGENTS",
            "ADT_USED_ANTAGONIST",
        )
    )


def compute_escalation_tier(meds: pl.DataFrame) -> pl.DataFrame:
    """Per-patient escalation exposure flags and first escalation dates."""
    def _exposed(med_set: set, alias: str) -> pl.Expr:
        return (
            pl.col("NCI_PREFERRED_MED_NM")
            .is_in(sorted(med_set))
            .any()
            .alias(alias)
        )

    def _first_date(med_set: set, alias: str) -> pl.Expr:
        return (
            pl.when(pl.col("NCI_PREFERRED_MED_NM").is_in(sorted(med_set)))
            .then(pl.col("MED_START_DT"))
            .otherwise(None)
            .min()
            .alias(alias)
        )

    return meds.group_by(ID_COL).agg(
        _exposed(DEFINITIVE_METASTATIC_MEDS, "HAS_DEFINITIVE_ESCALATION"),
        _exposed(ARPI_METASTATIC_MEDS, "HAS_ARPI"),
        _exposed(BONE_AGENT_MEDS, "HAS_BONE_AGENT"),
        _first_date(DEFINITIVE_METASTATIC_MEDS, "FIRST_DEFINITIVE_ESCALATION_DATE"),
        _first_date(ARPI_METASTATIC_MEDS, "FIRST_ARPI_DATE"),
    )


def compute_antiandrogen_pattern(
    meds: pl.DataFrame,
    flare_max_days: int = FLARE_PROPHYLAXIS_MAX_DAYS,
    flare_window_days: int = FLARE_PROPHYLAXIS_WINDOW_DAYS,
) -> pl.DataFrame:
    """Flag antiandrogen use that is pure flare prophylaxis.

    A short antiandrogen course beginning within `flare_window_days` of the
    first depot start is standard flare prophylaxis and says nothing about
    intent. Longer courses, or ones begun well after castration, suggest
    combined androgen blockade for advancing disease.
    """
    aa = meds.filter(
        pl.col("NCI_PREFERRED_MED_NM").is_in(sorted(ANTIANDROGEN_MEDS))
    )
    depot = meds.filter(pl.col("NCI_PREFERRED_MED_NM").is_in(sorted(DEPOT_ADT_MEDS)))

    if aa.height == 0:
        return pl.DataFrame(
            schema={
                ID_COL: pl.Int64,
                "AA_SPAN_DAYS": pl.Float64,
                "AA_ONLY_FLARE_PROPHYLAXIS": pl.Boolean,
            }
        )

    aa_summary = aa.group_by(ID_COL).agg(
        pl.col("MED_START_DT").min().alias("_aa_first"),
        pl.col("MED_START_DT").max().alias("_aa_last"),
    )
    depot_first = depot.group_by(ID_COL).agg(
        pl.col("MED_START_DT").min().alias("_depot_first")
    )

    joined = aa_summary.join(depot_first, on=ID_COL, how="left").with_columns(
        (pl.col("_aa_last") - pl.col("_aa_first"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("AA_SPAN_DAYS"),
        (pl.col("_aa_first") - pl.col("_depot_first"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("_aa_offset_days"),
    )

    return joined.with_columns(
        (
            (pl.col("AA_SPAN_DAYS") <= flare_max_days)
            & pl.col("_aa_offset_days").is_not_null()
            & (pl.col("_aa_offset_days").abs() <= flare_window_days)
        )
        .fill_null(False)
        .alias("AA_ONLY_FLARE_PROPHYLAXIS")
    ).select(ID_COL, "AA_SPAN_DAYS", "AA_ONLY_FLARE_PROPHYLAXIS")


def classify_adt_intent(
    meds: pl.DataFrame,
    follow_up: pl.DataFrame | None = None,
    gap_threshold_days: int = GAP_THRESHOLD_DAYS,
    min_followup_days: int = MIN_FOLLOWUP_DAYS,
    adjuvant_max_span_days: int = ADJUVANT_MAX_SPAN_DAYS,
) -> pl.DataFrame:
    """Label each patient's ADT treatment intent from medication history.

    `follow_up` is an optional frame of [DFCI_MRN, FOLLOW_UP_END_DATE] (death
    date, else last contact -- see `load_patient_status` /
    `FOLLOW_UP_END_DATE` in compile_COMPASS_cohort_data.py). It is only used to
    detect ongoing therapy and administrative censoring; when it is absent no
    patient can be labelled LOCALIZED_ADJUVANT, because a short observed course
    cannot then be distinguished from a truncated one.

    The label is framed as an *exclusion*: the question asked of each patient
    is "is there positive evidence this was a completed adjuvant course?", and
    only patients that clear that bar are set aside. Everything else is
    retained, so the burden of proof sits on removing a patient rather than on
    keeping one -- the conservative direction when the label feeds a cohort
    filter.

    Three columns come out of this:

      IS_LOCALIZED_ADJUVANT  bool  -- positive evidence of a completed course
      ADT_EXCLUSION_REASON   str   -- why a patient was, or was not, excluded
      ADT_INTENT             str   -- LOCALIZED_ADJUVANT / METASTATIC /
                                      INDETERMINATE, retained for continuity

    A patient is LOCALIZED_ADJUVANT only when every one of these holds:

      - no definitive escalation (taxane, radium-223, sipuleucel-T,
        estramustine, mitoxantrone) at any time
      - ADT span <= `adjuvant_max_span_days` (default 1095, i.e. 36 months)
      - a single episode -- a restart after a gap > `gap_threshold_days` is
        salvage for recurrence, not adjuvant therapy
      - therapy demonstrably stopped: no fill within ONGOING_WINDOW_DAYS of
        last contact
      - at least `min_followup_days` (default 730) observed after ADT start,
        so a truncated course cannot masquerade as a completed one

    Failing any one of them leaves the patient retained. ARPI exposure is
    recorded (HAS_ARPI, FIRST_ARPI_DATE) but is deliberately not part of the
    test: ARPIs are approved in non-metastatic CRPC and used in high-risk
    localized disease, so they neither prove nor disprove adjuvant intent.

    ADT_INTENT sub-divides the retained group into METASTATIC (positive
    evidence of advanced disease: escalation, sustained ADT over <= 2 episodes,
    or ongoing ADT past 2 years) and INDETERMINATE (retained, but no positive
    evidence either way). That split is descriptive only -- both are kept by an
    exclusion filter, and no downstream filter should depend on the boundary
    between them.
    """
    prepared = load_medications_for_intent(meds)
    episodes = build_adt_episodes(prepared, gap_threshold_days=gap_threshold_days)
    escalation = compute_escalation_tier(prepared)
    antiandrogen = compute_antiandrogen_pattern(prepared)

    df = (
        episodes.join(escalation, on=ID_COL, how="left")
        .join(antiandrogen, on=ID_COL, how="left")
        .with_columns(
            pl.col("HAS_DEFINITIVE_ESCALATION").fill_null(False),
            pl.col("HAS_ARPI").fill_null(False),
            pl.col("HAS_BONE_AGENT").fill_null(False),
            pl.col("AA_ONLY_FLARE_PROPHYLAXIS").fill_null(False),
        )
    )

    if follow_up is not None and follow_up.height > 0:
        fu = follow_up.select(
            pl.col(ID_COL)
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64, strict=False)
            .alias(ID_COL),
            pl.col("FOLLOW_UP_END_DATE").cast(pl.Datetime, strict=False),
        )
        df = df.join(fu, on=ID_COL, how="left")
    else:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Datetime).alias("FOLLOW_UP_END_DATE")
        )

    df = df.with_columns(
        (pl.col("FOLLOW_UP_END_DATE") - pl.col("ADT_FIRST_DATE"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("FOLLOWUP_DAYS_FROM_ADT"),
        (pl.col("FOLLOW_UP_END_DATE") - pl.col("ADT_LAST_DATE"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("_days_from_last_adt_to_end"),
    ).with_columns(
        (pl.col("_days_from_last_adt_to_end") <= ONGOING_WINDOW_DAYS)
        .fill_null(False)
        .alias("ADT_ONGOING_AT_LAST_CONTACT")
    )

    # Missing follow-up is treated as insufficient: it cannot rule censoring out.
    insufficient_followup = (
        pl.col("FOLLOWUP_DAYS_FROM_ADT").is_null()
        | (pl.col("FOLLOWUP_DAYS_FROM_ADT") < min_followup_days)
    )

    # ---------------------------------------------------------------------
    # The exclusion test. Framed positively -- each term is a thing that must
    # be TRUE for a patient to be set aside as a completed adjuvant course --
    # so that missing or ambiguous evidence retains the patient by default.
    # ARPI exposure is deliberately absent: it is approved in non-metastatic
    # CRPC and used in high-risk localized disease, so it neither proves nor
    # disproves adjuvant intent. It is still recorded for audit.
    # ---------------------------------------------------------------------
    fits_adjuvant_duration = pl.col("ADT_SPAN_DAYS") <= adjuvant_max_span_days
    single_episode = pl.col("ADT_N_EPISODES") <= 1
    therapy_stopped = ~pl.col("ADT_ONGOING_AT_LAST_CONTACT")
    no_escalation = ~pl.col("HAS_DEFINITIVE_ESCALATION")
    observed_long_enough = ~insufficient_followup

    is_localized_adjuvant = (
        no_escalation
        & fits_adjuvant_duration
        & single_episode
        & therapy_stopped
        & observed_long_enough
    ).fill_null(False)

    # Why a patient was kept. Ordered most- to least-decisive so the reason
    # names the strongest disqualifier rather than whichever is checked first.
    exclusion_reason = (
        pl.when(is_localized_adjuvant)
        .then(pl.lit("excluded_completed_adjuvant_course"))
        .when(pl.col("HAS_DEFINITIVE_ESCALATION"))
        .then(pl.lit("retained_definitive_escalation"))
        .when(~fits_adjuvant_duration)
        .then(pl.lit("retained_adt_span_too_long"))
        .when(~single_episode)
        .then(pl.lit("retained_multiple_episodes"))
        .when(~therapy_stopped)
        .then(pl.lit("retained_adt_ongoing"))
        .when(insufficient_followup)
        .then(pl.lit("retained_insufficient_followup"))
        .otherwise(pl.lit("retained_unresolved"))
        .alias("ADT_EXCLUSION_REASON")
    )

    # ADT_INTENT is descriptive: it sub-divides the retained group for the
    # validation reports. An exclusion filter uses IS_LOCALIZED_ADJUVANT and
    # must not depend on the METASTATIC / INDETERMINATE boundary.
    has_positive_metastatic_evidence = (
        pl.col("HAS_DEFINITIVE_ESCALATION")
        | (
            (pl.col("ADT_SPAN_DAYS") > SUSTAINED_ADT_DAYS)
            & (pl.col("ADT_N_EPISODES") <= 2)
        )
        | (
            pl.col("ADT_ONGOING_AT_LAST_CONTACT")
            & (pl.col("ADT_SPAN_DAYS") > 730)
        )
    ).fill_null(False)

    intent = (
        pl.when(is_localized_adjuvant)
        .then(pl.lit(INTENT_LOCALIZED))
        .when(has_positive_metastatic_evidence)
        .then(pl.lit(INTENT_METASTATIC))
        .otherwise(pl.lit(INTENT_INDETERMINATE))
        .alias("ADT_INTENT")
    )

    return df.with_columns(
        is_localized_adjuvant.alias("IS_LOCALIZED_ADJUVANT"),
        exclusion_reason,
        intent,
    ).drop("_days_from_last_adt_to_end")


def summarize_intent(labelled: pl.DataFrame) -> pl.DataFrame:
    """Class counts with shares, for the run log and quick sanity checks."""
    total = labelled.height
    return (
        labelled.group_by("ADT_INTENT")
        .agg(pl.len().alias("n_patients"))
        .with_columns(
            (pl.col("n_patients") / max(total, 1) * 100).round(1).alias("pct")
        )
        .sort("n_patients", descending=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify ADT treatment intent from medication history."
    )
    parser.add_argument(
        "--medications-path",
        required=True,
        help="Raw OncDRS MEDICATIONS table (CSV or merged Parquet).",
    )
    parser.add_argument(
        "--patient-status-path",
        default=None,
        help=(
            "Optional CSV/Parquet with DFCI_MRN and FOLLOW_UP_END_DATE. "
            "Without it no patient can be labelled LOCALIZED_ADJUVANT."
        ),
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--gap-threshold-days", type=int, default=GAP_THRESHOLD_DAYS)
    parser.add_argument("--min-followup-days", type=int, default=MIN_FOLLOWUP_DAYS)
    parser.add_argument(
        "--adjuvant-max-span-days", type=int, default=ADJUVANT_MAX_SPAN_DAYS
    )
    args = parser.parse_args()

    from data_preprocessing_common.oncdrs_sources import scan_source

    meds = scan_source(args.medications_path).collect()

    follow_up = None
    if args.patient_status_path:
        follow_up = scan_source(args.patient_status_path).collect()
        if "FOLLOW_UP_END_DATE" not in follow_up.columns:
            raise SystemExit(
                f"{args.patient_status_path} has no FOLLOW_UP_END_DATE column; "
                f"found {sorted(follow_up.columns)}"
            )
        follow_up = follow_up.with_columns(
            parse_mixed_datetime_expr("FOLLOW_UP_END_DATE").alias(
                "FOLLOW_UP_END_DATE"
            )
        )
    else:
        print(
            "WARNING: no --patient-status-path given. Administrative censoring "
            "cannot be ruled out, so every patient will fall to INDETERMINATE "
            "rather than LOCALIZED_ADJUVANT.",
            file=sys.stderr,
        )

    labelled = classify_adt_intent(
        meds,
        follow_up=follow_up,
        gap_threshold_days=args.gap_threshold_days,
        min_followup_days=args.min_followup_days,
        adjuvant_max_span_days=args.adjuvant_max_span_days,
    )

    labelled.write_csv(args.output_path)
    print(f"Wrote {labelled.height} ADT-intent labels to {args.output_path}")
    print(summarize_intent(labelled))


if __name__ == "__main__":
    main()
