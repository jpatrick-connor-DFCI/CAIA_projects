"""Tests for medication-only ADT treatment-intent classification.

The two guards these tests exist to protect (see classify_adt_intent module
docstring) are administrative censoring -- a truncated course must never be
read as a completed adjuvant course -- and upstream dedup, which collapses
same-day duplicate rows so record counts cannot stand in for duration.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import polars as pl

from COMPASS.data_preprocessing.classify_adt_intent import (
    ADT_ANCHOR_MEDS,
    ADT_FIRST_DATE_MEDS,
    ARPI_METASTATIC_MEDS,
    DEFINITIVE_METASTATIC_MEDS,
    INTENT_INDETERMINATE,
    INTENT_LOCALIZED,
    INTENT_METASTATIC,
    build_adt_episodes,
    classify_adt_intent,
    compute_antiandrogen_pattern,
    load_medications_for_intent,
    parse_mixed_datetime_expr,
)
from COMPASS.data_preprocessing.validate_adt_intent import (
    compute_first_metastasis_icd_date,
    compute_psa_nadir_features,
    compute_met_burden_at_adt,
    load_met_burden_reference,
    load_stage_max_around_adt,
    load_stage_nearest_adt,
    load_stage_reference,
    report_against_met_burden,
    report_stage_contradictions,
    report_stage_max,
)

ADT_START = datetime(2015, 1, 1)


def _meds(rows: list[tuple[int, str, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "DFCI_MRN": [r[0] for r in rows],
            "NCI_PREFERRED_MED_NM": [r[1] for r in rows],
            "MED_START_DT": [r[2].strftime("%Y-%m-%d") for r in rows],
        }
    )


def _depot_course(mrn: int, months: int, start: datetime = ADT_START) -> list[tuple]:
    """Quarterly leuprolide fills spanning `months` months."""
    return [
        (mrn, "LEUPROLIDE ACETATE", start + timedelta(days=90 * i))
        for i in range(months // 3 + 1)
    ]


def _follow_up(mrn: int, end: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {"DFCI_MRN": [mrn], "FOLLOW_UP_END_DATE": [end]}
    ).with_columns(pl.col("FOLLOW_UP_END_DATE").cast(pl.Datetime))


def _follow_up_many(mrns: list[int], end: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {"DFCI_MRN": mrns, "FOLLOW_UP_END_DATE": [end] * len(mrns)}
    ).with_columns(pl.col("FOLLOW_UP_END_DATE").cast(pl.Datetime))


def _intent(labelled: pl.DataFrame, mrn: int) -> str:
    return labelled.filter(pl.col("DFCI_MRN") == mrn)["ADT_INTENT"][0]


def _reason(labelled: pl.DataFrame, mrn: int) -> str:
    return labelled.filter(pl.col("DFCI_MRN") == mrn)["ADT_EXCLUSION_REASON"][0]


def _excluded(labelled: pl.DataFrame, mrn: int) -> bool:
    return labelled.filter(pl.col("DFCI_MRN") == mrn)["IS_LOCALIZED_ADJUVANT"][0]


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_definitive_escalation_is_metastatic():
    """Docetaxel after 8 months of ADT is definitive, whatever the duration."""
    meds = _meds(
        _depot_course(1, months=8)
        + [(1, "DOCETAXEL", ADT_START + timedelta(days=240))]
    )
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, ADT_START + timedelta(days=2000))
    )
    assert _intent(labelled, 1) == INTENT_METASTATIC
    assert _reason(labelled, 1) == "retained_definitive_escalation"


def test_arpi_alone_with_short_adt_does_not_force_metastatic():
    """ARPIs are approved in nmCRPC and high-risk localized disease, so an ARPI
    with only a short ADT course must not be read as definitive evidence."""
    meds = _meds(
        _depot_course(1, months=6)
        + [(1, "DAROLUTAMIDE", ADT_START + timedelta(days=30))]
    )
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, ADT_START + timedelta(days=2000))
    )
    assert _intent(labelled, 1) != INTENT_METASTATIC


def test_arpi_never_classifies_in_either_direction():
    """ARPI is not part of the exclusion test in either direction. A 24-month
    single course is judged on its duration alone, so adding an ARPI must not
    change the outcome -- under the old rules this patient was METASTATIC."""
    course = _depot_course(1, months=24)
    fu = _follow_up(1, ADT_START + timedelta(days=2500))

    without_arpi = classify_adt_intent(_meds(course), follow_up=fu)
    with_arpi = classify_adt_intent(
        _meds(course + [(1, "ENZALUTAMIDE", ADT_START + timedelta(days=400))]),
        follow_up=fu,
    )

    assert _intent(with_arpi, 1) == _intent(without_arpi, 1)
    assert _reason(with_arpi, 1) == _reason(without_arpi, 1)
    assert _intent(with_arpi, 1) == INTENT_LOCALIZED
    assert _reason(with_arpi, 1) == "excluded_completed_adjuvant_course"


def test_arpi_exposure_still_recorded_as_an_audit_feature():
    """ARPI stops driving the label but must remain in the output, so a
    downstream user can re-introduce it or audit the patients it would have
    reclassified."""
    labelled = classify_adt_intent(
        _meds(
            _depot_course(1, months=24)
            + [(1, "ENZALUTAMIDE", ADT_START + timedelta(days=400))]
        ),
        follow_up=_follow_up(1, ADT_START + timedelta(days=2500)),
    )
    assert labelled["HAS_ARPI"][0]
    assert labelled["FIRST_ARPI_DATE"][0] == ADT_START + timedelta(days=400)


def test_every_reason_states_whether_the_patient_was_kept():
    """ADT_EXCLUSION_REASON must read as a decision, so each value says plainly
    whether the patient was excluded or retained."""
    meds = _meds(
        _depot_course(1, months=24)
        + [(1, "ENZALUTAMIDE", ADT_START + timedelta(days=400))]
        + _depot_course(2, months=48)
    )
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up_many([1, 2], ADT_START + timedelta(days=2500))
    )
    for r in labelled["ADT_EXCLUSION_REASON"].to_list():
        assert r.startswith("excluded_") or r.startswith("retained_")
    # The reason must agree with the boolean it explains.
    disagree = labelled.filter(
        pl.col("IS_LOCALIZED_ADJUVANT")
        != pl.col("ADT_EXCLUSION_REASON").str.starts_with("excluded_")
    )
    assert disagree.height == 0


def test_bone_agent_alone_never_classifies():
    """Bone agents also treat ADT-induced osteoporosis in localized disease."""
    meds = _meds(
        _depot_course(1, months=12)
        + [(1, "DENOSUMAB", ADT_START + timedelta(days=60))]
    )
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, ADT_START + timedelta(days=2000))
    )
    assert _intent(labelled, 1) == INTENT_LOCALIZED
    assert labelled.filter(pl.col("DFCI_MRN") == 1)["HAS_BONE_AGENT"][0]


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


def test_continuous_five_year_adt_is_metastatic():
    meds = _meds(_depot_course(1, months=60))
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, ADT_START + timedelta(days=2200))
    )
    assert _intent(labelled, 1) == INTENT_METASTATIC
    assert _reason(labelled, 1) == "retained_adt_span_too_long"


def test_short_course_with_full_followup_is_adjuvant():
    """18 months of ADT, then a permanent stop, observed for years afterwards."""
    meds = _meds(_depot_course(1, months=18))
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, ADT_START + timedelta(days=2500))
    )
    assert _intent(labelled, 1) == INTENT_LOCALIZED
    assert _reason(labelled, 1) == "excluded_completed_adjuvant_course"


def test_ongoing_adt_at_last_contact_no_longer_classifies():
    """Still filling ADT at last contact is a metastatic pattern clinically, but
    the two-term rule does not test for it: a 30-month course is under the
    36-month bound and carries no escalation, so it is labelled adjuvant despite
    never having stopped. The signal survives only in the audit columns."""
    meds = _meds(_depot_course(1, months=30))
    last_fill = ADT_START + timedelta(days=90 * (30 // 3))
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, last_fill + timedelta(days=30))
    )
    row = labelled.filter(pl.col("DFCI_MRN") == 1)
    assert _intent(labelled, 1) == INTENT_LOCALIZED
    # Ongoing therapy is still measured, just not acted on.
    assert row["ADT_ONGOING_AT_LAST_CONTACT"][0]
    assert not row["ADJUVANT_THERAPY_STOPPED"][0]


# ---------------------------------------------------------------------------
# Administrative censoring: measured, no longer guarded against
# ---------------------------------------------------------------------------


def test_short_course_truncated_by_data_cutoff_is_now_labelled_adjuvant():
    """This is the cost of the two-term rule, pinned so it cannot regress
    silently. An 18-month course whose follow-up ends right at the last fill is
    indistinguishable from ongoing therapy -- it may well be a metastatic
    patient caught at the data cutoff -- but escalation and duration are the
    only tests, so it is excluded as a completed adjuvant course anyway.
    ADJUVANT_OBSERVED_LONG_ENOUGH is what flags it."""
    meds = _meds(_depot_course(1, months=18))
    last_fill = ADT_START + timedelta(days=90 * (18 // 3))
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, last_fill + timedelta(days=5))
    )
    row = labelled.filter(pl.col("DFCI_MRN") == 1)
    assert _intent(labelled, 1) == INTENT_LOCALIZED
    assert _excluded(labelled, 1)
    assert _reason(labelled, 1) == "excluded_completed_adjuvant_course"
    # The two dropped guards both still fire as columns.
    assert not row["ADJUVANT_OBSERVED_LONG_ENOUGH"][0]
    assert not row["ADJUVANT_THERAPY_STOPPED"][0]


def test_missing_follow_up_still_yields_adjuvant():
    """Follow-up no longer bears on the label at all: with follow_up=None the
    duration test alone decides, so an 18-month course is excluded even though
    censoring cannot be ruled out. Previously this was impossible by
    construction."""
    meds = _meds(_depot_course(1, months=18))
    labelled = classify_adt_intent(meds, follow_up=None)
    row = labelled.filter(pl.col("DFCI_MRN") == 1)
    assert _intent(labelled, 1) == INTENT_LOCALIZED
    assert _excluded(labelled, 1)
    # Unknown observation window reads as not-long-enough, as before.
    assert not row["ADJUVANT_OBSERVED_LONG_ENOUGH"][0]
    # Not metastatic-evidenced either -- absent follow-up is absent evidence.
    assert not labelled["HAS_POSITIVE_METASTATIC_EVIDENCE"][0]


# ---------------------------------------------------------------------------
# Guard 2: dedup / episode reconstruction
# ---------------------------------------------------------------------------


def test_same_day_duplicate_rows_do_not_change_span():
    """Upstream dedup collapses identical rows; duplicates that survive (e.g.
    two agents on one day) must not alter span or episode structure."""
    base = _depot_course(1, months=12)
    meds_plain = _meds(base)
    meds_dupes = _meds(base + base + [(1, "DEGARELIX", ADT_START)])

    span_plain = build_adt_episodes(load_medications_for_intent(meds_plain))
    span_dupes = build_adt_episodes(load_medications_for_intent(meds_dupes))

    assert span_plain["ADT_SPAN_DAYS"][0] == span_dupes["ADT_SPAN_DAYS"][0]
    assert span_plain["ADT_N_EPISODES"][0] == span_dupes["ADT_N_EPISODES"][0]


def test_two_courses_years_apart_split_into_episodes():
    """A re-challenge four years later is two episodes, not one long span."""
    rows = _depot_course(1, months=6) + _depot_course(
        1, months=6, start=ADT_START + timedelta(days=4 * 365)
    )
    episodes = build_adt_episodes(load_medications_for_intent(_meds(rows)))
    assert episodes["ADT_N_EPISODES"][0] == 2
    # Span still measures first-to-last; the episode count is what distinguishes
    # this from continuous therapy.
    assert episodes["ADT_SPAN_DAYS"][0] > 1095


def test_intermittent_adt_is_not_labelled_adjuvant():
    """Multiple separated courses are not a single adjuvant course."""
    rows = _depot_course(1, months=6) + _depot_course(
        1, months=6, start=ADT_START + timedelta(days=4 * 365)
    )
    labelled = classify_adt_intent(
        _meds(rows), follow_up=_follow_up(1, ADT_START + timedelta(days=7 * 365))
    )
    assert _intent(labelled, 1) != INTENT_LOCALIZED


def test_gap_threshold_is_configurable():
    """A 200-day gap is one episode at the default and two at a tighter one."""
    rows = [
        (1, "LEUPROLIDE ACETATE", ADT_START),
        (1, "LEUPROLIDE ACETATE", ADT_START + timedelta(days=200)),
    ]
    prepared = load_medications_for_intent(_meds(rows))
    assert build_adt_episodes(prepared, gap_threshold_days=270)["ADT_N_EPISODES"][0] == 1
    assert build_adt_episodes(prepared, gap_threshold_days=180)["ADT_N_EPISODES"][0] == 2


# ---------------------------------------------------------------------------
# Antiandrogen pattern
# ---------------------------------------------------------------------------


def test_short_antiandrogen_at_depot_start_is_flare_prophylaxis():
    rows = _depot_course(1, months=18) + [
        (1, "BICALUTAMIDE", ADT_START - timedelta(days=7)),
        (1, "BICALUTAMIDE", ADT_START + timedelta(days=20)),
    ]
    pattern = compute_antiandrogen_pattern(load_medications_for_intent(_meds(rows)))
    assert pattern["AA_ONLY_FLARE_PROPHYLAXIS"][0]


def test_long_running_antiandrogen_is_not_flare_prophylaxis():
    rows = _depot_course(1, months=24) + [
        (1, "BICALUTAMIDE", ADT_START),
        (1, "BICALUTAMIDE", ADT_START + timedelta(days=600)),
    ]
    pattern = compute_antiandrogen_pattern(load_medications_for_intent(_meds(rows)))
    assert not pattern["AA_ONLY_FLARE_PROPHYLAXIS"][0]


def test_antiandrogen_only_patient_has_no_depot_span():
    """Oral-antiandrogen-only patients have no depot exposure, so duration is
    undefined and they must not be labelled adjuvant on an empty span."""
    meds = _meds([(1, "BICALUTAMIDE", ADT_START)])
    episodes = build_adt_episodes(load_medications_for_intent(meds))
    assert episodes.height == 0


# ---------------------------------------------------------------------------
# Loader and constants
# ---------------------------------------------------------------------------


def test_loader_keeps_drugs_the_survival_loader_would_drop():
    """load_medications_for_survival filters to ADT+ARPI+platinum; the intent
    loader must retain the rest of the definitive tier and relugolix."""
    rows = [
        (1, "SIPULEUCEL-T", ADT_START),
        (2, "MITOXANTRONE HYDROCHLORIDE", ADT_START),
        (3, "ESTRAMUSTINE PHOSPHATE SODIUM", ADT_START),
        (4, "RELUGOLIX", ADT_START),
        (5, "DENOSUMAB", ADT_START),
    ]
    kept = set(
        load_medications_for_intent(_meds(rows))["NCI_PREFERRED_MED_NM"].to_list()
    )
    assert kept == {
        "SIPULEUCEL-T",
        "MITOXANTRONE HYDROCHLORIDE",
        "ESTRAMUSTINE PHOSPHATE SODIUM",
        "RELUGOLIX",
        "DENOSUMAB",
    }


def test_relugolix_counts_as_adt_exposure_without_mutating_anchor_set():
    """Relugolix is real ADT for duration purposes, but ADT_ANCHOR_MEDS -- which
    an upstream test asserts excludes it -- must stay untouched."""
    assert "RELUGOLIX" not in ADT_ANCHOR_MEDS
    meds = _meds(
        [
            (1, "RELUGOLIX", ADT_START + timedelta(days=30 * i))
            for i in range(13)
        ]
    )
    episodes = build_adt_episodes(load_medications_for_intent(meds))
    assert episodes.height == 1
    assert episodes["ADT_SPAN_DAYS"][0] == 360


def test_first_date_med_set_matches_the_upstream_anchor_set():
    """ADT_FIRST_DATE is cross-checked against TREATMENT_ANCHOR_DATE, which the
    longitudinal builder derives from ADT_ANCHOR_MEDS. If the sets drift apart
    the two dates diverge for real patients and the anchor guard fires."""
    assert ADT_FIRST_DATE_MEDS == set(ADT_ANCHOR_MEDS)


def test_antiandrogen_lead_in_sets_first_date():
    """The standard flare-prophylaxis bicalutamide lead-in precedes the first
    depot injection and is in the anchor set, so it -- not the injection --
    sets ADT_FIRST_DATE. This is the case that made 30% of patients mismatch."""
    meds = _meds(
        [
            (1, "BICALUTAMIDE", ADT_START - timedelta(days=14)),
            (1, "LEUPROLIDE ACETATE", ADT_START),
            (1, "LEUPROLIDE ACETATE", ADT_START + timedelta(days=180)),
        ]
    )
    episodes = build_adt_episodes(load_medications_for_intent(meds))
    assert episodes["ADT_FIRST_DATE"][0] == ADT_START - timedelta(days=14)
    # ADT_LAST_DATE stays a depot measurement.
    assert episodes["ADT_LAST_DATE"][0] == ADT_START + timedelta(days=180)


def test_relugolix_only_patient_keeps_span_without_an_anchor_drug():
    """Relugolix is out of the anchor set, so a relugolix-only patient has no
    anchor-set record; ADT_FIRST_DATE falls back to their first depot date so
    duration is still measured over real exposure."""
    meds = _meds(
        [(1, "RELUGOLIX", ADT_START + timedelta(days=30 * i)) for i in range(13)]
    )
    episodes = build_adt_episodes(load_medications_for_intent(meds))
    assert episodes["ADT_FIRST_DATE"][0] == ADT_START
    assert episodes["ADT_SPAN_DAYS"][0] == 360


def test_exclusion_requires_exactly_the_two_criteria():
    """Two criteria classify -- escalation and duration -- and only those two.
    Breaking either retains the patient; breaking any of the three audit-only
    terms does not. Both halves are asserted, because the second half is what
    distinguishes this rule from the earlier five-term one."""
    fu_end = ADT_START + timedelta(days=2500)

    # Baseline: 24-month single course, stopped, long follow-up -> excluded.
    base = _depot_course(1, months=24)
    labelled = classify_adt_intent(_meds(base), follow_up=_follow_up(1, fu_end))
    assert _excluded(labelled, 1)

    # --- The two terms that classify -------------------------------------

    # Break duration: 48 months exceeds the adjuvant bound.
    long_course = classify_adt_intent(
        _meds(_depot_course(1, months=48)), follow_up=_follow_up(1, fu_end)
    )
    assert not _excluded(long_course, 1)
    assert _reason(long_course, 1) == "retained_adt_span_too_long"

    # Break escalation: a taxane disqualifies regardless of duration.
    escalated = classify_adt_intent(
        _meds(base + [(1, "DOCETAXEL", ADT_START + timedelta(days=200))]),
        follow_up=_follow_up(1, fu_end),
    )
    assert not _excluded(escalated, 1)
    assert _reason(escalated, 1) == "retained_definitive_escalation"

    # --- The three that no longer classify -------------------------------

    # Multiple episodes: a second course after a >270d gap is salvage for
    # recurrence, but total span stays under the bound, so it is excluded.
    two_ep = classify_adt_intent(
        _meds(
            _depot_course(1, months=6)
            + [
                (1, "LEUPROLIDE ACETATE", ADT_START + timedelta(days=720 + 90 * i))
                for i in range(3)
            ]
        ),
        follow_up=_follow_up(1, fu_end),
    )
    assert _excluded(two_ep, 1)
    assert not two_ep.filter(pl.col("DFCI_MRN") == 1)["ADJUVANT_SINGLE_EPISODE"][0]

    # Short follow-up: not observed long enough to call the course complete,
    # yet excluded anyway.
    short_fu = classify_adt_intent(
        _meds(base), follow_up=_follow_up(1, ADT_START + timedelta(days=700))
    )
    assert _excluded(short_fu, 1)
    assert not short_fu.filter(pl.col("DFCI_MRN") == 1)[
        "ADJUVANT_OBSERVED_LONG_ENOUGH"
    ][0]

    # Ongoing therapy: last fill sits inside the ongoing window of last
    # contact, and it still does not retain the patient.
    last_fill = ADT_START + timedelta(days=90 * (24 // 3))
    ongoing = classify_adt_intent(
        _meds(base), follow_up=_follow_up(1, last_fill + timedelta(days=30))
    )
    assert _excluded(ongoing, 1)
    assert not ongoing.filter(pl.col("DFCI_MRN") == 1)["ADJUVANT_THERAPY_STOPPED"][0]


def test_exclusion_without_follow_up_rests_on_duration_alone():
    """With follow_up=None every follow-up-derived term is unknown, so the
    duration test decides by itself. Both short courses here are excluded --
    the opposite of the previous rule, where absent observation windows made
    exclusion impossible."""
    meds = _meds(_depot_course(1, months=24) + _depot_course(2, months=6))
    labelled = classify_adt_intent(meds, follow_up=None)
    assert labelled["IS_LOCALIZED_ADJUVANT"].all()
    # Nothing was observed long enough; it simply no longer matters.
    assert not labelled["ADJUVANT_OBSERVED_LONG_ENOUGH"].any()

    # A course over the bound is still retained without follow-up.
    long_course = classify_adt_intent(_meds(_depot_course(3, months=48)), follow_up=None)
    assert not _excluded(long_course, 3)
    assert _reason(long_course, 3) == "retained_adt_span_too_long"


def test_intent_and_exclusion_flag_agree_on_the_adjuvant_class():
    """ADT_INTENT is retained for continuity, so LOCALIZED_ADJUVANT must mean
    exactly IS_LOCALIZED_ADJUVANT -- otherwise the two disagree downstream."""
    meds = _meds(
        _depot_course(1, months=24)
        + _depot_course(2, months=48)
        + _depot_course(3, months=6)
        + [(4, "DOCETAXEL", ADT_START)]
        + _depot_course(4, months=12)
    )
    labelled = classify_adt_intent(
        meds,
        follow_up=_follow_up_many([1, 2, 3, 4], ADT_START + timedelta(days=2500)),
    )
    mismatched = labelled.filter(
        (pl.col("ADT_INTENT") == INTENT_LOCALIZED)
        != pl.col("IS_LOCALIZED_ADJUVANT")
    )
    assert mismatched.height == 0


def test_label_is_binary_and_mirrors_the_exclusion_flag():
    """No INDETERMINATE class: every patient is one of two labels, and the
    label is exactly IS_LOCALIZED_ADJUVANT restated."""
    meds = _meds(
        _depot_course(1, months=24)          # excludable
        + _depot_course(2, months=48)        # too long
        + _depot_course(3, months=6)         # short but under-observed below
        + [(4, "DOCETAXEL", ADT_START)]
        + _depot_course(4, months=12)        # escalation
    )
    labelled = classify_adt_intent(
        meds,
        follow_up=_follow_up_many([1, 2, 3, 4], ADT_START + timedelta(days=2500)),
    )
    assert set(labelled["ADT_INTENT"].to_list()) <= {
        INTENT_METASTATIC,
        INTENT_LOCALIZED,
    }
    mismatched = labelled.filter(
        (pl.col("ADT_INTENT") == INTENT_LOCALIZED)
        != pl.col("IS_LOCALIZED_ADJUVANT")
    )
    assert mismatched.height == 0


def test_metastatic_separates_evidenced_from_assumed():
    """The assumption has to stay auditable: a patient with a taxane and one
    merely lacking adjuvant evidence share a label but not the audit flag.

    Patient 2 is three widely-spaced short courses. Total span exceeds the
    36-month bound, so the duration test retains them, but the episode count
    keeps them outside HAS_POSITIVE_METASTATIC_EVIDENCE (which wants <= 2
    episodes) -- metastatic by assumption only."""
    meds = _meds(
        _depot_course(1, months=12)
        + [(1, "DOCETAXEL", ADT_START + timedelta(days=200))]
        + _depot_course(2, months=3)
        + [(2, "LEUPROLIDE ACETATE", ADT_START + timedelta(days=720 + 90 * i))
           for i in range(2)]
        + [(2, "LEUPROLIDE ACETATE", ADT_START + timedelta(days=1500 + 90 * i))
           for i in range(2)]
    )
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up_many([1, 2], ADT_START + timedelta(days=2500))
    )
    assert _intent(labelled, 1) == INTENT_METASTATIC
    assert _intent(labelled, 2) == INTENT_METASTATIC
    assert _reason(labelled, 2) == "retained_adt_span_too_long"

    def flag(mrn):
        return labelled.filter(pl.col("DFCI_MRN") == mrn)[
            "HAS_POSITIVE_METASTATIC_EVIDENCE"
        ][0]

    assert flag(1)       # taxane: affirmatively evidenced
    assert not flag(2)   # two short episodes: metastatic by assumption only


def test_escalation_tiers_are_disjoint():
    assert not (DEFINITIVE_METASTATIC_MEDS & ARPI_METASTATIC_MEDS)


def test_endpoint_drugs_excluded_from_escalation():
    """Platinum is the modelled endpoint and PARPi exposure is a cohort
    exclusion; classifying on either would leak the endpoint into the label."""
    for drug in ("CARBOPLATIN", "CISPLATIN", "OLAPARIB", "TALAZOPARIB"):
        assert drug not in DEFINITIVE_METASTATIC_MEDS


def test_unparseable_dates_are_dropped_not_fatal():
    meds = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1],
            "NCI_PREFERRED_MED_NM": ["LEUPROLIDE ACETATE", "LEUPROLIDE ACETATE"],
            "MED_START_DT": ["2015-01-01", "not-a-date"],
        }
    )
    assert load_medications_for_intent(meds).height == 1


def test_mixed_iso_and_slash_dates_are_parsed():
    """Realistic mixed-format column: polars infers month-first and both
    formats land correctly, which is the case the pipeline actually sees."""
    meds = pl.DataFrame(
        {
            "DFCI_MRN": [1] * 3,
            "NCI_PREFERRED_MED_NM": ["LEUPROLIDE ACETATE"] * 3,
            "MED_START_DT": ["2015-01-01", "04/01/2015", "07/15/2015"],
        }
    )
    episodes = build_adt_episodes(load_medications_for_intent(meds))
    assert episodes["ADT_FIRST_DATE"][0] == datetime(2015, 1, 1)
    assert episodes["ADT_LAST_DATE"][0] == datetime(2015, 7, 15)
    assert episodes["ADT_SPAN_DAYS"][0] == 195


def test_slash_only_column_raises_in_shared_date_parser():
    """Characterization test for a pre-existing fragility in the shared parser.

    `parse_mixed_datetime_expr` coalesces `str.to_datetime` ahead of the
    explicit `%m/%d/%Y` pattern, but `str.to_datetime(strict=False)` RAISES
    rather than returning nulls when it cannot infer one format for the whole
    column. So the coalesce never reaches its fallback: a slash-only column
    whose days are all > 12 (no month-first/day-first tiebreak available)
    aborts the run instead of degrading.

    Mixed ISO+slash columns -- what the real tables contain -- infer fine, so
    this is a narrow edge case rather than an active data corruption. It is
    asserted here so that if it is ever fixed upstream this test fails loudly
    and ADT-intent spans get re-verified. Fixing it is out of scope: the helper
    backs 18 call sites across ICD, medication and patient-status dates.
    """
    import pytest

    with pytest.raises(pl.exceptions.ComputeError):
        pl.DataFrame({"d": ["01/15/2015", "07/15/2015"]}).with_columns(
            parse_mixed_datetime_expr("d").alias("parsed")
        )


def test_multiple_patients_classified_independently():
    rows = (
        _depot_course(1, months=60)
        + _depot_course(2, months=18)
        + _depot_course(3, months=8)
        + [(3, "CABAZITAXEL", ADT_START + timedelta(days=200))]
    )
    follow_up = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "FOLLOW_UP_END_DATE": [ADT_START + timedelta(days=2500)] * 3,
        }
    ).with_columns(pl.col("FOLLOW_UP_END_DATE").cast(pl.Datetime))

    labelled = classify_adt_intent(_meds(rows), follow_up=follow_up)
    assert _intent(labelled, 1) == INTENT_METASTATIC
    assert _intent(labelled, 2) == INTENT_LOCALIZED
    assert _intent(labelled, 3) == INTENT_METASTATIC


# ---------------------------------------------------------------------------
# Validation harness (independent reference standard)
# ---------------------------------------------------------------------------


def test_metastasis_icd_matches_dotted_and_dotless_codes():
    """Prefix matching must catch both `C78.01` and `C7801`.

    The existing `mark_non_prostate_primary_icd` numeric-range regex captures
    `^[A-Z](\\d{2,3})`, giving 781 for `C7801`, which silently fails its
    `77 <= n <= 79` test. This is why the harness matches on prefixes.
    """
    icds = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4, 5],
            "DIAGNOSIS_ICD10_CD": ["C78.01", "C7801", "C79.51", "C7B.00", "C61.9"],
            "START_DT": ["2016-01-01"] * 5,
        }
    )
    got = compute_first_metastasis_icd_date(icds)
    assert set(got["DFCI_MRN"].to_list()) == {1, 2, 3, 4}


def test_metastasis_icd_takes_earliest_and_drops_undated():
    icds = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2],
            "DIAGNOSIS_ICD10_CD": ["C79.51", "C78.01", "C78.01"],
            "START_DT": ["2018-05-01", "2016-02-01", None],
        }
    )
    got = compute_first_metastasis_icd_date(icds)
    assert got.to_dicts() == [
        {"DFCI_MRN": 1, "FIRST_METASTASIS_ICD_DATE": datetime(2016, 2, 1)}
    ]


def test_psa_nadir_rise_is_measured_after_the_nadir():
    """A high pre-nadir PSA must not be counted as a rise from nadir."""
    labs = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1, 1, 2, 2, 2],
            "LAB_NAME": ["PSA"] * 6,
            "LAB_VALUE": [50.0, 0.02, 0.03, 50.0, 0.02, 9.0],
            "t_lab": [10.0, 200.0, 400.0, 10.0, 200.0, 400.0],
        }
    )
    got = compute_psa_nadir_features(labs).sort("DFCI_MRN")
    # Patient 1: deep nadir, no meaningful rise -> adjuvant-like.
    assert got["PSA_DEEP_NADIR_NO_RISE"][0]
    assert got["PSA_RISE_FROM_NADIR"][0] < 1.0
    # Patient 2: same nadir, then biochemical progression.
    assert not got["PSA_DEEP_NADIR_NO_RISE"][1]


def test_psa_nadir_ignores_pre_adt_measurements():
    """Only post-anchor PSA (t_lab >= 0) describes response to ADT."""
    labs = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1],
            "LAB_NAME": ["PSA"] * 2,
            "LAB_VALUE": [0.001, 5.0],
            "t_lab": [-100.0, 50.0],
        }
    )
    got = compute_psa_nadir_features(labs)
    assert got["PSA_NADIR"][0] == 5.0
    assert got["PSA_N_POST_ADT"][0] == 1


# ---------------------------------------------------------------------------
# Cross-reference against clinical_text_embedding_project stage / met burden
# ---------------------------------------------------------------------------


def _xref_labelled() -> pl.DataFrame:
    """Four labelled patients spanning both stage cells that matter."""
    return pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4],
            "ADT_INTENT": [
                INTENT_LOCALIZED,
                INTENT_METASTATIC,
                INTENT_METASTATIC,
                INTENT_LOCALIZED,
            ],
            "IS_LOCALIZED_ADJUVANT": [True, False, False, True],
            "ADT_EXCLUSION_REASON": ["excluded_completed_adjuvant_course"] * 4,
            "ADT_SPAN_DAYS": [400, 2000, 1500, 300],
        }
    )


def test_stage_reference_normalizes_and_dedupes():
    """Unknown stages drop out; duplicate MRNs collapse to one row."""
    raw = pl.DataFrame(
        {
            "DFCI_MRN": [1.0, 2.0, 2.0, 3.0],
            "CANCER_STAGE": ["I", "IV", "IV", "Unknown"],
        }
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cancer_stage_df.csv")
        raw.write_csv(path)
        got = load_stage_reference(path).sort("DFCI_MRN")
    assert got["DFCI_MRN"].to_list() == [1, 2]
    assert got["IS_STAGE_IV"].to_list() == [False, True]


def test_stage_reference_reconstructs_legacy_one_hot():
    """Older files carry only drop-first dummies; all-zero means stage I."""
    raw = pl.DataFrame(
        {
            "DFCI_MRN": [7, 8, 9],
            "CANCER_STAGE_II": [0, 1, 0],
            "CANCER_STAGE_III": [0, 0, 0],
            "CANCER_STAGE_IV": [0, 0, 1],
        }
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "legacy_stage.csv")
        raw.write_csv(path)
        got = load_stage_reference(path).sort("DFCI_MRN")
    assert got["CANCER_STAGE"].to_list() == ["I", "II", "IV"]


def test_only_stage_iv_among_excluded_counts_as_a_contradiction():
    """Stage I-III + METASTATIC is progression, not error; the reverse is.

    The label adjudicates in one direction only, so the contradiction report
    must pick up the excluded stage IV patient and nothing else.
    """
    stage = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4],
            "CANCER_STAGE": ["I", "IV", "III", "IV"],
            "IS_STAGE_IV": [False, True, False, True],
        }
    )
    labelled = _xref_labelled().join(stage, on="DFCI_MRN", how="left")
    got = report_stage_contradictions(labelled)
    assert got.height == 1
    assert got["DFCI_MRN"][0] == 4


def test_met_burden_coverage_gaps_are_not_counted_as_zero_burden():
    """A patient missing from the burden file is uncovered, not met-free.

    Treating an absent row as zero would silently understate burden in the
    excluded group -- exactly the direction that would hide a bad exclusion.
    """
    raw = pl.DataFrame(
        {
            "DFCI_MRN": [1.0, 2.0],
            "N_MET_SITES": [0, 3],
            "MET_SITE_bone": [0, 1],
            "MET_SITE_liver": [0, 1],
        }
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "met_burden_df.csv")
        raw.write_csv(path)
        burden = load_met_burden_reference(path)

    labelled = _xref_labelled().join(burden, on="DFCI_MRN", how="left")
    got = report_against_met_burden(labelled)
    # Patients 3 and 4 have no burden row and must not enter any denominator.
    assert got["n_covered"].sum() == 2
    localized = got.filter(pl.col("ADT_INTENT") == INTENT_LOCALIZED)
    assert localized["n_covered"][0] == 1
    assert localized["pct_zero_sites"][0] == 100.0


def test_cross_reference_reports_are_silent_without_the_files():
    """Both reports degrade to empty frames when the columns are absent."""
    labelled = _xref_labelled()
    assert report_against_met_burden(labelled).height == 0
    assert report_stage_contradictions(labelled).height == 0


# ---------------------------------------------------------------------------
# ADT-anchored stage and metastatic burden
# ---------------------------------------------------------------------------


def _note_level(tmpdir, rows):
    """Write a CANCER_STAGE_NOTE_LEVEL-shaped parquet and return its path."""
    path = os.path.join(tmpdir, "note_level.parquet")
    pl.DataFrame(
        {
            "DFCI_MRN": [r[0] for r in rows],
            "EVENT_DATE": [r[1] for r in rows],
            "DERIVED_STAGE_MERGED": [r[2] for r in rows],
        }
    ).write_parquet(path)
    return path


def test_stage_picks_the_observation_nearest_adt_not_the_earliest():
    """The whole point of the anchored loader.

    The shipped CANCER_STAGE.parquet carries NOTE_STAGE_EARLIEST, which for
    this patient is stage II from two years before ADT. Stage only ratchets
    upward, so the earliest note understates disease at treatment start; the
    nearest one (IV, 61 days before) is what the label must be judged against.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(
            d,
            [
                (1, datetime(2013, 1, 1), 2),
                (1, datetime(2014, 11, 1), 4),
            ],
        )
        labelled = pl.DataFrame({"DFCI_MRN": [1], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        got = load_stage_nearest_adt(path, labelled)

    assert got["CANCER_STAGE"][0] == "IV"
    assert got["STAGE_DAYS_FROM_ADT"][0] == -61


def test_stage_default_excludes_post_adt_observations():
    """A note written after ADT start can reflect progression under therapy.

    The prescriber could not have known it, so it must not be used to judge
    the intent decision. prefer="any" opts back in explicitly.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(d, [(2, datetime(2015, 6, 1), 4)])
        labelled = pl.DataFrame({"DFCI_MRN": [2], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        assert load_stage_nearest_adt(path, labelled).height == 0
        assert load_stage_nearest_adt(path, labelled, prefer="any").height == 1


def test_stage_outside_the_matching_window_is_dropped():
    """A stale observation is worse than no observation."""
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(d, [(3, datetime(2012, 10, 1), 3)])
        labelled = pl.DataFrame({"DFCI_MRN": [3], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        assert load_stage_nearest_adt(path, labelled).height == 0
        assert load_stage_nearest_adt(path, labelled, window_days=1000).height == 1


def test_max_stage_splits_before_and_after_adt_start():
    """The headline behaviour: worst stage on each side, kept separate."""
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(
            d,
            [
                (1, datetime(2014, 1, 1), 2),
                (1, datetime(2014, 6, 1), 3),
                (1, datetime(2015, 6, 1), 4),
            ],
        )
        labelled = pl.DataFrame({"DFCI_MRN": [1], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        got = load_stage_max_around_adt(path, labelled)

    assert got["MAX_STAGE_BEFORE"][0] == "III"
    assert got["MAX_STAGE_AFTER"][0] == "IV"
    assert got["STAGE_N_OBS_BEFORE"][0] == 2
    assert got["STAGE_N_OBS_AFTER"][0] == 1
    assert got["STAGE_UPSTAGED_AFTER_ADT"][0] is True


def test_max_stage_is_not_windowed_by_default():
    """A maximum means "worst ever recorded", so a distant stage IV must count.

    Windowing it would report III for this patient and quietly contradict the
    column name. The nearest-stage loader windows on purpose; this one must
    not inherit that.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(
            d,
            [
                (2, datetime(2012, 1, 1), 4),   # ~1100 days pre-ADT
                (2, datetime(2014, 11, 1), 3),
            ],
        )
        labelled = pl.DataFrame({"DFCI_MRN": [2], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        assert load_stage_max_around_adt(path, labelled)["MAX_STAGE_BEFORE"][0] == "IV"
        # Opting into a window is allowed, and then it does truncate.
        windowed = load_stage_max_around_adt(path, labelled, window_days=365)
        assert windowed["MAX_STAGE_BEFORE"][0] == "III"


def test_max_stage_day_count_marks_first_attainment_not_last():
    """Among observations tied at the max, the earliest is reported, so an
    early stage IV is distinguishable from a late restatement of it."""
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(
            d,
            [
                (3, datetime(2014, 1, 1), 4),
                (3, datetime(2014, 12, 1), 4),
            ],
        )
        labelled = pl.DataFrame({"DFCI_MRN": [3], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        got = load_stage_max_around_adt(path, labelled)

    assert got["MAX_STAGE_BEFORE_DAYS_FROM_ADT"][0] == -365


def test_max_stage_on_adt_start_date_counts_as_before():
    """The stage was on the record when treatment began."""
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(d, [(4, datetime(2015, 1, 1), 4)])
        labelled = pl.DataFrame({"DFCI_MRN": [4], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        got = load_stage_max_around_adt(path, labelled)

    assert got["MAX_STAGE_BEFORE"][0] == "IV"
    assert got["STAGE_N_OBS_AFTER"][0] == 0
    assert got["MAX_STAGE_AFTER"][0] is None


def test_max_stage_absent_side_is_null_not_a_floor_value():
    """No pre-ADT staging note is "unknown", not stage I, and an unknown side
    must not produce a confident upstaging verdict."""
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(d, [(5, datetime(2015, 6, 1), 3)])
        labelled = pl.DataFrame({"DFCI_MRN": [5], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        got = load_stage_max_around_adt(path, labelled)

    assert got["MAX_STAGE_BEFORE"][0] is None
    assert got["IS_MAX_STAGE_IV_BEFORE"][0] is None
    assert got["STAGE_N_OBS_BEFORE"][0] == 0
    assert got["MAX_STAGE_AFTER"][0] == "III"
    assert got["STAGE_UPSTAGED_AFTER_ADT"][0] is None


def test_max_stage_empty_input_keeps_schema():
    """The notebook joins this frame unconditionally, so an empty result still
    has to carry every column."""
    with tempfile.TemporaryDirectory() as d:
        path = _note_level(d, [(9, datetime(2015, 1, 1), 4)])
        labelled = pl.DataFrame({"DFCI_MRN": [404], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
        got = load_stage_max_around_adt(path, labelled)

    assert got.height == 0
    for col in ("MAX_STAGE_BEFORE", "MAX_STAGE_AFTER", "STAGE_UPSTAGED_AFTER_ADT"):
        assert col in got.columns


def test_max_stage_report_denominators_are_covered_patients():
    """A patient with no pre-ADT note must drop out of the pre-ADT
    denominator rather than count as not-stage-IV."""
    labelled = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "ADT_INTENT": ["METASTATIC", "METASTATIC"],
            "MAX_STAGE_BEFORE_INT": [4, None],
            "IS_MAX_STAGE_IV_BEFORE": [True, None],
            "MAX_STAGE_AFTER_INT": [4, 4],
            "IS_MAX_STAGE_IV_AFTER": [True, True],
            "STAGE_UPSTAGED_AFTER_ADT": [False, None],
        }
    )
    rep = report_stage_max(labelled)

    assert rep["n"][0] == 2
    assert rep["n_before"][0] == 1
    assert rep["pct_iv_before"][0] == 100.0   # 1/1 covered, not 1/2
    assert rep["n_after"][0] == 2


def test_met_burden_counts_only_codes_on_or_before_adt_start():
    """Anchoring is the fix for the shipped file being pre-index on a
    different project's first_treatment_date."""
    icds = pl.DataFrame(
        {
            "DFCI_MRN": [1, 1, 1],
            "DIAGNOSIS_ICD10_CD": ["C7951", "C787", "C7931"],
            "START_DT": ["2014-06-01", "2014-07-01", "2016-01-01"],
        }
    )
    labelled = pl.DataFrame({"DFCI_MRN": [1], "ADT_FIRST_DATE": [datetime(2015, 1, 1)]})
    got = compute_met_burden_at_adt(icds, labelled)

    # bone + liver counted; the 2016 brain code is after ADT start.
    assert got["N_MET_SITES"][0] == 2
    assert got["MET_SITE_bone"][0] == 1
    assert got["MET_SITE_liver"][0] == 1
    assert got["MET_SITE_brain"][0] == 0


def test_met_burden_emits_every_site_column_even_when_unobserved():
    """The column set must be fixed, not data-dependent, so downstream
    selects and plots do not break on a cohort that lacks a rare site."""
    icds = pl.DataFrame(
        {"DFCI_MRN": [1], "DIAGNOSIS_ICD10_CD": ["C7951"], "START_DT": ["2014-06-01"]}
    )
    labelled = pl.DataFrame(
        {"DFCI_MRN": [1, 2], "ADT_FIRST_DATE": [datetime(2015, 1, 1)] * 2}
    )
    got = compute_met_burden_at_adt(icds, labelled)

    for group in ("brain", "bone", "liver", "lung", "node", "adrenal", "peritoneal", "other"):
        assert f"MET_SITE_{group}" in got.columns
    # Patient 2 has no codes at all -> a real zero, since the frame is
    # zero-filled only across patients this pipeline observes.
    assert got.filter(pl.col("DFCI_MRN") == 2)["N_MET_SITES"][0] == 0
