"""Tests for medication-only ADT treatment-intent classification.

The two guards these tests exist to protect (see classify_adt_intent module
docstring) are administrative censoring -- a truncated course must never be
read as a completed adjuvant course -- and upstream dedup, which collapses
same-day duplicate rows so record counts cannot stand in for duration.
"""

from __future__ import annotations

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
    return labelled.filter(pl.col("DFCI_MRN") == mrn)["ADT_INTENT_REASON"][0]


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
    assert _reason(labelled, 1) == "rule1_definitive_escalation"


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
    """Rule 2 is retired and rule 6 no longer excludes ARPI patients. A 24-month
    single course is judged on its duration alone, so adding an ARPI must not
    change the label -- in the old rules this exact patient was METASTATIC."""
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
    assert _reason(with_arpi, 1) == "rule6_single_short_course"


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


def test_no_reason_string_mentions_the_retired_rule2():
    """Rule numbers are not renumbered, so the gap must simply stay empty."""
    meds = _meds(
        _depot_course(1, months=24)
        + [(1, "ENZALUTAMIDE", ADT_START + timedelta(days=400))]
        + _depot_course(2, months=48)
    )
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up_many([1, 2], ADT_START + timedelta(days=2500))
    )
    assert not any(
        "rule2" in r for r in labelled["ADT_INTENT_REASON"].to_list()
    )


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
    assert _reason(labelled, 1) == "rule3_continuous_adt_over_3y"


def test_short_course_with_full_followup_is_adjuvant():
    """18 months of ADT, then a permanent stop, observed for years afterwards."""
    meds = _meds(_depot_course(1, months=18))
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, ADT_START + timedelta(days=2500))
    )
    assert _intent(labelled, 1) == INTENT_LOCALIZED
    assert _reason(labelled, 1) == "rule6_single_short_course"


def test_ongoing_adt_at_last_contact_is_metastatic():
    """Still filling ADT at last contact after 2+ years is a metastatic pattern."""
    meds = _meds(_depot_course(1, months=30))
    last_fill = ADT_START + timedelta(days=90 * (30 // 3))
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, last_fill + timedelta(days=30))
    )
    assert _intent(labelled, 1) == INTENT_METASTATIC
    assert labelled.filter(pl.col("DFCI_MRN") == 1)["ADT_ONGOING_AT_LAST_CONTACT"][0]


# ---------------------------------------------------------------------------
# Guard 1: administrative censoring
# ---------------------------------------------------------------------------


def test_short_course_truncated_by_data_cutoff_is_indeterminate():
    """The single most important rule: an 18-month course whose follow-up ends
    right at the last fill is indistinguishable from ongoing therapy, and must
    never be labelled adjuvant."""
    meds = _meds(_depot_course(1, months=18))
    last_fill = ADT_START + timedelta(days=90 * (18 // 3))
    labelled = classify_adt_intent(
        meds, follow_up=_follow_up(1, last_fill + timedelta(days=5))
    )
    assert _intent(labelled, 1) == INTENT_INDETERMINATE
    assert _reason(labelled, 1) == "rule5_insufficient_followup"


def test_missing_follow_up_cannot_yield_adjuvant():
    """Without follow-up dates, censoring cannot be ruled out for anyone."""
    meds = _meds(_depot_course(1, months=18))
    labelled = classify_adt_intent(meds, follow_up=None)
    assert _intent(labelled, 1) == INTENT_INDETERMINATE
    assert _reason(labelled, 1) == "rule5_insufficient_followup"


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
