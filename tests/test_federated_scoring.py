"""Building a scoring frame locally must reproduce the federated pipeline.

:mod:`federated_scoring` reimplements the feature derivation that lives in
``caia-project-compass/rhino_scripts/*/preprocessing.py``, because the bundle
carries the pooled transform but not the derivation.  Two copies of a rule is
exactly the situation that drifts silently, so these tests pin the rules that
change predictions:

* the pre-landmark filter is STRICT, so a lab drawn on the landmark day is not
  visible to the model;
* feature names are ``{sanitized_lab}__{stat}``, and ``delta`` needs two
  observations;
* a patient with no pre-landmark labs is KEPT, all-NaN, because that is what
  makes the bundle's ``__missing`` indicators informative;
* durations are shifted onto the landmark clock, non-positive ones dropped, and
  an event past the horizon becomes a censored observation AT the horizon;
* the castrate exclusion matches on the RAW lab name and is strict about
  same-day results.

The federated repo is not importable from here (it needs NVFLARE), so these
tests restate the expected values rather than importing the other copy.  The
round-trip parity against the real federated modules was verified separately;
see the module docstring of ``federated_scoring``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common import federated_scoring as fs
from survival_common.federated_inference import BundleError

ANCHOR = pd.Timestamp("2018-06-01")
DX = pd.Timestamp("2018-01-01")
HGB = "Hemoglobin [Mass/volume] in Blood"
TESTO = "Testosterone [Mass/volume] in Serum or Plasma"


def _rows(records) -> pd.DataFrame:
    """A long frame with the real COMPASS column names.

    ``records`` are ``(person_id, lab_name, lab_value, days_from_anchor)``;
    per-patient dates are filled in so ``derive_analysis_columns`` has work to do.
    """
    out = []
    for pid, lab, value, day in records:
        out.append({
            "person_id": pid,
            "age_at_diagnosis": 65.0,
            "diagnosis_date": DX,
            "adt_start_date_post_diagnosis": ANCHOR,
            "platinum_start_date": pd.NaT,
            "last_followup_date": ANCHOR + pd.Timedelta(days=2000),
            "lab_name": lab,
            "lab_value": value,
            "measurement_date": (
                ANCHOR + pd.Timedelta(days=day) if day is not None else pd.NaT
            ),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------- #
# Derived timing columns
# --------------------------------------------------------------------- #
def test_age_at_anchor_advances_from_age_at_diagnosis():
    """Age is at the ADT anchor, not at diagnosis -- a 151-day gap here."""
    df = fs.derive_analysis_columns(_rows([(1, HGB, 10.0, -5)]))
    expected = 65.0 + (ANCHOR - DX).days / fs.DAYS_PER_YEAR
    assert df[fs.AGE_COL].iloc[0] == pytest.approx(expected)


def test_event_is_recomputed_from_the_platinum_date():
    """A stale indicator must never survive: the date defines the event."""
    df = _rows([(1, HGB, 10.0, -5)])
    df["event_platinum"] = 1          # wrong on purpose
    df["platinum_start_date"] = pd.NaT
    out = fs.derive_analysis_columns(df)
    assert out[fs.EVENT_COL].iloc[0] == 0


def test_duration_falls_back_to_last_followup_when_no_platinum():
    df = fs.derive_analysis_columns(_rows([(1, HGB, 10.0, -5)]))
    assert df[fs.DURATION_COL].iloc[0] == 2000


# --------------------------------------------------------------------- #
# The pre-landmark filter
# --------------------------------------------------------------------- #
def test_landmark_day_measurement_is_excluded():
    """Strictly before: a lab drawn ON the landmark day is not available.

    This is the leakage guard.  At landmark 0 it means the labs drawn the day
    ADT started cannot be used to predict the outcome.
    """
    df = _rows([(1, HGB, 10.0, 90), (1, HGB, 20.0, 89)])
    pre = fs.filter_pre_landmark(fs.derive_analysis_columns(df), 90)
    assert list(pre["lab_value"]) == [20.0]


def test_landmark_zero_excludes_the_anchor_day():
    df = _rows([(1, HGB, 10.0, 0), (1, HGB, 20.0, -1)])
    pre = fs.filter_pre_landmark(fs.derive_analysis_columns(df), 0)
    assert list(pre["lab_value"]) == [20.0]


# --------------------------------------------------------------------- #
# Lab aggregation
# --------------------------------------------------------------------- #
def test_last_is_chronological_not_row_order():
    """Rows arrive unsorted; ``last`` must still be the latest measurement."""
    df = fs.derive_analysis_columns(
        _rows([(1, HGB, 5.0, -10), (1, HGB, 9.0, -1), (1, HGB, 7.0, -5)])
    )
    agg = fs.aggregate_lab_features(fs.filter_pre_landmark(df, 0))
    row = agg.iloc[0]
    assert row["last"] == 9.0
    assert row["delta"] == 9.0 - 5.0
    assert row["n_observations"] == 3
    assert row["min"] == 5.0 and row["max"] == 9.0


def test_delta_needs_two_observations():
    df = fs.derive_analysis_columns(_rows([(1, HGB, 5.0, -10)]))
    agg = fs.aggregate_lab_features(fs.filter_pre_landmark(df, 0))
    assert np.isnan(agg.iloc[0]["delta"])


def test_feature_names_are_sanitized():
    """XGBoost rejects ``[``, ``]`` and ``<``, so names are ``[^\\w] -> _``."""
    wide, _ = fs.build_landmark_frame(
        _rows([(1, HGB, 5.0, -10), (1, HGB, 6.0, -2)]), 0
    )
    assert "Hemoglobin__Mass_volume__in_Blood__mean" in wide.columns
    assert not any("[" in c or "]" in c for c in wide.columns)


# --------------------------------------------------------------------- #
# Keeping lab-free patients
# --------------------------------------------------------------------- #
def test_patient_with_no_pre_landmark_labs_is_kept_all_nan():
    """The reason the bundle's ``__missing`` indicators carry information.

    Patient 2's only lab is after the landmark, so they have no features -- but
    dropping them would change the cohort the federated run scored.
    """
    df = _rows([(1, HGB, 5.0, -10), (1, HGB, 6.0, -2), (2, HGB, 7.0, 200)])
    wide, _ = fs.build_landmark_frame(df, 90)
    assert set(wide["person_id"]) == {1, 2}
    p2 = wide[wide["person_id"] == 2].iloc[0]
    assert np.isnan(p2["Hemoglobin__Mass_volume__in_Blood__mean"])


def test_patient_with_no_labs_at_all_is_kept():
    df = _rows([(1, HGB, 5.0, -10), (2, None, np.nan, None)])
    wide, _ = fs.build_landmark_frame(df, 0)
    assert set(wide["person_id"]) == {1, 2}


# --------------------------------------------------------------------- #
# Outcome construction
# --------------------------------------------------------------------- #
def test_duration_is_shifted_onto_the_landmark_clock():
    static = pd.DataFrame({
        fs.ID_COL: [1], fs.AGE_COL: [70.0],
        fs.EVENT_COL: [1], fs.DURATION_COL: [500.0],
    })
    out, _ = fs.make_outcome(static, 90)
    assert out[fs.DURATION_COL].iloc[0] == 410.0


def test_non_positive_shifted_duration_is_dropped():
    """An event at or before the landmark is not predictable from it."""
    static = pd.DataFrame({
        fs.ID_COL: [1, 2, 3], fs.AGE_COL: [70.0] * 3,
        fs.EVENT_COL: [1, 1, 1], fs.DURATION_COL: [50.0, 90.0, 120.0],
    })
    out, stats = fs.make_outcome(static, 90)
    assert list(out[fs.ID_COL]) == [3]
    assert stats["n_dropped_non_positive_duration"] == 2


def test_event_past_the_horizon_becomes_censored_at_the_horizon():
    static = pd.DataFrame({
        fs.ID_COL: [1], fs.AGE_COL: [70.0],
        fs.EVENT_COL: [1], fs.DURATION_COL: [5000.0],
    })
    out, stats = fs.make_outcome(static, 0, max_followup_days=3650)
    assert out[fs.EVENT_COL].iloc[0] == 0
    assert out[fs.DURATION_COL].iloc[0] == 3650.0
    assert stats["n_admin_censored"] == 1


def test_frame_without_outcome_columns_keeps_every_patient():
    """Scoring patients whose outcome is unknown is the point of inference."""
    static = pd.DataFrame({fs.ID_COL: [1, 2], fs.AGE_COL: [70.0, 71.0]})
    out, stats = fs.make_outcome(static, 90)
    assert len(out) == 2
    assert stats["has_outcome"] is False


def test_patient_without_age_is_dropped():
    """No age means no usable anchor row, so no time origin."""
    df = _rows([(1, HGB, 5.0, -10), (2, HGB, 6.0, -10)])
    df.loc[df["person_id"] == 2, "age_at_diagnosis"] = np.nan
    static = fs.extract_static_frame(df)
    assert list(static[fs.ID_COL]) == [1]


# --------------------------------------------------------------------- #
# Cohort exclusion
# --------------------------------------------------------------------- #
def test_pre_anchor_castrate_drops_the_patient():
    df = _rows([(1, TESTO, 20.0, -30), (2, TESTO, 300.0, -30)])
    out, stats = fs.apply_exclusion(df, "pre_anchor_castrate")
    assert set(out["person_id"]) == {2}
    assert stats["n_patients_excluded"] == 1


def test_same_day_castrate_result_is_not_prior_treatment():
    """anchor_days == 0 may already reflect the first ADT dose, so ``< 0``."""
    df = _rows([(1, TESTO, 20.0, 0)])
    out, stats = fs.apply_exclusion(df, "pre_anchor_castrate")
    assert set(out["person_id"]) == {1}
    assert stats["n_patients_excluded"] == 0


def test_exclusion_matches_the_raw_lab_name():
    """Matching happens before sanitization; a sanitized name must not match."""
    df = _rows([(1, TESTO.replace("[", "_").replace("]", "_"), 20.0, -30)])
    out, stats = fs.apply_exclusion(df, "pre_anchor_castrate")
    assert stats["n_patients_excluded"] == 0
    assert set(out["person_id"]) == {1}


def test_unknown_exclusion_is_rejected():
    with pytest.raises(ValueError, match="unknown exclusion"):
        fs.apply_exclusion(_rows([(1, HGB, 5.0, -10)]), "something_else")


# --------------------------------------------------------------------- #
# Scoring a bundle
# --------------------------------------------------------------------- #
def _enet_bundle(covariates, coefficients, *, landmark=0, baseline=True):
    """A minimal elastic-net bundle over explicit covariates."""
    model = {
        "landmark_days": landmark,
        "config": "both",
        "hyperparameters": {"penalizer": 0.01, "l1_ratio": 0.5, "n_iter": 5},
        "converged": True,
        "note": "",
        "covariate_cols": list(covariates),
        "coefficients": [float(c) for c in coefficients],
        "preprocessing": {
            "covariate_cols": list(covariates),
            "base_feature_cols": [c for c in covariates if c != "age"],
            "missing_indicator_cols": [],
            "static_covariate_cols": [],
            "impute_means": {c: 0.0 for c in covariates},
            "centers": {c: 0.0 for c in covariates},
            "scales": {c: 1.0 for c in covariates},
            "source_cols": {"age": fs.AGE_COL},
        },
    }
    if baseline:
        model["baseline_cumhaz"] = {
            "times": [0.0, 52.0, 104.0],
            "cumhaz": [0.0, 0.1, 0.3],
            "time_unit_days": 7.0,
        }
    return {
        "format": "caia-federated-model-bundle",
        "format_version": 1,
        "model_family": "elastic_net_cox",
        "analysis_label": "adt",
        "endpoint": "platinum",
        "time_unit_days": 7.0,
        "models": [model],
    }


def _cohort(n=40, event_frac=0.4):
    """A scoreable cohort in which some patients reach the endpoint.

    ``_rows`` censors everyone, which leaves the C-index undefined, so a
    fraction of patients are given a platinum date here.
    """
    rng = np.random.default_rng(0)
    records = []
    for pid in range(n):
        for k in range(3):
            records.append((pid, HGB, float(rng.normal(10, 2)), -10 * (k + 1)))
    df = _rows(records)
    n_event = int(n * event_frac)
    for pid in range(n_event):
        rows = df["person_id"] == pid
        df.loc[rows, "platinum_start_date"] = ANCHOR + pd.Timedelta(
            days=200 + 30 * pid
        )
    return df


def test_score_bundle_returns_one_row_per_patient_per_model():
    df = _cohort()
    wide, _ = fs.build_landmark_frame(df, 0)
    cov = ["Hemoglobin__Mass_volume__in_Blood__mean", "age"]
    scores, metrics, coverage = fs.score_bundle(
        df, _enet_bundle(cov, [0.5, 0.2]), configs=("both",), verbose=False
    )
    assert len(scores) == len(wide)
    assert len(metrics) == 1
    assert len(coverage) == len(cov)
    assert list(scores["person_id"]) == list(wide["person_id"])


def test_scored_risk_is_the_linear_predictor():
    """With identity preprocessing the score is exactly X @ beta."""
    df = _cohort()
    wide, _ = fs.build_landmark_frame(df, 0)
    col = "Hemoglobin__Mass_volume__in_Blood__mean"
    scores, _, _ = fs.score_bundle(
        df, _enet_bundle([col, "age"], [0.5, 0.2]), configs=("both",), verbose=False
    )
    expected = 0.5 * wide[col].to_numpy(float) + 0.2 * wide[fs.AGE_COL].to_numpy(float)
    np.testing.assert_allclose(scores["risk_score"].to_numpy(float), expected)


def test_coverage_flags_a_covariate_absent_locally():
    """The diagnostic that catches a mismatched lab dictionary."""
    df = _cohort()
    bundle = _enet_bundle(["NotAMeasuredLab__mean", "age"], [0.5, 0.2])
    _, metrics, coverage = fs.score_bundle(df, bundle, configs=("both",), verbose=False)
    absent = coverage[~coverage["in_local_frame"]]
    assert list(absent["covariate"]) == ["NotAMeasuredLab__mean"]
    assert metrics["n_covariates_absent_locally"].iloc[0] == 1


def test_survival_probabilities_are_added_when_a_baseline_exists():
    df = _cohort()
    cov = ["Hemoglobin__Mass_volume__in_Blood__mean", "age"]
    scores, _, _ = fs.score_bundle(
        df, _enet_bundle(cov, [0.5, 0.2]), configs=("both",),
        horizons_days=(365.0, 730.0), verbose=False,
    )
    assert {"survival_365d", "survival_730d"} <= set(scores.columns)
    s = scores[["survival_365d", "survival_730d"]].to_numpy(float)
    assert ((s >= 0) & (s <= 1)).all()
    # Non-increasing in time, for every patient.
    assert (np.diff(s, axis=1) <= 1e-12).all()


def test_survival_is_skipped_without_a_baseline():
    """An XGBoost bundle carries no baseline, so only relative risk exists."""
    df = _cohort()
    cov = ["Hemoglobin__Mass_volume__in_Blood__mean", "age"]
    scores, _, _ = fs.score_bundle(
        df, _enet_bundle(cov, [0.5, 0.2], baseline=False), configs=("both",),
        horizons_days=(365.0,), verbose=False,
    )
    assert not any(c.startswith("survival_") for c in scores.columns)


def test_no_matching_model_is_an_error_that_says_what_is_available():
    df = _cohort()
    bundle = _enet_bundle(["age"], [0.2], landmark=0)
    with pytest.raises(BundleError, match=r"bundle has \[\(0, 'both'\)\]"):
        fs.score_bundle(df, bundle, landmarks=(180,), verbose=False)


def test_evaluate_reports_a_c_index_from_the_local_outcome():
    df = _cohort()
    cov = ["Hemoglobin__Mass_volume__in_Blood__mean", "age"]
    _, metrics, _ = fs.score_bundle(
        df, _enet_bundle(cov, [0.5, 0.2]), configs=("both",),
        evaluate=True, verbose=False,
    )
    c = metrics["c_index"].iloc[0]
    assert 0.0 <= c <= 1.0


def test_c_index_is_oriented_so_higher_risk_means_shorter_survival():
    """A score that is the true risk must land above 0.5, not below."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(0, 1, n)
    duration = np.exp(-0.9 * x) * rng.exponential(300, n) + 1.0
    event = np.ones(n, dtype=int)
    out = fs.evaluate_risk(x, duration, event)
    assert out["c_index"] > 0.6


def test_evaluate_risk_survives_a_cohort_with_no_events():
    out = fs.evaluate_risk(np.zeros(5), np.arange(1, 6, dtype=float), np.zeros(5))
    assert np.isnan(out["c_index"])
    assert out["n_events"] == 0
