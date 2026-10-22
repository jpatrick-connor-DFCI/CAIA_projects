"""Numerical checks for the checklist survival extension and comparators."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis.predictive_checklists.baselines import _Preprocessor, fit_baselines
from COMPASS.survival_analysis.predictive_checklists.evaluation import (
    ProbabilityCalibrator,
    evaluate_horizon,
    fit_calibrator,
    weighted_auc,
)
from COMPASS.survival_analysis.predictive_checklists.survival import (
    CensoringDistribution,
    UnsupportedHorizonError,
    horizon_targets,
)


def test_reverse_km_event_before_censor_ties_and_serialization():
    fitted = CensoringDistribution.fit([1, 1, 2, 3, 4], [1, 0, 1, 0, 1])
    np.testing.assert_allclose(fitted.predict([0, 1, 2, 3, 4]), [1, 0.75, 0.75, 0.375, 0.375])
    restored = CensoringDistribution.from_dict(json.loads(json.dumps(fitted.to_dict())))
    np.testing.assert_equal(restored.predict([0, 1, 2, 3]), fitted.predict([0, 1, 2, 3]))
    with pytest.raises(UnsupportedHorizonError):
        restored.predict(5)


def test_reverse_km_matches_sksurv_with_ties():
    nonparametric = pytest.importorskip("sksurv.nonparametric")
    rng = np.random.default_rng(10)
    duration = rng.integers(1, 20, 150).astype(float)
    event = rng.random(150) < 0.45
    times, expected = nonparametric.kaplan_meier_estimator(event, duration, reverse=True)
    actual = CensoringDistribution.fit(duration, event)
    np.testing.assert_allclose(actual.predict(times), expected)


def test_horizon_boundaries_unknown_censoring_and_unclipped_weights():
    fitted = CensoringDistribution.fit([1, 1, 2, 3, 4], [1, 0, 1, 0, 1])
    labels, weights, audit = horizon_targets([1, 2, 2, 3, 1.5], [1, 1, 0, 0, 0], 2, fitted)
    np.testing.assert_equal(labels, [1, 1, -1, 0, -1])
    np.testing.assert_allclose(weights, [4 / 3, 4 / 3, 0, 4 / 3, 0])
    assert audit["n_unknown"] == 2
    assert audit["weight_truncation"] is False
    with pytest.raises(UnsupportedHorizonError, match="strictly below"):
        horizon_targets([1, 5], [1, 0], 4, fitted)
    with pytest.raises(UnsupportedHorizonError, match="not clipped"):
        horizon_targets([1, 5], [1, 0], 3, fitted, min_g=0.4)


def test_brier_denominator_includes_unknown_patients():
    fitted = CensoringDistribution.fit([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    result = evaluate_horizon([1, 1.5, 3], [1, 0, 0], [2, 1, 0], [1, 0, 0], [.8, .4, .2], 2, fitted)
    assert result["brier"] == pytest.approx((.2 ** 2 + .2 ** 2) / 3)
    assert result["auc"] == 1
    assert result["n_unknown"] == 1


def test_auc_ties_and_ipcw_match_sksurv_metrics():
    metrics = pytest.importorskip("sksurv.metrics")
    util = pytest.importorskip("sksurv.util")
    duration = np.array([1, 1, 2, 3, 4, 5, 6, 7, 9, 10], dtype=float)
    event = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
    evaluation_t = np.array([1, 2, 2, 3, 4, 6, 7, 8], dtype=float)
    evaluation_e = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
    score = np.array([2, 2, 1, 1, 1, 0, 0, 2])
    probability = np.array([.8, .6, .6, .4, .7, .2, .3, .7])
    fitted = CensoringDistribution.fit(duration, event)
    result = evaluate_horizon(evaluation_t, evaluation_e, score, probability >= .5, probability, 4.5, fitted)
    train = util.Surv.from_arrays(event, duration)
    test = util.Surv.from_arrays(evaluation_e, evaluation_t)
    expected_auc, _ = metrics.cumulative_dynamic_auc(train, test, score, [4.5])
    _, expected_brier = metrics.brier_score(train, test, (1 - probability)[:, None], [4.5])
    assert result["auc"] == pytest.approx(expected_auc[0])
    assert result["brier"] == pytest.approx(expected_brier[0])


def test_weighted_auc_pair_counting_ties():
    assert weighted_auc(np.array([1, 0]), np.array([1, 1]), np.array([3, 8])) == .5
    assert weighted_auc(np.array([1, 1, 0]), np.array([2, 0, 1]), np.array([3, 1, 2])) == .75


def test_calibration_monotone_serialization_and_no_class_balancing():
    for method in ["logistic", "isotonic"]:
        fitted = fit_calibrator([0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 0, 1, 0, 1, 1, 1],
                                [2, 2, 2, 1, 2, 1, 1, 1], method)
        grid = np.linspace(-1, 4, 50)
        assert (np.diff(fitted.predict(grid)) >= -1e-12).all()
        restored = ProbabilityCalibrator.from_dict(json.loads(json.dumps(fitted.to_dict())))
        np.testing.assert_equal(restored.predict(grid), fitted.predict(grid))
        assert fitted.metadata["class_balancing"] is False
        assert fitted.metadata["weighted_prevalence"] == pytest.approx(1 / 3)
    constant = fit_calibrator([0, 0, 0], [0, 1, -1], [3, 1, 0])
    np.testing.assert_equal(constant.predict([0, 1]), [.25, .25])


def test_calibrator_reversed_score_cannot_fit_negative_slope():
    fitted = fit_calibrator([0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1])
    assert fitted.parameters["slope"] == 0
    np.testing.assert_allclose(fitted.predict([0, 1]), [.5, .5])


def test_evaluation_reports_support_failure_and_reproducible_conditional_bootstrap():
    fitted = CensoringDistribution.fit([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    arguments = ([1, 1.5, 3, 4], [1, 0, 1, 0], [2, 1, 0, 0], [1, 0, 0, 0], [.8, .4, .2, .2])
    result = evaluate_horizon(*arguments, 2, fitted, bootstrap_samples=50, seed=47)
    repeated = evaluate_horizon(*arguments, 2, fitted, bootstrap_samples=50, seed=47)
    assert result == repeated
    assert result["bootstrap"]["intervals"]["auc"]["valid_samples"] < 50
    json.dumps(result, allow_nan=False)
    assert evaluate_horizon(*arguments, 5, fitted)["status"] == "unsupported"


def test_invalid_outcomes_and_predictions_fail_explicitly():
    with pytest.raises(ValueError, match="binary"):
        CensoringDistribution.fit([1, 2], [1, 2])
    with pytest.raises(ValueError, match="nonnegative"):
        CensoringDistribution.fit([-1, 2], [1, 0])
    fitted = CensoringDistribution.fit([1, 2, 4], [1, 1, 1])
    with pytest.raises(ValueError, match="probabilities"):
        evaluate_horizon([1, 3], [1, 0], [1, 0], [1, 0], [1.1, .2], 2, fitted)


def _baseline_data():
    rng = np.random.default_rng(21)
    age, lab = rng.normal(65, 8, 100), rng.normal(0, 1, 100)
    event_time = rng.exponential(7 * np.exp(-.04 * (age - 65) - .4 * lab)) + .2
    censor_time = rng.uniform(3, 14, 100)
    duration, event = np.minimum(event_time, censor_time), event_time <= censor_time
    frame = pd.DataFrame({"age": age, "lab": lab})
    frame.loc[::11, "lab"] = np.nan
    return frame, duration, event


@pytest.mark.parametrize("model_name", ["age_cox", "elastic_net", "xgboost"])
def test_baselines_training_cv_probabilities_and_pickle_roundtrip(model_name):
    try:
        pytest.importorskip("xgboost" if model_name == "xgboost" else "sksurv")
    except (OSError, ValueError) as exc:
        # A Python wheel may be present while its optional native runtime is
        # absent (e.g. XGBoost/OpenMP on macOS). Check the explicit skip path.
        frame, duration, event = _baseline_data()
        unavailable = fit_baselines(frame, duration, event, age_column="age", horizons=[4],
                                    models=[model_name])[model_name]
        assert unavailable.status == "skipped"
        assert unavailable.reason
        pytest.skip(f"optional native survival runtime unavailable: {exc}")
    frame, duration, event = _baseline_data()
    fitted = fit_baselines(frame, duration, event, age_column="age", horizons=[4], n_splits=2,
                           seed=11, models=[model_name])[model_name]
    assert fitted.status == "ok", fitted.reason
    prediction = fitted.predict(frame.iloc[:4], 4)
    assert np.isfinite(prediction["score"]).all()
    assert ((prediction["probability"] >= 0) & (prediction["probability"] <= 1)).all()
    np.testing.assert_equal(prediction["flag"], prediction["probability"] >= .5)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_allclose(restored.predict(frame.iloc[:4], 4)["probability"], prediction["probability"])
    if model_name != "age_cox":
        assert fitted.metadata["cv_candidates"]
        assert fitted.metadata["best_cv_auc"] > 0


def test_constant_age_is_reported_as_skipped_without_crashing_comparators():
    pytest.importorskip("sksurv")
    frame, duration, event = _baseline_data()
    frame["age"] = 60
    result = fit_baselines(frame, duration, event, age_column="age", horizons=[4], models=["age_cox"])
    assert result["age_cox"].status == "skipped"
    assert "constant" in result["age_cox"].reason


def test_baseline_feature_coverage_and_imputation_are_fit_on_training_fold_only():
    training = pd.DataFrame({"age": [60, 70, 80, 90, 100],
                             "rare_lab": [1, np.nan, np.nan, np.nan, np.nan],
                             "usual_lab": [1, 2, 3, np.nan, np.nan]})
    processor = _Preprocessor.fit(training, min_patient_coverage=.4, age_column="age")
    assert processor.columns == ["age", "usual_lab"]
    validation = pd.DataFrame({"age": [80], "rare_lab": [1], "usual_lab": [np.nan]})
    transformed = processor.transform(validation)
    expected = (2 - processor.means[1]) / processor.scales[1]
    assert transformed[0, 1] == pytest.approx(expected)
    assert processor.medians[1] == 2
