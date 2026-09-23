"""Scoring a federated model bundle must reproduce the federated model exactly.

The bundle is the only artifact that crosses from the federated runs to this
repo, so these tests pin the contract it carries: the design matrix is rebuilt
with the POOLED preprocessing (never refit locally), the identifier never enters
it, and an exported booster predicts identically after a JSON round trip.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from survival_common.federated_inference import (
    BundleError,
    FederatedCoxModel,
    _check_xgboost_version,
    _model_base_score,
    build_design_matrix,
    load_bundle,
    load_fitted_model,
    load_model,
    predict_risk,
    predict_survival,
    select_model,
)

LAB_COLS = ["Hemoglobin__mean", "Glucose__max", "Platelets__last"]
COVARIATES = LAB_COLS + ["Glucose__max__missing", "age"]
N_MISSING = 30


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(10.0, 2.0, n) for c in LAB_COLS})
    df["age_at_anchor"] = rng.normal(65.0, 8.0, n)
    df["person_id"] = np.arange(n)
    df.loc[df.index[:N_MISSING], "Glucose__max"] = np.nan
    return df


def _preprocessing(df: pd.DataFrame) -> dict:
    centers = {c: float(df[c].mean()) for c in LAB_COLS}
    centers.update({"Glucose__max__missing": 0.15, "age": 65.0})
    scales = {c: float(df[c].std(ddof=0)) for c in LAB_COLS}
    scales.update({"Glucose__max__missing": 1.0, "age": 8.0})
    return {
        "covariate_cols": COVARIATES,
        "base_feature_cols": LAB_COLS,
        "missing_indicator_cols": ["Glucose__max__missing"],
        "static_covariate_cols": [],
        "impute_means": {c: float(df[c].mean()) for c in LAB_COLS},
        "centers": centers,
        "scales": scales,
        "source_cols": {"age": "age_at_anchor"},
    }


def _enet_bundle(df: pd.DataFrame, beta=(0.3, -0.2, 0.1, 0.4, 0.5)) -> dict:
    return {
        "format": "caia-federated-model-bundle",
        "format_version": 1,
        "model_family": "elastic_net_cox",
        "analysis_label": "adt",
        "endpoint": "platinum",
        "time_unit_days": 1.0,
        "models": [{
            "landmark_days": 90,
            "config": "both",
            "hyperparameters": {"penalizer": 0.01, "l1_ratio": 0.5, "n_iter": 12},
            "converged": True,
            "note": "",
            "covariate_cols": COVARIATES,
            "coefficients": list(beta),
            "preprocessing": _preprocessing(df),
            "baseline_cumhaz": {
                "times": [0.0, 100.0, 365.0, 730.0],
                "cumhaz": [0.0, 0.05, 0.2, 0.45],
                "time_unit_days": 1.0,
            },
        }],
    }


# --------------------------------------------------------------------- #
# Bundle validation
# --------------------------------------------------------------------- #
def test_rejects_a_file_that_is_not_a_bundle(tmp_path):
    p = tmp_path / "nope.json"
    p.write_text(json.dumps({"format": "something-else", "format_version": 1}))
    with pytest.raises(BundleError, match="format"):
        load_bundle(p)


def test_rejects_an_unsupported_format_version(tmp_path):
    p = tmp_path / "future.json"
    p.write_text(json.dumps({
        "format": "caia-federated-model-bundle",
        "format_version": 99,
        "models": [{}],
    }))
    with pytest.raises(BundleError, match="format_version"):
        load_bundle(p)


def test_rejects_a_bundle_with_no_models(tmp_path):
    """An aborted run writes the file with an empty model list; scoring it must
    fail loudly rather than silently produce nothing."""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({
        "format": "caia-federated-model-bundle",
        "format_version": 1,
        "models": [],
    }))
    with pytest.raises(BundleError, match="no fitted models"):
        load_bundle(p)


def test_select_model_reports_what_is_available(tmp_path):
    df = _frame()
    p = tmp_path / "b.json"
    p.write_text(json.dumps(_enet_bundle(df)))
    bundle = load_bundle(p)
    with pytest.raises(BundleError, match=r"\(90, 'both'\)"):
        select_model(bundle, 180, "both")


# --------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------- #
def test_design_matrix_has_exactly_the_model_covariates_in_order():
    """XGBoost scores by position, so a reordered matrix returns wrong risks
    without raising."""
    df = _frame()
    model = _enet_bundle(df)["models"][0]
    X = build_design_matrix(df, model, age_col="age_at_anchor")
    assert list(X.columns) == COVARIATES


def test_identifier_never_enters_the_design_matrix():
    df = _frame()
    model = _enet_bundle(df)["models"][0]
    X = build_design_matrix(df, model, age_col="age_at_anchor")
    assert "person_id" not in X.columns


def test_missing_indicator_describes_the_raw_data_not_the_imputed_data():
    """The indicator must be computed before imputation, or it is all zeros."""
    df = _frame()
    model = _enet_bundle(df)["models"][0]
    X = build_design_matrix(df, model, age_col="age_at_anchor")
    centre = model["preprocessing"]["centers"]["Glucose__max__missing"]
    expected = df["Glucose__max"].isna().to_numpy(dtype=float) - centre
    assert np.allclose(X["Glucose__max__missing"].to_numpy(), expected)


def test_pooled_scaling_is_reused_not_refit():
    """A locally refit scaler would give the local cohort mean zero.  The pooled
    centre differs from the local mean, so the scaled column must not."""
    df = _frame()
    model = _enet_bundle(df)["models"][0]
    model["preprocessing"]["centers"]["Hemoglobin__mean"] = 999.0
    model["preprocessing"]["scales"]["Hemoglobin__mean"] = 1.0
    X = build_design_matrix(df, model, age_col="age_at_anchor")
    assert X["Hemoglobin__mean"].mean() < -900


def test_absent_covariate_is_imputed_with_the_pooled_mean():
    df = _frame().drop(columns=["Platelets__last"])
    model = _enet_bundle(_frame())["models"][0]
    X = build_design_matrix(df, model, age_col="age_at_anchor")
    pre = model["preprocessing"]
    expected = (
        pre["impute_means"]["Platelets__last"] - pre["centers"]["Platelets__last"]
    ) / pre["scales"]["Platelets__last"]
    assert np.allclose(X["Platelets__last"].to_numpy(), expected)


def test_design_matrix_is_finite():
    df = _frame()
    model = _enet_bundle(df)["models"][0]
    X = build_design_matrix(df, model, age_col="age_at_anchor")
    assert np.isfinite(X.to_numpy(dtype=float)).all()


# --------------------------------------------------------------------- #
# Elastic-net scoring
# --------------------------------------------------------------------- #
def test_elasticnet_risk_is_the_linear_predictor():
    df = _frame()
    bundle = _enet_bundle(df)
    model = bundle["models"][0]
    X = build_design_matrix(df, model, age_col="age_at_anchor")
    risk = predict_risk(df, bundle, model, age_col="age_at_anchor")
    assert np.allclose(risk, X.to_numpy(dtype=float) @ np.asarray(model["coefficients"]))


def test_coefficient_count_mismatch_is_caught():
    df = _frame()
    bundle = _enet_bundle(df)
    bundle["models"][0]["coefficients"] = [0.1, 0.2]
    with pytest.raises(BundleError, match="coefficients"):
        predict_risk(df, bundle, bundle["models"][0], age_col="age_at_anchor")


def test_survival_is_a_probability_and_non_increasing_in_time():
    df = _frame()
    bundle = _enet_bundle(df)
    model = bundle["models"][0]
    risk = predict_risk(df, bundle, model, age_col="age_at_anchor")
    surv = predict_survival(risk, model, [365, 730])
    values = surv.to_numpy(dtype=float)
    assert ((values > 0) & (values <= 1)).all()
    assert (surv["survival_365d"] >= surv["survival_730d"]).all()


def test_higher_risk_means_lower_survival():
    df = _frame()
    bundle = _enet_bundle(df)
    model = bundle["models"][0]
    risk = predict_risk(df, bundle, model, age_col="age_at_anchor")
    surv = predict_survival(risk, model, [365])["survival_365d"].to_numpy()
    assert surv[np.argmax(risk)] < surv[np.argmin(risk)]


def test_survival_requires_a_baseline():
    df = _frame()
    bundle = _enet_bundle(df)
    model = bundle["models"][0]
    model.pop("baseline_cumhaz")
    risk = predict_risk(df, bundle, model, age_col="age_at_anchor")
    with pytest.raises(BundleError, match="baseline_cumhaz"):
        predict_survival(risk, model, [365])


# --------------------------------------------------------------------- #
# XGBoost scoring
# --------------------------------------------------------------------- #
xgb = pytest.importorskip("xgboost", reason="xgboost not installed")


def test_exported_booster_predicts_identically_after_a_json_round_trip():
    """This is the whole promise of the bundle: the model that scored patients
    inside the federation scores them the same way here."""
    df = _frame(n=300, seed=1)
    pre = _preprocessing(df)
    X = build_design_matrix(df, {"preprocessing": pre}, age_col="age_at_anchor")

    rng = np.random.default_rng(7)
    t = rng.exponential(300.0, len(df))
    e = rng.integers(0, 2, len(df))
    dtrain = xgb.DMatrix(
        X.to_numpy(dtype=float),
        label=np.where(e == 1, t, -t),
        feature_names=COVARIATES,
    )
    booster = xgb.train(
        {"objective": "survival:cox", "max_depth": 3, "eta": 0.1},
        dtrain,
        num_boost_round=20,
    )
    direct = booster.predict(dtrain, output_margin=True)

    bundle = {
        "format": "caia-federated-model-bundle",
        "format_version": 1,
        "model_family": "xgboost_cox",
        "objective": "survival:cox",
        "analysis_label": "adt",
        "endpoint": "platinum",
        "time_unit_days": 1.0,
        "models": [{
            "landmark_days": 90,
            "config": "both",
            "hyperparameters": {
                "max_depth": 3, "eta": 0.1, "min_child_weight": 1.0, "n_rounds": 20,
            },
            "n_trees_used": None,
            "feature_names": COVARIATES,
            "preprocessing": pre,
            "model_json": booster.save_raw("json").decode(),
        }],
    }
    scored = predict_risk(df, bundle, bundle["models"][0], age_col="age_at_anchor")
    assert np.allclose(scored, direct, atol=1e-6)


# --------------------------------------------------------------------- #
# Loading the trained object
# --------------------------------------------------------------------- #
def test_elasticnet_loads_as_an_estimator_with_named_coefficients(tmp_path):
    df = _frame()
    p = tmp_path / "cox.json"
    p.write_text(json.dumps(_enet_bundle(df)))

    model, bundle, entry = load_fitted_model(p, 90)
    assert isinstance(model, FederatedCoxModel)
    assert list(model.params_.index) == COVARIATES
    assert np.allclose(model.params_.to_numpy(), entry["coefficients"])
    assert np.allclose(model.hazard_ratios_.to_numpy(), np.exp(entry["coefficients"]))


def test_estimator_predict_agrees_with_the_functional_api():
    df = _frame()
    bundle = _enet_bundle(df)
    model = load_model(bundle, bundle["models"][0])
    assert np.allclose(
        model.predict(df, age_col="age_at_anchor"),
        predict_risk(df, bundle, bundle["models"][0], age_col="age_at_anchor"),
    )


def test_estimator_rejects_a_coefficient_count_mismatch():
    df = _frame()
    bundle = _enet_bundle(df)
    bundle["models"][0]["coefficients"] = [0.1, 0.2]
    with pytest.raises(BundleError, match="coefficients"):
        load_model(bundle, bundle["models"][0])


def test_zero_coefficients_survive_the_round_trip():
    """The coefficient CSV drops zeros; the bundle must not, or the covariate
    vector no longer lines up with its coefficients."""
    df = _frame()
    bundle = _enet_bundle(df, beta=(0.3, 0.0, 0.0, 0.4, 0.5))
    model = load_model(bundle, bundle["models"][0])
    assert len(model.coefficients) == len(COVARIATES)
    assert float(model.params_["Glucose__max"]) == 0.0


def _xgb_bundle(df):
    pre = _preprocessing(df)
    X = build_design_matrix(df, {"preprocessing": pre}, age_col="age_at_anchor")
    rng = np.random.default_rng(11)
    t = rng.exponential(300.0, len(df))
    e = rng.integers(0, 2, len(df))
    dtrain = xgb.DMatrix(
        X.to_numpy(dtype=float),
        label=np.where(e == 1, t, -t),
        feature_names=COVARIATES,
    )
    booster = xgb.train(
        {"objective": "survival:cox", "max_depth": 3, "eta": 0.1},
        dtrain,
        num_boost_round=12,
    )
    bundle = {
        "format": "caia-federated-model-bundle",
        "format_version": 1,
        "model_family": "xgboost_cox",
        "objective": "survival:cox",
        "xgboost_version": xgb.__version__,
        "analysis_label": "adt",
        "endpoint": "platinum",
        "time_unit_days": 1.0,
        "models": [{
            "landmark_days": 90,
            "config": "both",
            "hyperparameters": {
                "max_depth": 3, "eta": 0.1, "min_child_weight": 1.0, "n_rounds": 12,
            },
            "n_trees_used": None,
            "feature_names": COVARIATES,
            "preprocessing": pre,
            "model_json": booster.save_raw("json").decode(),
        }],
    }
    return bundle, booster, dtrain


def test_xgboost_loads_as_a_real_booster(tmp_path):
    """Not a wrapper: the caller gets the estimator itself, so every Booster
    method (trees, gain, SHAP, save_model) is available."""
    df = _frame(n=150, seed=4)
    bundle, _, _ = _xgb_bundle(df)
    p = tmp_path / "xgb.json"
    p.write_text(json.dumps(bundle))

    booster, _, _ = load_fitted_model(p, 90)
    assert isinstance(booster, xgb.Booster)
    assert booster.feature_names == COVARIATES
    assert booster.num_boosted_rounds() == 12
    assert not booster.trees_to_dataframe().empty
    assert booster.get_score(importance_type="gain")


def test_loaded_booster_supports_shap_and_resaving(tmp_path):
    df = _frame(n=150, seed=4)
    bundle, _, _ = _xgb_bundle(df)
    booster = load_model(bundle, bundle["models"][0])

    X = build_design_matrix(df, bundle["models"][0], age_col="age_at_anchor")
    dm = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=COVARIATES)
    contribs = booster.predict(dm, pred_contribs=True)
    assert contribs.shape == (len(df), len(COVARIATES) + 1)

    out = tmp_path / "resaved.json"
    booster.save_model(str(out))
    assert out.stat().st_size > 0


def test_loaded_booster_predicts_what_the_federated_model_predicted():
    df = _frame(n=150, seed=4)
    bundle, trained, dtrain = _xgb_bundle(df)
    booster = load_model(bundle, bundle["models"][0])
    X = build_design_matrix(df, bundle["models"][0], age_col="age_at_anchor")
    dm = xgb.DMatrix(X.to_numpy(dtype=float), feature_names=COVARIATES)
    assert np.allclose(
        booster.predict(dm, output_margin=True),
        trained.predict(dtrain, output_margin=True),
        atol=1e-6,
    )


def test_unknown_family_is_rejected():
    df = _frame()
    bundle = _enet_bundle(df)
    bundle["model_family"] = "random_forest"
    with pytest.raises(BundleError, match="model_family"):
        load_model(bundle, bundle["models"][0])


# ---------------------------------------------------------------------------
# Cross-version guard.
#
# A model trained by xgboost 3.x records the fitted Cox intercept in
# ``base_score``; 2.x ignores that field and substitutes 0.5, so the model
# loads without complaint and every prediction is off by a constant.  Rankings
# (and so C-index/AUC) survive, absolute survival does not.  The cluster pins
# 2.1.1 while the federated image allows >=2.0, so this is a live mismatch,
# not a hypothetical one.
# ---------------------------------------------------------------------------


def test_base_score_is_parsed_from_the_shape_xgboost_actually_writes():
    """XGBoost stores it as a bracketed *string*, not a number or a list.

    This is the shape that broke the guard: ``float("[2.4610445E0]")`` raises,
    the exception was swallowed, and the guard silently stopped checking.  A
    real booster is round-tripped here so the test tracks whatever xgboost
    writes rather than a literal copied from one version's output.
    """
    df = _frame(n=120, seed=11)
    _, trained, _ = _xgb_bundle(df)
    model_json = trained.save_raw("json").decode()

    assert json.loads(model_json)["learner"]["learner_model_param"]["base_score"]
    parsed = _model_base_score(model_json)
    assert parsed is not None, "guard would be silently disabled"
    assert np.isfinite(parsed)


def test_base_score_parser_accepts_every_serialised_form():
    def wrap(value):
        return json.dumps({"learner": {"learner_model_param": {"base_score": value}}})

    assert _model_base_score(wrap("[2.4610445E0]")) == pytest.approx(2.4610445)
    assert _model_base_score(wrap("5E-1")) == pytest.approx(0.5)
    assert _model_base_score(wrap([2.5, 1.0])) == pytest.approx(2.5)
    assert _model_base_score(wrap(0.5)) == pytest.approx(0.5)
    # Unparseable input disables the guard rather than crashing the load.
    assert _model_base_score("not json") is None
    assert _model_base_score(wrap("")) is None


def test_guard_refuses_an_older_xgboost_for_a_fitted_intercept():
    bundle = {"xgboost_version": "99.0.0"}
    model = {"model_json": json.dumps(
        {"learner": {"learner_model_param": {"base_score": "[2.4610445E0]"}}}
    )}
    with pytest.raises(BundleError, match="base_score"):
        _check_xgboost_version(bundle, model)


def test_guard_allows_a_newer_or_equal_xgboost():
    model = {"model_json": json.dumps(
        {"learner": {"learner_model_param": {"base_score": "[2.4610445E0]"}}}
    )}
    _check_xgboost_version({"xgboost_version": "0.1.0"}, model)
    _check_xgboost_version({"xgboost_version": str(xgb.__version__)}, model)


def test_guard_ignores_a_default_base_score_and_an_unstamped_bundle():
    """0.5 is what the old version would substitute, so nothing can go wrong."""
    default = {"model_json": json.dumps(
        {"learner": {"learner_model_param": {"base_score": "5E-1"}}}
    )}
    _check_xgboost_version({"xgboost_version": "99.0.0"}, default)
    # Written by an exporter predating the version stamp: nothing to compare.
    _check_xgboost_version({}, {"model_json": json.dumps(
        {"learner": {"learner_model_param": {"base_score": "[2.46E0]"}}}
    )})
