"""Adjust-only covariates: fit in training, dropped from the deployed model.

``--adjust-only-mode drop`` treats ``__n_observations``, ``__missing`` and the
static covariates as batch/density adjusters: they are fit jointly (Cox), as
an offset (XGBoost) or in the calibrator (checklists), and never appear in
coefficients, importances, checklist items or scores. ``none`` (the default)
must reproduce the existing behaviour exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import compass_pipeline as cp
from COMPASS.survival_analysis import cox_aggregated as ca
from survival_common.cox_engine import adjust_only_columns, predictive_columns

FEATURES = ["LAB_A__mean", "LAB_A__n_observations"]
STATIC = ("PANEL_VERSION_2",)
# The model matrices add a __missing indicator for the partly missing lab.
ADJUSTERS = ["LAB_A__n_observations", "LAB_A__mean__missing", "PANEL_VERSION_2"]


@pytest.fixture(scope="module")
def frames():
    """n_obs and the panel dummy strongly drive the hazard; the lab weakly."""
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.Index([f"P{i}" for i in range(n)], name=ca.ID_COL)
    nobs = rng.poisson(5, n).astype(float)
    panel = rng.binomial(1, 0.5, n).astype(float)
    lab = rng.normal(size=n)
    lab[rng.random(n) < 0.2] = np.nan
    lp = nobs / 3 + panel + 0.3 * np.nan_to_num(lab)
    t = rng.exponential(400 * np.exp(-lp))
    c = rng.exponential(600, n)
    df = pd.DataFrame({
        ca.AGE_COL: rng.normal(66, 8, n),
        "LAB_A__mean": lab,
        "LAB_A__n_observations": nobs,
        "PANEL_VERSION_2": panel,
        "t_platinum": np.minimum(t, c) + 1,
        "PLATINUM": (t <= c).astype(int),
    }, index=idx)
    return df.iloc[:240], df.iloc[240:]


def _perturb_adjusters(frame):
    out = frame.copy()
    out["LAB_A__n_observations"] += 7
    out["PANEL_VERSION_2"] = 1 - out["PANEL_VERSION_2"]
    return out


def test_column_classification():
    cols = ["age", "A__mean", "A__n_observations", "A__missing", "PANEL_VERSION_2"]
    assert adjust_only_columns(cols, ("PANEL_VERSION_2",), "none") == []
    assert set(adjust_only_columns(cols, ("PANEL_VERSION_2",), "drop")) == {
        "A__n_observations", "A__missing", "PANEL_VERSION_2"}
    drop = adjust_only_columns(cols, ("PANEL_VERSION_2",), "drop")
    assert predictive_columns(cols, drop) == ["age", "A__mean"]
    with pytest.raises(ValueError):
        adjust_only_columns(cols, (), "zero")


# --------------------------------------------------------------------- Cox

def _fit_cox(train_val, test, mode):
    return ca.fit_final_multivariable_model(
        train_val, test, feature_cols=FEATURES, endpoint="platinum",
        penalizer=0.01, l1_ratio=0.5, split_stratification="x", cv_stratification="y",
        auc_time_unit_days=30, auc_max_time_units=None, horizon_grid=np.arange(3, 20),
        canonical_labs=["LAB_A"], static_covariate_cols=STATIC, adjust_only_mode=mode,
    )


def test_cox_drop_splits_summary_and_scores_features_only(frames):
    train_val, test = frames
    m_none, s_none, p_none, *_ = _fit_cox(train_val, test, "none")
    m_drop, s_drop, p_drop, *_ = _fit_cox(train_val, test, "drop")

    # none: unchanged schema.
    assert "is_adjust_only" not in s_none.columns
    assert "n_adjust_only" not in m_none and "adjust_only_mode" not in m_none

    # Same joint fit, so the coefficients agree; only their labelling differs.
    coefs = s_none.set_index("feature")["coef"]
    np.testing.assert_allclose(s_drop.set_index("feature")["coef"].loc[coefs.index], coefs)
    assert set(s_drop.loc[s_drop["is_adjust_only"], "feature"]) == set(ADJUSTERS)
    assert m_drop["n_adjust_only"] == 3
    assert m_drop["n_covariates"] == m_none["n_covariates"] - 3

    # The deployed score ignores the adjusters entirely.
    _, _, p_pert, *_ = _fit_cox(train_val, _perturb_adjusters(test), "drop")
    np.testing.assert_allclose(p_pert["risk_score"], p_drop["risk_score"])
    _, _, p_none_pert, *_ = _fit_cox(train_val, _perturb_adjusters(test), "none")
    assert not np.allclose(p_none_pert["risk_score"], p_none["risk_score"])


def test_cox_runner_writes_adjuster_audit_table(frames):
    from survival_common.cox_runners import _collect_multivariable_outputs

    train_val, test = frames
    metrics, summary, predictions, auc_df, brier_df = _fit_cox(train_val, test, "drop")
    out = {key: [] for key in ("metric_rows", "frames", "patient_risk_frames",
                               "test_auc_frames", "test_brier_frames")}
    _collect_multivariable_outputs(out, 0, metrics, summary, predictions, auc_df, brier_df)
    reported, = out["frames"]
    audit, = out["adjuster_frames"]
    assert "is_adjust_only" not in reported.columns
    assert not set(reported["feature"]) & set(ADJUSTERS)
    assert set(audit["feature"]) == set(ADJUSTERS)


# ----------------------------------------------------------------- XGBoost

def _fit_xgb(train_val, mode):
    from survival_common.xgboost_engine import fit_xgb_cox

    args = argparse.Namespace(
        num_boost_round=50, eta=0.1, max_depth=2, min_child_weight=1.0, subsample=1.0,
        colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0, seed=1, tree_method="hist",
        verbose_eval=0, early_stopping_rounds=10, adjust_only_mode=mode,
    )
    return fit_xgb_cox(
        train_val.iloc[:200], train_val.iloc[200:], feature_cols=FEATURES,
        duration_col="t_platinum", event_col="PLATINUM", args=args,
        age_col=ca.AGE_COL, static_cols=STATIC,
    )


def test_xgboost_drop_uses_offset_and_predictive_trees(frames):
    pytest.importorskip("xgboost")
    from survival_common.xgboost_engine import feature_importance_frame, predict_risk

    train_val, test = frames
    model_none, cols_none, _, pre_none = _fit_xgb(train_val, "none")
    model, cols, _, pre = _fit_xgb(train_val, "drop")

    assert set(ADJUSTERS) <= set(cols_none)
    assert not set(cols) & set(ADJUSTERS)
    assert set(pre["adjust_only_cols"]) == set(ADJUSTERS)
    assert any(abs(v) > 0 for v in pre["adjuster_coefs"].values())

    importance = feature_importance_frame(
        model, covariate_cols=cols, xgb_feature_names=pre["xgb_feature_names"], endpoint="platinum")
    assert not set(importance["feature"]) & set(ADJUSTERS)

    risk, _ = predict_risk(model, test, preprocessor=pre)
    risk_pert, _ = predict_risk(model, _perturb_adjusters(test), preprocessor=pre)
    np.testing.assert_allclose(risk, risk_pert)
    risk_none, _ = predict_risk(model_none, test, preprocessor=pre_none)
    risk_none_pert, _ = predict_risk(model_none, _perturb_adjusters(test), preprocessor=pre_none)
    assert not np.allclose(risk_none, risk_none_pert)


# -------------------------------------------------------------- checklists

def _checklist_config(**kwargs):
    from COMPASS.survival_analysis.predictive_checklists.config import RunConfig

    values = dict(landmarks=(0,), horizons=(365,), max_items=(2,), n_folds=2,
                  quantiles=(.25, .5, .75), min_events=2, min_controls=2,
                  bootstrap_samples=0, baselines=(), solver_time_limit=10,
                  resolved_nepc_sensitivity=False)
    values.update(kwargs)
    return RunConfig(**values).validate()


@pytest.fixture(scope="module")
def checklist_train(tmp_path_factory):
    from COMPASS.survival_analysis.predictive_checklists.data import load_landmark
    from COMPASS.survival_analysis.predictive_checklists.demo import build_demo_inputs

    inputs = build_demo_inputs(tmp_path_factory.mktemp("checklist") / "inputs", landmarks=(0,), n=160)
    data = load_landmark(inputs / "prediction_inputs_adt", "platinum", 0)
    return data.frame.loc[data.frame.split != "test"].copy()


def test_checklist_drop_excludes_adjuster_items(checklist_train):
    from COMPASS.survival_analysis.predictive_checklists.features import FeatureBinarizer

    def items(exclude):
        fitted = FeatureBinarizer(exclude_adjusters=exclude).fit(checklist_train).items
        return [i["name"] for i in fitted], {i["op"] for i in fitted}

    names_none, ops_none = items(False)
    names_drop, ops_drop = items(True)
    assert any("__n_observations" in n for n in names_none) and "is_missing" in ops_none
    assert not any("__n_observations" in n for n in names_drop) and "is_missing" not in ops_drop
    # Dropping adjusters only removes items; the rest are unchanged.
    assert set(names_drop) < set(names_none)
    # none mode serializes exactly as before (no new key).
    assert "exclude_adjusters" not in FeatureBinarizer().fit(checklist_train).to_dict()


def test_checklist_calibrator_records_gamma_and_ignores_z():
    from COMPASS.survival_analysis.predictive_checklists.evaluation import fit_calibrator

    rng = np.random.default_rng(1)
    n = 400
    scores = rng.integers(0, 4, n) / 3
    z = rng.normal(size=(n, 1))
    labels = (rng.random(n) < 1 / (1 + np.exp(-(scores * 2 - 1 + 1.5 * z[:, 0])))).astype(int)
    weights = np.ones(n)
    plain = fit_calibrator(scores, labels, weights)
    adjusted = fit_calibrator(scores, labels, weights, adjusters=z, adjuster_names=["PSA__n_observations"])
    assert "adjust_only" not in plain.metadata
    gamma = adjusted.metadata["adjust_only"]["coefficients"]["PSA__n_observations"]
    assert gamma > 0.5
    # Prediction is a function of the score only (Z evaluated at its training mean, 0).
    grid = np.linspace(0, 1, 4)
    assert set(adjusted.to_dict()["parameters"]) == {"intercept", "slope"}
    assert np.all(np.diff(adjusted.predict(grid)) >= 0)
    with pytest.raises(ValueError):
        fit_calibrator(scores, labels, weights, method="isotonic", adjusters=z)


def test_checklist_drop_bundle_round_trips(tmp_path):
    from COMPASS.survival_analysis.predictive_checklists.data import load_landmark
    from COMPASS.survival_analysis.predictive_checklists.demo import build_demo_inputs
    from COMPASS.survival_analysis.predictive_checklists.inference import predict_bundle
    from COMPASS.survival_analysis.predictive_checklists.runner import run

    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    dirs = {"platinum": inputs / "prediction_inputs_adt"}
    out = tmp_path / "results"
    assert run(_checklist_config(endpoints=("platinum",), adjust_only_mode="drop"), dirs, out) == 0
    folder = out / "platinum" / "landmark_0" / "full" / "horizon_365"
    bundle = json.loads((folder / "model.json").read_text())
    assert bundle["adjuster_design"]["columns"]
    assert bundle["binarizer"]["exclude_adjusters"] is True
    assert "adjust_only" in bundle["calibrator"]["metadata"]
    items = [item["name"] for item in bundle["binarizer"]["items"]]
    assert not any("__n_observations" in n or n.endswith(" is missing") for n in items)

    data = load_landmark(dirs["platinum"], "platinum", 0)
    held_out = data.frame.loc[data.frame.split == "test"]
    scored = predict_bundle(folder / "model.json", held_out)
    saved = pd.read_csv(folder / "predictions.csv")
    saved = saved.loc[saved.dataset == "test"]
    np.testing.assert_allclose(scored.probability, saved.probability)


# ---------------------------------------------------------------- pipeline

def test_pipeline_drop_runs_get_parallel_tree_and_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)
    kwargs = dict(endpoints=("platinum",), cohorts=("all",), exclusions=("none",))
    default, = cp.make_endpoint_runs(["adt"], **kwargs)
    drop, = cp.make_endpoint_runs(["adt"], adjust_only_mode="drop", **kwargs)

    assert default["adjust_only_mode"] == "none"
    assert default["output_dir"] == tmp_path / "survival_analysis" / "local_runs_adt"
    assert drop["output_dir"] == tmp_path / "survival_analysis" / "local_runs_adt_adjust_only"
    assert drop["inputs_dir"] == default["inputs_dir"]
    assert cp.run_key(drop) == cp.run_key(default)
    with pytest.raises(ValueError):
        cp.make_endpoint_runs(["adt"], adjust_only_mode="zero", **kwargs)

    legacy = {k: v for k, v in default.items() if k != "adjust_only_mode"}
    for model in ("elastic-net", "xgboost"):
        for config_dir in ("both", "baseline"):
            row = Path("/x")
            base = cp.build_model_command(model, 90, config_dir, row, default)
            assert base == cp.build_model_command(model, 90, config_dir, row, legacy)
            assert "--adjust-only-mode" not in base
            flagged = cp.build_model_command(model, 90, config_dir, row, drop)
            assert flagged[-2:] == ["--adjust-only-mode", "drop"]
