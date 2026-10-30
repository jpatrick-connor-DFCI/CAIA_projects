"""Vitals and statistic restrictions on the checklist pool, and paired held-out contrasts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import cox_aggregated as ca
from COMPASS.survival_analysis.predictive_checklists import features
from COMPASS.survival_analysis.predictive_checklists.__main__ import main
from COMPASS.survival_analysis.predictive_checklists.config import RunConfig
from COMPASS.survival_analysis.predictive_checklists.data import load_landmark
from COMPASS.survival_analysis.predictive_checklists.demo import build_demo_inputs
from COMPASS.survival_analysis.predictive_checklists.evaluation import evaluate_horizon, paired_bootstrap
from COMPASS.survival_analysis.predictive_checklists.features import (
    AdjusterDesign, FeatureBinarizer, schema_feature_columns,
)
from COMPASS.survival_analysis.predictive_checklists.inference import predict_bundle
from COMPASS.survival_analysis.predictive_checklists.runner import run
from COMPASS.survival_analysis.predictive_checklists.survival import CensoringDistribution

AGE = "AGE_AT_TREATMENTSTART"
RESTRICTED = dict(exclude_vitals=True, feature_stats="mean-n-obs")


def small_config(**kwargs):
    values = dict(endpoints=("platinum",), landmarks=(0,), horizons=(365,), max_items=(2,), n_folds=2,
                  quantiles=(.25, .5, .75), min_events=2, min_controls=2, bootstrap_samples=0,
                  baselines=(), solver_time_limit=10, resolved_nepc_sensitivity=False)
    values.update(kwargs)
    return RunConfig(**values).validate()


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Demo inputs plus the columns the restrictions act on: a vital, lab means, a panel dummy."""
    folder = build_demo_inputs(tmp_path_factory.mktemp("restrictions") / "inputs", landmarks=(0,), n=160)
    folder = folder / "prediction_inputs_adt"
    path = folder / "aggregated_landmark0.csv"
    frame = pd.read_csv(path)
    rng = np.random.default_rng(3)
    frame["HEMOGLOBIN__mean"] = frame["HEMOGLOBIN__last"] + rng.normal(0, .2, len(frame))
    frame["Body weight__mean"] = rng.normal(85, 10, len(frame))
    frame["Body weight__n_observations"] = rng.integers(1, 8, len(frame))
    frame["PANEL_VERSION_2"] = rng.integers(0, 2, len(frame))
    frame.loc[frame.index[::7], "HEMOGLOBIN__mean"] = np.nan
    frame.to_csv(path, index=False)
    return folder


@pytest.fixture(scope="module")
def train(inputs):
    data = load_landmark(inputs, "platinum", 0)
    return data.frame.loc[data.frame.split != "test"].copy()


def _stat(column):
    return column.rpartition("__")[2]


def test_vital_names_match_the_survival_package():
    assert features.VITAL_MEASUREMENTS == ca.VITAL_MEASUREMENTS


def test_default_pool_still_has_every_statistic_and_vital(train):
    columns = FeatureBinarizer().fit(train).feature_columns
    assert {"last", "mean", "n_observations"} <= {_stat(c) for c in columns}
    assert "Body weight__mean" in columns


def test_exclude_vitals_keeps_labs_and_removes_vitals(train):
    columns = FeatureBinarizer(exclude_vitals=True).fit(train).feature_columns
    assert not any(c.startswith("Body weight") for c in columns)
    assert "PSA__last" in columns and "HEMOGLOBIN__mean" in columns and AGE in columns


def test_mean_n_obs_keeps_only_those_statistics_and_age(train):
    columns = FeatureBinarizer(feature_stats="mean-n-obs").fit(train).feature_columns
    assert {_stat(c) for c in columns if c != AGE} == {"mean", "n_observations"}
    assert AGE in columns


def test_restricted_drop_pool_is_age_and_non_vital_means(train):
    binarizer = FeatureBinarizer(exclude_adjusters=True, **RESTRICTED).fit(train)
    assert set(binarizer.feature_columns) == {AGE, "PSA__mean", "HEMOGLOBIN__mean"}
    assert "is_missing" not in {item["op"] for item in binarizer.items}


@pytest.mark.parametrize("kwargs", [RESTRICTED, dict(exclude_adjusters=True, **RESTRICTED)])
def test_baselines_use_the_same_pool(train, kwargs):
    assert set(schema_feature_columns(train, age_col=AGE, **kwargs)) == set(
        FeatureBinarizer(**kwargs).fit(train).feature_columns)


def test_restrictions_survive_serialization(train):
    fitted = FeatureBinarizer(**RESTRICTED).fit(train)
    payload = json.loads(json.dumps(fitted.to_dict()))
    assert payload["exclude_vitals"] is True and payload["feature_stats"] == "mean-n-obs"
    restored = FeatureBinarizer.from_dict(payload)
    assert (restored.exclude_vitals, restored.feature_stats) == (True, "mean-n-obs")
    assert restored.items == fitted.items
    np.testing.assert_array_equal(restored.transform(train), fitted.transform(train))


def test_defaults_add_nothing_to_saved_settings(train):
    """Absent keys are what keep old bundles loadable and old manifests resumable."""
    saved = FeatureBinarizer().fit(train).to_dict()
    assert not {"exclude_vitals", "feature_stats"} & set(saved)
    assert (FeatureBinarizer.from_dict(saved).exclude_vitals, FeatureBinarizer.from_dict(saved).feature_stats) == (
        False, "all")
    assert not {"exclude_vitals", "feature_stats", "adjust_only_mode"} & set(small_config().to_dict())
    assert {"exclude_vitals", "feature_stats"} <= set(small_config(**RESTRICTED).to_dict())


def test_unknown_statistic_is_refused(train):
    with pytest.raises(ValueError, match="feature_stats"):
        small_config(feature_stats="median")
    with pytest.raises(ValueError, match="feature_stats"):
        FeatureBinarizer(feature_stats="median").fit(train)


def test_calibration_adjusters_leave_vitals_out(train):
    binarizer = FeatureBinarizer(exclude_adjusters=True, **RESTRICTED).fit(train)
    names = {column["name"] for column in AdjusterDesign().fit(train, binarizer).columns}
    assert not any(name.startswith("Body weight") for name in names)
    assert {"PSA__n_observations", "PANEL_VERSION_2"} <= names


def test_cli_passes_restrictions_to_the_manifest(inputs, tmp_path):
    out = tmp_path / "preflight"
    code = main(["--platinum-inputs", str(inputs), "--output-dir", str(out), "--endpoints", "platinum",
                 "--landmarks", "0", "--horizons", "365", "--no-baselines", "--no-nepc-sensitivity",
                 "--exclude-vitals", "--feature-stats", "mean-n-obs", "--adjust-only-mode", "drop",
                 "--preflight-only"])
    assert code == 0
    config = json.loads((out / "run_manifest.json").read_text())["config"]
    assert config["exclude_vitals"] is True
    assert config["feature_stats"] == "mean-n-obs"
    assert config["adjust_only_mode"] == "drop"


def test_cli_defaults_leave_json_settings_alone(inputs, tmp_path):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(dict(exclude_vitals=True, feature_stats="mean-n-obs")))
    out = tmp_path / "preflight"
    assert main(["--config", str(source), "--platinum-inputs", str(inputs), "--output-dir", str(out),
                 "--endpoints", "platinum", "--landmarks", "0", "--horizons", "365", "--no-baselines",
                 "--no-nepc-sensitivity", "--preflight-only"]) == 0
    config = json.loads((out / "run_manifest.json").read_text())["config"]
    assert (config["exclude_vitals"], config["feature_stats"]) == (True, "mean-n-obs")


def test_restricted_drop_run_stays_in_pool_and_reloads(inputs, tmp_path):
    out = tmp_path / "results"
    config = small_config(adjust_only_mode="drop", **RESTRICTED)
    assert run(config, {"platinum": inputs}, out) == 0
    folder = out / "platinum" / "landmark_0" / "full" / "horizon_365"
    bundle = json.loads((folder / "model.json").read_text())
    assert set(bundle["binarizer"]["feature_columns"]) == {AGE, "PSA__mean", "HEMOGLOBIN__mean"}
    assert {item["feature"] for item in bundle["binarizer"]["items"]} <= {AGE, "PSA__mean", "HEMOGLOBIN__mean"}

    data = load_landmark(inputs, "platinum", 0)
    scored = predict_bundle(folder / "model.json", data.frame.loc[data.frame.split == "test"])
    saved = pd.read_csv(folder / "predictions.csv")
    np.testing.assert_allclose(scored.probability, saved.loc[saved.dataset == "test"].probability)

    # A resumed run with the same settings reuses the finished horizon.
    assert run(config, {"platinum": inputs}, out, resume=True) == 0
    with pytest.raises(ValueError, match="Cannot resume"):
        run(small_config(adjust_only_mode="drop"), {"platinum": inputs}, out, resume=True)


# ---------------------------------------------------------------- paired bootstrap

@pytest.fixture(scope="module")
def heldout():
    rng = np.random.default_rng(5)
    n = 300
    duration = rng.uniform(30, 1500, n)
    event = rng.integers(0, 2, n)
    score = rng.integers(0, 5, n).astype(float)
    predictions = {"a": (score, score >= 2, (score + 1) / 7),
                   "b": (score[::-1], score[::-1] >= 3, (score[::-1] + 1) / 6)}
    return duration, event, predictions, CensoringDistribution.fit(duration, event)


def test_paired_bootstrap_observed_matches_evaluate_horizon(heldout):
    duration, event, predictions, censoring = heldout
    observed, _ = paired_bootstrap(duration, event, predictions, 365, censoring, bootstrap_samples=2)
    for name, values in predictions.items():
        expected = evaluate_horizon(duration, event, *values, 365, censoring)
        assert observed[name]["auc"] == pytest.approx(expected["auc"])
        assert observed[name]["brier"] == pytest.approx(expected["brier"])


def test_paired_bootstrap_draws_are_shared_and_reproducible(heldout):
    duration, event, predictions, censoring = heldout
    same = {"a": predictions["a"], "copy": predictions["a"]}
    _, rows = paired_bootstrap(duration, event, same, 365, censoring, bootstrap_samples=25, seed=7)
    draws = pd.DataFrame(rows).pivot(index="draw", columns="fit", values="auc")
    assert len(draws) == 25
    assert draws["a"].nunique() > 1
    np.testing.assert_array_equal(draws["a"], draws["copy"])

    _, again = paired_bootstrap(duration, event, same, 365, censoring, bootstrap_samples=25, seed=7)
    _, other = paired_bootstrap(duration, event, same, 365, censoring, bootstrap_samples=25, seed=8)
    assert rows == again and rows != other
