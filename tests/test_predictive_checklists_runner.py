"""End-to-end synthetic integration, export parity, and held-out isolation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis.predictive_checklists.config import RunConfig
from COMPASS.survival_analysis.predictive_checklists.data import load_landmark
from COMPASS.survival_analysis.predictive_checklists.demo import build_demo_inputs
from COMPASS.survival_analysis.predictive_checklists.inference import predict_bundle
from COMPASS.survival_analysis.predictive_checklists.runner import fit_horizon, run


def small_config(**kwargs):
    values = dict(landmarks=(0,), horizons=(365,), max_items=(2,), n_folds=2,
                  quantiles=(.25, .5, .75), min_events=2, min_controls=2,
                  bootstrap_samples=0, baselines=(), solver_time_limit=10,
                  resolved_nepc_sensitivity=False)
    values.update(kwargs)
    return RunConfig(**values).validate()


def test_both_endpoints_export_and_reload(tmp_path):
    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    dirs = {ep: inputs / ("prediction_inputs_adt" + ("_nepc" if ep == "nepc" else ""))
            for ep in ("platinum", "nepc")}
    out = tmp_path / "results"
    assert run(small_config(), dirs, out) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert {row["endpoint"] for row in metrics if row["status"] == "ok"} == {"platinum", "nepc"}
    for endpoint in dirs:
        folder = out / endpoint / "landmark_0" / "full" / "horizon_365"
        data = load_landmark(dirs[endpoint], endpoint, 0)
        held_out = data.frame.loc[data.frame.split == "test"]
        scored = predict_bundle(folder / "model.json", held_out)
        saved = pd.read_csv(folder / "predictions.csv")
        saved = saved.loc[saved.dataset == "test"]
        np.testing.assert_array_equal(scored.score, saved.score)
        np.testing.assert_allclose(scored.probability, saved.probability)
        manifest = json.loads((out / "run_manifest.json").read_text())
        assert all(source["synthetic"] for source in manifest["inputs"])
    with pytest.raises(FileExistsError):
        run(small_config(), dirs, out)


def test_training_rejects_test_rows_and_ignores_unrelated_columns(tmp_path):
    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    data = load_landmark(inputs / "prediction_inputs_adt", "platinum", 0)
    with pytest.raises(ValueError, match="only train/valid"):
        fit_horizon(data.frame, "platinum", 365, small_config())
    train = data.frame.loc[data.frame.split != "test"].copy()
    first = fit_horizon(train, "platinum", 365, small_config())
    train["NEPC"] = np.arange(len(train)) % 2
    train["ADT_INTENT"] = "RETROSPECTIVE"
    train["t_last_contact"] = np.arange(len(train)) * 10000
    second = fit_horizon(train, "platinum", 365, small_config())
    assert first["binarizer"].to_dict() == second["binarizer"].to_dict()
    assert first["model"].selected_indices == second["model"].selected_indices
    assert first["model"].threshold == second["model"].threshold
    np.testing.assert_allclose(first["oof"], second["oof"])


def test_preflight_and_unsupported_horizon_are_explicit(tmp_path):
    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=100)
    dirs = {"platinum": inputs / "prediction_inputs_adt"}
    config = small_config(endpoints=("platinum",), horizons=(2000,))
    assert run(config, dirs, tmp_path / "audit", preflight_only=True) == 0
    assert not list((tmp_path / "audit").rglob("model.json"))
    assert run(config, dirs, tmp_path / "unsupported") == 2
    rows = json.loads((tmp_path / "unsupported" / "metrics.json").read_text())
    assert rows[0]["status"] == "unsupported"


def test_invalid_later_landmark_prevents_all_fits(tmp_path):
    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0, 90), n=100)
    folder = inputs / "prediction_inputs_adt"
    file = folder / "aggregated_landmark90.csv"
    frame = pd.read_csv(file)
    index = frame.index[frame.split == "train"][0]
    frame.loc[index, "split"] = "test"
    frame.to_csv(file, index=False)
    out = tmp_path / "results"
    assert run(small_config(endpoints=("platinum",), landmarks=(0, 90)),
               {"platinum": folder}, out) == 1
    assert not list(out.rglob("model.json"))
    assert json.loads((out / "run_manifest.json").read_text())["status"] == "incomplete"


def test_failure_keeps_cv_diagnostics_and_overwrite_archives(tmp_path, monkeypatch):
    import COMPASS.survival_analysis.predictive_checklists.runner as runner

    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    dirs = {"platinum": inputs / "prediction_inputs_adt"}
    out = tmp_path / "results"
    config = small_config(endpoints=("platinum",))
    assert run(config, dirs, out) == 0

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(runner, "fit_checklist", fail)
    assert run(config, dirs, out, overwrite=True) == 2
    cell = out / "platinum/landmark_0/full/horizon_365"
    assert not (cell / "model.json").exists()
    assert (cell / "failure.json").exists()
    folds = json.loads((cell / "cv_folds.json").read_text())
    assert len(folds) == 2 and all(x["reason"] == "synthetic solver failure" for x in folds)
    assert len(list((out / "previous_runs").rglob("model.json"))) == 1
    assert json.loads((out / "run_manifest.json").read_text())["status"] == "no_supported_models"


def test_demo_refusal_does_not_modify_existing_inputs(tmp_path):
    from COMPASS.survival_analysis.predictive_checklists.__main__ import main
    out = tmp_path / "existing"
    out.mkdir()
    (out / "run_manifest.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(["--demo", "--output-dir", str(out)])
    assert exc.value.code == 2
    assert not (out / "synthetic_inputs").exists()


def test_resume_reuses_completed_cells_and_refits_partial_ones(tmp_path, monkeypatch):
    import COMPASS.survival_analysis.predictive_checklists.runner as runner

    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0, 90), n=160)
    dirs = {"platinum": inputs / "prediction_inputs_adt"}
    out = tmp_path / "results"
    config = small_config(endpoints=("platinum",), landmarks=(0, 90))
    assert run(config, dirs, out) == 0
    first = json.loads((out / "metrics.json").read_text())
    created = json.loads((out / "run_manifest.json").read_text())["created_utc"]
    # Simulate an interruption inside landmark 90: no cell metrics, stale files.
    partial = out / "platinum/landmark_90/full"
    (partial / "metrics.json").unlink()
    (partial / "horizon_365" / "stale.json").write_text("{}")
    fitted = []
    original = runner.run_cell
    monkeypatch.setattr(runner, "run_cell", lambda data, endpoint, landmark, *rest:
                        fitted.append(landmark) or original(data, endpoint, landmark, *rest))
    assert run(config, dirs, out, resume=True) == 0
    assert fitted == [90]
    assert not (partial / "horizon_365" / "stale.json").exists()
    assert (partial / "horizon_365" / "model.json").exists()
    assert json.loads((out / "metrics.json").read_text()) == first
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["created_utc"] == created and len(manifest["resumed_utc"]) == 1
    assert manifest["status"] == "complete"


def test_resume_refuses_changed_settings_without_touching_run(tmp_path):
    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    dirs = {"platinum": inputs / "prediction_inputs_adt"}
    out = tmp_path / "results"
    assert run(small_config(endpoints=("platinum",)), dirs, out) == 0
    before = (out / "run_manifest.json").read_text()
    with pytest.raises(ValueError, match="configuration differs"):
        run(small_config(endpoints=("platinum",), seed=7), dirs, out, resume=True)
    frame = pd.read_csv(dirs["platinum"] / "aggregated_landmark0.csv")
    frame.loc[0, "t_platinum"] += 1
    frame.to_csv(dirs["platinum"] / "aggregated_landmark0.csv", index=False)
    with pytest.raises(ValueError, match="inputs changed"):
        run(small_config(endpoints=("platinum",)), dirs, out, resume=True)
    assert (out / "run_manifest.json").read_text() == before
    # Without an existing run, resume simply starts one.
    assert run(small_config(endpoints=("platinum",)), dirs, tmp_path / "fresh", resume=True) == 0
