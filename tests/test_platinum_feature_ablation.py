"""Predictor ablations must preserve patients and reach both model families."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import cox_aggregated as ca
from survival_common.projects.compass_profile import CONFIG
from survival_common.xgboost_engine import fit_preprocessor


@pytest.mark.parametrize("exclude_vitals", [False, True])
@pytest.mark.parametrize("feature_stats", ["all", "mean-n-obs"])
def test_candidates_and_missingness_preserve_risk_set(monkeypatch, tmp_path, exclude_vitals, feature_stats):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(index=pd.Index(range(50), name=ca.ID_COL))
    frame[ca.AGE_COL] = rng.uniform(50, 80, len(frame))
    frame["t_platinum"] = rng.uniform(50, 900, len(frame))
    frame["PLATINUM"] = np.arange(len(frame)) % 2
    stats = ["mean", "min", "max", "last", "delta", "n_observations"]
    names = ["PSA", *sorted(ca.VITAL_MEASUREMENTS)]
    for name in names:
        for stat in stats:
            frame[f"{name}__{stat}"] = rng.uniform(1, 10, len(frame))
            frame.loc[[0, 4, 45], f"{name}__{stat}"] = np.nan
    train, test = frame.iloc[:40].copy(), frame.iloc[40:].copy()
    long = pd.DataFrame([
        {ca.ID_COL: mrn, "LAB_NAME": name, "LAB_VALUE": float(mrn + 1), "t_lab": -10.0}
        for mrn in frame.index for name in names
    ])
    monkeypatch.setattr(ca, "_load_prebuilt_landmark", lambda *_: (frame, train, test, long))
    ctx = ca.prepare_landmark_context(
        tmp_path, 0, min_patient_coverage=0.2,
        exclude_vitals=exclude_vitals, feature_stats=feature_stats,
    )
    pd.testing.assert_frame_equal(ctx.train_val, train)
    pd.testing.assert_frame_equal(ctx.test, test)
    expected_names = {"PSA"} if exclude_vitals else set(names)
    expected_stats = {"mean", "n_observations"} if feature_stats == "mean-n-obs" else set(stats)
    expected = {f"{name}__{stat}" for name in expected_names for stat in expected_stats}
    assert set(ctx.raw_feature_cols) == expected
    assert set(ctx.selected_feature_cols) == expected
    assert set(ctx.pre_treatment_lab_df["LAB_NAME"]) == expected_names
    # Both engines must still derive missing indicators from the retained inputs.
    _, _, cox_columns = ca.build_model_matrices(
        train, test, feature_cols=ctx.selected_feature_cols,
        duration_col="t_platinum", event_col="PLATINUM",
    )
    xgb = fit_preprocessor(train, feature_cols=ctx.selected_feature_cols, age_col=ca.AGE_COL)
    for columns in (cox_columns, xgb["covariate_cols"]):
        assert "PSA__mean__missing" in columns
        assert "PSA__n_observations__missing" in columns
        if exclude_vitals:
            assert not any(c.startswith("Body weight__") for c in columns)
        if feature_stats == "mean-n-obs":
            assert not any("__delta" in c for c in columns)


def test_cli_options_reach_context():
    parser = argparse.ArgumentParser()
    CONFIG.add_cli_args(parser, ca)
    assert CONFIG.prepare_context_kwargs(parser.parse_args([])) == {
        "feature_set": "labs", "exclude_vitals": False, "feature_stats": "all",
    }
    args = parser.parse_args(["--exclude-vitals", "--feature-stats", "mean-n-obs"])
    assert CONFIG.prepare_context_kwargs(args) == {
        "feature_set": "labs", "exclude_vitals": True, "feature_stats": "mean-n-obs",
    }


def test_notebook_grid_and_summary(tmp_path, monkeypatch):
    """Execute orchestration and reporting with fake fits, never cluster data."""
    import matplotlib
    matplotlib.use("Agg")
    import subprocess

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "COMPASS/survival_analysis/10_platinum_feature_ablation.ipynb").read_text())
    monkeypatch.chdir(root)
    scope = {}
    setup = "".join(notebook["cells"][1]["source"])
    setup = setup.replace(
        'Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")', repr(tmp_path)
    )
    # repr(Path) uses PosixPath; provide it without changing notebook imports.
    scope["PosixPath"] = Path
    exec(compile(setup, "notebook_setup", "exec"), scope)
    tasks = scope["TASKS"]
    assert len(tasks) == len({t["folder"] for t in tasks}) == 24
    inputs = scope["INPUTS_DIR"]
    inputs.mkdir()
    manifest = {"auc_horizons_by_landmark": {str(lm): {"platinum": [52, 104]} for lm in (0, 90, 180)}}
    (inputs / "build_manifest.json").write_text(json.dumps(manifest))
    for lm in (0, 90, 180):
        pd.DataFrame(columns=["PSA__mean", "PSA__n_observations", "PSA__delta", "Body weight__mean"]).to_csv(
            inputs / f"aggregated_landmark{lm}.csv", index=False)
        (inputs / f"pre_treatment_lab_long_landmark{lm}.csv").touch()
    commands = []
    clock = [0.0]
    monkeypatch.setattr(scope["time"], "perf_counter", lambda: clock[0])

    def fake_fit(command, **kwargs):
        commands.append(command)
        task = next(t for t in tasks if t["command"] == command)
        assert ("--exclude-vitals" in command) == (task["vitals"] == "without_vitals")
        assert command[command.index("--feature-stats") + 1] == task["stats"]
        assert command[command.index("--endpoints") + 1] == "platinum"
        assert "--baseline" not in command
        effect = 0.02 * (task["vitals"] == "with_vitals") + 0.01 * (task["stats"] == "all")
        clock[0] += 120.0 + 1000 * effect
        pd.DataFrame([dict(endpoint="platinum", n_test=10, n_events_test=4,
                           test_c_index=0.7, test_mean_auc_t=0.75 + effect,
                           test_integrated_brier=0.15)]).to_csv(task["metrics"], index=False)

    monkeypatch.setattr(subprocess, "run", fake_fit)
    for cell in notebook["cells"][2:]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "notebook_cell", "exec"), scope)
    assert len(commands) == 24
    assert len(scope["summary"]) == 24
    assert len(scope["contrast_df"]) == 36  # six pairs per model/landmark
    assert scope["contrast_df"].groupby(["model", "landmark"]).size().eq(6).all()
    assert len(scope["cross_comparison"]) == 6
    assert scope["cross_comparison"].shape[1] == 16  # four metrics x four combinations
    diagonal = scope["contrast_df"].query(
        "vitals_a == 'with_vitals' and stats_a == 'all' and "
        "vitals_b == 'without_vitals' and stats_b == 'mean-n-obs'"
    )
    assert len(diagonal) == 6
    assert np.allclose(diagonal["delta_test_mean_auc_t"], 0.03)
    assert np.allclose(diagonal["delta_training_seconds"], 30.0)
    assert scope["summary"]["status"].eq("ok").all()
    assert scope["summary"]["timing_status"].eq("completed").all()
    assert scope["summary"]["training_seconds"].between(120, 150).all()
    assert np.allclose(scope["summary"]["training_minutes"], scope["summary"]["training_seconds"] / 60)
    timing_paths = [t["folder"] / "training_time.json" for t in tasks]
    timings = [p.read_text() for p in timing_paths]
    # Resume does not rerun completed fits.
    exec(compile("".join(notebook["cells"][5]["source"]), "notebook_resume", "exec"), scope)
    assert len(commands) == 24
    assert [p.read_text() for p in timing_paths] == timings
    # Older fits have unknown time, never a fabricated zero or skip duration.
    timing_paths[0].unlink()
    exec(compile("".join(notebook["cells"][7]["source"]), "notebook_summary", "exec"), scope)
    assert scope["summary"].iloc[0]["timing_status"] == "not_recorded"
    assert np.isnan(scope["summary"].iloc[0]["training_seconds"])
    # A failed overwrite must not report its elapsed time as successful training.
    scope["TASKS"] = tasks[:1]
    scope["OVERWRITE"] = True

    def failed_fit(command, **kwargs):
        clock[0] += 5.0
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", failed_fit)
    with pytest.raises(subprocess.CalledProcessError):
        exec(compile("".join(notebook["cells"][5]["source"]), "notebook_failure", "exec"), scope)
    failure = json.loads(timing_paths[0].read_text())
    assert failure["status"] == "failed"
    assert failure["training_seconds"] == 5.0
    assert failure["finished_at_utc"] is not None
    exec(compile("".join(notebook["cells"][7]["source"]), "notebook_failed_summary", "exec"), scope)
    assert scope["summary"].iloc[0]["status"] == "training_failed"
    assert np.isnan(scope["summary"].iloc[0]["training_seconds"])
