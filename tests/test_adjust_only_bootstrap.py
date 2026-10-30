"""Default vs adjust-only intervals must pair patients and leave default commands alone."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import compass_pipeline as cp
from survival_common.heldout_bootstrap import percentile_ci, score_heldout
from survival_common.helper import survival_columns

HORIZONS = np.array([10.0, 20.0, 30.0])
LANDMARKS = [0, 90]
TASKS = [(model, landmark, config) for model, config, _ in cp.MULTIVARIATE_TASK_SPECS for landmark in LANDMARKS]
N_BOOT = 40


def _run(tmp_path, label):
    return dict(
        label=label, endpoint="platinum", landmarks=LANDMARKS,
        inputs_dir=tmp_path / "inputs", output_dir=tmp_path / label,
    )


def _build_inputs(inputs_dir):
    """Prediction inputs shared by both modes, one outcome table per landmark."""
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for landmark in LANDMARKS:
        rng = np.random.default_rng(landmark)
        n = 300
        signal = rng.normal(size=n)
        outcomes[landmark] = pd.DataFrame({
            cp._ca.ID_COL: np.arange(n),
            "split": np.where(np.arange(n) < 200, "train", "test"),
            "t_platinum": np.ceil(rng.exponential(700 * np.exp(-0.7 * signal))).clip(1, 1500),
            "PLATINUM": rng.integers(0, 2, size=n),
            "signal": signal,
        })
        outcomes[landmark].drop(columns="signal").to_csv(
            inputs_dir / f"aggregated_landmark{landmark}.csv", index=False,
        )
    (inputs_dir / "build_manifest.json").write_text(json.dumps({
        "auc_timeline_schema_version": 3,
        "auc_time_unit_days": 7,
        "auc_max_time_units": 200,
        "auc_horizons_by_landmark": {
            str(landmark): {"platinum": HORIZONS.tolist()} for landmark in LANDMARKS
        },
    }))
    return outcomes


def _write_fit(run, task, outcomes, *, save_survival, weight):
    """A fit's risk file and metrics, scored the way the engines report them."""
    model, landmark, config = task
    frame = outcomes[landmark]
    test = frame.loc[frame["split"] == "test"]
    reference = frame.loc[frame["split"] != "test"].rename(
        columns={"t_platinum": "duration_days", "PLATINUM": "event"}
    )
    duration, event = test["t_platinum"].to_numpy(dtype=float), test["PLATINUM"].to_numpy(dtype=int)
    noise = np.random.default_rng(TASKS.index(task)).normal(size=len(test))
    risk = weight * test["signal"].to_numpy() + noise
    surv = np.exp(-np.outer(np.exp(risk), HORIZONS / 100.0))
    scores = score_heldout(
        duration, event, risk, surv if save_survival else None, reference=reference,
        horizons=HORIZONS, time_unit_days=7, max_time_unit=200,
    )
    risk_path = cp.patient_risk_path(run, model, landmark, config)
    risk_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_name = next(name for m, c, name in cp.MULTIVARIATE_TASK_SPECS if (m, c) == (model, config))
    pd.DataFrame([dict(endpoint="platinum", n_test=len(test), n_events_test=int(event.sum()), **scores)]).to_csv(
        risk_path.parent / metrics_name, index=False,
    )
    predictions = pd.DataFrame({
        cp._ca.ID_COL: test[cp._ca.ID_COL], "endpoint": "platinum", "dataset": "test",
        # Each engine's own duration column.
        "duration_days" if model == "elastic-net" else "duration": duration,
        "event": event, "risk_score": risk,
    })
    if save_survival:
        predictions = predictions.assign(**survival_columns(surv, HORIZONS))
    # Rows the bootstrap must never read: another endpoint, and the OOF scheme.
    other_endpoint = predictions.assign(endpoint="nepc", risk_score=0.0)
    out_of_fold = predictions.assign(dataset="cv_oof", risk_score=0.0)
    pd.concat([predictions, other_endpoint, out_of_fold]).to_csv(risk_path, index=False)


@pytest.fixture()
def runs(tmp_path):
    outcomes = _build_inputs(tmp_path / "inputs")
    default_run, drop_run = _run(tmp_path, "default"), _run(tmp_path, "drop")
    for task in TASKS:
        _write_fit(default_run, task, outcomes, save_survival=True, weight=1.0)
        _write_fit(drop_run, task, outcomes, save_survival=True, weight=0.8)
    return default_run, drop_run, outcomes


def test_survival_knob_only_changes_the_two_fit_commands(monkeypatch):
    run = dict(label="x", inputs_dir=Path("/i"), output_dir=Path("/o"), endpoint="platinum")
    for model in ("elastic-net", "xgboost"):
        for config in ("both", "baseline"):
            off = cp.build_model_command(model, 90, config, Path("/x"), run)
            assert "--save-test-survival" not in off
            monkeypatch.setattr(cp, "SAVE_TEST_SURVIVAL", True)
            on = cp.build_model_command(model, 90, config, Path("/x"), run)
            monkeypatch.setattr(cp, "SAVE_TEST_SURVIVAL", False)
            assert [part for part in on if part != "--save-test-survival"] == off
            assert len(on) == len(off) + 1


def test_intervals_pair_each_task_across_modes(runs):
    default_run, drop_run, _ = runs
    intervals, draws = cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)
    assert len(intervals) == len(TASKS)
    assert len(draws) == len(TASKS) * 2 * N_BOOT
    assert set(draws["endpoint"]) == {"platinum"}
    for metric in ("c_index", "mean_auc_t", "integrated_brier"):
        assert intervals[f"delta_{metric}_ci_low"].notna().all()
        assert intervals[f"{metric}_default_ci_low"].notna().all()
        assert intervals[f"{metric}_adjust_only_ci_high"].notna().all()

    row = intervals.iloc[0]
    task = draws.loc[
        (draws["model"] == row["model"]) & (draws["landmark"] == row["landmark"]) & (draws["config"] == row["config"])
    ].pivot(index="draw", columns="mode", values="c_index")
    low, high, n_valid = percentile_ci(task["adjust_only"] - task["default"])
    assert (row["delta_c_index_ci_low"], row["delta_c_index_ci_high"]) == (low, high)
    assert row["delta_c_index_n_valid_draws"] == n_valid
    # Same patients in both modes: the paired delta is tighter than the unpaired one.
    unpaired = task["adjust_only"].to_numpy() - task["default"].sample(frac=1, random_state=0).to_numpy()
    assert (high - low) < percentile_ci(unpaired)[1] - percentile_ci(unpaired)[0]


def test_seed_is_shared_within_a_landmark_and_reproducible(runs, monkeypatch):
    default_run, drop_run, _ = runs
    seeds = []
    real_bootstrap = cp.bootstrap_heldout

    def recording_bootstrap(fits, **kwargs):
        seeds.append(kwargs["seed"])
        return real_bootstrap(fits, **kwargs)

    monkeypatch.setattr(cp, "bootstrap_heldout", recording_bootstrap)
    first, _ = cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)
    second, _ = cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert seeds[: len(TASKS)] == [42 + landmark for _, landmark, _ in TASKS]


def test_missing_survival_blanks_only_integrated_brier(tmp_path):
    outcomes = _build_inputs(tmp_path / "inputs")
    default_run, drop_run = _run(tmp_path, "default"), _run(tmp_path, "drop")
    for task in TASKS:
        # The 03 tree predates the knob; metrics there still report a Brier value.
        _write_fit(default_run, task, outcomes, save_survival=False, weight=1.0)
        _write_fit(drop_run, task, outcomes, save_survival=True, weight=0.8)
    intervals, _ = cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)
    assert intervals["integrated_brier_default_ci_low"].isna().all()
    assert intervals["delta_integrated_brier_ci_low"].isna().all()
    assert intervals["integrated_brier_adjust_only_ci_low"].notna().all()
    assert intervals["delta_c_index_ci_low"].notna().all()
    assert intervals["delta_mean_auc_t_ci_low"].notna().all()


def test_one_mode_gets_its_own_intervals_and_no_delta(runs):
    default_run, drop_run, _ = runs
    model, landmark, config = TASKS[0]
    cp.patient_risk_path(default_run, model, landmark, config).unlink()
    intervals, _ = cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)
    row = intervals.set_index(["model", "landmark", "config"]).loc[TASKS[0]]
    assert np.isnan(row["c_index_default_ci_low"])
    assert np.isfinite(row["c_index_adjust_only_ci_low"])
    assert np.isnan(row["delta_c_index_ci_low"])


def test_task_without_stored_metrics_is_left_out(runs):
    default_run, drop_run, _ = runs
    model, landmark, config = TASKS[0]
    metrics_name = next(name for m, c, name in cp.MULTIVARIATE_TASK_SPECS if (m, c) == (model, config))
    for run in (default_run, drop_run):
        (cp.patient_risk_path(run, model, landmark, config).parent / metrics_name).unlink()
    intervals, _ = cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)
    assert len(intervals) == len(TASKS) - 1


def test_empty_trees_give_empty_frames_that_still_merge(tmp_path):
    _build_inputs(tmp_path / "inputs")
    intervals, draws = cp.bootstrap_adjust_only_comparison(
        _run(tmp_path, "default"), _run(tmp_path, "drop"), n_boot=N_BOOT, seed=42,
    )
    assert intervals.empty and draws.empty
    comparison = pd.DataFrame({
        "model": ["xgboost"], "landmark": [0], "config": ["both"], "endpoint": ["platinum"],
    })
    merged = comparison.merge(intervals, on=["model", "landmark", "config", "endpoint"], how="left")
    assert merged["delta_c_index_ci_low"].isna().all()


def test_stale_risk_file_stops_the_section(runs):
    default_run, drop_run, _ = runs
    path = cp.patient_risk_path(default_run, *TASKS[0])
    stale = pd.read_csv(path)
    stale["risk_score"] = -stale["risk_score"]
    stale.to_csv(path, index=False)
    with pytest.raises(ValueError, match="default default elastic-net"):
        cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)


def test_modes_with_different_test_patients_are_refused(runs):
    default_run, drop_run, _ = runs
    path = cp.patient_risk_path(default_run, *TASKS[0])
    frame = pd.read_csv(path)
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="differ in test patients"):
        cp.bootstrap_adjust_only_comparison(default_run, drop_run, n_boot=N_BOOT, seed=42)


def _notebook():
    path = Path(cp.SURVIVAL_DIR) / "03c_multivariate_adjust_only.ipynb"
    return {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]}


def test_notebook_stores_survival_and_writes_only_to_the_adjust_only_tree():
    cells = _notebook()
    assert "cp.SAVE_TEST_SURVIVAL = True" in cells["c03c01"]
    source = cells["c03c13b"]
    assert source.count(".to_csv(") == 2
    assert source.count('drop_run["output_dir"]') == 2
    assert "default_run[" not in source


def test_notebook_table_formats_every_arm(runs):
    default_run, drop_run, _ = runs
    shown = []
    keys = ["model", "landmark", "config", "endpoint"]
    metrics = ["c_index", "mean_auc_t", "integrated_brier"]
    frames = []
    for mode, run in (("default", default_run), ("adjust_only", drop_run)):
        frames.append(cp.summarize_outputs(run)[keys + metrics].rename(
            columns={metric: f"{metric}_{mode}" for metric in metrics}
        ))
    comparison = frames[0].merge(frames[1], on=keys, how="outer")
    for metric in metrics:
        comparison[f"delta_{metric}"] = comparison[f"{metric}_adjust_only"] - comparison[f"{metric}_default"]
    drop_run["output_dir"].mkdir(parents=True, exist_ok=True)
    scope = dict(
        cp=cp, pd=pd, display=shown.append, KEYS=keys, METRICS=metrics, N_BOOTSTRAP=N_BOOT, SEED=42,
        DEFAULT_RUNS=[default_run], RUNS=[drop_run],
        comparison_dfs={cp.run_key(drop_run): comparison},
    )
    exec(compile(_notebook()["c03c13b"], "c03c13b", "exec"), scope)
    table = shown[0]
    assert table.shape == (len(TASKS), 9)
    assert table.iloc[0][("c_index", "delta")].count("(") == 1
    assert not table.apply(lambda column: column.str.contains("no CI")).any().any()
    assert (drop_run["output_dir"] / "default_vs_adjust_only_bootstrap_ci.csv").exists()
    assert (drop_run["output_dir"] / "default_vs_adjust_only_bootstrap_draws.csv").exists()
    assert not list(default_run["output_dir"].glob("default_vs_adjust_only*"))
