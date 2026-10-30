"""Notebook 10c compares adjust-only fits against notebook 10's full fits."""

import json
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import cox_aggregated as ca
from COMPASS.survival_analysis.predictive_checklists.evaluation import evaluate_horizon
from COMPASS.survival_analysis.predictive_checklists.survival import CensoringDistribution
from survival_common.heldout_bootstrap import score_heldout
from survival_common.helper import survival_columns

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "COMPASS/survival_analysis"
LANDMARKS = (0, 90, 180)
HORIZONS = np.array([52.0, 104.0])
TIME_UNIT_DAYS = 7
MANIFEST = {
    "auc_time_unit_days": TIME_UNIT_DAYS,
    "auc_max_time_units": 522,
    "auc_horizons_by_landmark": {str(lm): {"platinum": HORIZONS.tolist()} for lm in LANDMARKS},
}


def _cells(name):
    return [("".join(c["source"]), c["cell_type"])
            for c in json.loads((NOTEBOOK_DIR / name).read_text())["cells"]]


def _code(cells, starts_with):
    """Locate by content: notebook 10's cell positions are not a stable contract."""
    return next(src for src, kind in cells if kind == "code" and src.startswith(starts_with))


def _load_setup(name, tmp_path):
    cells = _cells(name)
    setup = next(src for src, kind in cells if kind == "code").replace(
        'Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")', f"Path({str(tmp_path)!r})"
    )
    scope = {}
    exec(compile(setup, f"{name}_setup", "exec"), scope)
    return cells, scope


def _write_inputs(inputs):
    rng = np.random.default_rng(0)
    inputs.mkdir()
    (inputs / "build_manifest.json").write_text(json.dumps(MANIFEST))
    n = 180
    frame = pd.DataFrame({
        ca.ID_COL: [f"P{i:03d}" for i in range(n)],
        "split": ["train"] * 120 + ["test"] * 60,
        "t_platinum": rng.uniform(30, 1500, n),
        "PLATINUM": rng.integers(0, 2, n),
        "PSA__mean": rng.normal(size=n),
        "PSA__n_observations": rng.integers(1, 9, n),
        "PSA__delta": rng.normal(size=n),
        "Body weight__mean": rng.normal(size=n),
    })
    for lm in LANDMARKS:
        frame.to_csv(inputs / f"aggregated_landmark{lm}.csv", index=False)
        (inputs / f"pre_treatment_lab_long_landmark{lm}.csv").touch()
    return frame


def _fake_fit(task, frame, *, leak=False):
    """Write what a real fit writes, with stored metrics the bootstrap can reproduce."""
    seed = LANDMARKS.index(task["landmark"]) + 10 * (task["model"] == "xgboost")
    rng = np.random.default_rng(seed + 100 * (task.get("arm", "full") == "full"))
    reference = frame.loc[frame["split"] == "train", ["t_platinum", "PLATINUM"]].rename(
        columns={"t_platinum": "duration_days", "PLATINUM": "event"})
    test = frame.loc[frame["split"] == "test"]
    duration = test["t_platinum"].to_numpy(dtype=float)
    event = test["PLATINUM"].to_numpy(dtype=int)
    risk = -duration / 500 + rng.normal(size=len(test))
    surv = np.exp(-np.outer(np.exp(risk) / 3000, HORIZONS * TIME_UNIT_DAYS))
    scores = score_heldout(duration, event, risk, surv, reference=reference, horizons=HORIZONS,
                           time_unit_days=TIME_UNIT_DAYS, max_time_unit=522)
    pd.DataFrame({ca.ID_COL: test[ca.ID_COL], "endpoint": "platinum", "dataset": "test",
                  "duration_days": duration, "event": event, "risk_score": risk,
                  **survival_columns(surv, HORIZONS)}).to_csv(task["risks"], index=False)
    pd.DataFrame([dict(endpoint="platinum", n_test=len(test), n_events_test=int(event.sum()),
                       **scores)]).to_csv(task["metrics"], index=False)
    folder, landmark = task["folder"], task["landmark"]
    reported = ["age", "PSA__mean", *(["PSA__n_observations"] if leak else [])]
    if task["model"] == "elastic-net":
        pd.DataFrame(dict(feature=reported, coef=0.1)).to_csv(
            folder / f"cox_agg_multivariable_landmark{landmark}.csv", index=False)
        pd.DataFrame(dict(feature=["PSA__n_observations"], coef=[0.2])).to_csv(
            folder / f"cox_agg_multivariable_adjusters_landmark{landmark}.csv", index=False)
    else:
        pd.DataFrame(dict(feature=reported, gain=1.0)).to_csv(
            folder / f"landmark_xgboost_feature_importance_landmark{landmark}.csv", index=False)


def _checklist_item(feature, op=">=", threshold=1.0):
    return dict(name=f"{feature} {op} {threshold}", feature=feature, op=op, threshold=threshold, group="labs")


def _fake_checklist(task, frame, *, extra_items=()):
    """Write what a real run writes; late horizons fail, as they can in a small cohort."""
    folder = task["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1 + 100 * (task["arm"] == "full"))
    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]
    duration = test["t_platinum"].to_numpy(dtype=float)
    event = test["PLATINUM"].to_numpy(dtype=int)
    censoring = CensoringDistribution.fit(train["t_platinum"].to_numpy(), train["PLATINUM"].to_numpy())
    items = [_checklist_item("AGE_AT_TREATMENTSTART"), _checklist_item("PSA__mean"), *extra_items]
    if task["arm"] == "full":
        items += [_checklist_item("PSA__n_observations"), _checklist_item("PSA__mean", "is_missing", None)]
    rows = []
    for landmark, horizon in product(task["settings"]["landmarks"], task["settings"]["horizons"]):
        cell = dict(endpoint="platinum", landmark_days=landmark, horizon_days=horizon, model="checklist")
        if horizon > 730:
            rows.append({**cell, "status": "unsupported", "reason": "Too few controls"})
            continue
        score = rng.integers(0, 4, len(test))
        flag = score >= 2
        probability = (score + 1) / 5
        metrics = evaluate_horizon(duration, event, score, flag, probability, horizon, censoring)
        rows.append({**cell, "status": "ok", "n": len(test), "auc": metrics["auc"], "brier": metrics["brier"],
                     "n_items": len(items), "threshold": 2})
        horizon_dir = folder / "platinum" / f"landmark_{landmark}" / "full" / f"horizon_{horizon}"
        horizon_dir.mkdir(parents=True)
        (horizon_dir / "checklist.json").write_text(json.dumps(dict(threshold=2, items=items)))
        # Shuffled, with in-sample rows first: the notebook must align on patient and dataset.
        tested = pd.DataFrame({ca.ID_COL: test[ca.ID_COL].to_numpy(), "split": "test", "dataset": "test",
                               "score": score, "flag": flag, "probability": probability})
        in_sample = pd.DataFrame({ca.ID_COL: train[ca.ID_COL].to_numpy(), "split": "train",
                                  "dataset": "train_val_in_sample", "score": 0, "flag": False, "probability": 0.5})
        pd.concat([in_sample, tested.sample(frac=1, random_state=0)]).to_csv(
            horizon_dir / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    # A real manifest omits settings left at their defaults.
    config = {key: value for key, value in task["settings"].items() if (key, value) != ("adjust_only_mode", "none")}
    (folder / "run_manifest.json").write_text(json.dumps(dict(status="complete", config=config)))


def _write_fits(scope, frame, *, checklist_arms=("full", "adjust_only"), extra_items=()):
    for task in scope["TASKS"]:
        task["folder"].mkdir(parents=True, exist_ok=True)
        _fake_fit(task, frame)
    for task in scope["CHECKLIST_TASKS"]:
        if task["arm"] in checklist_arms:
            _fake_checklist(task, frame, extra_items=extra_items if task["arm"] == "adjust_only" else ())


def _is_checklist(command):
    return "COMPASS.survival_analysis.predictive_checklists" in command


def _horizon_dir(task, landmark, horizon):
    return task["folder"] / "platinum" / f"landmark_{landmark}" / "full" / f"horizon_{horizon}"


def _run_cells(cells, scope, *, skip_fit=False):
    for src, kind in cells:
        if kind != "code" or src.startswith("import json"):
            continue
        if skip_fit and src.startswith(("fit_counts", "checklist_counts")):
            continue
        exec(compile(src, "notebook_cell", "exec"), scope)


@pytest.fixture
def notebooks(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.chdir(ROOT)
    cells_10, scope_10 = _load_setup("10_platinum_feature_ablation.ipynb", tmp_path)
    cells_c, scope_c = _load_setup("10c_platinum_adjust_only_vs_full.ipynb", tmp_path)
    scope_c["N_BOOTSTRAP"] = 20
    frame = _write_inputs(scope_c["INPUTS_DIR"])
    return cells_10, scope_10, cells_c, scope_c, frame


def test_full_arm_is_notebook_10_and_adjust_only_is_parallel(notebooks):
    _, scope_10, _, scope_c, _ = notebooks
    tasks = scope_c["TASKS"]
    assert len(tasks) == len({t["folder"] for t in tasks}) == 12
    ablation = {t["folder"]: t for t in scope_10["TASKS"]}
    for task in tasks:
        assert task["vitals"] == "without_vitals" and task["stats"] == "mean-n-obs"
        command = task["command"]
        assert "--exclude-vitals" in command
        assert command[command.index("--feature-stats") + 1] == "mean-n-obs"
        if task["arm"] == "full":
            # Equal commands and folders are what let the notebooks resume each other.
            assert command == ablation[task["folder"]]["command"]
            assert task["risks"] == ablation[task["folder"]]["risks"]
            assert "--adjust-only-mode" not in command
        else:
            assert task["folder"] not in ablation
            assert "local_runs_adt_adjust_only" in str(task["folder"])
            assert command[-2:] == ["--adjust-only-mode", "drop"]


def test_reuses_notebook_10_fits_and_reports_paired_deltas(notebooks, monkeypatch):
    cells_10, scope_10, cells_c, scope_c, frame = notebooks
    full = {t["folder"] for t in scope_c["TASKS"] if t["arm"] == "full"}
    scope_10["TASKS"] = [t for t in scope_10["TASKS"] if t["folder"] in full]
    scope_10["build_manifest"] = MANIFEST
    commands = []

    def fit(command, **kwargs):
        commands.append(command)
        for scope in (scope_10, scope_c):
            for task in scope["TASKS"]:
                if task["command"] == command:
                    _fake_fit(task, frame)
                    return
        task = next(t for t in scope_c["CHECKLIST_TASKS"] if t["command"] == command)
        _fake_checklist(task, frame)

    monkeypatch.setattr(subprocess, "run", fit)
    exec(compile(_code(cells_10, "for task in TASKS"), "notebook_10_fit", "exec"), scope_10)
    assert len(commands) == 6
    commands.clear()
    _run_cells(cells_c, scope_c)
    # Every full fit resumes from notebook 10: its settings must compare equal.
    survival = [command for command in commands if not _is_checklist(command)]
    assert len(survival) == 6
    assert all("--adjust-only-mode" in command for command in survival)
    assert scope_c["fit_counts"] == dict(completed=6, skipped=6)
    assert scope_c["checklist_counts"] == dict(completed=2, skipped=0)

    comparison = scope_c["comparison"]
    assert len(comparison) == 6
    assert np.allclose(comparison["delta_test_mean_auc_t"],
                       comparison["test_mean_auc_t_adjust_only"] - comparison["test_mean_auc_t_full"])
    for metric in scope_c["metrics"]:
        assert comparison[f"delta_{metric}_ci_low"].notna().all()
        assert (comparison[f"delta_{metric}_ci_low"] <= comparison[f"delta_{metric}_ci_high"]).all()
    root = scope_c["ADJUST_ONLY_ROOT"]
    for name in ("summary", "comparison", "bootstrap_ci", "bootstrap_draws"):
        assert (root / f"adjust_only_vs_full_{name}.csv").is_file()
    assert (root / "adjust_only_vs_full_mean_auc.png").is_file()
    assert pd.read_csv(root / "adjust_only_vs_full_comparison.csv")["delta_test_mean_auc_t_ci_low"].notna().all()
    assert all(not row["non_mean_predictors"] for row in scope_c["checks"])


def test_reported_adjuster_stops_the_comparison(notebooks, monkeypatch):
    _, _, cells_c, scope_c, frame = notebooks

    def fit(command, **kwargs):
        task = next(t for t in scope_c["TASKS"] if t["command"] == command)
        _fake_fit(task, frame, leak=True)

    monkeypatch.setattr(subprocess, "run", fit)
    with pytest.raises(ValueError, match="Adjusters reported"):
        _run_cells(cells_c, scope_c)


def test_one_finished_arm_shows_values_without_deltas(notebooks, monkeypatch, capsys):
    _, _, cells_c, scope_c, frame = notebooks
    for task in scope_c["TASKS"]:
        if task["arm"] == "full":
            task["folder"].mkdir(parents=True)
            _fake_fit(task, frame)
    _run_cells(cells_c, scope_c, skip_fit=True)
    assert "6 of 12 fits are not comparable yet" in capsys.readouterr().out
    comparison = scope_c["comparison"]
    assert len(comparison) == 6
    assert comparison["test_mean_auc_t_full"].notna().all()
    assert comparison["delta_test_mean_auc_t"].isna().all()
    assert comparison["delta_test_mean_auc_t_ci_low"].isna().all()


def test_dry_run_writes_nothing(notebooks, monkeypatch):
    _, _, cells_c, scope_c, _ = notebooks
    scope_c["DRY_RUN"] = True
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("dry run must not fit"))
    _run_cells(cells_c, scope_c)
    assert not scope_c["ADJUST_ONLY_ROOT"].exists()
    assert not scope_c["FULL_ROOT"].exists()
    assert scope_c["comparison"].empty
    assert scope_c["checklist_comparison"].empty


def test_checklist_arms_differ_only_in_mode_and_folder(notebooks):
    *_, scope_c, _ = notebooks
    full, adjust_only = scope_c["CHECKLIST_TASKS"]
    assert (full["arm"], adjust_only["arm"]) == ("full", "adjust_only")
    assert full["folder"] != adjust_only["folder"]
    assert scope_c["CHECKLIST_HORIZONS"] == [365, 1825]
    for task, mode in ((full, "none"), (adjust_only, "drop")):
        command = task["command"]
        assert _is_checklist(command)
        assert command[command.index("--adjust-only-mode") + 1] == mode
        assert command[command.index("--feature-stats") + 1] == "mean-n-obs"
        assert command[command.index("--endpoints") + 1:command.index("--landmarks")] == ["platinum"]
        assert "--exclude-vitals" in command and "--no-baselines" in command
    different = [(a, b) for a, b in zip(full["command"], adjust_only["command"]) if a != b]
    assert different == [(str(full["folder"].resolve()), str(adjust_only["folder"].resolve())), ("none", "drop")]


def test_checklists_report_paired_deltas(notebooks, capsys):
    _, _, cells_c, scope_c, frame = notebooks
    _write_fits(scope_c, frame)
    _run_cells(cells_c, scope_c, skip_fit=True)

    assert "6 of 12 checklist cells are not comparable" in capsys.readouterr().out
    comparison = scope_c["checklist_comparison"]
    assert len(comparison) == 3  # three landmarks at the one supported horizon
    assert np.allclose(comparison["delta_auc"], comparison["auc_adjust_only"] - comparison["auc_full"])
    assert (comparison["delta_n_items"] == -2).all()
    for metric in ("auc", "brier"):
        assert comparison[f"delta_{metric}_ci_low"].notna().all()
        assert (comparison[f"delta_{metric}_ci_low"] <= comparison[f"delta_{metric}_ci_high"]).all()
    root = scope_c["ADJUST_ONLY_ROOT"]
    for name in ("summary", "items", "comparison", "bootstrap_ci", "bootstrap_draws"):
        assert (root / f"adjust_only_vs_full_checklist_{name}.csv").is_file()
    assert (root / "adjust_only_vs_full_checklist_auc.png").is_file()
    draws = pd.read_csv(root / "adjust_only_vs_full_checklist_bootstrap_draws.csv")
    assert len(draws) == 3 * 2 * scope_c["N_BOOTSTRAP"]


@pytest.mark.parametrize("item", [
    _checklist_item("PSA__n_observations"),
    _checklist_item("PSA__mean", "is_missing", None),
    _checklist_item("Body weight__mean"),
    _checklist_item("PSA__delta"),
])
def test_checklist_item_outside_its_pool_stops_the_comparison(notebooks, item):
    _, _, cells_c, scope_c, frame = notebooks
    _write_fits(scope_c, frame, extra_items=[item])
    with pytest.raises(ValueError, match="outside the candidate pool"):
        _run_cells(cells_c, scope_c, skip_fit=True)


def test_one_finished_checklist_arm_shows_values_without_deltas(notebooks):
    _, _, cells_c, scope_c, frame = notebooks
    _write_fits(scope_c, frame, checklist_arms=("full",))
    _run_cells(cells_c, scope_c, skip_fit=True)
    comparison = scope_c["checklist_comparison"]
    assert len(comparison) == 3
    assert comparison["auc_full"].notna().all()
    assert comparison["delta_auc"].isna().all()
    assert comparison["delta_auc_ci_low"].isna().all()


def test_finished_checklist_runs_are_skipped_unless_settings_changed(notebooks, monkeypatch):
    _, _, cells_c, scope_c, frame = notebooks
    _write_fits(scope_c, frame)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("finished run must not refit"))
    fit_cell = compile(_code(cells_c, "checklist_counts"), "checklist_fit", "exec")
    exec(fit_cell, scope_c)
    assert scope_c["checklist_counts"] == dict(completed=0, skipped=2)

    manifest_path = scope_c["CHECKLIST_TASKS"][1]["folder"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["adjust_only_mode"] = "none"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Settings changed"):
        exec(fit_cell, scope_c)


def test_interrupted_checklist_run_goes_back_to_the_cli(notebooks, monkeypatch):
    _, _, cells_c, scope_c, frame = notebooks
    _write_fits(scope_c, frame)
    task = scope_c["CHECKLIST_TASKS"][0]
    manifest_path = task["folder"] / "run_manifest.json"
    manifest_path.write_text(json.dumps(dict(status="running", config=task["settings"])))
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    exec(compile(_code(cells_c, "checklist_counts"), "checklist_fit", "exec"), scope_c)
    assert commands == [task["command"]] and "--resume" in task["command"]
    assert scope_c["checklist_counts"] == dict(completed=1, skipped=1)


def test_checklist_metrics_must_reproduce_from_saved_predictions(notebooks):
    _, _, cells_c, scope_c, frame = notebooks
    _write_fits(scope_c, frame)
    path = scope_c["CHECKLIST_TASKS"][0]["folder"] / "metrics.csv"
    stored = pd.read_csv(path)
    stored.loc[stored["status"] == "ok", "auc"] += 0.01
    stored.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Recomputed metrics differ"):
        _run_cells(cells_c, scope_c, skip_fit=True)


def test_checklist_predictions_must_cover_the_test_set(notebooks):
    _, _, cells_c, scope_c, frame = notebooks
    _write_fits(scope_c, frame)
    path = _horizon_dir(scope_c["CHECKLIST_TASKS"][0], 0, 365) / "predictions.csv"
    saved = pd.read_csv(path)
    saved.drop(saved.index[-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Test patients are missing"):
        _run_cells(cells_c, scope_c, skip_fit=True)
