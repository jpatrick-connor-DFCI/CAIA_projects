"""The checklist supplement tables flatten finished runs without refitting."""
from __future__ import annotations

from COMPASS.survival_analysis.predictive_checklists.collect_supplement import collect, write
from COMPASS.survival_analysis.predictive_checklists.config import RunConfig
from COMPASS.survival_analysis.predictive_checklists.demo import build_demo_inputs
from COMPASS.survival_analysis.predictive_checklists.runner import run


def test_collect_flattens_items_and_metrics(tmp_path):
    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    dirs = {ep: inputs / ("prediction_inputs_adt" + ("_nepc" if ep == "nepc" else ""))
            for ep in ("platinum", "nepc")}
    config = RunConfig(landmarks=(0,), horizons=(365,), max_items=(2,), n_folds=2,
                       quantiles=(.25, .5, .75), min_events=2, min_controls=2,
                       bootstrap_samples=20, baselines=(), solver_time_limit=10,
                       resolved_nepc_sensitivity=False).validate()
    assert run(config, dirs, tmp_path / "labs") == 0

    cells, items, metrics = collect({"labs": tmp_path / "labs"}, horizon=365)
    assert set(cells.endpoint) == {"platinum", "nepc"}
    assert (cells.status == "ok").all()
    # Every fitted cell lists exactly n_items checklist rows.
    counts = items.groupby(["endpoint", "landmark_days"]).size()
    for row in cells.itertuples():
        assert counts[(row.endpoint, row.landmark_days)] == row.n_items
    held_out = metrics[(metrics.model == "checklist") & (metrics.dataset == "test")]
    assert {"auc", "brier", "auc_lower", "auc_upper"} <= set(held_out.columns)

    write({"labs": tmp_path / "labs"}, tmp_path / "out", horizon=365)
    for name in ("checklist_cells", "checklist_items_long", "checklist_metrics"):
        assert (tmp_path / "out" / f"{name}.csv").exists()


def test_missing_horizon_is_reported_not_dropped(tmp_path):
    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    dirs = {ep: inputs / ("prediction_inputs_adt" + ("_nepc" if ep == "nepc" else ""))
            for ep in ("platinum", "nepc")}
    config = RunConfig(landmarks=(0,), horizons=(365,), max_items=(2,), n_folds=2,
                       quantiles=(.25, .5, .75), min_events=2, min_controls=2,
                       bootstrap_samples=0, baselines=(), solver_time_limit=10,
                       resolved_nepc_sensitivity=False).validate()
    run(config, dirs, tmp_path / "labs")
    cells, items, _ = collect({"labs": tmp_path / "labs"}, horizon=1825)
    assert len(cells) == 2 and (cells.status != "ok").all() and items.empty


def test_prepare_checklist_supplement_caches_and_skips(tmp_path):
    from COMPASS.survival_analysis.prepare_figure_data import prepare_checklist_supplement

    config = {"cache_root": str(tmp_path / "cache"), "checklist_horizon": 365,
              "checklist_labs_run": str(tmp_path / "labs"),
              "checklist_trio_run": str(tmp_path / "trio")}
    assert prepare_checklist_supplement(config) is None  # no runs delivered yet

    inputs = build_demo_inputs(tmp_path / "inputs", landmarks=(0,), n=160)
    dirs = {ep: inputs / ("prediction_inputs_adt" + ("_nepc" if ep == "nepc" else ""))
            for ep in ("platinum", "nepc")}
    run_config = RunConfig(landmarks=(0,), horizons=(365,), max_items=(2,), n_folds=2,
                           quantiles=(.25, .5, .75), min_events=2, min_controls=2,
                           bootstrap_samples=20, baselines=(), solver_time_limit=10,
                           resolved_nepc_sensitivity=False).validate()
    assert run(run_config, dirs, tmp_path / "labs") == 0

    first = prepare_checklist_supplement(config)  # trio run still absent
    assert first["key"] and all(o.get("bytes") for o in first["outputs"])
    assert any(s.get("missing") for s in first["sources"])
    again = prepare_checklist_supplement(config)
    assert again["key"] == first["key"]
