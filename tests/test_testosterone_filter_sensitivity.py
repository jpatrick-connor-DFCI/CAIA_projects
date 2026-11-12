"""Synthetic checks for notebook 09's eligibility and cache lifecycle."""
import ast
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from COMPASS.pipeline import compass_pipeline as cp


NOTEBOOK = Path(__file__).resolve().parents[1] / (
    "COMPASS/notebooks/testing/09_cohort_sensitivity_testosterone_filter.ipynb"
)


def run_cell(env, index, *, functions_only=False):
    cell = json.loads(NOTEBOOK.read_text())["cells"][index]
    tree = ast.parse("".join(cell["source"]))
    if functions_only:
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, f"{NOTEBOOK}:cell_{index}", "exec"), env)


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    env = dict(np=np, pd=pd, pl=pl, Path=Path, json=json, hashlib=hashlib, uuid=uuid,
               cp=cp, ID_COL="DFCI_MRN", TESTOSTERONE_LAB=cp.TESTOSTERONE_LAB_NAME,
               LABS_OF_INTEREST=("PSA", cp.TESTOSTERONE_LAB_NAME),
               TESTOSTERONE_WINDOW_DAYS=183, ABOVE_CASTRATE_NG_DL=50.0,
               QUALIFYING_IDS={1}, EXCLUDED_IDS={2}, REBUILD_INPUTS=False, OVERWRITE=False)
    for key in ("STAGE1_PATH", "QUALIFYING_PATH"):
        env[key] = tmp_path / f"{key}.csv"
        env[key].write_text("DFCI_MRN\n1\n")
    run = dict(label="adt_noprecastrate_t6mo", title="Synthetic sensitivity",
               cohort="all", endpoint="platinum", anchor_col="none", landmarks=[180],
               inputs_dir=tmp_path / "inputs", output_dir=tmp_path / "outputs",
               input_csv=tmp_path / "longitudinal.csv", restrict_to_mrns=env["STAGE1_PATH"],
               exclude_mrns=tmp_path / "excluded.csv")
    run["input_csv"].write_text("synthetic-source-1\n")
    run["exclude_mrns"].write_text("DFCI_MRN\n2\n")
    for key in ("inputs_dir", "output_dir"):
        run[key].mkdir()
    env["NEW_RUNS"] = [run]
    calls = {"build": 0, "fit": [], "fail_build": False, "fail_fit": False}

    def build(run):
        calls["build"] += 1
        if calls["fail_build"]:
            raise RuntimeError("synthetic build failure")
        (run["inputs_dir"] / "build_manifest.json").write_text("{}")
        pd.DataFrame({"DFCI_MRN": sorted(env["QUALIFYING_IDS"])}).to_csv(
            run["inputs_dir"] / "aggregated_landmark180.csv", index=False)
        (run["inputs_dir"] / "pre_treatment_lab_long_landmark180.csv").write_text("DFCI_MRN\n")

    def subprocess_run(cmd, dry_run=False):
        # Use the real task skip/overwrite logic, replacing only the costly fit.
        calls["fit"].append("--overwrite" in cmd)
        if calls["fail_fit"]:
            return 1
        output = Path(cmd[cmd.index("--output-dir") + 1])
        (output / "cox_agg_univariate_nobs_adjusted.csv").write_text("synthetic-result\n")
        return 0

    monkeypatch.setattr(cp, "BUILD_FULL_FOLLOWUP", False)
    monkeypatch.setattr(cp, "FORCE_RERUN", False)
    monkeypatch.setattr(cp, "build_prediction_inputs", build)
    monkeypatch.setattr(cp, "_run", subprocess_run)
    run_cell(env, 3, functions_only=True)
    return env, run, calls


def test_filter_boundaries_and_nonfinite_measurements(workflow, tmp_path):
    env, _, _ = workflow
    rows = [(1, 50, -183), (2, 100, 0), (3, 100, -184),
            (4, 100, -10), (4, 49, -500), (5, 100, -10),
            (6, np.inf, -10), (7, 100, -np.inf), (8, np.nan, -10),
            (9, 100, np.nan), (10, -np.inf, -10)]
    labs = pd.DataFrame(rows, columns=["DFCI_MRN", "LAB_VALUE", "t_lab"])
    labs["LAB_NAME"] = cp.TESTOSTERONE_LAB_NAME
    path = tmp_path / "labs.csv"
    labs.to_csv(path, index=False)
    loaded = env["load_longitudinal_labs"](path)
    assert np.isfinite(loaded[["LAB_VALUE", "t_lab"]]).all().all()
    for frame in (labs, loaded):
        qualifying, excluded, _ = env["derive_testosterone_filter"](
            set(range(1, 11)), frame, window_days=183, threshold=50)
        assert set(qualifying.DFCI_MRN) == {1, 5}
        assert excluded == set(range(1, 11)) - {1, 5}


def test_unchanged_inputs_and_completed_fits_are_reused(workflow):
    env, _, calls = workflow
    for _ in range(2):
        run_cell(env, 7)
        run_cell(env, 8)
    assert calls["build"] == 1
    assert calls["fit"] == [True]
    assert cp.FORCE_RERUN is False


@pytest.mark.parametrize("source", ["input_csv", "restrict_to_mrns", "exclude_mrns", "qualifying"])
def test_source_content_change_invalidates_inputs_and_models(workflow, source):
    env, run, calls = workflow
    run_cell(env, 7)
    run_cell(env, 8)
    path = env["QUALIFYING_PATH"] if source == "qualifying" else run[source]
    stat = path.stat()
    path.write_text(path.read_text().replace("1", "3").replace("2", "4"))
    # Fingerprints must detect content changes even when size and mtime match.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    run_cell(env, 7)
    run_cell(env, 8)
    assert calls["build"] == 2
    assert calls["fit"] == [True, True]


def test_wider_filter_rebuilds_instead_of_accepting_old_subset(workflow):
    env, run, calls = workflow
    run_cell(env, 7)
    run_cell(env, 8)
    env.update(TESTOSTERONE_WINDOW_DAYS=365, QUALIFYING_IDS={1, 2}, EXCLUDED_IDS=set())
    run_cell(env, 7)
    run_cell(env, 8)
    actual = pd.read_csv(run["inputs_dir"] / "aggregated_landmark180.csv")
    assert set(actual.DFCI_MRN) == {1, 2}
    assert calls["build"] == 2 and calls["fit"] == [True, True]


def test_forced_rebuild_requires_refit_even_after_restart(workflow):
    env, run, calls = workflow
    run_cell(env, 7)
    run_cell(env, 8)
    env["REBUILD_INPUTS"] = True
    run_cell(env, 7)
    assert not (run["output_dir"] / env["FIT_PROVENANCE_FILENAME"]).exists()
    # Simulate resuming with default flags: no in-memory rebuilt-run list is needed.
    env["REBUILD_INPUTS"] = False
    run_cell(env, 7)
    run_cell(env, 8)
    assert calls["build"] == 2 and calls["fit"] == [True, True]


def test_failed_fit_does_not_allow_old_results_to_be_reused(workflow):
    env, run, calls = workflow
    run_cell(env, 7)
    run_cell(env, 8)
    env["OVERWRITE"] = True
    calls["fail_fit"] = True
    with pytest.raises(RuntimeError, match="Univariate fitting failed"):
        run_cell(env, 8)
    assert cp.FORCE_RERUN is False
    assert not (run["output_dir"] / env["FIT_PROVENANCE_FILENAME"]).exists()
    env["OVERWRITE"] = False
    calls["fail_fit"] = False
    run_cell(env, 8)
    assert calls["fit"] == [True, True, True]


def test_interrupted_rebuild_invalidates_both_completion_records(workflow):
    env, run, calls = workflow
    run_cell(env, 7)
    run_cell(env, 8)
    env["REBUILD_INPUTS"] = True
    calls["fail_build"] = True
    with pytest.raises(RuntimeError, match="synthetic build failure"):
        run_cell(env, 7)
    with pytest.raises(RuntimeError, match="build cell before fitting"):
        run_cell(env, 8)
    assert not (run["output_dir"] / env["FIT_PROVENANCE_FILENAME"]).exists()
    env["REBUILD_INPUTS"] = False
    calls["fail_build"] = False
    run_cell(env, 7)
    run_cell(env, 8)
    assert calls["build"] == 3 and calls["fit"] == [True, True]


def test_missing_companion_input_triggers_rebuild_and_refit(workflow):
    env, run, calls = workflow
    run_cell(env, 7)
    run_cell(env, 8)
    (run["inputs_dir"] / "pre_treatment_lab_long_landmark180.csv").unlink()
    run_cell(env, 7)
    run_cell(env, 8)
    assert calls["build"] == 2 and calls["fit"] == [True, True]
