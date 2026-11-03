"""Notebook 03 drives both predictive-checklist configs through the CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "COMPASS/notebooks"
INPUTS_ROOT = 'Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")'


def _code(name):
    return ["".join(c["source"]) for c in json.loads((NOTEBOOK_DIR / name).read_text())["cells"]
            if c["cell_type"] == "code"]


def _checklist_cells(name):
    """Locate by content: cell positions are not a stable contract."""
    cells = _code(name)
    start = next(i for i, src in enumerate(cells) if "CHECKLIST_RUNS = {" in src)
    return cells[start:]


def _run_notebook_03(tmp_path, monkeypatch, *, train=False, overwrite=False):
    commands = []
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    scope = {"sys": sys, "pd": pd, "display": lambda frame: None}
    for source in _checklist_cells("03_multivariate.ipynb"):
        source = (source.replace(INPUTS_ROOT, f"Path({str(tmp_path)!r})")
                  .replace("RUN_CHECKLISTS = False", f"RUN_CHECKLISTS = {train}")
                  .replace("CHECKLIST_OVERWRITE = False", f"CHECKLIST_OVERWRITE = {overwrite}"))
        exec(compile(source, "notebook_cell", "exec"), scope)
    return commands, scope


def _option(command, flag):
    return command[command.index(flag) + 1]


def test_default_validates_both_inputs_without_training(tmp_path, monkeypatch):
    commands, _ = _run_notebook_03(tmp_path, monkeypatch)
    assert [Path(_option(c, "--output-dir")).name for c in commands] == [
        "preflight", "preflight_gleason_trio"]
    for command in commands:
        assert command[:3] == [sys.executable, "-m", "COMPASS.multivariable_models.predictive_checklists"]
        # Preflight writes run_manifest.json, so a rerun without --overwrite exits with an error.
        assert {"--preflight-only", "--overwrite"} <= set(command)
    assert _option(commands[0], "--inputs-root") == str(tmp_path)
    subset = "somatic_gleason/gleason_somatic_available_case"
    assert _option(commands[1], "--platinum-inputs") == str(tmp_path / "prediction_inputs_adt" / subset)
    assert _option(commands[1], "--nepc-inputs") == str(tmp_path / "prediction_inputs_adt_nepc" / subset)


@pytest.mark.parametrize("overwrite, flag", [(False, "--resume"), (True, "--overwrite")])
def test_training_resumes_unless_overwriting(tmp_path, monkeypatch, overwrite, flag):
    commands, _ = _run_notebook_03(tmp_path, monkeypatch, train=True, overwrite=overwrite)
    training = [c for c in commands if "--preflight-only" not in c]
    assert [Path(_option(c, "--output-dir")).name for c in training] == ["full_adt", "gleason_trio_labs"]
    other = ({"--overwrite", "--resume"} - {flag}).pop()
    for command in training:
        assert flag in command and other not in command


def test_results_cell_renders_no_standalone_figures(tmp_path, monkeypatch):
    # Checklist figures come from COMPASS/figures (the federated checklist
    # supplement); the results cell only displays metrics and checklists.
    finished = tmp_path / "predictive_checklists" / "full_adt"
    finished.mkdir(parents=True)
    pd.DataFrame({"endpoint": ["platinum"], "model": ["checklist"], "status": ["ok"]}).to_csv(
        finished / "metrics.csv", index=False)
    commands, _ = _run_notebook_03(tmp_path, monkeypatch)
    assert not [c for c in commands if c[0] == "Rscript"]


def test_run_definitions_match_configs_and_adjust_only_notebook(tmp_path, monkeypatch):
    _, scope = _run_notebook_03(tmp_path, monkeypatch)
    adjust_only = {"sys": sys}
    setup = next(src for src in _checklist_cells("03c_multivariate_adjust_only.ipynb"))
    exec(compile(setup.replace(INPUTS_ROOT, f"Path({str(tmp_path)!r})"), "notebook_cell", "exec"), adjust_only)
    assert scope["CHECKLIST_RUNS"].keys() == adjust_only["CHECKLIST_RUNS"].keys()
    for name, run in scope["CHECKLIST_RUNS"].items():
        config, input_args, landmarks, horizons, _, output = run
        # 03c's default arm reads the run 03 writes, and the two must fit the same settings.
        assert adjust_only["CHECKLIST_RUNS"][name][:5] == (config, input_args, landmarks, horizons, output)
        # The CLI overrides must restate the config: a mismatch would refuse --resume.
        settings = json.loads(config.read_text())
        assert (settings["landmarks"], settings["horizons"]) == (landmarks, horizons)
