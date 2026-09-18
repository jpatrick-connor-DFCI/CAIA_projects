"""Phase 5 wiring: the incremental-value step reaches the dynamic arm's output.

The failure this guards against is silent. `incremental_risk.py` defaults
`--predictions` to a conventional filename *inside* `--output-dir`, so a command
pointed at the wrong directory does not crash with a bad flag -- it reports "no
predictions found" and the analysis simply never runs. These tests pin the
directory, the flag spellings and the opt-in gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
SURV = REPO / "COMPASS" / "survival_analysis"
sys.path.insert(0, str(SURV))
sys.path.insert(0, str(SURV / "multivariate_longitudinal"))

pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def cp():
    # The configured data root is cluster-only; importing builds run dicts that
    # would try to create it. Only command construction is under test.
    with mock.patch.object(Path, "mkdir", lambda *a, **k: None):
        import compass_pipeline

    return compass_pipeline


@pytest.fixture
def run(cp, tmp_path):
    return {
        "label": "adt",
        "endpoint": "platinum",
        "cohort": "all",
        "landmarks": [0, 90],
        "output_dir": tmp_path / "local_runs_adt",
        "inputs_dir": tmp_path / "prediction_inputs_adt",
    }


def _capture(cp, run, **attrs):
    """Run the step with _run stubbed; return the argv lists it would execute."""
    seen = []

    def fake_run(cmd, dry_run=False):
        seen.append([str(c) for c in cmd])
        return 0

    with mock.patch.object(cp, "_run", fake_run), mock.patch.object(
        Path, "mkdir", lambda *a, **k: None
    ), mock.patch.multiple(cp, **attrs):
        cp.run_incremental_risk(run, dry_run=True)
    return seen


class TestOptIn:
    def test_off_by_default_runs_nothing(self, cp, run):
        assert _capture(cp, run, RUN_DYNAMIC=False) == []

    def test_on_runs_one_command_per_landmark_and_config(self, cp, run):
        seen = _capture(cp, run, RUN_DYNAMIC=True)
        n_configs = len(cp._LONGITUDINAL_CONFIGS_BY_ENDPOINT["platinum"])
        assert len(seen) == len(run["landmarks"]) * n_configs

    def test_disabled_message_names_the_toggle(self, cp, run, capsys):
        _capture(cp, run, RUN_DYNAMIC=False)
        out = capsys.readouterr().out
        assert "RUN_DYNAMIC" in out, "a silent no-op is indistinguishable from a bug"


class TestCommandTargetsTheDynamicArm:
    def test_output_dir_is_the_dynamic_arms_directory(self, cp, run):
        """The load-bearing assertion: pointed at the landmark arm's directory
        instead, the step would find no predictions and silently do nothing."""
        seen = _capture(cp, run, RUN_DYNAMIC=True)
        out_dir = seen[0][seen[0].index("--output-dir") + 1]
        assert cp.model_output_dir("dynamic-deephit-dyn") in out_dir
        assert cp.model_output_dir("dynamic-deephit") + "/" not in out_dir

    def test_output_dir_matches_where_the_dynamic_arm_writes(self, cp, run):
        """Derive the expected path the way the model task does, rather than
        restating this module's own literal."""
        seen = _capture(cp, run, RUN_DYNAMIC=True)
        for landmark in run["landmarks"]:
            expected = {
                str(
                    run["output_dir"]
                    / cp.model_output_dir("dynamic-deephit-dyn")
                    / f"landmark_{landmark}"
                    / cfg
                )
                for cfg in cp._LONGITUDINAL_CONFIGS_BY_ENDPOINT["platinum"]
            }
            got = {
                c[c.index("--output-dir") + 1]
                for c in seen
                if f"landmark_{landmark}" in c[c.index("--output-dir") + 1]
            }
            assert got == expected

    def test_invokes_the_incremental_risk_script(self, cp, run):
        seen = _capture(cp, run, RUN_DYNAMIC=True)
        assert seen[0][1].endswith("incremental_risk.py")

    def test_carries_endpoint_landmark_config_and_cohort(self, cp, run):
        seen = _capture(cp, run, RUN_DYNAMIC=True)
        cmd = seen[0]
        assert cmd[cmd.index("--endpoint") + 1] == "platinum"
        assert cmd[cmd.index("--cohort") + 1] == "all"
        assert cmd[cmd.index("--landmark-day") + 1] == "0"
        assert cmd[cmd.index("--config") + 1] in cp._LONGITUDINAL_CONFIGS_BY_ENDPOINT[
            "platinum"
        ]


class TestTheGeneratedCommandActuallyParses:
    """A wiring test that only greps strings would miss a renamed flag. Feed the
    generated argv to the real parser."""

    def test_every_generated_command_parses(self, cp, run):
        import incremental_risk as ir

        seen = _capture(cp, run, RUN_DYNAMIC=True)
        assert seen, "anti-vacuity"
        for cmd in seen:
            # cmd[0] is the interpreter, cmd[1] the script.
            args = ir.build_parser().parse_args(cmd[2:])
            assert args.endpoint == "platinum"
            assert args.output_dir

    def test_a_renamed_flag_would_be_caught(self, cp, run):
        """Proves the parse above has teeth: the plural spelling must fail."""
        import incremental_risk as ir

        with pytest.raises(SystemExit):
            ir.build_parser().parse_args(
                ["--output-dir", "/tmp/x", "--landmark-days", "90"]
            )


class TestResume:
    def test_skips_when_the_ablation_table_exists(self, cp, run):
        landmark, cfg = 0, cp._LONGITUDINAL_CONFIGS_BY_ENDPOINT["platinum"][0]
        done = (
            run["output_dir"]
            / cp.model_output_dir("dynamic-deephit-dyn")
            / f"landmark_{landmark}"
            / cfg
            / "incremental_risk_ablation.csv"
        )
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("x\n")
        seen = _capture(cp, run, RUN_DYNAMIC=True, FORCE_RERUN=False)
        targets = [c[c.index("--output-dir") + 1] for c in seen]
        assert str(done.parent) not in targets
        assert len(targets) > 0, "only the completed cell should be skipped"

    def test_force_rerun_ignores_existing_outputs(self, cp, run):
        landmark, cfg = 0, cp._LONGITUDINAL_CONFIGS_BY_ENDPOINT["platinum"][0]
        done = (
            run["output_dir"]
            / cp.model_output_dir("dynamic-deephit-dyn")
            / f"landmark_{landmark}"
            / cfg
            / "incremental_risk_ablation.csv"
        )
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("x\n")
        seen = _capture(cp, run, RUN_DYNAMIC=True, FORCE_RERUN=True)
        targets = [c[c.index("--output-dir") + 1] for c in seen]
        assert str(done.parent) in targets
        assert "--overwrite" in seen[0]


class TestLongitudinalDriverCallsIt:
    def test_run_multivariate_longitudinal_invokes_the_step(self, cp, run):
        """Otherwise the notebook's existing cell would never reach Phase 5."""
        with mock.patch.object(cp, "_run_tasks", return_value=[]), mock.patch.object(
            cp, "run_incremental_risk", return_value=[]
        ) as step, mock.patch.object(Path, "mkdir", lambda *a, **k: None):
            cp.run_multivariate_longitudinal(run, dry_run=True)
        step.assert_called_once()


class TestResultLoader:
    def test_missing_outputs_give_an_empty_frame(self, cp, run):
        """A notebook cell displays this unconditionally; it must not raise."""
        with mock.patch.multiple(cp, RUN_DYNAMIC=True):
            assert cp.load_incremental_risk_results(run, "ablation").empty

    def test_reads_back_and_stamps_the_run(self, cp, run):
        import pandas as pd

        landmark = 90
        cfg = cp._LONGITUDINAL_CONFIGS_BY_ENDPOINT["platinum"][0]
        path = (
            run["output_dir"]
            / cp.model_output_dir("dynamic-deephit-dyn")
            / f"landmark_{landmark}"
            / cfg
            / "incremental_risk_ablation.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prediction_day": [91.0], "auc_gain": [0.1]}).to_csv(
            path, index=False
        )
        with mock.patch.multiple(cp, RUN_DYNAMIC=True):
            got = cp.load_incremental_risk_results(run, "ablation")
        assert len(got) == 1
        assert got.iloc[0]["run"] == "adt"
        assert got.columns[0] == "run"

    def test_rejects_an_unknown_table_name(self, cp, run):
        with pytest.raises(ValueError, match="ablation"):
            cp.load_incremental_risk_results(run, "nonsense")
