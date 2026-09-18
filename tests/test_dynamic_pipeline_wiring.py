"""The dynamic DeepHit arm's wiring through compass_pipeline.

Three couplings here span two files and would all fail *silently* rather than
loudly:

1. The metrics filename the pipeline looks for must equal the one the runner
   writes. A mismatch shows up as a permanently "missing" summary row for a
   model that in fact ran and succeeded.
2. The dynamic arm must ADD to the landmark arm, never replace it -- the
   landmark row is the one comparable to Cox/XGBoost, and every published
   number came from it.
3. The two arms must not share an output directory, or the second to run
   overwrites the first's artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "COMPASS" / "survival_analysis"))

import compass_pipeline as cp  # noqa: E402

ENDPOINT = "platinum"
LANDMARK = 0


@pytest.fixture
def run() -> dict:
    return {
        "inputs_dir": "/data/prediction_inputs_adt",
        "output_dir": Path("/out"),
        "label": "adt",
        "title": "ADT",
        "landmarks": [LANDMARK],
        "endpoint": ENDPOINT,
        "cohort": "all",
    }


@pytest.fixture
def dynamic_on(monkeypatch):
    monkeypatch.setattr(cp, "RUN_DYNAMIC", True)


def _specs(**kw):
    return cp.longitudinal_task_specs(ENDPOINT, include_survlatent=False, **kw)


def _cmd_strings(model, run, cfg=ENDPOINT):
    return [str(c) for c in cp.build_model_command(model, LANDMARK, cfg, "/out/x", run)]


class TestTheFlagIsOptIn:
    def test_default_has_no_dynamic_arm(self, monkeypatch):
        monkeypatch.setattr(cp, "RUN_DYNAMIC", False)
        models = {m for m, _, _ in _specs()}
        assert models == {"dynamic-deephit"}

    def test_enabling_adds_rather_than_replaces(self, dynamic_on):
        models = [m for m, _, _ in _specs()]
        assert "dynamic-deephit" in models, "the landmark arm must survive"
        assert "dynamic-deephit-dyn" in models

    def test_explicit_argument_overrides_the_module_toggle(self, monkeypatch):
        monkeypatch.setattr(cp, "RUN_DYNAMIC", False)
        models = {m for m, _, _ in _specs(include_dynamic=True)}
        assert "dynamic-deephit-dyn" in models

    def test_every_config_gets_a_dynamic_counterpart(self, dynamic_on):
        specs = _specs()
        landmark_cfgs = {c for m, c, _ in specs if m == "dynamic-deephit"}
        dyn_cfgs = {c for m, c, _ in specs if m == "dynamic-deephit-dyn"}
        assert landmark_cfgs == dyn_cfgs


class TestCommandCarriesTheFlag:
    def test_dynamic_arm_passes_dynamic(self, run):
        assert "--dynamic" in _cmd_strings("dynamic-deephit-dyn", run)

    def test_landmark_arm_never_passes_dynamic(self, run):
        """The regression that matters: the landmark arm must be untouched."""
        assert "--dynamic" not in _cmd_strings("dynamic-deephit", run)

    def test_both_arms_share_every_other_argument(self, run):
        """The two arms must differ by exactly the one flag, or the comparison
        between them is confounded by some other difference."""
        landmark = _cmd_strings("dynamic-deephit", run)
        dyn = [a for a in _cmd_strings("dynamic-deephit-dyn", run) if a != "--dynamic"]
        assert landmark == dyn


class TestOutputsCannotCollide:
    def test_distinct_output_directories(self):
        assert cp.model_output_dir("dynamic-deephit") != cp.model_output_dir(
            "dynamic-deephit-dyn"
        )

    def test_distinct_metrics_filenames(self, dynamic_on):
        names = [
            cp.longitudinal_metrics_filename(m, c, LANDMARK, f)
            for m, c, f in _specs()
        ]
        assert len(names) == len(set(names))


class TestFilenameMatchesTheRunner:
    """Pins the cross-file contract by deriving the name the way the runner
    does, rather than restating the literal the pipeline uses."""

    @pytest.mark.parametrize(
        "model,dynamic", [("dynamic-deephit", False), ("dynamic-deephit-dyn", True)]
    )
    @pytest.mark.parametrize("config", ["platinum", "competing"])
    def test_pipeline_looks_for_what_the_runner_writes(self, model, dynamic, config):
        # Mirrors survival_common.longitudinal_runners.run_deephit.
        prefix = "dynamic_deephit_dyn" if dynamic else "dynamic_deephit"
        written = f"{prefix}_metrics_{config}.csv"

        spec = next(
            (m, c, f)
            for m, c, f in _specs(include_dynamic=True)
            if m == model and c == config
        )
        looked_for = cp.longitudinal_metrics_filename(*spec[:2], LANDMARK, spec[2])
        assert looked_for == written

    def test_the_runner_prefix_is_what_we_think_it_is(self):
        """Anti-drift: read the prefix out of the runner itself. If someone
        renames it there, this test fails instead of the summary silently
        going blank."""
        src = (REPO / "survival_common" / "longitudinal_runners.py").read_text()
        assert 'prefix = "dynamic_deephit_dyn" if dynamic else "dynamic_deephit"' in src
        assert 'f"{prefix}_metrics_{' in src


class TestSummaryReadsBothArms:
    def test_dynamic_arm_is_parsed_like_the_landmark_arm(self):
        """Both write the canonical block, so summarize_longitudinal_outputs
        must not fall through its elif chain and drop the dynamic rows."""
        src = (
            REPO / "COMPASS" / "survival_analysis" / "compass_pipeline.py"
        ).read_text()
        assert 'if model in ("dynamic-deephit", "dynamic-deephit-dyn"):' in src
