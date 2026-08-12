"""SurvLatent upstream-directory preparation tests."""

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "COMPASS"
    / "survival_analysis"
    / "multivariate_longitudinal"
    / "survlatent_ode.py"
)
SPEC = importlib.util.spec_from_file_location("survlatent_adapter", MODULE_PATH)
survlatent_adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(survlatent_adapter)


def test_prepare_run_artifacts_creates_upstream_parent_dirs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    survlatent_adapter.prepare_run_artifacts(
        "prostate_platinum_landmark0_v1", overwrite=False, resume=False
    )

    assert (tmp_path / "model_performance").is_dir()
    assert (tmp_path / "surv_curves").is_dir()
    assert (tmp_path / "experiments").is_dir()


def test_overwrite_removes_run_specific_survival_curves(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "surv_curves" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "stale.csv").write_text("stale")

    survlatent_adapter.prepare_run_artifacts(
        "run_1", overwrite=True, resume=False
    )

    assert not run_dir.exists()
    assert (tmp_path / "surv_curves").is_dir()
