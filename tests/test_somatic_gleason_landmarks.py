"""Somatic/Gleason pipeline tasks are intentionally baseline-only."""

from pathlib import Path
import sys
from types import ModuleType

# compass_pipeline only needs these names to construct source paths at import
# time. Stub the optional polars-backed scanner so this orchestration test does
# not require the preprocessing environment.
_source_module_name = "data_preprocessing_common.oncdrs_sources"
_inserted_source_stub = _source_module_name not in sys.modules
if _inserted_source_stub:
    _source_stub = ModuleType(_source_module_name)
    _source_stub.TABLE_FILES = {
        name: (f"{name}.parquet", f"{name}.csv")
        for name in (
            "EHR_DIAGNOSES",
            "MEDICATIONS",
            "LABS",
            "HEALTH_HISTORY",
            "PT_INFO_STATUS_REGISTRATION",
        )
    }
    _source_stub.scan_source = lambda *_args, **_kwargs: None
    sys.modules[_source_module_name] = _source_stub

from COMPASS.survival_analysis import compass_pipeline  # noqa: E402

if _inserted_source_stub:
    del sys.modules[_source_module_name]


def _run(tmp_path: Path) -> dict:
    return {
        "label": "adt",
        "title": "ADT",
        "landmarks": [0, 90, 180],
        "inputs_dir": tmp_path / "prediction_inputs_adt",
        "output_dir": tmp_path / "local_runs_adt",
    }


def test_somatic_gleason_input_build_requests_only_landmark_zero(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        compass_pipeline,
        "_run",
        lambda command, *, dry_run=False: commands.append(command) or 0,
    )

    compass_pipeline.build_somatic_gleason_inputs(_run(tmp_path), dry_run=True)

    assert len(commands) == 1
    landmark_arg = commands[0].index("--landmark-days")
    assert commands[0][landmark_arg + 1 :] == ["0"]


def test_somatic_gleason_univariate_runs_only_landmark_zero(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        compass_pipeline,
        "_run",
        lambda command, *, dry_run=False: commands.append(command) or 0,
    )

    summary = compass_pipeline.run_somatic_gleason_univariate(
        _run(tmp_path), dry_run=True
    )

    assert len(commands) == 1
    landmark_arg = commands[0].index("--landmark-days")
    assert commands[0][landmark_arg + 1] == "0"
    assert len(summary) == 1
