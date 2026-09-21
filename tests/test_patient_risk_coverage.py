"""Held-out risk-score coverage must distinguish "stale fit" from "fit failed".

The pipeline's resume logic keys on the metrics file, so a task fit before the
risk-score output existed keeps being skipped while its metrics still read fine.
summarize_patient_risks is what makes that state visible in 03_multivariate;
if it silently reported such a task as fine, the notebook would claim coverage
it does not have and the stratified figures would be built on a subset.
"""

from pathlib import Path
import sys
from types import ModuleType

import pandas as pd

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

from COMPASS.survival_analysis import compass_pipeline as cp  # noqa: E402

if _inserted_source_stub:
    del sys.modules[_source_module_name]


def _run(tmp_path: Path) -> dict:
    return {
        "label": "adt",
        "endpoint": "platinum",
        "landmarks": [0, 180],
        "output_dir": tmp_path,
        "cohort": "all",
    }


def _write_risks(run, model, landmark, config, *, endpoint="platinum", n=3):
    path = cp.patient_risk_path(run, model, landmark, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "DFCI_MRN": [str(i) for i in range(n)],
        "endpoint": endpoint,
        "dataset": "test",
        "duration_days": range(10, 10 * (n + 1), 10),
        "event": [1, 0] * (n // 2) + [1] * (n % 2),
        "risk_score": [i / 10 for i in range(n)],
    }).to_csv(path, index=False)
    return path


def test_every_multivariable_task_has_a_known_risk_filename():
    """No task may fall through as "not applicable" -- that would hide a gap."""
    for model, config_dir, _metrics in cp.MULTIVARIATE_TASK_SPECS:
        assert (model, config_dir) in cp.PATIENT_RISK_FILENAMES


def test_risk_filenames_sit_beside_their_metrics(tmp_path):
    run = _run(tmp_path)
    for model, config_dir, metrics_filename in cp.MULTIVARIATE_TASK_SPECS:
        risk = cp.patient_risk_path(run, model, 0, config_dir)
        metrics = (
            run["output_dir"] / cp.model_output_dir(model)
            / "landmark_0" / config_dir / metrics_filename
        )
        assert risk.parent == metrics.parent
        # The two differ only in the trailing component of the stem.
        assert risk.name.replace("_patient_risks", "_metrics") == metrics.name


def test_missing_risk_file_is_reported_as_missing(tmp_path):
    run = _run(tmp_path)
    _write_risks(run, "elastic-net", 180, "both")
    table = cp.summarize_patient_risks(run)
    assert (table["status"] == "ok").sum() == 1
    # 4 specs x 2 landmarks = 8 tasks; the other 7 have no file.
    assert (table["status"] == "missing").sum() == 7


def test_counts_only_held_out_rows_of_the_runs_endpoint(tmp_path):
    run = _run(tmp_path)
    path = cp.patient_risk_path(run, "elastic-net", 0, "both")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "DFCI_MRN": ["1", "2", "3", "4"],
        "endpoint": ["platinum", "platinum", "nepc", "platinum"],
        "dataset": ["test", "test", "test", "train"],
        "duration_days": [10, 20, 30, 40],
        "event": [1, 0, 1, 1],
        "risk_score": [0.1, 0.2, 0.3, 0.4],
    }).to_csv(path, index=False)
    row = cp.summarize_patient_risks(run).query(
        "model == 'elastic-net' and landmark == 0 and config == 'both'"
    ).iloc[0]
    # Only the two held-out platinum rows count: the nepc row and the train row
    # are both excluded.
    assert row["n_patients"] == 2
    assert row["n_events"] == 1


def test_wrong_endpoint_file_does_not_read_as_ok(tmp_path):
    run = _run(tmp_path)
    _write_risks(run, "xgboost", 0, "both", endpoint="nepc")
    row = cp.summarize_patient_risks(run).query(
        "model == 'xgboost' and landmark == 0 and config == 'both'"
    ).iloc[0]
    assert row["status"] == "no platinum row"


def test_reports_every_task_once(tmp_path):
    run = _run(tmp_path)
    table = cp.summarize_patient_risks(run)
    expected = len(cp.MULTIVARIATE_TASK_SPECS) * len(run["landmarks"])
    assert len(table) == expected
    assert not table.duplicated(["model", "landmark", "config"]).any()
