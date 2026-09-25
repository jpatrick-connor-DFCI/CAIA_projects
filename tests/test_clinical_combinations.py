"""Gleason/somatic/labs combination runner and summary (Plan §2c).

run_multivariate_clinical_combinations fits every non-empty combination of
{gleason, somatic, labs} components on the two available-case cohorts, and
summarize_clinical_combinations reports each arm's metrics plus a paired
bootstrap delta-C-index against that cohort's labs arm, computed on the
shared test-block patients from the patient-risk files.
"""

from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

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

pytest.importorskip("lifelines")


def _run(tmp_path: Path, inputs_dir: Path) -> dict:
    return {
        "label": "adt",
        "endpoint": "platinum",
        "landmarks": [0],
        "output_dir": tmp_path,
        "inputs_dir": inputs_dir,
        "cohort": "all",
    }


def test_arm_lists_match_the_plans_table():
    assert cp.CLINICAL_COMBINATION_SPECS["gleason"] == ("labs", "gleason", "gleason-labs")
    assert set(cp.CLINICAL_COMBINATION_SPECS["gleason_somatic"]) == {
        "labs", "gleason", "somatic",
        "gleason-labs", "somatic-labs", "gleason-somatic", "gleason-somatic-labs",
    }
    assert len(cp.CLINICAL_COMBINATION_SPECS["gleason_somatic"]) == 7


def test_dry_run_reports_missing_inputs_without_raising(tmp_path, monkeypatch):
    run = _run(tmp_path, tmp_path / "inputs")
    run["output_dir"].mkdir(parents=True, exist_ok=True)
    (run["inputs_dir"] / "adt_index_report.json").parent.mkdir(parents=True, exist_ok=True)
    # This test only checks the command/skip shape, not the adt-index gate
    # itself (covered elsewhere), so patch it out for this test only.
    monkeypatch.setattr(cp, "_require_adt_index_run", lambda run: None)
    summary = cp.run_multivariate_clinical_combinations(run, dry_run=True)
    statuses = {status for _tag, status, _elapsed in summary}
    assert statuses == {"missing inputs"}
    # 1 landmark x (3 gleason arms + 7 gleason_somatic arms) x 2 models = 20
    assert len(summary) == 20


def _write_risks(path: Path, mrns, risk_scores, *, seed_events=None, endpoint="platinum"):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(mrns)
    rng = np.random.default_rng(0)
    event = seed_events if seed_events is not None else rng.binomial(1, 0.6, size=n)
    pd.DataFrame({
        "DFCI_MRN": mrns,
        "endpoint": endpoint,
        "dataset": "test",
        "duration_days": np.linspace(50, 500, n),
        "event": event,
        "risk_score": risk_scores,
    }).to_csv(path, index=False)


def test_summarize_reports_missing_when_no_metrics(tmp_path):
    run = _run(tmp_path, tmp_path / "inputs")
    table = cp.summarize_clinical_combinations(run)
    assert (table["status"] == "missing").all()
    assert len(table) == 1 * (3 + 7) * 2  # landmarks x arms x models


def test_summarize_computes_paired_bootstrap_delta_against_labs(tmp_path):
    run = _run(tmp_path, tmp_path / "inputs")
    rng = np.random.default_rng(1)
    n = 80
    mrns = [str(i) for i in range(n)]
    event = rng.binomial(1, 0.6, size=n)

    labs_dir = cp._clinical_combination_dir(run, "gleason", "labs", "cox", 0)
    gleason_dir = cp._clinical_combination_dir(run, "gleason", "gleason", "cox", 0)
    labs_dir.mkdir(parents=True, exist_ok=True)
    gleason_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "endpoint": ["platinum"], "n_test": [n], "n_events_test": [int(event.sum())],
        "test_c_index": [0.60], "test_mean_auc_t": [0.62], "test_integrated_brier": [0.18],
    }).to_csv(labs_dir / "cox_agg_multivariable_metrics.csv", index=False)
    pd.DataFrame({
        "endpoint": ["platinum"], "n_test": [n], "n_events_test": [int(event.sum())],
        "test_c_index": [0.65], "test_mean_auc_t": [0.66], "test_integrated_brier": [0.17],
    }).to_csv(gleason_dir / "cox_agg_multivariable_metrics.csv", index=False)

    labs_scores = rng.normal(size=n)
    # gleason arm is more concordant with the (shared) event/duration pairing
    duration = np.linspace(50, 500, n)
    gleason_scores = -duration + rng.normal(scale=5, size=n)

    _write_risks(
        labs_dir / "cox_agg_multivariable_patient_risks.csv", mrns, labs_scores,
        seed_events=event,
    )
    _write_risks(
        gleason_dir / "cox_agg_multivariable_patient_risks.csv", mrns, gleason_scores,
        seed_events=event,
    )

    table = cp.summarize_clinical_combinations(run)
    labs_row = table.query("cohort == 'gleason' and arm == 'labs' and model == 'elastic-net'").iloc[0]
    gleason_row = table.query("cohort == 'gleason' and arm == 'gleason' and model == 'elastic-net'").iloc[0]

    assert pd.isna(labs_row["delta_c_index"])
    assert not pd.isna(gleason_row["delta_c_index"])
    assert gleason_row["n_paired"] == n
    assert gleason_row["delta_c_index_ci_low"] <= gleason_row["delta_c_index"] <= gleason_row["delta_c_index_ci_high"]


def test_bootstrap_delta_is_paired_on_shared_patients():
    rng = np.random.default_rng(2)
    n = 60
    mrns = [str(i) for i in range(n)]
    duration = np.linspace(10, 400, n)
    event = rng.binomial(1, 0.5, size=n)

    arm = pd.DataFrame({
        "DFCI_MRN": mrns, "duration_days": duration, "event": event,
        "risk_score": -duration + rng.normal(scale=2, size=n),
    })
    labs = pd.DataFrame({
        "DFCI_MRN": mrns, "duration_days": duration, "event": event,
        "risk_score": rng.normal(size=n),
    })

    observed, lo, hi, n_paired = cp._paired_bootstrap_delta_c_index(
        arm, labs, id_col="DFCI_MRN", n_boot=200, seed=0
    )
    assert n_paired == n
    assert lo <= observed <= hi
    assert observed > 0  # the arm is far more concordant than pure noise


def test_bootstrap_returns_nan_on_disjoint_patients():
    arm = pd.DataFrame({
        "DFCI_MRN": ["1", "2"], "duration_days": [10.0, 20.0], "event": [1, 0],
        "risk_score": [0.1, 0.2],
    })
    labs = pd.DataFrame({
        "DFCI_MRN": ["3", "4"], "duration_days": [10.0, 20.0], "event": [1, 0],
        "risk_score": [0.1, 0.2],
    })
    observed, lo, hi, n_paired = cp._paired_bootstrap_delta_c_index(
        arm, labs, id_col="DFCI_MRN"
    )
    assert n_paired == 0
    assert np.isnan(observed) and np.isnan(lo) and np.isnan(hi)
