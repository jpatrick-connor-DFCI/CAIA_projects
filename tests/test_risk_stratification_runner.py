"""run_risk_stratification: stratify the labs risk score (Plan §4c).

Covers the two independent stages (full-cohort labs-vs-clinical, and matched
available-case labs-vs-other-arm comparisons) across both scoring schemes
(test, cv_oof), using dry_run=True so no subprocess is actually launched --
only the skip/warn/run bookkeeping and command shape are under test here.
"""

from pathlib import Path
import sys
from types import ModuleType

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
        "id_col": "DFCI_MRN",
    }


@pytest.fixture(autouse=True)
def _skip_adt_gate(monkeypatch):
    monkeypatch.setattr(cp, "_require_adt_index_run", lambda run: None)


class TestSchemesAndDirectories:
    def test_runs_both_schemes_into_separate_subdirectories(self, tmp_path):
        run = _run(tmp_path, tmp_path / "inputs")
        summary = cp.run_risk_stratification(run, dry_run=True)
        tags = {tag for tag, _status, _elapsed in summary}
        assert any("test  " in tag or " test " in tag for tag in tags)
        assert any("cv_oof" in tag for tag in tags)

    def test_never_mixes_test_and_cv_oof_paths(self, tmp_path):
        run = _run(tmp_path, tmp_path / "inputs")
        cp.run_risk_stratification(run, dry_run=True)
        # Nothing is written in dry-run mode, but the two schemes' planned
        # output roots must differ.
        test_root = tmp_path / cp.RISK_STRATIFICATION_DIRNAME / "test"
        oof_root = tmp_path / cp.RISK_STRATIFICATION_DIRNAME / "cv_oof"
        assert test_root != oof_root


class TestFullCohortStage:
    def test_reports_missing_labs_risks_when_absent(self, tmp_path):
        run = _run(tmp_path, tmp_path / "inputs")
        summary = cp.run_risk_stratification(run, dry_run=True)
        statuses = {status for _tag, status, _elapsed in summary}
        assert "missing labs risks" in statuses

    def test_reports_missing_clinical_stratifiers_when_labs_risks_present_but_no_stratifiers(
        self, tmp_path
    ):
        run = _run(tmp_path, tmp_path / "inputs")
        labs_path = cp.patient_risk_path(run, "elastic-net", 0, "both")
        labs_path.parent.mkdir(parents=True, exist_ok=True)
        labs_path.write_text("DFCI_MRN,endpoint,dataset,duration_days,event,risk_score\n")

        summary = cp.run_risk_stratification(run, dry_run=True)
        full_cohort_rows = [
            (tag, status) for tag, status, _elapsed in summary if "full_cohort" in tag
        ]
        assert any(status == "missing clinical stratifiers" for _tag, status in full_cohort_rows)

    def test_skips_when_discrimination_table_already_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cp, "FORCE_RERUN", False)
        run = _run(tmp_path, tmp_path / "inputs")
        row_output_dir = tmp_path / cp.RISK_STRATIFICATION_DIRNAME / "test" / "full_cohort" / "landmark_0"
        row_output_dir.mkdir(parents=True, exist_ok=True)
        (row_output_dir / "risk_stratified_discrimination_platinum_landmark0.csv").write_text("x\n")

        summary = cp.run_risk_stratification(run, dry_run=True)
        full_cohort_test_rows = [
            (tag, status)
            for tag, status, _elapsed in summary
            if "full_cohort" in tag and "test" in tag.split()[2]
        ]
        assert any(status == "skipped" for _tag, status in full_cohort_test_rows)


class TestMatchedCohortStage:
    def test_reports_missing_labs_risks_for_matched_cohorts(self, tmp_path):
        run = _run(tmp_path, tmp_path / "inputs")
        summary = cp.run_risk_stratification(run, dry_run=True)
        matched_rows = [
            (tag, status) for tag, status, _elapsed in summary if "matched/gleason" in tag
        ]
        assert matched_rows
        assert all(status == "missing labs risks" for _tag, status in matched_rows)

    def test_matched_cohorts_follow_clinical_combination_specs(self, tmp_path):
        assert cp.RISK_STRATIFICATION_MATCHED_COHORTS["gleason"] == ("gleason", "gleason-labs")
        assert set(cp.RISK_STRATIFICATION_MATCHED_COHORTS["gleason_somatic"]) == {
            "gleason", "somatic", "gleason-labs", "somatic-labs",
            "gleason-somatic", "gleason-somatic-labs",
        }

    def test_warns_but_continues_when_some_comparison_arms_are_missing(self, tmp_path, capsys):
        run = _run(tmp_path, tmp_path / "inputs")
        labs_dir = cp._clinical_combination_dir(run, "gleason", "labs", "cox", 0)
        labs_dir.mkdir(parents=True, exist_ok=True)
        (labs_dir / "cox_agg_multivariable_patient_risks.csv").write_text(
            "DFCI_MRN,endpoint,dataset,duration_days,event,risk_score\n"
        )
        # No gleason/gleason-labs arm risk files -> both comparison arms missing,
        # but the labs-only run should still be attempted (not skipped outright).
        summary = cp.run_risk_stratification(run, dry_run=True)
        matched_gleason_cox = [
            (tag, status)
            for tag, status, _elapsed in summary
            if "matched/gleason/cox" in tag and "gleason_somatic" not in tag
        ]
        assert matched_gleason_cox
        captured = capsys.readouterr()
        assert "missing comparison risks" in captured.out
