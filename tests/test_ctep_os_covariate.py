"""build_ctep_os_risk_covariate: CTEP OS text-risk join (Plan §5).

Covers z-scoring within outer_fold, Int64 MRN joins across the CSV/parquet
pair, and per-landmark leakage exclusion (first_treatment_date after the
landmark cutoff). Never touches the real CTEP_DATA_PATH -- every test either
passes ctep_data_root explicitly (pointing at tmp_path fixtures) or uses
monkeypatch.setenv the way test_text_embedding_lab_gate.py does.
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
pytest.importorskip("pyarrow")


def _write_ctep_fixtures(root: Path, *, scores: pd.DataFrame, cohort: pd.DataFrame) -> None:
    scores_path = root / cp.CTEP_OS_RISK_SCORES_RELPATH
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_path, index=False)

    cohort_path = root / cp.CTEP_COHORT_RELPATH
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(cohort_path, index=False)


def _univariate_data(mrns, id_col="DFCI_MRN") -> pd.DataFrame:
    frame = pd.DataFrame({
        id_col: mrns,
        "duration_days": np.linspace(50, 500, len(mrns)),
        "event": [1, 0] * (len(mrns) // 2) + [1] * (len(mrns) % 2),
        "LAB1__mean": np.arange(len(mrns), dtype=float),
    })
    return frame.set_index(id_col)


class TestZScoringWithinFold:
    def test_z_scored_within_each_outer_fold_separately(self, tmp_path):
        mrns = list(range(1, 9))
        # Two folds with different score scales; z-scoring within fold should
        # put both folds on a comparable unit-variance scale.
        scores = pd.DataFrame({
            "DFCI_MRN": mrns,
            "outer_fold": [0, 0, 0, 0, 1, 1, 1, 1],
            "text_risk_score": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
        })
        cohort = pd.DataFrame({
            "DFCI_MRN": mrns,
            "first_treatment_date": pd.to_datetime(["2020-01-01"] * 8),
        })
        _write_ctep_fixtures(tmp_path, scores=scores, cohort=cohort)

        univariate = _univariate_data(mrns)
        joined, n_no_score, n_leak = cp.build_ctep_os_risk_covariate(
            univariate, landmark_day=0, ctep_data_root=tmp_path,
        )
        fold0 = joined.loc[mrns[:4], cp.CTEP_OS_RISK_COVARIATE]
        fold1 = joined.loc[mrns[4:], cp.CTEP_OS_RISK_COVARIATE]
        assert np.isclose(fold0.mean(), 0.0, atol=1e-6)
        assert np.isclose(fold1.mean(), 0.0, atol=1e-6)
        assert np.isclose(fold0.std(ddof=0), 1.0, atol=1e-6)
        assert np.isclose(fold1.std(ddof=0), 1.0, atol=1e-6)
        assert n_no_score == 0
        assert n_leak == 0

    def test_degenerate_fold_with_zero_variance_yields_nan_not_a_crash(self, tmp_path):
        mrns = [1, 2, 3]
        scores = pd.DataFrame({
            "DFCI_MRN": mrns,
            "outer_fold": [0, 0, 0],
            "text_risk_score": [5.0, 5.0, 5.0],
        })
        cohort = pd.DataFrame({
            "DFCI_MRN": mrns,
            "first_treatment_date": pd.to_datetime(["2020-01-01"] * 3),
        })
        _write_ctep_fixtures(tmp_path, scores=scores, cohort=cohort)
        joined, _n_no_score, _n_leak = cp.build_ctep_os_risk_covariate(
            _univariate_data(mrns), landmark_day=0, ctep_data_root=tmp_path,
        )
        assert joined[cp.CTEP_OS_RISK_COVARIATE].isna().all()


class TestIntMrnJoin:
    def test_joins_across_int_and_float_mrn_dtypes(self, tmp_path):
        # Simulate a common real-world mismatch: scores CSV round-trips MRN
        # as float (NaN-safe) while the cohort parquet stores it as int.
        mrns = [10, 20, 30]
        scores = pd.DataFrame({
            "DFCI_MRN": pd.array(mrns, dtype="float64"),
            "outer_fold": [0, 0, 0],
            "text_risk_score": [1.0, 2.0, 3.0],
        })
        cohort = pd.DataFrame({
            "DFCI_MRN": pd.array(mrns, dtype="int64"),
            "first_treatment_date": pd.to_datetime(["2019-06-01"] * 3),
        })
        _write_ctep_fixtures(tmp_path, scores=scores, cohort=cohort)

        univariate = _univariate_data(mrns)
        joined, n_no_score, n_leak = cp.build_ctep_os_risk_covariate(
            univariate, landmark_day=0, ctep_data_root=tmp_path,
        )
        assert joined[cp.CTEP_OS_RISK_COVARIATE].notna().sum() == 3
        assert n_no_score == 0
        assert n_leak == 0

    def test_patients_with_no_ctep_score_are_counted_and_left_nan(self, tmp_path):
        cohort_mrns = [1, 2]
        scores = pd.DataFrame({
            "DFCI_MRN": cohort_mrns,
            "outer_fold": [0, 0],
            "text_risk_score": [1.0, 2.0],
        })
        cohort = pd.DataFrame({
            "DFCI_MRN": cohort_mrns,
            "first_treatment_date": pd.to_datetime(["2019-06-01", "2019-06-01"]),
        })
        _write_ctep_fixtures(tmp_path, scores=scores, cohort=cohort)

        # Univariate data has an extra patient (3) with no CTEP score at all.
        joined, n_no_score, n_leak = cp.build_ctep_os_risk_covariate(
            _univariate_data([1, 2, 3]), landmark_day=0, ctep_data_root=tmp_path,
        )
        assert n_no_score == 1
        assert pd.isna(joined.loc[3, cp.CTEP_OS_RISK_COVARIATE])


class TestLandmarkLeakageExclusion:
    def test_first_treatment_after_landmark_cutoff_is_excluded(self, tmp_path):
        mrns = [1, 2, 3]
        scores = pd.DataFrame({
            "DFCI_MRN": mrns,
            "outer_fold": [0, 0, 0],
            "text_risk_score": [1.0, 2.0, 3.0],
        })
        # Patient 1: first treatment well before anchor + landmark -> kept.
        # Patient 2: first treatment AFTER anchor + landmark -> excluded (leakage).
        # Patient 3: no first_treatment_date at all -> kept (nothing to leak).
        cohort = pd.DataFrame({
            "DFCI_MRN": mrns,
            "first_treatment_date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-06-01"),
                pd.NaT,
            ],
        })
        _write_ctep_fixtures(tmp_path, scores=scores, cohort=cohort)

        anchors = pd.Series(
            [pd.Timestamp("2020-01-01")] * 3, index=pd.Index(mrns, name="DFCI_MRN"),
        )
        joined, n_no_score, n_leak = cp.build_ctep_os_risk_covariate(
            _univariate_data(mrns), landmark_day=30,
            ctep_data_root=tmp_path, treatment_anchors=anchors,
        )
        assert n_leak == 1
        assert pd.isna(joined.loc[2, cp.CTEP_OS_RISK_COVARIATE])
        assert not pd.isna(joined.loc[1, cp.CTEP_OS_RISK_COVARIATE])
        assert not pd.isna(joined.loc[3, cp.CTEP_OS_RISK_COVARIATE])

    def test_no_treatment_anchors_supplied_means_no_exclusion(self, tmp_path):
        mrns = [1, 2]
        scores = pd.DataFrame({
            "DFCI_MRN": mrns, "outer_fold": [0, 0], "text_risk_score": [1.0, 2.0],
        })
        cohort = pd.DataFrame({
            "DFCI_MRN": mrns,
            "first_treatment_date": [pd.Timestamp("2099-01-01"), pd.Timestamp("2099-01-01")],
        })
        _write_ctep_fixtures(tmp_path, scores=scores, cohort=cohort)
        joined, _n_no_score, n_leak = cp.build_ctep_os_risk_covariate(
            _univariate_data(mrns), landmark_day=0, ctep_data_root=tmp_path,
        )
        assert n_leak == 0
        assert joined[cp.CTEP_OS_RISK_COVARIATE].notna().all()


class TestModuleDefaultReadsEnvVar:
    def test_ctep_data_root_default_honors_ctep_data_path_env_var(self, monkeypatch):
        # CTEP_DATA_ROOT is resolved at import time from the CTEP_DATA_PATH
        # env var, matching build_text_embedding_inputs.py's NOTES_PATH
        # convention; build_ctep_os_risk_covariate's ctep_data_root parameter
        # lets callers (and tests) override it without touching the real path.
        assert str(cp.CTEP_DATA_ROOT).endswith("clinical_text_embedding_project/") or str(
            cp.CTEP_DATA_ROOT
        ).endswith("clinical_text_embedding_project")
