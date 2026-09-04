"""Integration test for build_prediction_inputs.py's longitudinal outputs
(Step 3): the person-period build wired into the per-landmark loop, the new
CLI flags, and additivity of the new build_manifest.json keys.

Runs the real script end-to-end (main()) against a small synthetic raw
cohort, matching the shape build_landmark_merged/build_pre_treatment_lab_long
already require (see tests/test_person_period_builder.py's fixtures), rather
than mocking the pipeline internals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_PREDICTION_INPUTS_DIR = REPO_ROOT / "COMPASS" / "data_preprocessing"
if str(BUILD_PREDICTION_INPUTS_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_PREDICTION_INPUTS_DIR))

import build_prediction_inputs as bpi  # noqa: E402

ID_COL = "DFCI_MRN"
AGE_COL = "AGE_AT_TREATMENTSTART"
N_PATIENTS = 40


def _synthetic_raw_df(n_patients: int = N_PATIENTS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for mrn in range(1, n_patients + 1):
        platinum = int(rng.random() < 0.4)
        death = int(rng.random() < 0.3) if not platinum else int(rng.random() < 0.2)
        t_platinum = float(rng.uniform(30, 900)) if platinum else np.nan
        t_death = float(rng.uniform(30, 1200)) if death else np.nan
        t_last_contact = float(rng.uniform(200, 1400))
        if death:
            t_last_contact = max(t_last_contact, t_death)
        if platinum:
            t_last_contact = max(t_last_contact, t_platinum)
        t_death_val = t_death if death else t_last_contact

        n_labs = int(rng.integers(3, 8))
        lab_times = sorted(rng.uniform(-180, -1, size=n_labs))
        lab_values = rng.uniform(1.0, 20.0, size=n_labs)
        for t_lab, val in zip(lab_times, lab_values):
            rows.append(
                {
                    ID_COL: mrn,
                    AGE_COL: float(rng.uniform(50, 85)),
                    "FIRST_RECORD_DATE": "2015-01-01",
                    "FIRST_TREATMENT": 1,
                    "t_first_treatment": 0.0,
                    "t_platinum": t_platinum,
                    "PLATINUM": platinum,
                    "t_death": t_death_val,
                    "DEATH": death,
                    "t_last_contact": t_last_contact,
                    "LAB_NAME": "PSA",
                    "LAB_VALUE": float(val),
                    "t_lab": float(t_lab),
                    "RAW_TEST_CODE": "PSA",
                }
            )
    return pd.DataFrame(rows)


def _base_args(output_dir: Path, data_path: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        id_col=ID_COL,
        age_col=AGE_COL,
        data=str(data_path),
        restrict_to_mrns=None,
        exclude_mrns=None,
        anchor_col="none",
        min_psa_count=0,
        exclude_parpi=False,
        landmark_days=[0],
        output_dir=str(output_dir),
        seed=42,
        test_frac=0.2,
        val_frac=0.2,
        min_patient_coverage=0.2,
        time_unit_days=7,
        auc_quantiles=[0.1, 0.5, 0.9],
        auc_max_time_units=260,
        max_followup_days=None,
        build_longitudinal=True,
        long_min_coverage=0.1,
        no_canonical_labs=False,
        max_longitudinal_labs=None,
        outlier_lo=0.005,
        outlier_hi=0.995,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture()
def raw_csv(tmp_path) -> Path:
    df = _synthetic_raw_df()
    path = tmp_path / "longitudinal_prediction_data.csv"
    df.to_csv(path, index=False)
    return path


class TestLongitudinalBuildIntegration:
    def test_longitudinal_outputs_written(self, tmp_path, raw_csv):
        output_dir = tmp_path / "out"
        bpi.main(_base_args(output_dir, raw_csv))

        long_csv = output_dir / bpi.longitudinal_filename(0)
        long_manifest_path = output_dir / bpi.longitudinal_manifest_filename(0)
        assert long_csv.exists()
        assert long_manifest_path.exists()

        wide = pd.read_csv(long_csv)
        assert {ID_COL, "TIME", "split", "PLATINUM", "DEATH", "t_platinum", "t_death"}.issubset(
            wide.columns
        )
        assert wide.duplicated(subset=[ID_COL, "TIME"]).sum() == 0

        manifest = json.loads(long_manifest_path.read_text())
        assert manifest["event_cols"] == ["PLATINUM", "DEATH"]
        assert manifest["time_to_event_cols"] == ["t_platinum", "t_death"]
        assert manifest["longitudinal_schema_version"] == 1
        assert manifest["time_unit_days"] == 7
        assert "max_landmark_time" in manifest
        assert "n_patients_without_selected_labs" in manifest

    def test_no_build_longitudinal_skips_outputs(self, tmp_path, raw_csv):
        output_dir = tmp_path / "out"
        bpi.main(_base_args(output_dir, raw_csv, build_longitudinal=False))

        assert not (output_dir / bpi.longitudinal_filename(0)).exists()
        assert not (output_dir / bpi.longitudinal_manifest_filename(0)).exists()

        manifest = json.loads((output_dir / bpi.BUILD_MANIFEST_FILENAME).read_text())
        assert manifest["longitudinal_landmarks_built"] == []
        assert "longitudinal_time_unit_days" not in manifest

    def test_build_manifest_additivity(self, tmp_path, raw_csv):
        # Additive-only contract (plan Sec.5): the new keys must not disturb
        # anything cox_runners.run_multivariable already reads, in particular
        # auc_timeline_schema_version must stay whatever the shared helper
        # module declares it to be.
        from survival_common.helper import AUC_TIMELINE_SCHEMA_VERSION

        output_dir = tmp_path / "out"
        bpi.main(_base_args(output_dir, raw_csv))

        manifest = json.loads((output_dir / bpi.BUILD_MANIFEST_FILENAME).read_text())
        assert manifest["auc_timeline_schema_version"] == AUC_TIMELINE_SCHEMA_VERSION
        # Pre-existing keys the cox/xgboost runners depend on must still be present.
        for key in (
            "auc_horizons_by_landmark",
            "auc_max_time_units",
            "auc_time_unit_days",
            "landmark_days",
        ):
            assert key in manifest
        # New additive keys.
        assert manifest["event_cols"] == ["PLATINUM", "DEATH"]
        assert manifest["time_to_event_cols"] == ["t_platinum", "t_death"]
        assert manifest["longitudinal_schema_version"] == 1
        assert manifest["longitudinal_landmarks_built"] == [0]
        assert manifest["longitudinal_time_unit_days"] == 7

    def test_split_agreement_holds(self, tmp_path, raw_csv):
        output_dir = tmp_path / "out"
        bpi.main(_base_args(output_dir, raw_csv))

        aggregated = pd.read_csv(output_dir / bpi.aggregated_filename(0))
        wide = pd.read_csv(output_dir / bpi.longitudinal_filename(0))

        agg_split = aggregated.set_index(ID_COL)["split"]
        long_split = wide.groupby(ID_COL)["split"].agg(lambda s: s.iloc[0])
        overlap = agg_split.index.intersection(long_split.index)
        assert len(overlap) > 0
        assert (agg_split.loc[overlap] == long_split.loc[overlap]).all()
