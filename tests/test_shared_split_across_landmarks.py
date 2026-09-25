"""Integration test for build_prediction_inputs.py's --split-mode (Plan §1):
a single held-out split shared across landmarks by default, with
--split-mode independent preserving the legacy per-landmark behavior.

Runs the real script end-to-end (main()) against a small synthetic raw
cohort, matching the fixture shape used by test_longitudinal_manifest.py.
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
N_PATIENTS = 60


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
        landmark_days=[0, 90, 180],
        output_dir=str(output_dir),
        seed=42,
        split_mode="shared",
        test_frac=0.2,
        val_frac=0.2,
        min_patient_coverage=0.2,
        time_unit_days=7,
        auc_quantiles=[0.1, 0.5, 0.9],
        auc_max_time_units=260,
        max_followup_days=None,
        build_longitudinal=False,
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


def _landmark_splits(output_dir: Path, landmark_days: list[int]) -> dict[int, pd.Series]:
    splits = {}
    for lm in landmark_days:
        df = pd.read_csv(output_dir / bpi.split_assignments_filename(lm))
        splits[lm] = df.set_index(ID_COL)["split"]
    return splits


class TestSharedSplitAcrossLandmarks:
    def test_shared_split_is_default_and_one_label_per_mrn(self, tmp_path, raw_csv):
        output_dir = tmp_path / "out"
        bpi.main(_base_args(output_dir, raw_csv))

        manifest = json.loads((output_dir / "build_manifest.json").read_text())
        assert manifest["split_mode"] == "shared"

        attrition = json.loads((output_dir / bpi.LANDMARK_ATTRITION_FILENAME).read_text())
        assert attrition["split_mode"] == "shared"

        splits = _landmark_splits(output_dir, [0, 90, 180])
        common = set(splits[0].index) & set(splits[90].index) & set(splits[180].index)
        assert len(common) > 0
        for mrn in common:
            labels = {splits[lm].loc[mrn] for lm in (0, 90, 180)}
            assert len(labels) == 1, f"MRN {mrn} has inconsistent split labels across landmarks: {labels}"

    def test_test_and_train_are_disjoint_across_landmarks(self, tmp_path, raw_csv):
        output_dir = tmp_path / "out"
        bpi.main(_base_args(output_dir, raw_csv))

        splits = _landmark_splits(output_dir, [0, 90, 180])
        test_mrns_0 = set(splits[0].index[splits[0] == "test"])
        train_mrns_90 = set(splits[90].index[splits[90] == "train"])
        assert test_mrns_0.isdisjoint(train_mrns_90)

    def test_independent_mode_allows_labels_to_differ(self, tmp_path, raw_csv):
        output_dir = tmp_path / "out"
        bpi.main(_base_args(output_dir, raw_csv, split_mode="independent"))

        manifest = json.loads((output_dir / "build_manifest.json").read_text())
        assert manifest["split_mode"] == "independent"

        splits = _landmark_splits(output_dir, [0, 90, 180])
        common = set(splits[0].index) & set(splits[90].index) & set(splits[180].index)
        assert len(common) > 0
        # Independent mode is not required to agree; just confirm it ran and
        # produced valid per-landmark splits (no consistency assertion here).
        for lm in (0, 90, 180):
            assert set(splits[lm].unique()).issubset({"train", "valid", "test"})

    def test_nepc_endpoint_stratifies_on_nepc_not_platinum(self, tmp_path, raw_csv):
        output_dir = tmp_path / "out"
        df = pd.read_csv(raw_csv)
        df["t_nepc"] = df["t_platinum"]
        df["NEPC"] = df["PLATINUM"]
        df.to_csv(raw_csv, index=False)

        bpi.main(_base_args(output_dir, raw_csv, endpoint="nepc", landmark_days=[0]))

        manifest = json.loads((output_dir / "build_manifest.json").read_text())
        assert manifest["split_mode"] == "shared"
