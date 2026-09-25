"""The age(+panel) baseline also emits cv_oof rows (Plan §3).

`_run_baseline_landmark` in survival_common/cox_runners.py fits the baseline
once on train_val/test with a fixed penalizer/l1_ratio (no CV search). Before
this change it had no out-of-fold path at all, unlike the elastic-net
multivariable runner, so stratification figures could compare a labs model's
`cv_oof` risk against a baseline that only had `test`-block scores -- the two
schemes must never be pooled, so the baseline needs its own `cv_oof` rows to
be a fair reference.

This exercises _run_baseline_landmark directly against a synthetic cohort
shaped like a COMPASS landmark context, with --out-of-fold-risks on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVIVAL_DIR = REPO_ROOT / "COMPASS" / "survival_analysis"
for _p in (str(REPO_ROOT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cox_aggregated as ca  # noqa: E402

from survival_common.cox_runners import _run_baseline_landmark  # noqa: E402
from survival_common.projects.compass_profile import CONFIG  # noqa: E402

ID_COL = ca.ID_COL
AGE_COL = ca.AGE_COL
N = 160


def _synthetic_cohort(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mrns = [f"P{i:04d}" for i in range(n)]
    return pd.DataFrame(
        {
            AGE_COL: rng.normal(66, 8, size=n),
            "t_platinum": np.clip(rng.normal(400, 150, size=n), 20, 900),
            "PLATINUM": rng.binomial(1, 0.4, size=n),
        },
        index=pd.Index(mrns, name=ID_COL),
    )


def _pre_treatment_lab_df(mrns, seed: int) -> pd.DataFrame:
    # select_canonical_labs is called unconditionally inside the nested-CV
    # tuner even when raw_feature_cols=[] (the baseline tests no labs as
    # features), so every MRN needs >=1 observation here or it raises.
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            ID_COL: list(mrns),
            "LAB_NAME": "DUMMY_LAB",
            "LAB_VALUE": rng.normal(size=len(mrns)),
        }
    )


def _ctx(seed: int) -> ca.LandmarkContext:
    cohort = _synthetic_cohort(N, seed)
    split = np.where(
        np.arange(N) < N * 0.8, "train_val", "test"
    )
    train_val = cohort.loc[split == "train_val"]
    test = cohort.loc[split == "test"]
    return ca.LandmarkContext(
        landmark_day=0,
        merged=cohort,
        train_val=train_val,
        test=test,
        pre_treatment_lab_df=_pre_treatment_lab_df(cohort.index, seed),
        raw_feature_cols=[],
        univariate_data=cohort,
        split_stratification="test_stratified",
        canonical_labs=[],
        selected_feature_cols=[],
        feature_meta_selected=pd.DataFrame(),
    )


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        baseline=True,
        cv_penalizers=[0.05],
        cv_l1_ratios=[0.5],
        seed=7,
        n_folds=3,
        oof_outer_folds=3,
        oof_inner_folds=2,
        out_of_fold_risks=True,
        id_col=ID_COL,
        age_col=AGE_COL,
        cohort=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestBaselineOutOfFold:
    def test_emits_cv_oof_rows_for_every_patient(self):
        ctx = _ctx(seed=3)
        args = _args()
        out = {
            "frames": [],
            "metric_rows": [],
            "test_auc_frames": [],
            "test_brier_frames": [],
            "patient_risk_frames": [],
            "canonical_labs_fold_rows": [],
        }
        horizon_grid = np.arange(30, 700, 30)
        _run_baseline_landmark(
            CONFIG,
            ca,
            ctx,
            {"platinum": horizon_grid},
            landmark_day=0,
            endpoints=["platinum"],
            args=args,
            auc_time_unit_days=30,
            auc_max_time_units=None,
            min_patient_coverage=0.0,
            out=out,
        )

        assert out["patient_risk_frames"], "no risk rows were produced"
        combined = pd.concat(out["patient_risk_frames"], ignore_index=True)
        # The always-on test-block score (dataset="test") lands here too;
        # cv_oof is an additional, separately-labeled set that must never be
        # pooled with it.
        oof = combined.loc[combined["dataset"] == "cv_oof"]
        assert not oof.empty, "no cv_oof rows were produced"
        assert set(oof[ID_COL]) == set(ctx.merged.index)
        assert oof["risk_score"].notna().all()
        assert (oof["landmark_days"] == 0).all()

        assert "test" in set(combined["dataset"])
        assert out["metric_rows"]
        assert out["frames"]

    def test_no_out_of_fold_flag_produces_no_cv_oof_rows(self):
        ctx = _ctx(seed=4)
        args = _args(out_of_fold_risks=False)
        out = {
            "frames": [],
            "metric_rows": [],
            "test_auc_frames": [],
            "test_brier_frames": [],
            "patient_risk_frames": [],
            "canonical_labs_fold_rows": [],
        }
        horizon_grid = np.arange(30, 700, 30)
        _run_baseline_landmark(
            CONFIG,
            ca,
            ctx,
            {"platinum": horizon_grid},
            landmark_day=0,
            endpoints=["platinum"],
            args=args,
            auc_time_unit_days=30,
            auc_max_time_units=None,
            min_patient_coverage=0.0,
            out=out,
        )
        # The always-on test-block score still lands here; only the cv_oof
        # scheme is gated behind the flag.
        if out["patient_risk_frames"]:
            combined = pd.concat(out["patient_risk_frames"], ignore_index=True)
            assert "cv_oof" not in set(combined["dataset"])
