"""XGBoost out-of-fold risk scores must cover every patient and stay honest.

Mirrors tests/test_out_of_fold_risk_scores.py (the elastic-net Cox scheme) for
`compute_out_of_fold_risk_scores_xgb` in
COMPASS/survival_analysis/multivariate_analysis.py (Plan §3). Before this
function existed, XGBoost had no held-out-for-every-patient scoring path at
all, so labs-vs-clinical stratification figures could only use the
test-block score for that arm while elastic-net had a full-cohort cv_oof
score -- an inconsistency this closes.

Three properties, as specified by the plan:
  * one score per patient (coverage)
  * a patient's own score does not depend on their own outcome (honesty)
  * a noise-only run gives a C-index near 0.5 (the same null test the
    elastic-net honesty test uses)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lifelines")

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVIVAL_DIR = REPO_ROOT / "COMPASS" / "survival_analysis"
for _p in (str(REPO_ROOT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lifelines.utils import concordance_index  # noqa: E402

import multivariate_analysis as mva  # noqa: E402

ID_COL = "DFCI_MRN"
AGE_COL = "AGE"


def _cohort(
    *,
    n: int,
    seed: int,
    n_features: int = 4,
    signal: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    mrns = [f"P{i:04d}" for i in range(n)]
    lab_names = [f"L{j}" for j in range(n_features)]

    features = {f"{lab}__mean": rng.normal(size=n) for lab in lab_names}
    n_obs = {f"{lab}__n_obs": rng.integers(2, 20, size=n).astype(float) for lab in lab_names}

    if signal:
        duration = np.clip(
            400 - 90 * features["L0__mean"] + rng.normal(0, 40, size=n), 20, 900
        )
    else:
        duration = rng.uniform(30, 800, size=n)

    cohort = pd.DataFrame(
        {
            AGE_COL: rng.normal(66, 8, size=n),
            **features,
            **n_obs,
            "t_platinum": duration,
            "PLATINUM": rng.binomial(1, 0.5, size=n),
        },
        index=pd.Index(mrns, name=ID_COL),
    )
    lab_long = pd.DataFrame(
        {
            ID_COL: np.repeat(mrns, n_features),
            "LAB_NAME": lab_names * n,
            "LAB_VALUE": rng.normal(size=n * n_features),
        }
    )
    return cohort, lab_long, [f"{lab}__mean" for lab in lab_names]


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        no_cv=True,
        baseline=False,
        min_patient_coverage=0.0,
        max_features=None,
        feature_set="labs",
        auc_time_unit_days=7,
        auc_quantiles=(0.25, 0.5, 0.75),
        auc_max_time_units=60,
        num_boost_round=50,
        early_stopping_rounds=10,
        eta=0.1,
        max_depth=2,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        tree_method="hist",
        verbose_eval=0,
        seed=7,
        oof_outer_folds=4,
        oof_inner_folds=2,
        out_of_fold_risks=True,
        id_col=ID_COL,
        age_col=AGE_COL,
        cohort=None,
        n_folds=3,
        cv_max_depths=[2],
        cv_etas=[0.1],
        cv_min_child_weights=[5.0],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _run(cohort, lab_long, feature_cols, *, seed: int, outer_folds: int = 4, **arg_overrides):
    mva.ID_COL = ID_COL
    mva.AGE_COL = AGE_COL
    args = _args(seed=seed, oof_outer_folds=outer_folds, **arg_overrides)
    return mva.compute_out_of_fold_risk_scores_xgb(
        cohort=cohort,
        raw_feature_cols=feature_cols,
        canonical_labs=feature_cols,
        pre_treatment_lab_df=lab_long,
        horizon_grid=np.arange(4, 60, 4),
        endpoint="platinum",
        landmark_day=0,
        args=args,
    )


class TestCoverage:
    def test_every_patient_is_scored_exactly_once(self):
        cohort, lab_long, feature_cols = _cohort(n=160, seed=5)
        out = _run(cohort, lab_long, feature_cols, seed=5)

        assert len(out) == len(cohort)
        assert out[ID_COL].nunique() == len(cohort)
        assert set(out[ID_COL]) == set(cohort.index)
        assert out["risk_score"].notna().all()

    def test_frame_matches_the_predictions_column_contract(self):
        cohort, lab_long, feature_cols = _cohort(n=160, seed=9)
        out = _run(cohort, lab_long, feature_cols, seed=9)

        for column in (ID_COL, "endpoint", "dataset", "outer_fold", "duration", "event", "risk_score"):
            assert column in out.columns
        assert (out["dataset"] == "cv_oof").all()
        assert (out["endpoint"] == "platinum").all()
        merged = out.set_index(ID_COL).loc[cohort.index]
        assert np.allclose(merged["duration"], cohort["t_platinum"])
        assert (merged["event"].to_numpy() == cohort["PLATINUM"].to_numpy()).all()


class TestHonesty:
    def test_a_patients_own_score_does_not_depend_on_their_own_outcome(self):
        """Change one patient's survival time; their OWN score must not move.

        Same rationale as the elastic-net version of this test: the perturbed
        patient sits in the training set of every outer fold that excluded
        them, so those folds' models are allowed to change. What must not
        happen is that patient's own held-out score moving, since the model
        that scores them never saw their outcome.
        """
        cohort, lab_long, feature_cols = _cohort(n=160, seed=13)
        baseline = _run(cohort, lab_long, feature_cols, seed=13)

        target = cohort.index[0]
        perturbed_cohort = cohort.copy()
        perturbed_cohort.loc[target, "t_platinum"] = (
            float(cohort.loc[target, "t_platinum"]) * 0.25
        )
        perturbed = _run(perturbed_cohort, lab_long, feature_cols, seed=13)

        base_i = baseline.set_index(ID_COL)
        pert_i = perturbed.set_index(ID_COL)
        assert base_i.loc[target, "risk_score"] == pytest.approx(
            pert_i.loc[target, "risk_score"]
        )

    def test_no_signal_gives_a_c_index_near_one_half(self):
        scores = []
        for seed in (3, 17):
            cohort, lab_long, feature_cols = _cohort(
                n=200, seed=seed, n_features=8, signal=False
            )
            out = _run(cohort, lab_long, feature_cols, seed=seed)
            assert out["risk_score"].notna().all()
            scores.append(
                concordance_index(out["duration"], -out["risk_score"], out["event"])
            )

        mean_c = float(np.mean(scores))
        assert 0.35 < mean_c < 0.65, (
            f"OOF C-index on pure noise is {mean_c:.3f}, not ~0.5."
        )
