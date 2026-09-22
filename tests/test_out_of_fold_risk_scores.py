"""Out-of-fold risk scores must cover every patient and stay honest.

`fit_final_multivariable_model` scores only the held-out test block, because
that is the one partition its model never saw. Scoring the whole cohort needs
a scheme where each patient is excluded from the model that scores them, which
is what `compute_out_of_fold_risk_scores` provides via nested CV.

Two properties matter, and each has a test below:

  * Coverage -- every patient gets exactly one score, attributed to the one
    outer fold that excluded them. A silent shortfall would mean the
    stratified figures are drawn on a subset while claiming the full cohort.

  * Honesty -- hyperparameters are re-tuned inside each outer fold rather than
    reused from a tuning pass that saw every patient. The cheap alternative
    (reusing the validation predictions already computed in
    `tune_multivariable_model`) is leaky: those same folds pick the winning
    penalizer/l1_ratio, so a patient's "held-out" score would come from a model
    tuned partly on that patient's own outcome. The null test pins the
    consequence -- on features with no signal, the OOF C-index must sit near
    0.5.

The nested scheme mirrors the clinical text embedding project's
semantic_search/train_prediction_models.py, including its insistence that
outer CV produce exactly one prediction per patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")

from lifelines.utils import concordance_index  # noqa: E402

from survival_common.cox_models import compute_out_of_fold_risk_scores  # noqa: E402

ENDPOINTS = {
    "platinum": {
        "duration_col": "t_platinum",
        "event_col": "PLATINUM",
        "description": "synthetic time-to-platinum",
    }
}

ID_COL = "DFCI_MRN"


def _cohort(
    *,
    n: int,
    seed: int,
    n_features: int = 4,
    signal: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """A landmark-shaped cohort plus the long lab frame canonical-lab selection reads.

    With `signal=False` the durations are drawn independently of every feature,
    so nothing is predictable and an honest scheme scores ~0.5.
    """
    rng = np.random.default_rng(seed)
    mrns = [f"P{i:04d}" for i in range(n)]
    lab_names = [f"L{j}" for j in range(n_features)]

    features = {f"{lab}__mean": rng.normal(size=n) for lab in lab_names}
    n_obs = {f"{lab}__n_obs": rng.integers(2, 20, size=n).astype(float) for lab in lab_names}

    if signal:
        # Higher L0 -> shorter time to event, so a working model has something
        # real to recover.
        duration = np.clip(
            400 - 90 * features["L0__mean"] + rng.normal(0, 40, size=n), 20, 900
        )
    else:
        duration = rng.uniform(30, 800, size=n)

    cohort = pd.DataFrame(
        {
            "AGE": rng.normal(66, 8, size=n),
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


def _run(cohort, lab_long, feature_cols, *, seed: int, outer_folds: int = 4, **kwargs):
    return compute_out_of_fold_risk_scores(
        cohort,
        raw_feature_cols=feature_cols,
        endpoint="platinum",
        penalizers=kwargs.pop("penalizers", [0.01, 0.1]),
        l1_ratios=kwargs.pop("l1_ratios", [0.5]),
        outer_folds=outer_folds,
        inner_folds=3,
        seed=seed,
        auc_time_unit_days=7,
        auc_max_time_units=60,
        pre_treatment_lab_df=lab_long,
        horizon_grid=np.arange(4, 60, 4),
        min_patient_coverage=0.0,
        endpoint_map=ENDPOINTS,
        id_col=ID_COL,
        age_col="AGE",
        **kwargs,
    )


class TestCoverage:
    def test_every_patient_is_scored_exactly_once(self):
        cohort, lab_long, feature_cols = _cohort(n=200, seed=5)
        out = _run(cohort, lab_long, feature_cols, seed=5)

        assert len(out) == len(cohort)
        assert out[ID_COL].nunique() == len(cohort)
        assert set(out[ID_COL]) == set(cohort.index)
        assert out["risk_score"].notna().all()

    def test_each_patient_is_attributed_to_one_outer_fold(self):
        cohort, lab_long, feature_cols = _cohort(n=200, seed=5)
        out = _run(cohort, lab_long, feature_cols, seed=5, outer_folds=4)

        assert set(out["outer_fold"]) == {1, 2, 3, 4}
        # Every fold scores a real share; no fold silently collapses to empty.
        assert out["outer_fold"].value_counts().min() >= len(cohort) // 8

    def test_frame_matches_the_predictions_column_contract(self):
        """Downstream readers key on these columns; dataset marks the scheme."""
        cohort, lab_long, feature_cols = _cohort(n=160, seed=9)
        out = _run(cohort, lab_long, feature_cols, seed=9)

        for column in (ID_COL, "endpoint", "dataset", "duration_days", "event", "risk_score"):
            assert column in out.columns
        assert (out["dataset"] == "cv_oof").all()
        assert (out["endpoint"] == "platinum").all()
        # Durations and events are carried through unchanged, so a joined
        # figure cannot silently disagree with the cohort it came from.
        merged = out.set_index(ID_COL).loc[cohort.index]
        assert np.allclose(merged["duration_days"], cohort["t_platinum"])
        assert (merged["event"].to_numpy() == cohort["PLATINUM"].to_numpy()).all()

    def test_scores_recover_a_planted_signal(self):
        cohort, lab_long, feature_cols = _cohort(n=240, seed=11)
        out = _run(cohort, lab_long, feature_cols, seed=11)

        c = concordance_index(out["duration_days"], -out["risk_score"], out["event"])
        assert c > 0.65, f"planted signal not recovered (C={c:.3f})"


class TestHonesty:
    def test_no_signal_gives_a_c_index_near_one_half(self):
        """The property nesting buys.

        25 pure-noise features and a grid wide enough to overfit: a scheme that
        tuned on the scored patients would drift above 0.5 here. Averaged over
        seeds to keep the bound tight without being flaky.
        """
        scores = []
        for seed in (3, 17, 29):
            cohort, lab_long, feature_cols = _cohort(
                n=300, seed=seed, n_features=25, signal=False
            )
            out = _run(
                cohort,
                lab_long,
                feature_cols,
                seed=seed,
                penalizers=[0.001, 0.01, 0.1],
                l1_ratios=[0.5, 1.0],
            )
            assert out["risk_score"].notna().all()
            scores.append(
                concordance_index(out["duration_days"], -out["risk_score"], out["event"])
            )

        mean_c = float(np.mean(scores))
        assert 0.42 < mean_c < 0.58, (
            f"OOF C-index on pure noise is {mean_c:.3f}, not ~0.5. A value "
            f"meaningfully above 0.5 means the scoring scheme saw the outcomes "
            f"of the patients it scored."
        )

    def test_a_patients_own_score_does_not_depend_on_their_own_outcome(self):
        """Change one patient's survival time; their OWN score must not move.

        This is the precise held-out property. It is deliberately not the
        stronger claim that *no* score moves: the perturbed patient sits in the
        training set of every fold that excluded them, so those folds' models
        are supposed to change. What must never happen is a patient's own score
        responding to their own outcome, since the model scoring them never saw
        it.

        Only the duration is perturbed, not the event indicator: the event
        drives `make_cv_splitter`'s stratification labels, so flipping it would
        repartition the whole cohort and compare two different fold structures.
        """
        cohort, lab_long, feature_cols = _cohort(n=200, seed=13)
        baseline = _run(cohort, lab_long, feature_cols, seed=13)

        target = cohort.index[0]
        perturbed_cohort = cohort.copy()
        perturbed_cohort.loc[target, "t_platinum"] = (
            float(cohort.loc[target, "t_platinum"]) * 0.25
        )
        perturbed = _run(perturbed_cohort, lab_long, feature_cols, seed=13)

        base_i = baseline.set_index(ID_COL)
        pert_i = perturbed.set_index(ID_COL)

        assert int(base_i.loc[target, "outer_fold"]) == int(
            pert_i.loc[target, "outer_fold"]
        ), "the fold partition moved; the comparison below would be meaningless"

        assert base_i.loc[target, "risk_score"] == pytest.approx(
            pert_i.loc[target, "risk_score"]
        ), (
            "a patient's own risk score changed when only that patient's own "
            "outcome changed -- the model scoring them saw their outcome"
        )


class TestFailureReporting:
    def test_a_configuration_error_is_raised_not_swallowed(self):
        """Every fold failing the same way is a config bug, not fold noise.

        _FOLD_FIT_ERRORS includes ValueError, which is also what a
        misconfiguration raises. Reporting that as N quiet "outer_fold_failed"
        notes would bury the real cause.
        """
        cohort, lab_long, feature_cols = _cohort(n=120, seed=21)
        # The lab frame is missing the columns canonical-lab selection requires.
        broken_labs = lab_long.rename(columns={"LAB_NAME": "lab_name"})

        with pytest.raises(RuntimeError) as excinfo:
            _run(cohort, broken_labs, feature_cols, seed=21)

        assert "Every outer fold failed" in str(excinfo.value)
        # The underlying cause is chained, not discarded.
        assert excinfo.value.__cause__ is not None
        assert "LAB_NAME" in str(excinfo.value.__cause__)
