"""Analytic-truth and invariant tests for survival_common.cox_engine.compute_ipcw_auc_t.

These are the ground-truth checks the metric core never had:
  * an uninformative (constant/random-noise, non-discriminating) risk score
    should give AUC(t) ~= 0.5 at every horizon;
  * a perfect risk score (monotone in true event time) should give AUC(t) ~= 1.0;
  * IPCW AUC should be approximately stable across different independent
    censoring rates applied to the same underlying event process -- this is
    the test that would catch Finding 2 (mislabeling competing events as
    censoring), since that mislabeling biases the KM censoring-weight
    estimator in exactly this dimension.

Requires scikit-survival; skipped entirely if it is not installed (the repo's
"torch absent" invariant does not extend to sksurv, which the metric core
hard-requires).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sksurv = pytest.importorskip("sksurv")

from survival_common.cox_engine import compute_ipcw_auc_t


def _make_cohort(
    n: int, seed: int, censor_rate: float, risk_sd: float = 1.0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate an exponential event process with independent censoring.

    Returns (df[duration, event], true_risk) where true_risk is the hazard
    multiplier that generated each patient's event time (higher = faster
    events = higher risk), giving us a known-perfect risk score for free.
    `risk_sd` controls how strongly risk separates event times -- a low value
    (the uninformative/censoring-invariance tests) keeps discrimination
    realistic; a high value (the perfect-predictor test) pushes AUC(t) close
    to its ceiling since even a "perfect" proportional-hazards risk score
    only achieves AUC(t) ~= 1.0 in the limit of strong separation, not for
    any finite hazard ratio.
    """
    rng = np.random.default_rng(seed)
    baseline_hazard = 1.0 / 200.0
    true_log_risk = rng.normal(0, risk_sd, size=n)
    hazard = baseline_hazard * np.exp(true_log_risk)
    event_time = rng.exponential(1.0 / hazard)
    censor_time = (
        rng.exponential(1.0 / (censor_rate * baseline_hazard), size=n)
        if censor_rate > 0
        else np.full(n, np.inf)
    )
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    df = pd.DataFrame({"DURATION": duration, "EVENT": event})
    return df, true_log_risk


def _fixed_horizons(df: pd.DataFrame) -> np.ndarray:
    events = df.loc[df["EVENT"] == 1, "DURATION"]
    q = np.quantile(events, [0.15, 0.30, 0.45])
    return np.unique(np.ceil(q / 7.0))


class TestUninformativeRisk:
    def test_constant_risk_gives_auc_near_half(self):
        df, _ = _make_cohort(n=800, seed=1, censor_rate=0.5)
        risk = np.zeros(len(df))  # perfectly uninformative
        horizons = _fixed_horizons(df)
        mean_auc, auc_df = compute_ipcw_auc_t(
            df,
            risk,
            duration_col="DURATION",
            event_col="EVENT",
            reference_df=df,
            fixed_horizons=horizons,
        )
        assert not np.isnan(mean_auc)
        assert abs(mean_auc - 0.5) < 0.05

    def test_random_noise_risk_gives_auc_near_half(self):
        df, _ = _make_cohort(n=800, seed=2, censor_rate=0.5)
        rng = np.random.default_rng(99)
        risk = rng.normal(size=len(df))  # independent of outcome
        horizons = _fixed_horizons(df)
        mean_auc, _ = compute_ipcw_auc_t(
            df,
            risk,
            duration_col="DURATION",
            event_col="EVENT",
            reference_df=df,
            fixed_horizons=horizons,
        )
        assert not np.isnan(mean_auc)
        assert abs(mean_auc - 0.5) < 0.07


class TestPerfectRisk:
    def test_perfect_predictor_gives_auc_near_one(self):
        df, true_log_risk = _make_cohort(n=800, seed=3, censor_rate=0.3, risk_sd=3.0)
        horizons = _fixed_horizons(df)
        # sksurv convention: higher risk score = higher risk = shorter event time.
        mean_auc, _ = compute_ipcw_auc_t(
            df,
            true_log_risk,
            duration_col="DURATION",
            event_col="EVENT",
            reference_df=df,
            fixed_horizons=horizons,
        )
        assert not np.isnan(mean_auc)
        assert mean_auc > 0.90


class TestCensoringRateInvariance:
    def test_auc_stable_across_censoring_rates(self):
        # Same underlying event process (same seed for the hazard draw),
        # only the independent censoring mechanism's rate changes. A metric
        # that correctly IPCW-weights should recover ~the same AUC(t) at
        # each rate; large drift indicates the censoring/event bookkeeping
        # is biased (e.g. by conflating a competing event with censoring).
        aucs = []
        for censor_rate in (0.2, 0.5, 0.9):
            df, true_log_risk = _make_cohort(n=1500, seed=42, censor_rate=censor_rate)
            horizons = _fixed_horizons(df)
            mean_auc, _ = compute_ipcw_auc_t(
                df,
                true_log_risk,
                duration_col="DURATION",
                event_col="EVENT",
                reference_df=df,
                fixed_horizons=horizons,
            )
            aucs.append(mean_auc)
        aucs = np.asarray(aucs)
        assert np.isfinite(aucs).all()
        assert (aucs.max() - aucs.min()) < 0.08
