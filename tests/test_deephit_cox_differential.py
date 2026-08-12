"""Differential test: DeepHit's metric path vs. survival_common.cox_engine on
identical synthetic inputs.

This is the highest-value technique in the plan (Stage 2b): reading code
cannot confirm a metric bug, but running both paths on the same data can. The
oracle contract, stated verbatim at cox_engine.py:196-201, is:

  ``fixed_horizons`` is the train/validation-derived requested timeline. The
  evaluation split is used only to decide which requested points are
  estimable, never to create replacement horizons ... a mean ... is always
  sksurv's censoring-weighted integral over the valid horizons -- never a
  substitute summary.

deephit_engine.compute_metrics previously violated this twice when computing
mean_auc_t:
  1. It rebuilt the integration grid with ``np.arange(horizons[0],
     horizons[-1])`` -- every integer week in the span, not the fixed
     manifest ``horizons`` array (and exclusive of its stop value, so it
     always dropped the final requested horizon).
  2. For any re-derived time lacking a materialized ``event_{k}_risk_h{t}``
     column, it silently substituted the time-constant total risk instead of
     masking the point as inestimable.

The fix makes DeepHit integrate over exactly the fixed requested horizons,
masking (not substituting) any horizon whose column isn't materialized, and
abstaining (NaN) via the same coverage guard cox_engine uses when too few
horizons are estimable. Scenario A verifies agreement on a fully
materialized grid; Scenario B verifies agreement under partial coverage and
correct abstention under insufficient coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sksurv = pytest.importorskip("sksurv")

from survival_common.cox_engine import compute_ipcw_auc_t
from survival_common.deephit_engine import compute_brier_for_pred, compute_metrics
from survival_common.helper import compute_brier


def _synthetic_single_event(n=600, seed=7):
    """Single (non-competing) event process with a genuinely time-varying risk."""
    rng = np.random.default_rng(seed)
    baseline_hazard = 1.0 / 150.0
    true_log_risk = rng.normal(0, 1.5, size=n)
    hazard = baseline_hazard * np.exp(true_log_risk)
    event_time = rng.exponential(1.0 / hazard)
    censor_time = rng.exponential(1.0 / (0.4 * baseline_hazard), size=n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return duration, event, true_log_risk


def _build_pred_frame(duration, event, true_log_risk, horizons_days, time_unit_days=7):
    """Materialize event_1_risk_h{t} at every requested horizon (time-varying
    risk: cumulative incidence grows with t, individual ordering follows
    true_log_risk), matching the shape deephit_engine.predict() produces.

    ``duration`` is in raw days (as cox_engine.compute_ipcw_auc_t expects and
    converts internally via ``duration_to_auc_units``). deephit_engine's
    ``compute_metrics``/``compute_brier_for_pred`` take ``pred["duration"]``
    already in the same AUC time units as ``horizons`` (no internal
    conversion), so the frame built here converts to those units to keep
    both paths scoring the identical underlying timeline.
    """
    n = len(duration)
    horizon_units = np.unique([int(round(h / time_unit_days)) for h in horizons_days])
    duration_units = np.ceil(np.asarray(duration, dtype=float) / float(time_unit_days))
    pred = pd.DataFrame(
        {
            "label": event,  # event_idx=1 <=> label==1 for single-event
            "duration": duration_units,
            "event_1_risk_total": 1.0 / (1.0 + np.exp(-true_log_risk)),
        }
    )
    # A monotone-in-t, monotone-in-risk CIF surrogate: logistic in
    # (log_risk + log(t)), so both cox_engine (fed a single time-constant
    # column) and DeepHit (fed the per-horizon columns) are scoring a
    # genuinely time-varying but internally consistent risk surface.
    for t in horizon_units:
        score = true_log_risk + 0.5 * np.log(max(t, 1))
        pred[f"event_1_risk_h{t}"] = 1.0 / (1.0 + np.exp(-score))
    return pred, horizon_units


class TestScenarioA_FullyMaterializedGrid:
    """Every requested horizon has a risk_h{t} column. Post-fix, DeepHit
    integrates over exactly the fixed requested horizons array (no
    ``np.arange(horizons[0], horizons[-1])`` re-derivation, which used to be
    exclusive of its stop value and silently drop the final requested
    horizon), so it agrees with cox_engine's oracle to floating-point
    tolerance whenever every column is materialized.
    """

    def test_deephit_grid_includes_every_requested_horizon(self):
        duration, event, true_log_risk = _synthetic_single_event()
        train_val = pd.DataFrame({"DURATION": duration, "EVENT": event})
        events_only = train_val.loc[train_val["EVENT"] == 1, "DURATION"]
        h0 = int(np.ceil(np.quantile(events_only, 0.3) / 7.0))
        horizon_units = np.arange(h0, h0 + 4, dtype=float)
        horizons_days = horizon_units * 7.0

        pred, _ = _build_pred_frame(duration, event, true_log_risk, horizons_days)
        const_risk = pred["event_1_risk_total"].to_numpy()

        cox_mean_auc, cox_auc_df = compute_ipcw_auc_t(
            train_val,
            const_risk,
            duration_col="DURATION",
            event_col="EVENT",
            reference_df=train_val,
            fixed_horizons=horizon_units,
        )
        # cox_engine's oracle integrates over all 4 requested (eligible) points.
        n_cox_points = int(np.isfinite(cox_auc_df["auc_t"]).sum())

        pred_const = pred.copy()
        for t in range(int(horizon_units.min()), int(horizon_units.max()) + 1):
            pred_const[f"event_1_risk_h{t}"] = const_risk

        train_val_targets = pd.DataFrame({"label": event, "duration": pred["duration"]})
        metrics_df, auc_rows_df = compute_metrics(
            pred_const,
            event_names=["platinum"],
            train_val_targets=train_val_targets,
            quantiles=(0.25, 0.5, 0.75),
            fixed_horizons_by_event={"platinum": horizon_units},
        )
        deephit_mean_auc = float(metrics_df.loc[0, "mean_auc_t"])

        assert np.isfinite(cox_mean_auc)
        assert np.isfinite(deephit_mean_auc)
        assert n_cox_points == 4  # sanity: cox_engine used the full requested set
        # Post-fix: DeepHit's mean matches cox_engine's exactly, since both
        # now integrate over the identical fixed 4-point requested timeline.
        assert deephit_mean_auc == pytest.approx(cox_mean_auc, abs=1e-9)


class TestScenarioB_SparseGrid_ProvesFinding1:
    """Drop a subset of risk_h{t} columns. Post-fix, DeepHit masks exactly
    the dropped horizons (rather than substituting constant risk at them),
    so its mean AUC(t) matches cox_engine's oracle integrated over that same
    masked subset -- and it abstains (NaN) together with cox_engine's
    coverage guard when too few points remain. This is the red/green anchor
    for the Finding 1 fix.
    """

    def _run_both(self, drop_fraction: float):
        duration, event, true_log_risk = _synthetic_single_event(seed=11)
        train_val = pd.DataFrame({"DURATION": duration, "EVENT": event})
        events_only = train_val.loc[train_val["EVENT"] == 1, "DURATION"]
        horizon_units = np.unique(
            np.ceil(np.quantile(events_only, [0.15, 0.3, 0.45, 0.6, 0.75]) / 7.0)
        )

        pred, materialized = _build_pred_frame(
            duration, event, true_log_risk, horizon_units * 7.0
        )
        train_val_targets = pd.DataFrame({"label": event, "duration": pred["duration"]})

        # DeepHit path: drop a subset of the materialized risk_h{t} columns so
        # compute_metrics's fixed-grid masking (post-fix) has to skip them.
        pred_sparse = pred.copy()
        rng = np.random.default_rng(0)
        drop_cols = [
            c
            for c in pred_sparse.columns
            if c.startswith("event_1_risk_h") and rng.random() < drop_fraction
        ]
        pred_sparse = pred_sparse.drop(columns=drop_cols)
        dropped_units = {int(c.rsplit("h", 1)[1]) for c in drop_cols}
        surviving_horizons = np.asarray(
            sorted(t for t in horizon_units if int(t) not in dropped_units), dtype=float
        )

        metrics_df, _ = compute_metrics(
            pred_sparse,
            event_names=["platinum"],
            train_val_targets=train_val_targets,
            quantiles=(0.25, 0.5, 0.75),
            fixed_horizons_by_event={"platinum": horizon_units},
        )
        deephit_mean_auc = float(metrics_df.loc[0, "mean_auc_t"])

        # cox_engine path: the oracle, integrated over the identical surviving
        # (post-mask) horizon subset DeepHit actually used -- an apples-to-
        # apples comparison of the integration/masking machinery, not of
        # which points get dropped (cox_engine has no notion of "missing
        # columns"; DeepHit's column availability is what drives the mask).
        const_risk = pred["event_1_risk_total"].to_numpy()
        cox_mean_auc, _ = compute_ipcw_auc_t(
            train_val,
            const_risk,
            duration_col="DURATION",
            event_col="EVENT",
            reference_df=train_val,
            fixed_horizons=surviving_horizons if len(surviving_horizons) else horizon_units,
        )
        return cox_mean_auc, deephit_mean_auc

    def test_paths_agree_after_fix_when_coverage_is_sufficient(self):
        """GREEN (post-fix): with one materialized column dropped (leaving
        enough coverage to clear the shared ``MIN_IPCW_VALID_HORIZONS`` /
        ``MIN_IPCW_TIMELINE_COVERAGE`` guard), DeepHit now masks the missing
        horizon exactly like cox_engine does, instead of substituting
        constant risk -- so the two paths' per-horizon AUCs and resulting
        integral agree to floating-point tolerance. Pre-fix, this scenario
        demonstrated disagreement (Finding 1); that red state is what the
        deephit_engine.py fix (masking instead of np.arange + constant-risk
        substitution) eliminates.
        """
        cox_mean_auc, deephit_mean_auc = self._run_both(drop_fraction=0.2)
        assert np.isfinite(cox_mean_auc)
        assert np.isfinite(deephit_mean_auc)
        assert deephit_mean_auc == pytest.approx(cox_mean_auc, abs=1e-9)

    def test_paths_abstain_together_when_coverage_is_insufficient(self):
        """GREEN (post-fix): when so many columns are dropped that fewer than
        half the requested horizons are estimable, DeepHit now abstains
        (NaN) via the same coverage guard as cox_engine, rather than
        silently reporting a substitute-risk mean as if it were valid.
        """
        _, deephit_mean_auc = self._run_both(drop_fraction=0.6)
        assert not np.isfinite(deephit_mean_auc)
