"""Differential check: deephit_engine.compute_brier_for_pred vs. the shared
helper.compute_brier oracle it wraps (deephit_engine.py:573-579 calls
helper.py:391's compute_brier directly, passing the fixed
``horizons_by_event`` array unmodified -- no np.arange re-derivation, no
constant-risk substitution).

Unlike the AUC path (Finding 1), the Brier path does NOT re-derive its own
grid: it is a thin per-cause wrapper around the shared oracle. This test is
the control that shows Brier agrees when fed a survival surface computed
identically to what compute_brier would score directly, confirming the bug
is isolated to the AUC path and not present here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sksurv = pytest.importorskip("sksurv")

from survival_common.deephit_engine import compute_brier_for_pred
from survival_common.helper import compute_brier


def test_deephit_brier_wrapper_matches_shared_oracle_directly():
    rng = np.random.default_rng(5)
    n = 500
    baseline_hazard = 1.0 / 150.0
    true_log_risk = rng.normal(0, 1.5, size=n)
    hazard = baseline_hazard * np.exp(true_log_risk)
    event_time = rng.exponential(1.0 / hazard)
    censor_time = rng.exponential(1.0 / (0.4 * baseline_hazard), size=n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    events_only = duration[event == 1]
    horizon_units = np.unique(np.ceil(np.quantile(events_only, [0.2, 0.4, 0.6]) / 7.0))

    # Survival surface: logistic in (log_risk + log(t)) -- same construction
    # style as the AUC differential test, giving a genuinely time-varying,
    # internally consistent CIF.
    cif_cols = []
    for t in horizon_units:
        score = true_log_risk + 0.5 * np.log(max(t, 1))
        cif_cols.append(1.0 / (1.0 + np.exp(-score)))
    surv_at_horizons = 1.0 - np.column_stack(cif_cols)

    # Oracle: call helper.compute_brier directly on the identical inputs.
    oracle_df, oracle_ibs = compute_brier(
        train_event=event,
        train_duration=duration,
        eval_event=event,
        eval_duration=duration,
        surv_at_horizons=surv_at_horizons,
        horizons=horizon_units,
    )

    # DeepHit wrapper: build a pred frame with the matching risk_h{t} = CIF
    # columns and identical train_val_targets, then call the wrapper.
    pred = pd.DataFrame({"label": event, "duration": duration})
    for t, cif in zip(horizon_units, cif_cols):
        pred[f"event_1_risk_h{int(t)}"] = cif
    train_val_targets = pd.DataFrame({"label": event, "duration": duration})

    wrapper_df, ibs_by_event = compute_brier_for_pred(
        pred,
        event_names=["platinum"],
        train_val_targets=train_val_targets,
        horizons_by_event={"platinum": horizon_units},
    )
    wrapper_ibs = ibs_by_event["platinum"]

    assert np.isfinite(oracle_ibs)
    assert np.isfinite(wrapper_ibs)
    assert wrapper_ibs == pytest.approx(oracle_ibs, abs=1e-9)

    oracle_by_h = dict(zip(oracle_df["horizon_time_unit"], oracle_df["brier"]))
    wrapper_by_h = dict(zip(wrapper_df["horizon_time_unit"], wrapper_df["brier"]))
    for h in horizon_units:
        assert wrapper_by_h[float(h)] == pytest.approx(oracle_by_h[float(h)], abs=1e-9)
