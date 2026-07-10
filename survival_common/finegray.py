"""Univariate Fine-Gray subdistribution-hazard fitting via IPCW-weighted Cox.

Implements Fine & Gray (1999) as a case-weighted Cox partial likelihood on the
subdistribution risk set (the "expanded risk set" reformulation), reusing
lifelines' ``CoxPHFitter`` (which supports ``weights_col``, ``entry_col``, and
``robust`` sandwich standard errors) rather than a from-scratch solver.

Scope: univariate association fitting only (mirrors survival_common.cox_engine's
fit_cox_with_fallback contract). Must not import from cox_aggregated.py or
cox_models.py -- this module is a leaf, consumed by cox_models.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.exceptions import ConvergenceError, ConvergenceWarning

    LIFELINES_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    CoxPHFitter = None
    KaplanMeierFitter = None
    ConvergenceError = RuntimeError
    ConvergenceWarning = RuntimeWarning
    LIFELINES_IMPORT_ERROR = exc

# See survival_common.cox_engine._MAX_NEWTON_STEPS: caps lifelines' Newton-
# Raphson step budget so rare/near-separated covariates fail fast instead of
# burning the full 500-step default, and escalates ConvergenceWarning (which
# lifelines raises instead of an exception on non-convergence) to an error so
# a capped, non-converged fit is treated as a failure rather than silently
# returning divergent coefficients.
_MAX_NEWTON_STEPS = 25

# Floor applied to the censoring survival function G(t) to avoid division by
# zero when weighting competing-event subjects past the last censoring time.
MIN_CENSORING_SURVIVAL = 1e-8


def require_lifelines() -> None:
    if CoxPHFitter is None or KaplanMeierFitter is None:
        raise ModuleNotFoundError(
            "lifelines is required for the Fine-Gray association pipeline."
        ) from LIFELINES_IMPORT_ERROR


class _StepFunction:
    """Right-continuous step function G(t), evaluated by right-side lookup.

    ``times`` must be sorted ascending; ``values`` is G(t) at each time (i.e.
    the survival probability immediately after that time). Evaluating at any
    t returns the value at the largest ``times`` entry <= t (or 1.0 before the
    first time), floored at MIN_CENSORING_SURVIVAL.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray):
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def __call__(self, t: np.ndarray | float) -> np.ndarray:
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.times, t_arr, side="right") - 1
        out = np.where(idx >= 0, self.values[np.clip(idx, 0, len(self.values) - 1)], 1.0)
        out = np.maximum(out, MIN_CENSORING_SURVIVAL)
        return out


def estimate_censoring_km(
    durations: np.ndarray,
    event_type: np.ndarray,
) -> _StepFunction:
    """Kaplan-Meier estimate of the censoring survival distribution G(t).

    The KM "event" here is censoring itself (event_type == 0) -- i.e. this is
    the reverse Kaplan-Meier estimator standard in IPCW weighting. Any
    non-zero event_type (event of interest or competing event) is treated as
    a KM censoring time for this fit.
    """
    require_lifelines()
    durations = np.asarray(durations, dtype=float)
    event_type = np.asarray(event_type)

    censoring_event = (event_type == 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=censoring_event)

    surv = kmf.survival_function_
    times = surv.index.to_numpy(dtype=float)
    values = surv.iloc[:, 0].to_numpy(dtype=float)
    # KM output already starts at t=0 -> G=1.0 (lifelines inserts this row).
    return _StepFunction(times, values)


def build_finegray_weighted_frame(
    df: pd.DataFrame,
    *,
    duration_col: str,
    event_type_col: str,
    covariate_cols: list[str],
    event_of_interest: int = 1,
    competing_event: int = 2,
    censoring_km: _StepFunction | None = None,
) -> pd.DataFrame:
    """Build the IPCW-weighted start-stop frame for Fine-Gray fitting.

    Subjects censored or experiencing the event of interest contribute a
    single row (start=0, stop=t_i, event=1{event_of_interest}, weight=1).
    Subjects experiencing the competing event are re-inserted into the
    subdistribution risk set at every distinct event-of-interest time after
    their own event time, with weight G(tau_k)/G(t_i) (both G's evaluated via
    the right-continuous step function so a value exactly at a jump uses the
    post-jump probability, matching the censoring-time convention used to fit
    G itself). Their `event` is always 0 (a competing-event subject never
    fails from the event of interest).

    Full expansion (strategy A from the design plan): one row per
    event-of-interest time per competing-event subject. Fine for the cohort
    sizes here (hundreds-low-thousands x tens of distinct event times).
    """
    if censoring_km is None:
        censoring_km = estimate_censoring_km(
            df[duration_col].to_numpy(dtype=float),
            df[event_type_col].to_numpy(),
        )

    durations = df[duration_col].to_numpy(dtype=float)
    event_type = df[event_type_col].to_numpy()
    covariates = df[covariate_cols]

    is_competing = event_type == competing_event
    is_direct = ~is_competing  # censored or event-of-interest: single row

    rows: list[pd.DataFrame] = []

    if is_direct.any():
        direct = covariates.loc[is_direct].copy()
        direct["_fg_start"] = 0.0
        direct["_fg_stop"] = durations[is_direct]
        direct["_fg_event"] = (event_type[is_direct] == event_of_interest).astype(int)
        direct["_fg_weight"] = 1.0
        rows.append(direct)

    if is_competing.any():
        event_times = np.sort(
            np.unique(durations[event_type == event_of_interest])
        )
        event_times = event_times[event_times > 0]
        if len(event_times) == 0:
            # No event-of-interest occurrences at all: competing-event subjects
            # contribute nothing beyond their own (non-informative) presence;
            # emit a single zero-weight row so the frame stays non-empty and
            # the caller's downstream fit fails loudly via "no events" rather
            # than silently dropping subjects.
            comp = covariates.loc[is_competing].copy()
            comp["_fg_start"] = 0.0
            comp["_fg_stop"] = durations[is_competing]
            comp["_fg_event"] = 0
            comp["_fg_weight"] = 1.0
            rows.append(comp)
        else:
            comp_idx = np.flatnonzero(is_competing)
            comp_t = durations[comp_idx]
            g_at_t = censoring_km(comp_t)
            g_at_tau = censoring_km(event_times)

            interval_starts = np.concatenate(([0.0], event_times[:-1]))
            for i, subj_pos in enumerate(comp_idx):
                t_i = comp_t[i]
                g_t_i = g_at_t[i]
                subj_cov = covariates.iloc[subj_pos]
                subj_rows: list[dict] = []
                for k, tau_k in enumerate(event_times):
                    start_k = interval_starts[k]
                    if tau_k <= start_k:
                        continue
                    if tau_k <= t_i:
                        weight = 1.0
                        stop_k = min(tau_k, t_i) if tau_k <= t_i else tau_k
                    else:
                        weight = float(g_at_tau[k] / g_t_i)
                        stop_k = tau_k
                    if start_k >= stop_k:
                        continue
                    row = {"_fg_start": start_k, "_fg_stop": stop_k, "_fg_event": 0, "_fg_weight": weight}
                    row.update(subj_cov.to_dict())
                    subj_rows.append(row)
                if subj_rows:
                    rows.append(pd.DataFrame(subj_rows))

    if not rows:
        raise ValueError("build_finegray_weighted_frame: no rows produced from input data.")

    weighted = pd.concat(rows, ignore_index=True)
    ordered_cols = list(covariate_cols) + ["_fg_start", "_fg_stop", "_fg_event", "_fg_weight"]
    return weighted[ordered_cols]


def fit_finegray_univariate_with_fallback(
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_type_col: str,
    covariate_cols: list[str],
    penalizers: list[float],
    event_of_interest: int = 1,
    competing_event: int = 2,
    censoring_km: _StepFunction | None = None,
) -> tuple[object | None, float, str]:
    """Fine-Gray fit mirroring fit_cox_with_fallback's (model, penalizer, note) contract.

    The returned model's .summary is indexed by covariate_cols (lifelines
    CoxPHFitter.summary is indexed by the fitted covariate names), so callers
    read model.summary.loc['feature_z'] exactly as with the plain-Cox path.
    """
    require_lifelines()

    try:
        weighted = build_finegray_weighted_frame(
            model_df,
            duration_col=duration_col,
            event_type_col=event_type_col,
            covariate_cols=covariate_cols,
            event_of_interest=event_of_interest,
            competing_event=competing_event,
            censoring_km=censoring_km,
        )
    except ValueError as exc:
        return None, float("nan"), f"finegray_fit_failed: {exc}"

    last_error = ""
    for penalizer in penalizers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("error", category=ConvergenceWarning)
                model = CoxPHFitter(penalizer=float(penalizer), l1_ratio=0.0)
                model.fit(
                    weighted,
                    duration_col="_fg_stop",
                    event_col="_fg_event",
                    entry_col="_fg_start",
                    weights_col="_fg_weight",
                    robust=True,
                    fit_options={"max_steps": _MAX_NEWTON_STEPS},
                )
            note = "finegray_fit_ok" if penalizer == 0 else f"finegray_fit_ok_penalizer_{penalizer:g}"
            return model, penalizer, note
        except (ConvergenceError, ConvergenceWarning, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)

    return None, float("nan"), f"finegray_fit_failed: {last_error}"
