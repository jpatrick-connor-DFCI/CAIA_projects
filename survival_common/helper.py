"""
Shared survival-analysis utilities used by the project adapters plus the
univariate and multivariate entry points so the pipelines agree on:

  * canonical lab selection (coverage-based)
  * AUC(t) / Brier horizon grid (fixed from train_val event times)
  * IPCW Brier (per-horizon + integrated) via sksurv
  * Breslow baseline survival for linear-predictor models (Cox / XGBoost-Cox)
  * 5-fold stratified CV iteration (endpoint-aware event labels)
  * train/test leakage guards

This module is standalone: it must not import from cox_aggregated.py so that
cox_aggregated → helper is a one-way dependency.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from sksurv.metrics import brier_score, cumulative_dynamic_auc, integrated_brier_score
    from sksurv.linear_model.coxph import BreslowEstimator

    SKSURV_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    brier_score = None
    cumulative_dynamic_auc = None
    integrated_brier_score = None
    BreslowEstimator = None
    SKSURV_IMPORT_ERROR = exc


DEFAULT_AUC_QUANTILES: tuple[float, ...] = (0.25, 0.375, 0.50, 0.625, 0.75)
DEFAULT_AUC_TIME_UNIT_DAYS: int = 7
DEFAULT_AUC_MAX_GRID_POINTS: int = 25
# Administrative censoring horizon for IPCW metrics (~5 years in weeks). The
# builders clamp the horizon grid to this so every requested horizon stays
# inside the window the runners later evaluate over.
DEFAULT_AUC_MAX_TIME_UNITS: int = 260
# v3 adds the required `auc_max_time_units` key and clamps the grid to it, so
# v2 manifests (grid built without the cap) must be rebuilt, not reused.
AUC_TIMELINE_SCHEMA_VERSION: int = 3
MIN_IPCW_VALID_HORIZONS: int = 2
MIN_IPCW_TIMELINE_COVERAGE: float = 0.50
DEFAULT_MIN_DISTINCT_LAB_VALUES: int = 2


def require_sksurv() -> None:
    if brier_score is None:
        raise ModuleNotFoundError(
            "sksurv is required for helper.py utilities."
        ) from SKSURV_IMPORT_ERROR


@contextmanager
def suppress_sksurv_warnings():
    """Silence RuntimeWarning/UserWarning/FutureWarning noise from sksurv calls.

    sksurv (and the numpy it drives internally) frequently warns on
    ill-conditioned folds/horizons -- e.g. division-by-zero in IPCW weighting
    or deprecation notices -- that are expected and already handled by the
    caller's own NaN/`note` fallback, so they're non-actionable clutter.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        yield


# ---------------------------------------------------------------------------
# Tiny shared primitives (kept here so this file is import-cycle-safe).
# ---------------------------------------------------------------------------

def _make_survival_array(event: np.ndarray, duration: np.ndarray) -> np.ndarray:
    survival = np.empty(
        dtype=[("event", bool), ("time", np.float64)],
        shape=len(duration),
    )
    survival["event"] = event.astype(bool)
    survival["time"] = duration.astype(float)
    return survival


def _duration_to_time_units(duration: pd.Series | np.ndarray, *, time_unit_days: int) -> np.ndarray:
    if time_unit_days <= 0:
        raise ValueError("time_unit_days must be positive.")
    arr = pd.to_numeric(pd.Series(duration), errors="coerce").to_numpy(dtype=float)
    return np.ceil(arr / float(time_unit_days))


def combined_event_label(df: pd.DataFrame) -> np.ndarray:
    """4-cell combined label: PLATINUM + 2*DEATH (00, 10, 01, 11)."""
    p = pd.to_numeric(df["PLATINUM"], errors="coerce").fillna(0).astype(int).to_numpy()
    d = pd.to_numeric(df["DEATH"], errors="coerce").fillna(0).astype(int).to_numpy()
    return p + 2 * d


def _binary_event_label(df: pd.DataFrame, event_col: str) -> np.ndarray:
    return pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int).to_numpy()


def choose_stratification_labels(
    df: pd.DataFrame,
    *,
    min_count: int,
    event_col: str | None = None,
) -> tuple[np.ndarray | None, str]:
    """Pick the strictest stratification label whose smallest cell >= min_count.

    When event_col is supplied, use only that endpoint event or fall back to
    unstratified. Without event_col, use the historical metadata-driven
    preference order: combined (4 cells) -> either -> platinum -> death ->
    unstratified.
    """
    if event_col is not None and event_col not in df.columns:
        return None, "unstratified"
    if event_col and event_col in df.columns:
        labels = _binary_event_label(df, event_col)
        counts = pd.Series(labels).value_counts()
        if len(counts) > 1 and counts.min() >= min_count:
            return labels, event_col.lower()
        return None, "unstratified"

    candidates: list[tuple[str, np.ndarray]] = []
    if {"PLATINUM", "DEATH"}.issubset(df.columns):
        candidates.append(("combined", combined_event_label(df)))
    if "EITHER" in df.columns:
        candidates.append(("either", _binary_event_label(df, "EITHER")))
    if "PLATINUM" in df.columns:
        candidates.append(("platinum", _binary_event_label(df, "PLATINUM")))
    if "DEATH" in df.columns:
        candidates.append(("death", _binary_event_label(df, "DEATH")))
    for label_name, labels in candidates:
        counts = pd.Series(labels).value_counts()
        if len(counts) > 1 and counts.min() >= min_count:
            return labels, label_name
    return None, "unstratified"


# ---------------------------------------------------------------------------
# Canonical lab selection.
# ---------------------------------------------------------------------------

def select_canonical_labs(
    long_df: pd.DataFrame,
    *,
    mrns: Iterable,
    min_coverage: float,
    id_col: str = "DFCI_MRN",
    min_distinct_values: int = DEFAULT_MIN_DISTINCT_LAB_VALUES,
) -> list[str]:
    """Coverage-based lab selection on a fixed MRN set.

    `long_df` must already represent the pre-landmark window (i.e. only rows
    that should count toward coverage). A lab is eligible iff:

      * fraction of `mrns` with >=1 observation >= min_coverage, AND
      * the lab has >= min_distinct_values distinct LAB_VALUE entries across
        the mrns rows (drops constant / flagged-only labs).

    Returned list is sorted alphabetically for determinism. The MRN set passed
    here defines the leakage boundary: callers must pass train_val MRNs at the
    pipeline level, fold_train MRNs inside CV, and never test MRNs.
    """
    required = {id_col, "LAB_NAME", "LAB_VALUE"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"select_canonical_labs missing columns: {sorted(missing)}")

    mrn_set = set(mrns)
    if not mrn_set:
        raise ValueError("select_canonical_labs received an empty MRN set.")

    sub = long_df.loc[long_df[id_col].isin(mrn_set)]
    if sub.empty:
        raise ValueError("select_canonical_labs: no observations for the given MRNs.")

    n = len(mrn_set)
    coverage = sub.groupby("LAB_NAME")[id_col].nunique() / float(n)
    distinct = sub.groupby("LAB_NAME")["LAB_VALUE"].nunique()
    eligible_mask = (
        coverage.ge(float(min_coverage))
        & distinct.reindex(coverage.index).fillna(0).ge(int(min_distinct_values))
    )
    eligible = coverage.index[eligible_mask].tolist()
    return sorted(str(lab) for lab in eligible)


# ---------------------------------------------------------------------------
# Fixed AUC / Brier horizon grid (derived from train_val event times only).
# ---------------------------------------------------------------------------

def resolve_auc_max_time_units(build_manifest: dict, requested: int | None) -> int:
    """Resolve the IPCW evaluation cap, preferring the cap the grid was built under.

    The builders clamp the horizon grid to their own ``--auc-max-time-units``
    and record it in the manifest. Evaluating with a different cap would mark
    otherwise-estimable horizons inestimable, so the manifest value is the
    default; an explicit request still wins but is called out.
    """
    manifest_cap = int(build_manifest["auc_max_time_units"])
    if requested is None:
        return manifest_cap
    requested = int(requested)
    if requested != manifest_cap:
        print(
            f"WARNING: --auc-max-time-units={requested} differs from the build "
            f"manifest's {manifest_cap}, which the horizon grid was clamped to. "
            "Horizons at or beyond the requested cap will be reported inestimable."
        )
    return requested


def compute_horizon_grid(
    train_val_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    quantiles: tuple[float, ...] = DEFAULT_AUC_QUANTILES,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
    admin_censor_days: int | float | None = None,
    max_grid_points: int = DEFAULT_AUC_MAX_GRID_POINTS,
) -> np.ndarray:
    """Build a bounded AUC(t) / Brier timeline from train_val only.

    The outer quantiles define a robust event-time interval after optional
    administrative censoring. Rather than evaluating only at the handful of
    quantile values (which often collapse at sparse late landmarks), the
    function fills that interval with evenly spaced integer time units, capped
    at ``max_grid_points``. The requested timeline is fixed once and reused
    across CV folds and held-out evaluation, eliminating test-driven horizon
    selection. Returns a strictly positive, sorted, deduplicated array.
    """
    if time_unit_days <= 0:
        raise ValueError("time_unit_days must be positive.")
    if max_grid_points < 2:
        raise ValueError("max_grid_points must be at least 2.")
    quantile_values = np.asarray(quantiles, dtype=float)
    if (
        quantile_values.size == 0
        or not np.isfinite(quantile_values).all()
        or (quantile_values < 0).any()
        or (quantile_values > 1).any()
    ):
        raise ValueError("quantiles must contain finite values between 0 and 1.")
    duration = pd.to_numeric(train_val_df[duration_col], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(train_val_df[event_col], errors="coerce").fillna(0).astype(int).to_numpy()
    if admin_censor_days is not None:
        cap = float(admin_censor_days)
        if cap <= 0:
            raise ValueError("admin_censor_days must be positive when provided.")
        late = duration > cap
        event = np.where(late, 0, event)
        duration = np.where(late, cap, duration)
    valid = (event == 1) & np.isfinite(duration) & (duration > 0)
    event_times = duration[valid]
    if len(event_times) == 0:
        raise ValueError(
            f"No observed events for {event_col!r} on train_val; cannot derive horizon grid."
        )
    event_times_units = np.ceil(event_times / float(time_unit_days))
    bounds = np.quantile(event_times_units, [quantile_values.min(), quantile_values.max()])
    lower = max(1, int(np.ceil(bounds[0])))
    upper = max(1, int(np.floor(bounds[1])))
    cap_units: float | None = None
    if admin_censor_days is not None:
        # compute_brier / compute_ipcw_auc_t treat h >= cap as inestimable, so
        # the requested timeline has to stay strictly inside the cap or those
        # horizons would only ever dilute the coverage denominator.
        cap_units = float(np.ceil(cap / float(time_unit_days)))
        upper = min(upper, int(cap_units) - 1)
        if upper < 1:
            raise ValueError(
                "admin_censor_days is smaller than one time unit; no horizon fits "
                "strictly inside the administrative censoring window."
            )
        lower = min(lower, upper)
    if upper < lower:
        midpoint = max(1, int(np.rint(np.mean(bounds))))
        lower = upper = midpoint
    if upper == lower:
        # Quantiles can coincide when late-landmark events are sparse or tied.
        # If the training block nevertheless contains distinct event weeks,
        # expand to the two observed weeks nearest the quantile center. This
        # preserves a train-only, data-supported interval without reaching for
        # held-out outcomes merely to manufacture a multi-time mean.
        unique_event_units = np.unique(event_times_units[event_times_units > 0])
        if len(unique_event_units) >= 2:
            center = float(np.mean(bounds))
            nearest = unique_event_units[
                np.argsort(np.abs(unique_event_units - center), kind="stable")[:2]
            ]
            lower = int(np.min(nearest))
            upper = int(np.max(nearest))

    available = np.arange(lower, upper + 1, dtype=float)
    if len(available) <= max_grid_points:
        horizons = available
    else:
        positions = np.linspace(0, len(available) - 1, num=max_grid_points)
        horizons = available[np.unique(np.rint(positions).astype(int))]
    horizons = horizons[horizons > 0]
    if cap_units is not None:
        # The tie-expansion branch above can reach back to observed event weeks,
        # so re-apply the cap here rather than trusting `upper` alone.
        horizons = horizons[horizons < cap_units]
    # `lower` is clamped to >= 1 and below the cap above, so the grid is
    # always non-empty here.
    return np.unique(horizons)


def horizon_grid_frame(
    horizons: np.ndarray,
    *,
    quantiles: tuple[float, ...] = DEFAULT_AUC_QUANTILES,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
    endpoint: str | None = None,
) -> pd.DataFrame:
    """Tidy the horizon grid for persistence (cox_agg_horizon_grid.csv)."""
    rows = []
    n = max(len(horizons), 1)
    # Quantiles are not 1:1 with horizons after de-dup; emit just horizons here.
    for h in horizons:
        rows.append(
            {
                "endpoint": endpoint,
                "horizon_time_unit": float(h),
                "horizon_days": float(h) * float(time_unit_days),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["quantiles"] = list(quantiles)
    df.attrs["n_horizons"] = n
    return df


# ---------------------------------------------------------------------------
# Breslow baseline survival for linear-predictor models (Cox / XGBoost-Cox).
# ---------------------------------------------------------------------------

def breslow_survival_at_horizons(
    *,
    train_event: np.ndarray,
    train_duration: np.ndarray,
    train_lp: np.ndarray,
    eval_lp: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """S(t|x) = exp(-H_0(t) * exp(lp)) via sksurv's BreslowEstimator.

    Train arrays define the censoring/event distribution and the baseline
    cumulative hazard; eval_lp is the linear predictor for the rows whose
    survival should be evaluated. Returns shape (n_eval, len(horizons)).
    """
    require_sksurv()
    train_event = np.asarray(train_event).astype(bool)
    train_duration = np.asarray(train_duration, dtype=float)
    train_lp = np.asarray(train_lp, dtype=float).reshape(-1)
    eval_lp = np.asarray(eval_lp, dtype=float).reshape(-1)
    horizons = np.asarray(horizons, dtype=float).reshape(-1)

    estimator = BreslowEstimator()
    with suppress_sksurv_warnings():
        estimator.fit(train_lp, train_event, train_duration)
        surv_funcs = estimator.get_survival_function(eval_lp)
    out = np.full((len(eval_lp), len(horizons)), np.nan, dtype=float)
    # StepFunction raises outside its fitted domain. Preserve requested-column
    # alignment but evaluate only supported points; compute_brier records why
    # every remaining NaN horizon was excluded.
    train_min = float(np.nanmin(train_duration))
    train_max = float(np.nanmax(train_duration))
    supported = np.isfinite(horizons) & (horizons > train_min) & (horizons < train_max)
    if not supported.any():
        return out
    for i, sf in enumerate(surv_funcs):
        out[i, supported] = sf(horizons[supported])
    return out


# ---------------------------------------------------------------------------
# IPCW Brier (per-horizon + integrated).
# ---------------------------------------------------------------------------

def compute_brier(
    *,
    train_event: np.ndarray,
    train_duration: np.ndarray,
    eval_event: np.ndarray,
    eval_duration: np.ndarray,
    surv_at_horizons: np.ndarray,
    horizons: np.ndarray,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
    max_time_unit: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Per-horizon IPCW Brier + integrated Brier.

    All durations / events must already be in the same time units as
    `horizons` (use _duration_to_time_units upstream when needed). The
    `train_*` arrays parameterize the IPCW censoring distribution and must
    come from training data only — never from test.

    `surv_at_horizons[i, j]` = predicted S(t=horizons[j] | x_i) for eval
    sample i. Returns (per_horizon_df, integrated_brier_scalar).

    The optional administrative cap is applied identically to the reference
    and evaluation outcomes. Requested horizons are never discarded: points
    outside the cap/follow-up support remain in the returned table with a
    reason. Integrated Brier is reported only with at least two valid points
    and at least 50% requested-timeline coverage.
    """
    require_sksurv()

    horizons = np.asarray(horizons, dtype=float).reshape(-1)
    surv_at_horizons = np.asarray(surv_at_horizons, dtype=float)
    if surv_at_horizons.ndim != 2:
        raise ValueError("surv_at_horizons must be a 2D array.")
    if surv_at_horizons.shape[1] != len(horizons):
        raise ValueError(
            "surv_at_horizons columns must align one-to-one with horizons "
            f"({surv_at_horizons.shape[1]} != {len(horizons)})."
        )
    positive = np.isfinite(horizons) & (horizons > 0)
    horizons = horizons[positive]
    surv_at_horizons = surv_at_horizons[:, positive]
    # sksurv's brier_score applies np.unique() to the requested times but returns
    # scores in that sorted order, so an unsorted or duplicated grid would pair
    # each score with the wrong survival column. Sort and dedupe up front, moving
    # the survival columns with their horizons, so the two can never diverge.
    horizons, first_idx = np.unique(horizons, return_index=True)
    surv_at_horizons = surv_at_horizons[:, first_idx]

    train_event_arr = np.asarray(train_event).astype(bool)
    train_duration_arr = np.asarray(train_duration, dtype=float).reshape(-1)
    eval_event_arr = np.asarray(eval_event).astype(bool)
    eval_duration_arr = np.asarray(eval_duration, dtype=float).reshape(-1)
    train_event_arr = train_event_arr.reshape(-1)
    eval_event_arr = eval_event_arr.reshape(-1)
    if len(train_event_arr) != len(train_duration_arr):
        raise ValueError("train_event and train_duration must have equal length.")
    if len(eval_event_arr) != len(eval_duration_arr):
        raise ValueError("eval_event and eval_duration must have equal length.")
    if surv_at_horizons.shape[0] != len(eval_duration_arr):
        raise ValueError(
            "surv_at_horizons rows must align one-to-one with evaluation outcomes "
            f"({surv_at_horizons.shape[0]} != {len(eval_duration_arr)})."
        )

    empty_cols = [
        "horizon_time_unit",
        "horizon_days",
        "brier",
        "n_eval",
        "n_eval_events",
        "reference_min_time_unit",
        "reference_max_time_unit",
        "evaluation_min_time_unit",
        "evaluation_max_time_unit",
        "eligible_by_support",
        "admin_censor_time_unit",
        "note",
    ]
    if (
        len(horizons) == 0
        or len(train_duration_arr) == 0
        or len(eval_duration_arr) == 0
        or surv_at_horizons.size == 0
    ):
        return pd.DataFrame(columns=empty_cols), float("nan")

    if max_time_unit is not None:
        cap = float(max_time_unit)
        if cap <= 0:
            raise ValueError("max_time_unit must be positive when provided.")
        train_late = train_duration_arr > cap
        eval_late = eval_duration_arr > cap
        train_event_arr = train_event_arr & ~train_late
        eval_event_arr = eval_event_arr & ~eval_late
        train_duration_arr = np.minimum(train_duration_arr, cap)
        eval_duration_arr = np.minimum(eval_duration_arr, cap)

    train_valid = np.isfinite(train_duration_arr) & (train_duration_arr > 0)
    eval_valid = np.isfinite(eval_duration_arr) & (eval_duration_arr > 0)
    if not train_valid.any() or not eval_valid.any():
        rows = []
        reason = "no_valid_reference_rows" if not train_valid.any() else "no_valid_evaluation_rows"
        for h in horizons:
            rows.append({
                "horizon_time_unit": float(h),
                "horizon_days": float(h) * float(time_unit_days),
                "brier": np.nan,
                "n_eval": int(eval_valid.sum()),
                "n_eval_events": int(eval_event_arr[eval_valid].sum()),
                "reference_min_time_unit": np.nan,
                "reference_max_time_unit": np.nan,
                "evaluation_min_time_unit": np.nan,
                "evaluation_max_time_unit": np.nan,
                "eligible_by_support": False,
                "admin_censor_time_unit": max_time_unit,
                "note": reason,
            })
        return pd.DataFrame(rows, columns=empty_cols), float("nan")

    train_event_arr = train_event_arr[train_valid]
    train_duration_arr = train_duration_arr[train_valid]
    eval_event_arr = eval_event_arr[eval_valid]
    eval_duration_arr = eval_duration_arr[eval_valid]
    surv_at_horizons = surv_at_horizons[eval_valid, :]

    train_min = float(train_duration_arr.min())
    train_max = float(train_duration_arr.max())
    eval_min = float(eval_duration_arr.min())
    eval_max = float(eval_duration_arr.max())

    ineligible_reasons: list[str] = []
    for h in horizons:
        reasons = []
        if max_time_unit is not None and h >= float(max_time_unit):
            reasons.append("at_or_beyond_admin_censor")
        if not (train_min < h < train_max):
            reasons.append("out_of_train_range")
        if h < eval_min or h >= eval_max:
            reasons.append("outside_evaluation_followup")
        ineligible_reasons.append(";".join(reasons))
    in_range = np.array([not reason for reason in ineligible_reasons], dtype=bool)

    train_surv = _make_survival_array(train_event_arr, train_duration_arr)
    eval_surv = _make_survival_array(eval_event_arr, eval_duration_arr)

    rows_by_horizon: dict[float, dict] = {}
    for h, reason in zip(horizons, ineligible_reasons):
        rows_by_horizon[float(h)] = {
            "horizon_time_unit": float(h),
            "horizon_days": float(h) * float(time_unit_days),
            "brier": np.nan,
            "n_eval": int(len(eval_event_arr)),
            "n_eval_events": int(eval_event_arr.sum()),
            "reference_min_time_unit": train_min,
            "reference_max_time_unit": train_max,
            "evaluation_min_time_unit": eval_min,
            "evaluation_max_time_unit": eval_max,
            "eligible_by_support": not bool(reason),
            "admin_censor_time_unit": max_time_unit,
            "note": str(reason),
        }

    integrated = float("nan")
    if in_range.any():
        h_in = horizons[in_range]
        # surv_at_horizons columns align with the input `horizons`; subset to in-range.
        col_idx = np.where(in_range)[0]
        surv_in = surv_at_horizons[:, col_idx]
        try:
            with suppress_sksurv_warnings():
                times, brier_vals = brier_score(train_surv, eval_surv, surv_in, h_in)
        except ValueError as exc:
            for h in h_in:
                rows_by_horizon[float(h)]["note"] = f"brier_failed: {exc}"
        else:
            for t, b in zip(times, brier_vals):
                rows_by_horizon[float(t)]["brier"] = float(b)

            n_requested = len(horizons)
            valid_count = int(np.isfinite(brier_vals).sum())
            coverage = valid_count / n_requested if n_requested else 0.0
            adequate = (
                valid_count >= MIN_IPCW_VALID_HORIZONS
                and coverage >= MIN_IPCW_TIMELINE_COVERAGE
            )
            if adequate:
                # `horizons` is sorted and unique, so `h_in` / `surv_in` are too and
                # `brier_vals` (sorted by sksurv) lines up with them positionally.
                valid_brier_mask = np.isfinite(brier_vals)
                h_integrate = h_in[valid_brier_mask]
                surv_integrate = surv_in[:, valid_brier_mask]
                try:
                    with suppress_sksurv_warnings():
                        integrated = float(
                            integrated_brier_score(
                                train_surv, eval_surv, surv_integrate, h_integrate
                            )
                        )
                except ValueError as exc:
                    integrated = float("nan")
                    for h in h_integrate:
                        current = rows_by_horizon[float(h)]["note"]
                        rows_by_horizon[float(h)]["note"] = (
                            f"{current};integrated_brier_failed: {exc}"
                            if current else f"integrated_brier_failed: {exc}"
                        )
            else:
                for h in h_in[np.isfinite(brier_vals)]:
                    rows_by_horizon[float(h)]["note"] = "insufficient_timeline_coverage_for_ibs"

    df = (
        pd.DataFrame(list(rows_by_horizon.values()), columns=empty_cols)
        .sort_values("horizon_time_unit")
        .reset_index(drop=True)
    )
    return df, integrated


def summarize_brier_timeline(brier_df: pd.DataFrame) -> dict[str, float | int | str]:
    """Return requested/eligible/valid Brier timeline diagnostics for metrics."""
    if brier_df.empty:
        return {
            "n_requested_brier_horizons": 0,
            "n_support_eligible_brier_horizons": 0,
            "n_valid_brier_horizons": 0,
            "brier_timeline_coverage": 0.0,
            "brier_valid_min_time_unit": np.nan,
            "brier_valid_max_time_unit": np.nan,
            "brier_timeline_status": "no_evaluable_timeline",
        }
    requested = int(brier_df["horizon_time_unit"].nunique())
    eligible = brier_df["eligible_by_support"].fillna(False).to_numpy(dtype=bool)
    support_eligible = int(brier_df.loc[eligible, "horizon_time_unit"].nunique())
    valid = brier_df.loc[np.isfinite(brier_df["brier"]), "horizon_time_unit"].drop_duplicates()
    n_valid = int(len(valid))
    coverage = n_valid / requested if requested else 0.0
    if n_valid < MIN_IPCW_VALID_HORIZONS:
        status = "insufficient_valid_horizons"
    elif coverage < MIN_IPCW_TIMELINE_COVERAGE:
        status = "insufficient_timeline_coverage"
    else:
        status = "ok"
    return {
        "n_requested_brier_horizons": requested,
        "n_support_eligible_brier_horizons": support_eligible,
        "n_valid_brier_horizons": n_valid,
        "brier_timeline_coverage": float(coverage),
        "brier_valid_min_time_unit": float(valid.min()) if n_valid else np.nan,
        "brier_valid_max_time_unit": float(valid.max()) if n_valid else np.nan,
        "brier_timeline_status": status,
    }


# ---------------------------------------------------------------------------
# Stratified CV iteration.
# ---------------------------------------------------------------------------

def iter_stratified_folds(
    train_val: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    event_col: str | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, str]]:
    """Yield (fold_idx, train_idx, val_idx, stratification_label).

    Indices are positional (compatible with .iloc). Stratification preference
    matches choose_stratification_labels (endpoint event_col first when supplied).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    labels, label_name = choose_stratification_labels(
        train_val,
        min_count=n_folds,
        event_col=event_col,
    )
    if labels is not None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_args = (np.arange(len(train_val)), labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_args = (np.arange(len(train_val)),)
    for fold, (tr, va) in enumerate(splitter.split(*split_args), start=1):
        yield fold, tr, va, label_name


# ---------------------------------------------------------------------------
# Leakage guards.
# ---------------------------------------------------------------------------

def _normalize_mrn(m) -> str:
    """Stringify an MRN for set comparison, collapsing int/float dtype splits.

    A merge that introduces a NaN elsewhere in an otherwise-integer MRN
    column upcasts the whole column to float64 in pandas, so the same
    patient can appear as ``12345`` on one side of a split and ``12345.0``
    on the other. Plain ``str(m)`` treats those as different keys, so a
    genuine overlap silently fails to compare equal and the guard passes
    when it should raise. Route anything that is an int-valued float through
    ``int()`` first so both forms normalize to the same string; fall back to
    ``str(m)`` for non-numeric or genuinely fractional IDs.
    """
    try:
        f = float(m)
    except (TypeError, ValueError):
        return str(m)
    if f.is_integer():
        return str(int(f))
    return str(m)


def assert_no_test_leakage(
    *,
    test_mrns: Iterable,
    train_mrns: Iterable,
    context: str = "",
) -> None:
    """Hard guard: test_mrns must be disjoint from train_mrns.

    Cheap to call at every pipeline stage. Raises RuntimeError on overlap so
    the run aborts loudly rather than silently leaking.
    """
    test_set = {_normalize_mrn(m) for m in test_mrns}
    train_set = {_normalize_mrn(m) for m in train_mrns}
    overlap = test_set & train_set
    if overlap:
        ctx = f" ({context})" if context else ""
        sample = sorted(overlap)[:5]
        raise RuntimeError(
            f"Test/train MRN leakage detected{ctx}: "
            f"{len(overlap)} overlapping MRNs (e.g. {sample})."
        )


def assert_disjoint_folds(
    *,
    fold_train_mrns: Iterable,
    fold_val_mrns: Iterable,
    fold: int,
) -> None:
    """Hard guard for CV folds: train and val MRNs within a fold must be disjoint."""
    train_set = {_normalize_mrn(m) for m in fold_train_mrns}
    val_set = {_normalize_mrn(m) for m in fold_val_mrns}
    overlap = train_set & val_set
    if overlap:
        sample = sorted(overlap)[:5]
        raise RuntimeError(
            f"Fold {fold}: train/val MRN overlap of {len(overlap)} MRNs (e.g. {sample})."
        )
