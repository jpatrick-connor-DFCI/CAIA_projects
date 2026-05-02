"""
Shared survival-analysis utilities used by cox_aggregated, landmark_xgboost,
and dynamic_deephit so the three pipelines agree on:

  * canonical lab selection (coverage-based)
  * AUC(t) / Brier horizon grid (fixed from train_val event times)
  * IPCW Brier (per-horizon + integrated) via sksurv
  * Breslow baseline survival for linear-predictor models (Cox / XGBoost-Cox)
  * 5-fold stratified CV iteration (combined PLATINUM+DEATH label)
  * train/test leakage guards

This module is standalone: it must not import from cox_aggregated.py so that
cox_aggregated → helper is a one-way dependency.
"""

from __future__ import annotations

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
DEFAULT_MIN_DISTINCT_LAB_VALUES: int = 2


def require_sksurv() -> None:
    if brier_score is None:
        raise ModuleNotFoundError(
            "sksurv is required for helper.py utilities."
        ) from SKSURV_IMPORT_ERROR


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


def choose_stratification_labels(
    df: pd.DataFrame, *, min_count: int
) -> tuple[np.ndarray | None, str]:
    """Pick the strictest stratification label whose smallest cell >= min_count.

    Preference order: combined (4 cells) -> either -> platinum -> death ->
    unstratified. Combined gives equal axis weight to PLATINUM and DEATH.
    """
    candidates: list[tuple[str, np.ndarray]] = []
    if {"PLATINUM", "DEATH"}.issubset(df.columns):
        candidates.append(("combined", combined_event_label(df)))
    if "EITHER" in df.columns:
        candidates.append(("either", pd.to_numeric(df["EITHER"], errors="coerce").fillna(0).astype(int).to_numpy()))
    if "PLATINUM" in df.columns:
        candidates.append(("platinum", pd.to_numeric(df["PLATINUM"], errors="coerce").fillna(0).astype(int).to_numpy()))
    if "DEATH" in df.columns:
        candidates.append(("death", pd.to_numeric(df["DEATH"], errors="coerce").fillna(0).astype(int).to_numpy()))
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
    required = {"DFCI_MRN", "LAB_NAME", "LAB_VALUE"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"select_canonical_labs missing columns: {sorted(missing)}")

    mrn_set = set(mrns)
    if not mrn_set:
        raise ValueError("select_canonical_labs received an empty MRN set.")

    sub = long_df.loc[long_df["DFCI_MRN"].isin(mrn_set)]
    if sub.empty:
        raise ValueError("select_canonical_labs: no observations for the given MRNs.")

    n = len(mrn_set)
    coverage = sub.groupby("LAB_NAME")["DFCI_MRN"].nunique() / float(n)
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

def compute_horizon_grid(
    train_val_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    quantiles: tuple[float, ...] = DEFAULT_AUC_QUANTILES,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
) -> np.ndarray:
    """AUC(t) / Brier horizon grid in time-units, derived from train_val ONLY.

    Quantiles are taken over observed event durations (event==1) on the
    train_val block. The grid is fixed once and reused across all CV folds and
    the held-out test evaluation, eliminating any test-driven horizon choice.
    Returns a strictly-positive, sorted, deduplicated array of horizons.
    """
    duration = pd.to_numeric(train_val_df[duration_col], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(train_val_df[event_col], errors="coerce").fillna(0).astype(int).to_numpy()
    valid = (event == 1) & np.isfinite(duration) & (duration > 0)
    event_times = duration[valid]
    if len(event_times) == 0:
        raise ValueError(
            f"No observed events for {event_col!r} on train_val; cannot derive horizon grid."
        )
    event_times_units = np.ceil(event_times / float(time_unit_days))
    raw = np.asarray(
        [int(v) for v in np.quantile(event_times_units, list(quantiles))],
        dtype=float,
    )
    horizons = np.unique(raw)
    horizons = horizons[horizons > 0]
    if len(horizons) == 0:
        raise ValueError(
            f"All horizon quantiles collapsed to non-positive values for {event_col!r}."
        )
    return horizons


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
    estimator.fit(train_lp, train_event, train_duration)
    surv_funcs = estimator.get_survival_function(eval_lp)
    out = np.empty((len(eval_lp), len(horizons)), dtype=float)
    for i, sf in enumerate(surv_funcs):
        out[i, :] = sf(horizons)
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
) -> tuple[pd.DataFrame, float]:
    """Per-horizon IPCW Brier + integrated Brier.

    All durations / events must already be in the same time units as
    `horizons` (use _duration_to_time_units upstream when needed). The
    `train_*` arrays parameterize the IPCW censoring distribution and must
    come from training data only — never from test.

    `surv_at_horizons[i, j]` = predicted S(t=horizons[j] | x_i) for eval
    sample i. Returns (per_horizon_df, integrated_brier_scalar).

    sksurv requires horizons strictly inside (min(train_time), max(train_time)).
    Out-of-range horizons are dropped from both per-horizon and integrated
    outputs and recorded with note='out_of_train_range'.
    """
    require_sksurv()

    horizons = np.asarray(horizons, dtype=float).reshape(-1)
    horizons = horizons[horizons > 0]
    surv_at_horizons = np.asarray(surv_at_horizons, dtype=float)

    train_event_arr = np.asarray(train_event).astype(bool)
    train_duration_arr = np.asarray(train_duration, dtype=float)
    eval_event_arr = np.asarray(eval_event).astype(bool)
    eval_duration_arr = np.asarray(eval_duration, dtype=float)

    empty_cols = [
        "horizon_time_unit",
        "horizon_days",
        "brier",
        "n_eval",
        "n_eval_events",
        "note",
    ]
    if (
        len(horizons) == 0
        or len(train_duration_arr) == 0
        or len(eval_duration_arr) == 0
        or surv_at_horizons.size == 0
    ):
        return pd.DataFrame(columns=empty_cols), float("nan")

    train_min = float(train_duration_arr.min())
    train_max = float(train_duration_arr.max())
    in_range = (horizons > train_min) & (horizons < train_max)

    train_surv = _make_survival_array(train_event_arr, train_duration_arr)
    eval_surv = _make_survival_array(eval_event_arr, eval_duration_arr)

    rows = []
    integrated = float("nan")
    if in_range.any():
        h_in = horizons[in_range]
        # surv_at_horizons columns align with the input `horizons`; subset to in-range.
        col_idx = np.where(in_range)[0]
        surv_in = surv_at_horizons[:, col_idx]
        try:
            times, brier_vals = brier_score(train_surv, eval_surv, surv_in, h_in)
        except ValueError as exc:
            for h in h_in:
                rows.append(
                    {
                        "horizon_time_unit": float(h),
                        "horizon_days": float(h) * float(time_unit_days),
                        "brier": float("nan"),
                        "n_eval": int(len(eval_event_arr)),
                        "n_eval_events": int(eval_event_arr.sum()),
                        "note": f"brier_failed: {exc}",
                    }
                )
        else:
            for t, b in zip(times, brier_vals):
                rows.append(
                    {
                        "horizon_time_unit": float(t),
                        "horizon_days": float(t) * float(time_unit_days),
                        "brier": float(b),
                        "n_eval": int(len(eval_event_arr)),
                        "n_eval_events": int(eval_event_arr.sum()),
                        "note": "",
                    }
                )
            if len(times) >= 2:
                try:
                    integrated = float(
                        integrated_brier_score(train_surv, eval_surv, surv_in, h_in)
                    )
                except ValueError:
                    integrated = float("nan")

    for h in horizons[~in_range]:
        rows.append(
            {
                "horizon_time_unit": float(h),
                "horizon_days": float(h) * float(time_unit_days),
                "brier": float("nan"),
                "n_eval": int(len(eval_event_arr)),
                "n_eval_events": int(eval_event_arr.sum()),
                "note": "out_of_train_range",
            }
        )

    df = pd.DataFrame(rows, columns=empty_cols).sort_values("horizon_time_unit").reset_index(drop=True)
    return df, integrated


# ---------------------------------------------------------------------------
# Stratified CV iteration.
# ---------------------------------------------------------------------------

def iter_stratified_folds(
    train_val: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, str]]:
    """Yield (fold_idx, train_idx, val_idx, stratification_label).

    Indices are positional (compatible with .iloc). Stratification preference
    matches choose_stratification_labels (combined PLATINUM+DEATH first).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    labels, label_name = choose_stratification_labels(train_val, min_count=n_folds)
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
    test_set = {str(m) for m in test_mrns}
    train_set = {str(m) for m in train_mrns}
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
    train_set = {str(m) for m in fold_train_mrns}
    val_set = {str(m) for m in fold_val_mrns}
    overlap = train_set & val_set
    if overlap:
        sample = sorted(overlap)[:5]
        raise RuntimeError(
            f"Fold {fold}: train/val MRN overlap of {len(overlap)} MRNs (e.g. {sample})."
        )
