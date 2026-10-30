"""Paired patient bootstrap of held-out survival metrics.

Scores saved test predictions with the same functions the engines use, so a
draw can differ from the stored metric only through the resampled patients.
The fitted model and the train-side IPCW reference stay fixed: the intervals
describe test-sampling uncertainty, not refitting or tuning variability.

Every fit in one call is scored on the same resampled patients, which is what
makes the difference between two fits a paired comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from survival_common.cox_engine import compute_ipcw_auc_t
from survival_common.helper import compute_brier, survival_column

METRICS = ("test_mean_auc_t", "test_c_index", "test_integrated_brier")
DURATION_COL = "duration_days"
EVENT_COL = "event"
MIN_VALID_DRAWS = 10


def load_test_predictions(path: str | Path) -> pd.DataFrame:
    """One fit's test rows, with the XGBoost column names normalized to Cox's."""
    frame = pd.read_csv(path).rename(columns={"duration": DURATION_COL})
    return frame.loc[frame["dataset"].astype(str) == "test"].reset_index(drop=True)


def survival_matrix(predictions: pd.DataFrame, horizons: np.ndarray) -> np.ndarray | None:
    """S(t|x) aligned to `horizons`, or None for a fit without saved survival."""
    columns = [survival_column(horizon) for horizon in horizons]
    if not set(columns) <= set(predictions.columns):
        return None
    return predictions[columns].to_numpy(dtype=float)


def score_heldout(
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    surv_at_horizons: np.ndarray | None,
    *,
    reference: pd.DataFrame,
    horizons: np.ndarray,
    time_unit_days: int,
    max_time_unit: int | None,
) -> dict[str, float]:
    """The engines' test metrics for one evaluation set.

    `reference` holds the training durations (days) and events behind the IPCW
    weights. Integrated Brier is NaN when `surv_at_horizons` is None.
    """
    c_index = np.nan
    valid = np.isfinite(duration) & (duration > 0) & np.isfinite(risk)
    if event[valid].sum():
        # A small resample can have events but no admissible pairs.
        try:
            c_index = float(concordance_index(duration[valid], -risk[valid], event[valid]))
        except ZeroDivisionError:
            pass

    mean_auc, _ = compute_ipcw_auc_t(
        pd.DataFrame({DURATION_COL: duration, EVENT_COL: event}),
        risk,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        reference_df=reference,
        time_unit_days=time_unit_days,
        max_time_unit=max_time_unit,
        fixed_horizons=horizons,
    )

    integrated_brier = np.nan
    if surv_at_horizons is not None:
        _, integrated_brier = compute_brier(
            train_event=reference[EVENT_COL].to_numpy(dtype=int),
            train_duration=np.ceil(reference[DURATION_COL].to_numpy(dtype=float) / float(time_unit_days)),
            eval_event=event,
            eval_duration=np.ceil(duration / float(time_unit_days)),
            surv_at_horizons=surv_at_horizons,
            horizons=horizons,
            time_unit_days=time_unit_days,
            max_time_unit=max_time_unit,
        )
    return {
        "test_mean_auc_t": float(mean_auc),
        "test_c_index": c_index,
        "test_integrated_brier": float(integrated_brier),
    }


def _align_fits(fits: Mapping[str, pd.DataFrame], id_col: str) -> dict[str, pd.DataFrame]:
    """Sort every fit by patient and require identical patients and outcomes."""
    aligned = {
        label: frame.sort_values(id_col).reset_index(drop=True) for label, frame in fits.items()
    }
    first_label, first = next(iter(aligned.items()))
    if first[id_col].duplicated().any():
        raise ValueError(f"Duplicate test patients in fit {first_label!r}.")
    outcome_cols = [id_col, DURATION_COL, EVENT_COL]
    for label, frame in aligned.items():
        if not frame[outcome_cols].equals(first[outcome_cols]):
            raise ValueError(
                f"Fits {first_label!r} and {label!r} differ in test patients or outcomes; "
                "paired draws would not be comparable."
            )
    return aligned


def bootstrap_heldout(
    fits: Mapping[str, pd.DataFrame],
    *,
    reference: pd.DataFrame,
    horizons: np.ndarray,
    time_unit_days: int,
    max_time_unit: int | None,
    n_boot: int,
    seed: int,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each fit on the observed patients and on shared bootstrap draws.

    `fits` maps a label to its `load_test_predictions` frame. Returns
    (observed, draws): one row per fit, and one row per (fit, draw). Patients
    are sorted by id first, so a given seed draws the same patients whichever
    order the fits arrive in. The horizon grid is fixed; the support and
    coverage gates are reapplied in each draw, and a draw that fails them is
    NaN rather than dropped, which keeps draws aligned across fits.
    """
    horizons = np.asarray(horizons, dtype=float).reshape(-1)
    aligned = _align_fits(fits, id_col)
    first = next(iter(aligned.values()))
    duration = first[DURATION_COL].to_numpy(dtype=float)
    event = first[EVENT_COL].to_numpy(dtype=int)
    n_patients = len(first)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, n_patients, size=(n_boot, n_patients))

    score_kwargs = dict(
        reference=reference, horizons=horizons,
        time_unit_days=time_unit_days, max_time_unit=max_time_unit,
    )
    observed_rows, draw_rows = [], []
    for label, frame in aligned.items():
        risk = frame["risk_score"].to_numpy(dtype=float)
        surv = survival_matrix(frame, horizons)
        observed_rows.append({"fit": label, **score_heldout(duration, event, risk, surv, **score_kwargs)})
        for draw, idx in enumerate(draw_indices):
            draw_surv = None if surv is None else surv[idx]
            scores = score_heldout(duration[idx], event[idx], risk[idx], draw_surv, **score_kwargs)
            draw_rows.append({"fit": label, "draw": draw, **scores})
    return pd.DataFrame(observed_rows), pd.DataFrame(draw_rows)


def percentile_ci(values: Sequence[float] | np.ndarray | pd.Series) -> tuple[float, float, int]:
    """95% percentile interval over finite draws, plus how many were finite."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_VALID_DRAWS:
        return float("nan"), float("nan"), len(values)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), len(values)


def assert_reproduces_stored(
    observed: Mapping[str, float],
    stored: Mapping[str, float],
    *,
    context: str,
    tolerance: float = 1e-8,
) -> None:
    """Refuse intervals around anything other than the metrics the fit reported.

    Integrated Brier is skipped when the fit saved no survival columns. A
    mismatch means the risk file, the reference or the grid is not what the fit
    used, for example stale outputs after the prediction inputs were rebuilt.
    """
    for metric in METRICS:
        recomputed, expected = float(observed[metric]), float(stored[metric])
        if metric == "test_integrated_brier" and np.isnan(recomputed):
            continue
        if np.isnan(recomputed) and np.isnan(expected):
            continue
        if not abs(recomputed - expected) <= tolerance:
            raise ValueError(
                f"{context}: recomputed {metric}={recomputed:.10g} but the fit stored "
                f"{expected:.10g}. Inspect stale outputs before trusting the bootstrap."
            )
