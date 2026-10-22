"""Horizon labels and training-only inverse-probability-of-censoring weights.

The reverse Kaplan–Meier tie convention follows scikit-survival: observed
events leave the risk set before censorings at the same time. We evaluate the
right-continuous G(t) for both event weights and horizon weights, as in its
time-dependent Brier score implementation. No extrapolation or weight clipping
is performed. This assumes independent censoring; it is not a competing-risk
estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class UnsupportedHorizonError(ValueError):
    """The training censoring distribution cannot support this horizon."""


def validate_outcomes(duration: Any, event: Any) -> tuple[np.ndarray, np.ndarray]:
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event)
    if duration.ndim != 1 or event.ndim != 1 or duration.shape != event.shape:
        raise ValueError("duration and event must be equally sized one-dimensional arrays")
    if not duration.size or not np.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("duration must be nonempty, finite, and nonnegative")
    if not np.isin(event, [0, 1, False, True]).all():
        raise ValueError("event must contain only binary 0/1 values")
    return duration, event.astype(bool)


@dataclass
class CensoringDistribution:
    times: np.ndarray
    survival: np.ndarray
    max_time: float
    n_samples: int

    @classmethod
    def fit(cls, duration: Any, event: Any) -> "CensoringDistribution":
        duration, event = validate_outcomes(duration, event)
        times, inverse, totals = np.unique(duration, return_inverse=True, return_counts=True)
        events = np.bincount(inverse, weights=event.astype(int), minlength=len(times))
        censorings = totals - events
        at_risk = len(duration) - np.r_[0, np.cumsum(totals[:-1])]
        denominator = at_risk - events  # observed events precede tied censorings
        decrement = np.divide(censorings, denominator, out=np.zeros(len(times)), where=denominator > 0)
        survival = np.cumprod(1.0 - decrement)
        return cls(times, survival, float(times[-1]), len(duration))

    def predict(self, time: Any) -> np.ndarray | float:
        time = np.asarray(time, dtype=float)
        if not np.isfinite(time).all() or (time < 0).any():
            raise ValueError("censoring prediction times must be finite and nonnegative")
        if (time > self.max_time).any():
            raise UnsupportedHorizonError("censoring distribution cannot be extrapolated beyond training follow-up")
        index = np.searchsorted(self.times, time, side="right") - 1
        result = np.where(index >= 0, self.survival[np.maximum(index, 0)], 1.0)
        return float(result) if result.ndim == 0 else result

    def to_dict(self) -> dict:
        return {"times": self.times.tolist(), "survival": self.survival.tolist(),
                "max_time": self.max_time, "n_samples": self.n_samples,
                "tie_convention": "events_before_censorings; right_continuous_G"}

    @classmethod
    def from_dict(cls, state: dict) -> "CensoringDistribution":
        times = np.asarray(state["times"], dtype=float)
        survival = np.asarray(state["survival"], dtype=float)
        if (times.ndim != 1 or not times.size or times.shape != survival.shape
                or not np.isfinite(times).all() or not np.isfinite(survival).all()
                or (times < 0).any() or (np.diff(times) <= 0).any()
                or ((survival < 0) | (survival > 1)).any() or (np.diff(survival) > 0).any()
                or float(state["max_time"]) != float(times[-1])
                or int(state["n_samples"]) < len(times)):
            raise ValueError("invalid serialized censoring distribution")
        return cls(times, survival, float(state["max_time"]), int(state["n_samples"]))


def horizon_targets(duration: Any, event: Any, horizon: float,
                    censoring: CensoringDistribution, min_g: float = 0.05
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return {-1 unknown, 0 control, 1 case}, unbalanced IPCW, and audit.

    An event exactly at the horizon is a case; censoring exactly at it is
    unknown. Controls require follow-up strictly beyond the horizon. Unknown
    rows keep weight zero, and remain in the denominator of the Brier score.
    ``min_g`` is a positivity gate, never a weight-truncation parameter.
    """
    duration, event = validate_outcomes(duration, event)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    if not 0 < min_g <= 1:
        raise ValueError("min_g must be in (0, 1]")
    if horizon >= censoring.max_time:
        raise UnsupportedHorizonError(
            f"horizon {horizon:g} must be strictly below maximum training follow-up {censoring.max_time:g}")
    g_horizon = float(censoring.predict(horizon))
    if g_horizon < min_g:
        raise UnsupportedHorizonError(
            f"G({horizon:g})={g_horizon:.6g} is below min_g={min_g:g}; weights are not clipped")
    cases = event & (duration <= horizon)
    controls = duration > horizon
    labels = np.full(len(duration), -1, dtype=int)
    labels[cases], labels[controls] = 1, 0
    weights = np.zeros(len(duration), dtype=float)
    event_g = np.asarray(censoring.predict(duration[cases]))
    if (event_g < min_g).any():
        raise UnsupportedHorizonError("event IPCW falls below the configured positivity support")
    weights[cases] = 1.0 / event_g
    weights[controls] = 1.0 / g_horizon
    known = labels >= 0
    known_weights = weights[known]
    total_weight = float(known_weights.sum())
    audit = {"status": "supported", "horizon": float(horizon), "n": len(duration),
             "n_cases": int(cases.sum()), "n_controls": int(controls.sum()),
             "n_unknown": int((~known).sum()), "g_horizon": g_horizon,
             "min_g": float(min_g), "max_training_followup": censoring.max_time,
             "weight_sum": total_weight,
             "max_weight": float(known_weights.max()) if known.any() else None,
             "effective_sample_size": (total_weight ** 2 / float(np.square(known_weights).sum()))
             if known.any() else 0.0,
             "weight_truncation": False}
    return labels, weights, audit
