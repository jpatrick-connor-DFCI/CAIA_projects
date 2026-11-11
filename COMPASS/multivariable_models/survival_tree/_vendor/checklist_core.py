"""Framework-free port of PROFILE's predictive-checklist method.

Source: ``PROFILE-testing/COMPASS/multivariable_models/predictive_checklists``
(written ``PC/`` below), a weighted re-implementation of Zhang et al. 2021,
*Learning Optimal Predictive Checklists*.  Every block names its source module.
Nothing here imports NVFlare, so the parity tests can import it directly.

What differs from PC/, and why:

* ``fit_checklist`` returns the raw stage-1 solver bound and the weight total,
  and takes optional restrictions (fixed item count, threshold cap).  The
  federation divides one pooled loss across sites, so each site needs its bound
  on the *global* weight scale.  With no restriction it is PC/ line for line.
* ``loss_bound`` is a stage-1-only routine for the secondary certification.
* ``FeatureBinarizer`` is built from a spec the server assembled from pooled
  statistics (``from_spec``) rather than fitted on rows: thresholds and medians
  come from exact pooled order statistics, not any one site's data.
* ``CensoringDistribution.from_counts`` builds the reverse-KM curve from pooled
  ``(duration, event) -> n`` counts, which determine it exactly.
* Metrics and the calibrator work on weighted cells ``(score, label, day) -> n``
  so the server never needs patient rows.  Expanding the cells reproduces the
  patient-level arrays, so the point estimates equal ``evaluate_horizon``.
* Everything runs in PC/'s ``adjust_only_mode = "drop"``: lab counts and
  ``is missing`` items are never candidates.  ``AdjusterDesign`` is PC/'s, but it
  is fitted per site, and ``fit_site_adjusters`` solves that site's calibration
  coefficients for a server-chosen intercept and slope, so they stay at the site.

Only the HiGHS backend is ported; CBC/CPLEX are not dependencies of this app.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

# --------------------------------------------------------------------------- #
# Feature schema                                          (PC/features.py:18-48)
# --------------------------------------------------------------------------- #

SUMMARY_STATS = frozenset({"mean", "min", "max", "last", "delta"})
# Static adjustment covariates carried by genomic inputs (PC/features.py).
STATIC_ADJUSTER_PREFIXES = ("PANEL_VERSION_",)
STATIC_ADJUSTER_COLUMNS = ("SOMATIC_N_GENES_ALTERED",)
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# How a threshold item treats a missing value.  "median" is PROFILE's rule
# (impute the pooled median, then compare); "item_false" leaves the item
# unchecked, so a missing lab never earns a point.  Missingness reaches the
# model only through the adjust-only indicators.
MISSING_POLICIES = frozenset({"median", "item_false"})
_NON_LAB_SOURCES = frozenset({
    "DFCI_MRN", "MRN", "ID", "SPLIT", "GENDER", "FIRST_TREATMENT",
    "FIRST_RECORD_DATE", "DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE",
    "LAST_CONTACT_DATE", "PRIMARY_LABEL", "EITHER", "PANEL_VERSION",
})


def feature_group(feature: str, age_col: str) -> str | None:
    """The item group a column belongs to, or None when it is not a candidate.

    This is the allowlist: age, or ``<lab>__<stat>`` with a known summary stat.
    ``person_id`` and the outcome columns have no stat suffix, so they can never
    become features (see the person_id leak this guards against).
    """
    if feature == age_col:
        return feature
    source, separator, stat = feature.rpartition("__")
    if not separator or not source or stat not in SUMMARY_STATS:
        return None
    normalized = source.upper()
    if normalized in _NON_LAB_SOURCES or normalized.startswith(
        ("T_", "NEPC", "AVPC", "PLATINUM", "DEATH")
    ):
        return None
    return source


def _numeric(frame, feature: str):
    import pandas as pd

    if feature not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[feature], errors="coerce").astype(float).replace(
        [np.inf, -np.inf], np.nan
    )


class FeatureBinarizer:
    """Frozen binary candidates (PC/features.py:84-225, ``fit`` removed).

    State is ``PC.FeatureBinarizer.to_dict`` plus keys PROFILE ignores, so
    PROFILE's own ``from_dict`` loads what this writes.  ``missing_policy``
    (see ``MISSING_POLICIES``) is "item_false" for every binarizer this app
    builds; ``falsifying_medians`` makes a bundle score the same way under
    PROFILE's median-imputing ``transform``.
    """

    def __init__(self, min_patient_coverage=0.20, quantiles=DEFAULT_QUANTILES,
                 age_col="age_at_anchor", canonical_labs=None, id_col="person_id",
                 clinical_features=(), missing_policy="median"):
        if missing_policy not in MISSING_POLICIES:
            raise ValueError(f"Unknown missing_policy {missing_policy!r}.")
        self.missing_policy = str(missing_policy)
        self.min_patient_coverage = float(min_patient_coverage)
        self.quantiles = tuple(sorted(set(float(q) for q in quantiles)))
        self.age_col = str(age_col)
        self.id_col = str(id_col)
        self.clinical_features = tuple(map(str, clinical_features))
        self.canonical_labs = None if canonical_labs is None else sorted(set(map(str, canonical_labs)))
        self.feature_columns: list[str] = []
        self.items: list[dict[str, Any]] = []
        self.groups: list[str] = []
        self.medians: dict[str, float] = {}
        self.coverage: dict[str, float] = {}
        self._fitted = False

    def transform(self, frame) -> np.ndarray:
        if not self._fitted:
            raise ValueError("FeatureBinarizer must be fitted before transform.")
        values = {feature: _numeric(frame, feature) for feature in self.feature_columns}
        result = np.zeros((len(frame), len(self.items)), dtype=np.int8)
        for i, item in enumerate(self.items):
            raw = values[item["feature"]]
            if item["op"] == "is_missing":
                result[:, i] = raw.isna().to_numpy(dtype=np.int8)
            else:
                threshold = item["threshold"]
                if self.missing_policy == "item_false":
                    observed = raw.to_numpy(dtype=float)
                    with np.errstate(invalid="ignore"):
                        hit = observed <= threshold if item["op"] == "<=" else observed >= threshold
                    result[:, i] = hit & ~np.isnan(observed)
                else:
                    imputed = raw.fillna(self.medians[item["feature"]]).to_numpy()
                    result[:, i] = imputed <= threshold if item["op"] == "<=" else imputed >= threshold
        return result

    def to_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise ValueError("FeatureBinarizer must be fitted before serialization.")
        return {
            "schema_version": 1, "age_col": self.age_col, "id_col": self.id_col,
            "min_patient_coverage": self.min_patient_coverage, "quantiles": list(self.quantiles),
            "canonical_labs": self.canonical_labs, "clinical_features": list(self.clinical_features),
            "feature_columns": list(self.feature_columns),
            "medians": dict(self.medians), "coverage": dict(self.coverage),
            "items": [dict(item) for item in self.items],
            "exclude_adjusters": True, "missing_policy": self.missing_policy,
        }

    def falsifying_medians(self, item_ids: Sequence[int]) -> dict[str, float]:
        """Imputation values that make each listed item false when its feature
        is missing, so a median-imputing scorer (PROFILE's ``transform``)
        reproduces ``missing_policy="item_false"`` on those items.

        Needs at most one listed item per feature, which the one-item-per-group
        constraint guarantees for a fitted checklist.  Other features keep their
        pooled medians.
        """
        medians = dict(self.medians)
        seen: set[str] = set()
        for i in item_ids:
            item = self.items[i]
            feature, threshold = item["feature"], float(item["threshold"])
            if feature in seen:
                raise ValueError(f"Two selected items share feature {feature!r}.")
            seen.add(feature)
            step = max(1.0, abs(threshold))
            medians[feature] = threshold + step if item["op"] == "<=" else threshold - step
        return medians

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "FeatureBinarizer":
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported checklist feature schema_version.")
        result = cls(**{key: state[key] for key in (
            "age_col", "id_col", "min_patient_coverage", "quantiles", "canonical_labs"
        )}, clinical_features=state.get("clinical_features", ()),
            missing_policy=state.get("missing_policy", "median"))
        result.feature_columns = list(state["feature_columns"])
        result.medians = {key: float(value) for key, value in state["medians"].items()}
        result.coverage = {key: float(value) for key, value in state["coverage"].items()}
        result.items = [dict(item) for item in state["items"]]
        for feature in result.feature_columns:
            if feature_group(feature, result.age_col) is None or not np.isfinite(result.medians[feature]):
                raise ValueError("Invalid serialized checklist feature or imputation value.")
        for item in result.items:
            if item["feature"] not in result.feature_columns or item["group"] != feature_group(item["feature"], result.age_col):
                raise ValueError("Invalid serialized checklist item feature/group.")
            if item["op"] not in {"<=", ">=", "is_missing"}:
                raise ValueError("Invalid serialized checklist comparison.")
            if item["op"] == "is_missing":
                raise ValueError("Missingness items are adjust-only in this bundle.")
            if not np.isfinite(float(item["threshold"])):
                raise ValueError("Invalid serialized checklist threshold.")
        result.groups = [item["group"] for item in result.items]
        result._fitted = True
        return result

    def subset(self, item_ids: Sequence[int]) -> "FeatureBinarizer":
        """The binarizer restricted to ``item_ids`` (order kept); used for the
        age-only checklist, whose dictionary is the age items of the main one."""
        state = self.to_dict()
        items = [state["items"][i] for i in item_ids]
        used = {item["feature"] for item in items}
        state["items"] = items
        state["feature_columns"] = [f for f in state["feature_columns"] if f in used]
        state["medians"] = {f: v for f, v in state["medians"].items() if f in used}
        state["coverage"] = {f: v for f, v in state["coverage"].items() if f in used}
        return FeatureBinarizer.from_dict(state)


def binarizer_from_spec(spec: dict[str, Any], *, age_col: str, id_col: str,
                        min_patient_coverage: float, quantiles) -> FeatureBinarizer:
    """Assemble a binarizer from the server's pooled spec.

    ``spec`` holds ``feature_columns``, ``medians``, ``coverage`` and ``items``
    (each ``name/feature/op/threshold/group``), all derived from pooled counts.
    """
    state = {
        "schema_version": 1, "age_col": age_col, "id_col": id_col,
        "min_patient_coverage": float(min_patient_coverage), "quantiles": list(quantiles),
        "canonical_labs": None, "clinical_features": [],
        "feature_columns": list(spec["feature_columns"]),
        "medians": {k: float(v) for k, v in spec["medians"].items()},
        "coverage": {k: float(v) for k, v in spec["coverage"].items()},
        "items": [dict(item) for item in spec["items"]],
        "missing_policy": "item_false",
    }
    return FeatureBinarizer.from_dict(state)


class AdjusterDesign:
    """Standardized adjust-only covariates (PC/features.py ``AdjusterDesign``).

    Columns are log1p lab counts for each lab the binarizer kept, missingness
    indicators for its features (identical training patterns collapsed), and
    any static panel/burden covariates.  Each site fits this on its own
    ``train_val`` rows, so a zero row is that site's training mean.
    """

    def __init__(self) -> None:
        self.columns: list[dict[str, Any]] = []
        self.means: list[float] = []
        self.scales: list[float] = []

    def _raw(self, frame, column: dict[str, Any]) -> np.ndarray:
        values = _numeric(frame, column["source"])
        if column["kind"] == "log1p_count":
            return np.log1p(values.fillna(0.0).clip(lower=0.0)).to_numpy(dtype=float)
        if column["kind"] == "missing":
            return values.isna().to_numpy(dtype=float)
        return values.fillna(column["fill"]).to_numpy(dtype=float)

    def fit(self, frame, binarizer: FeatureBinarizer) -> "AdjusterDesign":
        candidates: list[dict[str, Any]] = []
        lab_groups = sorted({feature_group(f, binarizer.age_col)
                             for f in binarizer.feature_columns if f != binarizer.age_col})
        for group in lab_groups:
            source = f"{group}__n_observations"
            if source in frame:
                candidates.append({"name": source, "source": source, "kind": "log1p_count"})
        seen_patterns: set[bytes] = set()
        for feature in binarizer.feature_columns:
            pattern = _numeric(frame, feature).isna().to_numpy()
            if pattern.any() and not pattern.all() and pattern.tobytes() not in seen_patterns:
                seen_patterns.add(pattern.tobytes())
                candidates.append({"name": f"{feature}__missing", "source": feature, "kind": "missing"})
        static = [str(c) for c in frame.columns
                  if str(c).startswith(STATIC_ADJUSTER_PREFIXES) or str(c) in STATIC_ADJUSTER_COLUMNS]
        for source in sorted(static):
            observed = _numeric(frame, source).dropna()
            if not observed.empty:
                candidates.append({"name": source, "source": source, "kind": "static",
                                   "fill": float(observed.mean())})
        self.columns, self.means, self.scales = [], [], []
        for column in candidates:
            raw = self._raw(frame, column)
            scale = float(raw.std())
            if scale > 1e-12:
                self.columns.append(column)
                self.means.append(float(raw.mean()))
                self.scales.append(scale)
        return self

    @property
    def names(self) -> list[str]:
        return [column["name"] for column in self.columns]

    def transform(self, frame) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(frame), 0), dtype=float)
        raw = np.column_stack([self._raw(frame, column) for column in self.columns])
        return (raw - np.asarray(self.means)) / np.asarray(self.scales)


# --------------------------------------------------------------------------- #
# Censoring distribution and horizon labels                  (PC/survival.py)
# --------------------------------------------------------------------------- #

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
    def from_counts(cls, times: Any, totals: Any, events: Any) -> "CensoringDistribution":
        """The same curve as ``PC.CensoringDistribution.fit`` on the expanded rows.

        ``totals[i]`` rows end at ``times[i]`` and ``events[i]`` of them are
        events; observed events leave the risk set before tied censorings.
        """
        times = np.asarray(times, dtype=float)
        totals = np.asarray(totals, dtype=float)
        events = np.asarray(events, dtype=float)
        order = np.argsort(times, kind="stable")
        times, totals, events = times[order], totals[order], events[order]
        if not times.size or (np.diff(times) <= 0).any():
            raise ValueError("censoring counts need nonempty, strictly increasing times")
        censorings = totals - events
        at_risk = totals.sum() - np.r_[0, np.cumsum(totals[:-1])]
        denominator = at_risk - events
        decrement = np.divide(censorings, denominator, out=np.zeros(len(times)), where=denominator > 0)
        survival = np.cumprod(1.0 - decrement)
        return cls(times, survival, float(times[-1]), int(totals.sum()))

    @classmethod
    def fit(cls, duration: Any, event: Any) -> "CensoringDistribution":
        duration, event = validate_outcomes(duration, event)
        times, inverse, totals = np.unique(duration, return_inverse=True, return_counts=True)
        events = np.bincount(inverse, weights=event.astype(int), minlength=len(times))
        return cls.from_counts(times, totals, events)

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
    """{-1 unknown, 0 control, 1 case}, unbalanced IPCW weights and an audit.

    An event exactly at the horizon is a case; censoring exactly at it is
    unknown.  Controls need follow-up strictly beyond the horizon.  ``min_g`` is
    a positivity gate, never a weight truncation (PC/survival.py:83-127).
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


# --------------------------------------------------------------------------- #
# The checklist MIP                                          (PC/optimizer.py)
# --------------------------------------------------------------------------- #

@dataclass
class ChecklistModel:
    selected_indices: List[int]
    threshold: int
    n_features: int
    diagnostics: Dict[str, Any]

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Raw unit-weight item counts, not probabilities."""
        matrix = _binary_matrix(X, allow_empty=True)
        if matrix.shape[1] != self.n_features:
            raise ValueError("Prediction matrix has a different number of features")
        return matrix[:, self.selected_indices].sum(axis=1, dtype=np.int64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) >= self.threshold).astype(np.uint8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "weighted_m_of_n_checklist",
            "schema_version": 1,
            "selected_indices": [int(i) for i in self.selected_indices],
            "threshold": int(self.threshold),
            "n_features": int(self.n_features),
            "diagnostics": deepcopy(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ChecklistModel":
        if value.get("model_type") != "weighted_m_of_n_checklist" or value.get("schema_version") != 1:
            raise ValueError("Unsupported checklist artifact")
        raw = value["selected_indices"]
        selected = [int(i) for i in raw]
        threshold, n_features = int(value["threshold"]), int(value["n_features"])
        if (selected != raw or threshold != value["threshold"]
                or n_features != value["n_features"] or n_features < 1
                or not selected or len(set(selected)) != len(selected)
                or min(selected) < 0 or max(selected) >= n_features
                or not 1 <= threshold <= len(selected)):
            raise ValueError("Invalid checklist parameters")
        return cls(selected, threshold, n_features, deepcopy(value.get("diagnostics", {})))


@dataclass
class _Problem:
    matrix: Any
    lower: np.ndarray
    upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray

    def constrain(self, coefficients: np.ndarray, lower: float, upper: float) -> "_Problem":
        return _Problem(
            sparse.vstack((self.matrix, sparse.csr_matrix(coefficients[None, :])), format="csc"),
            np.append(self.lower, lower), np.append(self.upper, upper),
            self.variable_lower, self.variable_upper,
        )


@dataclass
class _Result:
    x: Optional[np.ndarray]
    status: str
    optimal: bool
    objective: Optional[float]
    bound: Optional[float]
    gap: Optional[float]
    message: str
    nodes: Optional[int] = None


def _finite(value: Any) -> Optional[float]:
    return float(value) if value is not None and np.isfinite(value) else None


def _binary_matrix(X: np.ndarray, allow_empty: bool = False) -> np.ndarray:
    matrix = np.asarray(X)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or (not allow_empty and not matrix.shape[0]):
        raise ValueError("X must be a nonempty two-dimensional binary matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("X must contain only binary 0/1 values")
    return matrix.astype(np.uint8, copy=False)


def _solve(problem: _Problem, c: np.ndarray, seconds: float) -> _Result:
    """HiGHS branch and cut (PC/optimizer.py:115-129)."""
    result = milp(
        c, integrality=np.ones(len(c), dtype=np.uint8),
        bounds=Bounds(problem.variable_lower, problem.variable_upper),
        constraints=LinearConstraint(problem.matrix, problem.lower, problem.upper),
        options={"time_limit": seconds, "mip_rel_gap": 0.0, "presolve": True},
    )
    return _Result(
        result.x, {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(result.status, "error"),
        result.status == 0, _finite(result.fun), _finite(getattr(result, "mip_dual_bound", None)),
        _finite(getattr(result, "mip_gap", None)), str(result.message),
        int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None,
    )


@dataclass
class _Built:
    problem: _Problem
    loss: np.ndarray
    size: np.ndarray
    threshold: np.ndarray
    X: np.ndarray
    y: np.ndarray
    normalized: np.ndarray
    grouped: dict
    n_features: int
    n_rows: int
    max_items: int
    total_weight: float
    n_samples: int
    n_positive: int


def _build(X, y, sample_weight, groups, max_items, compress,
           fixed_items: int | None = None, max_threshold: int | None = None) -> _Built:
    """The MIP of PC/optimizer.py:215-263, plus two optional restrictions.

    ``fixed_items`` forces ``sum(lambda) == fixed_items`` and ``max_threshold``
    caps M; the unrestricted problem is untouched.
    """
    matrix = _binary_matrix(X)
    target = np.asarray(y)
    n_samples, n_features = matrix.shape
    if target.shape != (n_samples,) or not np.all((target == 0) | (target == 1)):
        raise ValueError("y must be a one-dimensional binary vector aligned with X")
    weights = np.ones(n_samples, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must be finite, nonnegative, and aligned with X")
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("sample_weight must have a positive finite sum")
    if isinstance(max_items, bool) or int(max_items) != max_items or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    group_ids = list(range(n_features)) if groups is None else list(groups)
    if len(group_ids) != n_features:
        raise ValueError("groups must have one source group ID per feature")
    grouped: Dict[Hashable, List[int]] = {}
    for j, group in enumerate(group_ids):
        grouped.setdefault(group, []).append(j)
    max_items = min(int(max_items), len(grouped), n_features)
    keep = weights > 0
    training_X, training_y = matrix[keep], target[keep].astype(np.uint8)
    normalized = weights[keep] / total_weight
    if compress:
        unique, inverse = np.unique(np.column_stack((training_X, training_y)), axis=0, return_inverse=True)
        normalized = np.bincount(inverse.reshape(-1), weights=normalized, minlength=len(unique))
        training_X, training_y = unique[:, :-1], unique[:, -1]
    n_rows = len(training_y)
    # Variables: selected items lambda[0:p], mistake indicators z[0:n], M.
    n_vars = n_features + n_rows + 1
    sign = np.where(training_y == 1, -1.0, 1.0)
    event_constraints = sparse.hstack((
        sparse.csr_matrix(training_X).multiply(sign[:, None]),
        -max_items * sparse.eye(n_rows, format="csr"),
        sparse.csr_matrix((-sign)[:, None]),
    ), format="csc")
    extra_rows, extra_upper = [], []
    size = np.zeros(n_vars); size[:n_features] = 1
    threshold = np.zeros(n_vars); threshold[-1] = 1
    extra_rows.extend((size, threshold - size))
    extra_upper.extend((max_items, 0))
    for indices in grouped.values():
        if len(indices) > 1:
            row = np.zeros(n_vars); row[indices] = 1
            extra_rows.append(row); extra_upper.append(1)
    problem_matrix = sparse.vstack((event_constraints, sparse.csr_matrix(np.asarray(extra_rows))), format="csc")
    upper = np.concatenate((np.where(training_y == 1, 0.0, -1.0), extra_upper))
    variable_lower, variable_upper = np.zeros(n_vars), np.ones(n_vars)
    variable_lower[-1], variable_upper[-1] = 1, max_items
    if max_threshold is not None:
        variable_upper[-1] = min(max_items, int(max_threshold))
    problem = _Problem(problem_matrix, np.full(len(upper), -np.inf), upper, variable_lower, variable_upper)
    if fixed_items is not None:
        problem = problem.constrain(size, int(fixed_items), int(fixed_items))
    loss = np.zeros(n_vars); loss[n_features:-1] = normalized
    return _Built(problem, loss, size, threshold, training_X, training_y, normalized,
                  grouped, n_features, n_rows, max_items, total_weight, n_samples, int(keep.sum()))


def _singleton_incumbent(b: _Built) -> np.ndarray:
    singleton_losses = b.normalized @ (b.X != b.y[:, None])
    best_index = int(np.argmin(singleton_losses))
    incumbent = np.zeros(len(b.loss)); incumbent[best_index] = 1; incumbent[-1] = 1
    incumbent[b.n_features:-1] = b.X[:, best_index] != b.y
    return incumbent


def fit_checklist(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                  groups: Optional[Sequence[Hashable]] = None, max_items: int = 5,
                  time_limit: float = 60, loss_tolerance: float = 1e-8,
                  compress: bool = True) -> ChecklistModel:
    """Weighted checklist with at most one item per group (PC/optimizer.py:196-376).

    Three lexicographic stages: weighted error, item count, threshold.  Later
    stages may raise the loss by at most ``loss_tolerance``.  The time limit is
    shared by all stages.  A deterministic best singleton is the fallback
    incumbent, and the diagnostics say so.  Added to the diagnostics:
    ``raw_stage1_bound`` (the unclipped solver bound, on the normalized scale)
    and ``stage_completed`` flags used by the one-site certification.
    """
    started = time.monotonic()
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    if not np.isfinite(loss_tolerance) or not 0 <= loss_tolerance < 1:
        raise ValueError("loss_tolerance must be finite and in [0, 1)")
    b = _build(X, y, sample_weight, groups, max_items, compress)
    problem, loss, size, threshold = b.problem, b.loss, b.size, b.threshold
    training_X, training_y = b.X, b.y
    n_features, grouped = b.n_features, b.grouped
    incumbent = _singleton_incumbent(b)
    source, stages = "single_item_fallback", []
    original_bound, loss_limit = None, None
    loss_optimal, lexicographic_complete = False, False
    deadline = started + float(time_limit)
    max_items = b.max_items

    for stage_index, (name, objective) in enumerate((("weighted_error", loss), ("item_count", size), ("threshold", threshold))):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        stage_started = time.monotonic()
        result = _solve(problem, objective, remaining)
        if result.status not in {"optimal", "limit"}:
            raise RuntimeError("Checklist %s stage failed (%s): %s" % (name, result.status, result.message))
        accepted = False
        if result.x is not None and np.all(np.isfinite(result.x)):
            candidate = np.rint(result.x)
            selected = np.flatnonzero(candidate[:n_features])
            M = int(candidate[-1])
            valid = (np.all(np.abs(result.x - candidate) <= 1e-5)
                     and np.all((candidate[:n_features] == 0) | (candidate[:n_features] == 1))
                     and 1 <= M <= len(selected) <= max_items
                     and all(candidate[indices].sum() <= 1 for indices in grouped.values()))
            if valid:
                candidate[n_features:-1] = (training_X[:, selected].sum(axis=1) >= M) != training_y
                candidate_loss = float(loss @ candidate)
                # Recheck against actual predictions, independent of solver z.
                valid = (loss_limit is None or candidate_loss <= loss_limit + 1e-12)
                if stage_index == 2:
                    valid = valid and int(size @ candidate) == int(size @ incumbent)
                if valid and float(objective @ candidate) <= float(objective @ incumbent) + 1e-12:
                    incumbent, source, accepted = candidate, "solver", True
        stage = {
            "name": name, "status": result.status, "optimal": result.optimal,
            "accepted_incumbent": accepted, "objective": result.objective,
            "lower_bound": result.bound, "relative_gap": result.gap,
            "message": result.message, "nodes": result.nodes,
            "elapsed_seconds": time.monotonic() - stage_started,
        }
        stages.append(stage)
        if stage_index == 0:
            original_bound = result.bound
            loss_optimal = bool(result.optimal and accepted and result.bound is not None
                                and float(loss @ incumbent) - result.bound <= loss_tolerance + 1e-12)
        # Do not trade primary loss for simplicity while its optimum is unproved.
        if not result.optimal or not accepted or (stage_index == 0 and not loss_optimal):
            break
        if stage_index == 0:
            # Anchor the tolerance to the certified lower bound as well as the
            # incumbent so solver and tie tolerances cannot accumulate.
            loss_limit = min(float(loss @ incumbent), max(0.0, result.bound)) + loss_tolerance
            problem = problem.constrain(loss, -np.inf, loss_limit)
        elif stage_index == 1:
            item_count = int(size @ incumbent)
            problem = problem.constrain(size, item_count, item_count)
        else:
            lexicographic_complete = True

    selected_indices = np.flatnonzero(incumbent[:n_features]).astype(int).tolist()
    error_rate = float(loss @ incumbent)
    bound = max(0.0, original_bound) if original_bound is not None else None
    absolute_gap = max(0.0, error_rate - bound) if bound is not None else None
    relative_gap = (absolute_gap / error_rate if error_rate else (0.0 if absolute_gap == 0 else None)) if absolute_gap is not None else None
    done = {s["name"]: bool(s["optimal"] and s["accepted_incumbent"]) for s in stages}
    diagnostics = {
        "implementation": "local_weighted_paper_formulation",
        "solver": "highs", "status": "optimal" if lexicographic_complete else "feasible_unproved",
        "has_feasible_solution": True, "solution_source": source,
        "optimal": lexicographic_complete, "loss_optimal": loss_optimal,
        "lexicographic_complete": lexicographic_complete,
        "items_stage_complete": bool(loss_optimal and done.get("item_count", False)),
        "loss_tolerance": float(loss_tolerance), "loss_normalization": "sum_of_sample_weights",
        "weighted_error": error_rate * b.total_weight, "weighted_error_rate": error_rate,
        "loss_lower_bound": bound, "raw_stage1_bound": original_bound, "loss_upper_bound": error_rate,
        "absolute_gap": absolute_gap, "optimality_gap": relative_gap,
        "n_samples": b.n_samples, "n_positive_weight_samples": b.n_positive,
        "n_compressed_rows": b.n_rows, "compression": bool(compress),
        "total_weight": b.total_weight, "n_features": n_features,
        "n_selected": len(selected_indices), "max_items": max_items,
        "time_limit_seconds": float(time_limit), "elapsed_seconds": time.monotonic() - started,
        "stages": stages,
    }
    return ChecklistModel(selected_indices, int(incumbent[-1]), n_features, diagnostics)


def loss_bound(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
               groups: Optional[Sequence[Hashable]], max_items: int, time_limit: float,
               *, fixed_items: int | None = None, max_threshold: int | None = None,
               incumbent_ids: Sequence[int] | None = None, incumbent_threshold: int = 1,
               compress: bool = True) -> dict:
    """Stage-1 only: a lower bound on the weighted error of a restricted MIP.

    Used for the secondary certification (fewest items, smallest M).  The
    feasible incumbent must satisfy the same restriction: PC's singleton
    fallback is infeasible at ``fixed_items > 1``, so the caller passes an
    explicit ``incumbent_ids``.  Returns the raw solver bound and the
    incumbent's loss, both on the normalized scale, plus ``total_weight`` to
    rescale to a global one.  A missing bound is reported as ``None``.
    """
    started = time.monotonic()
    b = _build(X, y, sample_weight, groups, max_items, compress, fixed_items, max_threshold)
    ids = [int(i) for i in (incumbent_ids if incumbent_ids is not None else [])]
    if ids:
        sel = np.asarray(ids)
        # Rescore against actual predictions on the compressed rows.
        pred = b.X[:, sel].sum(axis=1) >= int(incumbent_threshold)
        incumbent_loss = float(b.normalized @ (pred != b.y))
        feasible = (1 <= incumbent_threshold <= len(ids) <= b.max_items
                    and all(len([j for j in ids if j in set(v)]) <= 1 for v in b.grouped.values())
                    and (fixed_items is None or len(ids) == fixed_items)
                    and (max_threshold is None or incumbent_threshold <= max_threshold))
    else:
        feasible, incumbent_loss = False, None
    result = _solve(b.problem, b.loss, max(time_limit, 1e-3))
    if result.status not in {"optimal", "limit"}:
        raise RuntimeError("restricted loss bound failed (%s): %s" % (result.status, result.message))
    solution_loss = None
    if result.x is not None and np.all(np.isfinite(result.x)):
        cand = np.rint(result.x)
        sel = np.flatnonzero(cand[:b.n_features])
        M = int(cand[-1])
        if 1 <= M <= len(sel) <= b.max_items and (fixed_items is None or len(sel) == fixed_items):
            pred = b.X[:, sel].sum(axis=1) >= M
            solution_loss = float(b.normalized @ (pred != b.y))
    return {
        "status": result.status, "optimal": result.optimal,
        "raw_bound": result.bound, "incumbent_loss": incumbent_loss,
        "incumbent_feasible": bool(feasible), "solution_loss": solution_loss,
        "total_weight": b.total_weight, "elapsed_seconds": time.monotonic() - started,
        "message": result.message,
    }


# --------------------------------------------------------------------------- #
# Calibration                                           (PC/evaluation.py:14-90)
# --------------------------------------------------------------------------- #

@dataclass
class ProbabilityCalibrator:
    method: str
    parameters: dict
    metadata: dict = field(default_factory=dict)

    def predict(self, scores: Any) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("calibration scores must be finite")
        if self.method == "constant":
            return np.full(scores.shape, self.parameters["probability"], dtype=float)
        if self.method == "logistic":
            from scipy.special import expit
            return expit(self.parameters["intercept"] + self.parameters["slope"] * scores)
        if self.method == "isotonic":
            return np.interp(scores, self.parameters["x"], self.parameters["y"])
        raise ValueError(f"unknown calibration method: {self.method}")

    def to_dict(self) -> dict:
        return {"method": self.method, "parameters": self.parameters, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, state: dict) -> "ProbabilityCalibrator":
        return cls(state["method"], dict(state["parameters"]), dict(state.get("metadata", {})))


# Ridge on the adjust-only calibration coefficients, per unit of normalized
# weight, on standardized adjusters (PC/evaluation.py).
ADJUSTER_RIDGE = 1e-2


def fit_calibrator(scores: Any, labels: Any, weights: Any, counts: Any = None,
                   method: str = "logistic") -> ProbabilityCalibrator:
    """Logistic calibrator on unbalanced out-of-fold IPCW weights.

    Identical to ``PC.fit_calibrator`` (the slope is constrained nonnegative; a
    tiny ridge keeps separable checklist counts finite) except that the inputs
    may be weighted *cells*: the loss is a weighted sum over rows, so identical
    ``(score, label)`` rows can be summed into one cell with the weights added.
    ``counts`` is the number of rows each cell stands for, used only for the
    metadata so it matches a patient-level fit.
    """
    scores, labels, weights = [np.asarray(value) for value in (scores, labels, weights)]
    if scores.ndim != 1 or scores.shape != labels.shape or scores.shape != weights.shape:
        raise ValueError("calibration arrays must be equally sized and one-dimensional")
    counts = np.ones(len(scores), dtype=int) if counts is None else np.asarray(counts, dtype=int)
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("calibration labels must be -1, 0, or 1")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("calibration weights must be finite and nonnegative")
    known = (labels >= 0) & (weights > 0)
    if not known.any() or not np.isfinite(scores[known]).all():
        raise ValueError("finite out-of-fold scores and positive weights are required")
    x, y, w = scores[known].astype(float), labels[known].astype(float), weights[known].astype(float)
    prevalence = float(np.average(y, weights=w))
    metadata = {"n_known": int(counts[known].sum()), "weight_sum": float(w.sum()),
                "weighted_prevalence": prevalence, "class_balancing": False,
                "score_source_required": "training_out_of_fold", "requested_method": method}
    if method not in {"logistic", "isotonic"}:
        raise ValueError("calibration method must be logistic or isotonic")
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        metadata["reason"] = "single outcome class or constant scores"
        return ProbabilityCalibrator("constant", {"probability": prevalence}, metadata)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y, sample_weight=w)
        return ProbabilityCalibrator("isotonic", {"x": fitted.X_thresholds_.tolist(),
                                                 "y": fitted.y_thresholds_.tolist()}, metadata)
    from scipy.optimize import minimize
    from scipy.special import expit, logit
    ridge = 1e-6
    w = w / w.sum()

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = beta[0] + beta[1] * x
        loss = np.dot(w, np.logaddexp(0.0, z) - y * z) + ridge * beta[1] ** 2 / 2
        residual = w * (expit(z) - y)
        return float(loss), np.array([residual.sum(), np.dot(residual, x) + ridge * beta[1]])

    fitted = minimize(objective, [logit(prevalence), 0.0], jac=True, method="L-BFGS-B",
                      bounds=[(None, None), (0.0, None)])
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"probability calibration failed: {fitted.message}")
    metadata["slope_ridge"] = ridge
    return ProbabilityCalibrator("logistic", {"intercept": float(fitted.x[0]),
                                             "slope": float(fitted.x[1])}, metadata)


def fit_site_adjusters(intercept: float, slope: float, scores: np.ndarray, labels: np.ndarray,
                       weights: np.ndarray, Z: np.ndarray, gamma: Optional[np.ndarray] = None,
                       max_iter: int = 50) -> tuple[float, list[float], np.ndarray]:
    """One site's term of the adjust-only calibration objective.

    PC/ fits ``logit p = a + b*score + Z @ gamma`` with a ridge on gamma, on one
    pooled block.  Here ``(a, b)`` are shared and each site keeps its own gamma,
    so this solves gamma for fixed ``(a, b)`` by damped Newton and returns the
    site's loss per unit of normalized weight, the loss's gradient in ``(a, b)``
    (at the solved gamma that is the gradient of the minimized objective), and
    gamma, which the caller must not send anywhere.  ``gamma`` warm-starts.
    """
    from scipy.special import expit

    w = weights / weights.sum()
    linear = intercept + slope * scores
    gamma = np.zeros(Z.shape[1]) if gamma is None else gamma

    def loss_at(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        eta = linear + Z @ candidate
        loss = np.dot(w, np.logaddexp(0.0, eta) - labels * eta)
        return float(loss + ADJUSTER_RIDGE * np.dot(candidate, candidate) / 2), eta

    loss, eta = loss_at(gamma)
    for _ in range(max_iter):
        probability = expit(eta)
        gradient = Z.T @ (w * (probability - labels)) + ADJUSTER_RIDGE * gamma
        hessian = (Z.T * (w * probability * (1 - probability))) @ Z
        step = np.linalg.solve(hessian + ADJUSTER_RIDGE * np.eye(len(gamma)), gradient)
        decrement = float(gradient @ step)
        if decrement < 1e-22:
            residual = w * (probability - labels)
            return loss, [float(residual.sum()), float(residual @ scores)], gamma
        scale = 1.0
        trial_loss, trial_eta = loss_at(gamma - step)
        # Below 1e-12 the loss can't resolve the decrease, and Newton is quadratic.
        while decrement > 1e-12 and trial_loss > loss - 1e-4 * scale * decrement:
            scale /= 2
            if scale < 1e-10:
                raise ArithmeticError("adjuster line search failed")
            trial_loss, trial_eta = loss_at(gamma - scale * step)
        gamma, loss, eta = gamma - scale * step, trial_loss, trial_eta
    raise ArithmeticError("adjuster fit did not converge")


# --------------------------------------------------------------------------- #
# Metrics                                               (PC/evaluation.py:93-197)
# --------------------------------------------------------------------------- #

def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float | None:
    """Weighted concordant case/control pairs with half credit for score ties."""
    known = (labels >= 0) & (weights > 0)
    labels, scores, weights = labels[known], scores[known], weights[known]
    total_positive, total_negative = weights[labels == 1].sum(), weights[labels == 0].sum()
    if total_positive <= 0 or total_negative <= 0:
        return None
    _, group = np.unique(scores, return_inverse=True)
    group = group.reshape(-1)
    positive = np.bincount(group, weights=weights * (labels == 1))
    negative = np.bincount(group, weights=weights * (labels == 0))
    below = np.cumsum(negative) - negative
    return float(np.dot(positive, below + 0.5 * negative) / (total_positive * total_negative))


def _metrics(labels: np.ndarray, weights: np.ndarray, score: np.ndarray,
             flag: np.ndarray, probability: np.ndarray) -> dict:
    positive, negative = labels == 1, labels == 0
    tp, fn = float(weights[positive & flag].sum()), float(weights[positive & ~flag].sum())
    fp, tn = float(weights[negative & flag].sum()), float(weights[negative & ~flag].sum())
    known = labels >= 0
    sensitivity, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {"auc": weighted_auc(labels, score, weights), "sensitivity": sensitivity,
            "specificity": specificity, "ppv": _ratio(tp, tp + fp), "npv": _ratio(tn, tn + fn),
            "balanced_accuracy": (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None else None,
            "brier": float(np.sum(weights[known] * np.square(labels[known] - probability[known])) / len(labels)),
            "weighted_tp": tp, "weighted_fn": fn, "weighted_fp": fp, "weighted_tn": tn,
            "mean_predicted_risk": float(probability.mean()),
            "ipcw_event_risk": float(weights[positive].sum() / len(labels)),
            "weighted_observed_risk": _ratio(tp + fn, tp + fn + fp + tn)}


def expand_cells(cells: dict, horizon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Patient-level (duration, event, score, cell index) from weighted cells.

    A cell is ``(score, label, case_event_day) -> n``.  ``horizon_targets`` only
    reads durations through the case/control/unknown split and the cases' G, so
    a control stands in as ``horizon + 1`` censored and an unknown as censored at
    the horizon -- labels and weights then equal the patient-level ones.
    """
    label = np.asarray(cells["label"], dtype=int)
    day = np.asarray(cells["day"], dtype=float)
    duration = np.where(label == 1, day, np.where(label == 0, horizon + 1.0, float(horizon)))
    event = (label == 1).astype(int)
    n = np.asarray(cells["n"], dtype=int)
    return duration, event, np.asarray(cells["score"], dtype=float), n


def evaluate_cells(cells: dict, probability_by_cell: np.ndarray, flag_by_cell: np.ndarray,
                   horizon: float, censoring: CensoringDistribution, min_g: float = 0.05,
                   bootstrap_samples: int = 0, seed: int = 42, calibration_bins: int = 5,
                   confidence_level: float = 0.95) -> dict:
    """``PC.evaluate_horizon`` computed from weighted cells.

    Point estimates are taken on the expanded patient-level arrays, so they equal
    ``evaluate_horizon`` on the same patients.  The bootstrap resamples *cells*
    multinomially: rows within a cell are exchangeable, so this has the same
    distribution as resampling patients, conditional on the fitted model and G.
    """
    duration_c, event_c, score_c, n_c = expand_cells(cells, horizon)
    if not len(n_c) or n_c.sum() == 0:
        return {"status": "unsupported", "reason": "empty evaluation scope",
                "horizon": float(horizon), "n": 0}
    reps = np.repeat(np.arange(len(n_c)), n_c)
    duration, event = duration_c[reps], event_c[reps]
    score = score_c[reps]
    probability = np.asarray(probability_by_cell, dtype=float)[reps]
    flag = np.asarray(flag_by_cell).astype(bool)[reps]
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    try:
        labels, weights, audit = horizon_targets(duration, event, horizon, censoring, min_g)
    except UnsupportedHorizonError as exc:
        return {"status": "unsupported", "reason": str(exc), "horizon": float(horizon), "n": len(duration)}
    result = {**audit, **_metrics(labels, weights, score, flag, probability)}
    result["status"] = "ok" if result["auc"] is not None else "single_class"
    if result["auc"] is None:
        result["reason"] = "both observed horizon cases and controls are required for AUC"
    result["calibration"] = []
    # Fixed equal-width probability bins avoid selecting cutpoints on test labels.
    bins = np.minimum((probability * calibration_bins).astype(int), calibration_bins - 1)
    for index in range(calibration_bins):
        all_rows = bins == index
        known = all_rows & (labels >= 0)
        if not all_rows.any():
            continue
        weight_sum = float(weights[known].sum())
        result["calibration"].append({"bin": index, "lower": index / calibration_bins,
                                      "upper": (index + 1) / calibration_bins,
                                      "n": int(all_rows.sum()), "n_known": int(known.sum()),
                                      "mean_predicted_risk": float(probability[all_rows].mean()),
                                      "weighted_mean_predicted_risk":
                                      _ratio(float(np.dot(weights[known], probability[known])), weight_sum),
                                      "weighted_observed_risk":
                                      _ratio(float(np.dot(weights[known], labels[known])), weight_sum)})
    if bootstrap_samples:
        labels_c, weights_c, _ = horizon_targets(duration_c, event_c, horizon, censoring, min_g)
        prob_c = np.asarray(probability_by_cell, dtype=float)
        flag_c = np.asarray(flag_by_cell).astype(bool)
        rng = np.random.default_rng(seed)
        total = int(n_c.sum())
        metric_names = ["auc", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "brier"]
        replicates = {name: [] for name in metric_names}
        for _ in range(bootstrap_samples):
            draw = rng.multinomial(total, n_c / total)
            rows = np.repeat(np.arange(len(n_c)), draw)
            sample = _metrics(labels_c[rows], weights_c[rows], score_c[rows], flag_c[rows], prob_c[rows])
            for name in metric_names:
                if sample[name] is not None and np.isfinite(sample[name]):
                    replicates[name].append(sample[name])
        alpha = (1 - confidence_level) / 2
        result["bootstrap"] = {"samples": bootstrap_samples, "seed": seed,
                               "confidence_level": confidence_level,
                               "method": "table_cell_multinomial_conditional_on_fitted_models_and_training_G",
                               "intervals": {}}
        for name, values in replicates.items():
            interval = np.quantile(values, [alpha, 1 - alpha]).tolist() if values else [None, None]
            result["bootstrap"]["intervals"][name] = {
                "lower": interval[0], "upper": interval[1], "valid_samples": len(values)}
    return result
