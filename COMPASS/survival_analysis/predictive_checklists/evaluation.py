"""Censoring-aware horizon evaluation and serializable monotone calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .survival import CensoringDistribution, UnsupportedHorizonError, horizon_targets, validate_outcomes


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


def fit_calibrator(scores: Any, labels: Any, weights: Any, method: str = "logistic"
                   ) -> ProbabilityCalibrator:
    """Fit to training out-of-fold scores with unbalanced censoring weights.

    Callers must supply out-of-fold scores; this function cannot establish
    their provenance. The logistic slope is constrained nonnegative. A tiny
    slope ridge ensures finite estimates for separable checklist counts.
    """
    scores, labels, weights = [np.asarray(value) for value in (scores, labels, weights)]
    if scores.ndim != 1 or scores.shape != labels.shape or scores.shape != weights.shape:
        raise ValueError("calibration arrays must be equally sized and one-dimensional")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("calibration labels must be -1, 0, or 1")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("calibration weights must be finite and nonnegative")
    known = (labels >= 0) & (weights > 0)
    if not known.any() or not np.isfinite(scores[known]).all():
        raise ValueError("finite out-of-fold scores and positive weights are required")
    x, y, w = scores[known].astype(float), labels[known].astype(float), weights[known].astype(float)
    prevalence = float(np.average(y, weights=w))
    metadata = {"n_known": int(known.sum()), "weight_sum": float(w.sum()),
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


def evaluate_horizon(duration: Any, event: Any, score: Any, flag: Any,
                     probability: Any, horizon: float, censoring: CensoringDistribution,
                     min_g: float = 0.05, bootstrap_samples: int = 0, seed: int = 42,
                     calibration_bins: int = 5, confidence_level: float = 0.95) -> dict:
    """Evaluate fixed predictions; bootstrap rows represent distinct patients.

    Bootstrap intervals condition on the fitted models and training censoring
    distribution. Models and calibration are not re-fit in these intervals.
    The caller must pass one row per patient within an endpoint/landmark.
    """
    duration, event = validate_outcomes(duration, event)
    score, probability = np.asarray(score, dtype=float), np.asarray(probability, dtype=float)
    flag_array = np.asarray(flag)
    if score.shape != duration.shape or probability.shape != duration.shape or flag_array.shape != duration.shape:
        raise ValueError("all evaluation arrays must have one value per patient")
    if not np.isfinite(score).all() or not np.isfinite(probability).all():
        raise ValueError("predictions must be finite")
    if ((probability < 0) | (probability > 1)).any() or not np.isin(flag_array, [0, 1]).all():
        raise ValueError("probabilities must be in [0, 1] and flags must be binary")
    if int(bootstrap_samples) != bootstrap_samples or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    if not 0 < confidence_level < 1 or int(calibration_bins) != calibration_bins or calibration_bins < 1:
        raise ValueError("invalid confidence_level or calibration_bins")
    flag = flag_array.astype(bool)
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
        rng = np.random.default_rng(seed)
        metric_names = ["auc", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "brier"]
        replicates = {name: [] for name in metric_names}
        for _ in range(bootstrap_samples):
            rows = rng.integers(0, len(duration), size=len(duration))
            sample = _metrics(labels[rows], weights[rows], score[rows], flag[rows], probability[rows])
            for name in metric_names:
                if sample[name] is not None and np.isfinite(sample[name]):
                    replicates[name].append(sample[name])
        alpha = (1 - confidence_level) / 2
        result["bootstrap"] = {"samples": bootstrap_samples, "seed": seed,
                               "confidence_level": confidence_level,
                               "method": "patient_percentile_conditional_on_fitted_models_and_training_G",
                               "intervals": {}}
        for name, values in replicates.items():
            interval = np.quantile(values, [alpha, 1 - alpha]).tolist() if values else [None, None]
            result["bootstrap"]["intervals"][name] = {
                "lower": interval[0], "upper": interval[1], "valid_samples": len(values)}
    return result
