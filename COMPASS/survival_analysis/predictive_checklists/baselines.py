"""Training-only Cox and XGBoost comparisons on the checklist patient splits.

Optional survival libraries are imported only when their model is requested.
Every CV fold learns its own imputation/scaling and censoring distribution.
These are compact comparator grids, not a reproduction of the project's
larger existing survival-model hyperparameter searches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

from .evaluation import weighted_auc
from .survival import CensoringDistribution, UnsupportedHorizonError, horizon_targets, validate_outcomes


@dataclass
class _Preprocessor:
    columns: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, min_patient_coverage: float = 0.20,
            age_column: str | None = None) -> "_Preprocessor":
        numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Eligibility is determined within this training fold.
        columns = [column for column in numeric if numeric[column].notna().any()
                   and (column == age_column or numeric[column].notna().mean() >= min_patient_coverage)]
        if not columns:
            raise ValueError("no predictor has observed training values")
        medians = numeric[columns].median().to_numpy(dtype=float)
        data = numeric[columns].fillna(dict(zip(columns, medians))).to_numpy(dtype=float)
        means, scales = data.mean(axis=0), data.std(axis=0)
        variable = scales > 1e-12
        if not variable.any():
            raise ValueError("all training predictors are constant after imputation")
        return cls(np.asarray(columns)[variable].tolist(), medians[variable], means[variable], scales[variable])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        absent = set(self.columns) - set(frame.columns)
        if absent:
            raise ValueError(f"missing baseline predictor columns: {sorted(absent)}")
        data = frame[self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        data = np.where(np.isfinite(data), data, self.medians)
        return (data - self.means) / self.scales


@dataclass
class BaselineResult:
    name: str
    status: str
    reason: str | None = None
    metadata: dict = field(default_factory=dict)
    model: Any = None
    preprocessing: _Preprocessor | None = None
    event_times: np.ndarray | None = None
    log_cumulative_hazard: np.ndarray | None = None
    max_training_followup: float | None = None

    def predict(self, frame: pd.DataFrame, horizon: float) -> dict[str, np.ndarray]:
        if self.status != "ok" or self.preprocessing is None:
            raise ValueError(f"baseline {self.name} is unavailable: {self.reason}")
        if not np.isfinite(horizon) or horizon <= 0 or horizon >= self.max_training_followup:
            raise UnsupportedHorizonError("baseline horizon must be positive and below maximum training follow-up")
        data = self.preprocessing.transform(frame)
        scores = _predict_score(self.name, self.model, data)
        index = np.searchsorted(self.event_times, horizon, side="right") - 1
        if index < 0:
            probability = np.zeros(len(data))
        else:
            # Log-space Breslow predictions remain stable for extreme margins.
            log_intensity = self.log_cumulative_hazard[index] + scores
            probability = -np.expm1(-np.exp(np.minimum(log_intensity, 709.0)))
        return {"score": scores, "probability": probability, "flag": (probability >= 0.5).astype(int)}


def _survival_array(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    result = np.empty(len(duration), dtype=[("event", bool), ("time", float)])
    result["event"], result["time"] = event, duration
    return result


def _fit_model(name: str, data: np.ndarray, duration: np.ndarray,
               event: np.ndarray, parameters: dict, seed: int) -> Any:
    if not event.any():
        raise ValueError("at least one observed training event is required")
    if name == "age_cox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        return CoxPHSurvivalAnalysis(alpha=1e-4, ties="breslow").fit(data, _survival_array(duration, event))
    if name == "elastic_net":
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        return CoxnetSurvivalAnalysis(alphas=[parameters["alpha"]], l1_ratio=parameters["l1_ratio"],
                                     fit_baseline_model=False, max_iter=100000).fit(
                                         data, _survival_array(duration, event))
    if name == "xgboost":
        import xgboost as xgb
        if (duration <= 0).any():
            raise ValueError("XGBoost Cox requires strictly positive durations for signed censoring labels")
        target = np.where(event, duration, -duration)
        dtrain = xgb.DMatrix(data, label=target)
        return xgb.train({"objective": "survival:cox", "eval_metric": "cox-nloglik",
                          "max_depth": parameters["max_depth"], "eta": 0.05,
                          "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 1.0,
                          "tree_method": "hist", "nthread": 1, "seed": seed, "verbosity": 0},
                         dtrain, num_boost_round=parameters["rounds"])
    raise ValueError(f"unknown baseline model: {name}")


def _predict_score(name: str, model: Any, data: np.ndarray) -> np.ndarray:
    if name == "xgboost":
        import xgboost as xgb
        score = model.predict(xgb.DMatrix(data), output_margin=True)
    else:
        score = model.predict(data)
    score = np.asarray(score, dtype=float)
    if score.shape != (len(data),) or not np.isfinite(score).all():
        raise ValueError("baseline returned nonfinite or misaligned risk scores")
    return score


def _breslow(duration: np.ndarray, event: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Breslow baseline cumulative hazard in log space, including tied events."""
    from scipy.special import logsumexp
    event_times, events = np.unique(duration[event], return_counts=True)
    log_increments = np.array([np.log(count) - logsumexp(score[duration >= time])
                               for time, count in zip(event_times, events)])
    return event_times, np.logaddexp.accumulate(log_increments)


def _candidate_grid(name: str) -> list[dict]:
    if name == "age_cox":
        return [{"ridge": 1e-4}]
    if name == "elastic_net":
        return [{"alpha": alpha, "l1_ratio": ratio}
                for alpha in (0.0001, 0.001, 0.01, 0.1) for ratio in (0.5, 1.0)]
    if name == "xgboost":
        return [{"max_depth": depth, "rounds": rounds} for depth in (2, 3) for rounds in (100, 300)]
    raise ValueError(f"unknown baseline model: {name}")


def fit_baselines(X_train: pd.DataFrame, duration: Any, event: Any, *,
                  age_column: str, horizons: Sequence[float], n_splits: int = 3,
                  seed: int = 42, models: Sequence[str] | None = None,
                  min_g: float = 0.05, min_patient_coverage: float = 0.20) -> dict[str, BaselineResult]:
    """Fit requested comparators using only the supplied development patients.

    The caller must supply the explicit age/laboratory feature whitelist and
    exclude all test patients. Model tuning uses mean validation horizon AUC;
    an unsupported horizon is recorded and omitted identically for every
    candidate in that fold. A candidate with a failed usable fold is ineligible.
    Model artifacts may be pickled by the runner for trusted local reuse.
    """
    from sklearn.model_selection import StratifiedKFold
    duration, event = validate_outcomes(duration, event)
    if len(X_train) != len(duration) or not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a DataFrame with one row per training patient")
    if X_train.columns.duplicated().any():
        raise ValueError("baseline predictor names must be unique")
    if int(n_splits) != n_splits or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if not 0 <= min_patient_coverage <= 1 or not 0 < min_g <= 1:
        raise ValueError("invalid min_patient_coverage or censoring min_g")
    horizons = [float(value) for value in horizons]
    if not horizons or not np.isfinite(horizons).all() or min(horizons) <= 0:
        raise ValueError("horizons must contain finite positive values")
    names = list(models) if models is not None else ["age_cox", "elastic_net", "xgboost"]
    invalid = set(names) - {"age_cox", "elastic_net", "xgboost"}
    if invalid:
        raise ValueError(f"unknown baseline models: {sorted(invalid)}")
    counts = np.bincount(event.astype(int), minlength=2)
    available_folds = min(n_splits, int(counts.min()))
    splits = list(StratifiedKFold(n_splits=available_folds, shuffle=True, random_state=seed).split(X_train, event)) \
        if available_folds >= 2 else []
    result: dict[str, BaselineResult] = {}
    for name in names:
        metadata: dict = {"prediction_cutoff": 0.5, "cutoff_selection": "fixed; not optimized",
                          "requested_horizons": horizons, "seed": seed,
                          "cv_folds": available_folds if name != "age_cox" else 0,
                          "censoring_min_g": min_g, "training_n": len(duration),
                          "min_patient_coverage": min_patient_coverage,
                          "baseline_hazard": "training_only_Breslow", "cv_candidates": []}
        try:
            if name in {"age_cox", "elastic_net"}:
                import sksurv  # noqa: F401
                metadata["library_version"] = sksurv.__version__
            else:
                import xgboost
                metadata["library_version"] = xgboost.__version__
            if name == "age_cox":
                if age_column not in X_train:
                    raise ValueError(f"age column {age_column!r} is missing")
                frame = X_train[[age_column]]
                best_parameters = _candidate_grid(name)[0]
            else:
                frame = X_train
                if not splits:
                    raise ValueError("at least two events and two censored patients are required for training-only CV")
                best_score, best_parameters = -np.inf, None
                for parameters in _candidate_grid(name):
                    fold_scores, fold_audit = [], []
                    failed_usable_fold = False
                    for fold, (train, valid) in enumerate(splits):
                        fold_info: dict = {"fold": fold, "horizons": []}
                        try:
                            censoring = CensoringDistribution.fit(duration[train], event[train])
                            supported_targets = []
                            for horizon in horizons:
                                try:
                                    labels, weights, audit = horizon_targets(duration[valid], event[valid], horizon,
                                                                             censoring, min_g)
                                    if len(np.unique(labels[labels >= 0])) < 2:
                                        audit["status"] = "single_class"
                                    else:
                                        supported_targets.append((labels, weights))
                                    fold_info["horizons"].append(audit)
                                except UnsupportedHorizonError as exc:
                                    fold_info["horizons"].append({"horizon": horizon, "status": "unsupported", "reason": str(exc)})
                            if not supported_targets:
                                fold_info["status"] = "no_supported_two_class_horizon"
                                fold_audit.append(fold_info)
                                continue
                            preprocessor = _Preprocessor.fit(frame.iloc[train], min_patient_coverage, age_column)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                model = _fit_model(name, preprocessor.transform(frame.iloc[train]), duration[train],
                                                   event[train], parameters, seed + fold)
                            scores = _predict_score(name, model, preprocessor.transform(frame.iloc[valid]))
                            aucs = [weighted_auc(labels, scores, weights) for labels, weights in supported_targets]
                            fold_score = float(np.mean(aucs))
                            fold_scores.append(fold_score)
                            fold_info.update({"status": "ok", "mean_auc": fold_score,
                                              "feature_columns": preprocessor.columns,
                                              "warnings": [str(w.message) for w in caught]})
                        except (ValueError, ArithmeticError, RuntimeError) as exc:
                            failed_usable_fold = True
                            fold_info.update({"status": "failed", "reason": str(exc)})
                        fold_audit.append(fold_info)
                    eligible = bool(fold_scores) and not failed_usable_fold
                    mean_score = float(np.mean(fold_scores)) if eligible else None
                    metadata["cv_candidates"].append({"parameters": parameters, "mean_auc": mean_score,
                                                      "eligible": eligible, "folds": fold_audit})
                    if eligible and mean_score > best_score:
                        best_score, best_parameters = mean_score, parameters
                if best_parameters is None:
                    raise ValueError("all baseline candidates lacked usable training CV results; inspect cv_candidates")
                metadata["best_cv_auc"] = best_score
            preprocessing = _Preprocessor.fit(frame, min_patient_coverage, age_column)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = _fit_model(name, preprocessing.transform(frame), duration, event, best_parameters, seed)
            training_score = _predict_score(name, model, preprocessing.transform(frame))
            event_times, log_hazard = _breslow(duration, event, training_score)
            metadata.update({"parameters": best_parameters, "feature_columns": preprocessing.columns,
                             "warnings": [str(w.message) for w in caught]})
            result[name] = BaselineResult(name, "ok", metadata=metadata, model=model,
                                           preprocessing=preprocessing, event_times=event_times,
                                           log_cumulative_hazard=log_hazard,
                                           max_training_followup=float(duration.max()))
        except (ImportError, OSError) as exc:
            result[name] = BaselineResult(name, "skipped", f"optional dependency unavailable: {exc}", metadata)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            result[name] = BaselineResult(name, "skipped", str(exc), metadata)
    return result
