"""Shared Cox model-fitting and evaluation primitives."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from survival_common.helper import (
    _make_survival_array,
    breslow_survival_at_horizons,
    choose_stratification_labels,
)

try:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError, ConvergenceWarning
    from lifelines.utils import concordance_index

    LIFELINES_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    CoxPHFitter = None
    ConvergenceError = RuntimeError
    ConvergenceWarning = RuntimeWarning
    concordance_index = None
    LIFELINES_IMPORT_ERROR = exc

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    SKSURV_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    CoxnetSurvivalAnalysis = None
    cumulative_dynamic_auc = None
    Surv = None
    SKSURV_IMPORT_ERROR = exc

DEFAULT_COXNET_MAX_ITER = 20000
DEFAULT_UNPENALIZED_ALPHA = 1e-6
DEFAULT_AUC_QUANTILES = (0.25, 0.375, 0.50, 0.625, 0.75)
DEFAULT_AUC_TIME_UNIT_DAYS = 7


def require_lifelines() -> None:
    if CoxPHFitter is None or concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to run the Cox association pipeline."
        ) from LIFELINES_IMPORT_ERROR


def require_sksurv() -> None:
    if cumulative_dynamic_auc is None:
        raise ModuleNotFoundError(
            "scikit-survival is required for Cox IPCW AUC(t) evaluation."
        ) from SKSURV_IMPORT_ERROR


def parse_feature_name(feature_name: str) -> tuple[str, str]:
    if "__" not in feature_name:
        return feature_name, "value"
    return feature_name.rsplit("__", 1)


def matching_n_obs_feature(feature_name: str) -> str:
    lab_name, _ = parse_feature_name(feature_name)
    return f"{lab_name}__n_observations"


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.notna()
    if not valid.any():
        return q_values

    valid_values = p_values.loc[valid].astype(float)
    order = np.argsort(valid_values.values)
    ordered_values = valid_values.values[order]
    ranks = np.arange(1, len(ordered_values) + 1, dtype=float)

    adjusted = ordered_values * len(ordered_values) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    ordered_index = valid_values.index[order]
    q_values.loc[ordered_index] = adjusted
    return q_values



# Caps lifelines' Newton-Raphson step budget (default 500) for univariate/
# per-fold fits. Rare binary indicators (e.g. low-prevalence genomic
# mutations) combined with sparse categorical adjustment covariates
# (CANCER_TYPE_* dummies) can quasi-separate, causing the unpenalized MLE to
# never converge -- each such fit would otherwise burn all 500 steps (a full
# Hessian solve per step) before lifelines gives up and merely *warns*
# (ConvergenceWarning), returning the best-effort, non-converged coefficients
# as if the fit had succeeded. We escalate that warning to an exception (see
# the catch_warnings block below) so a capped, non-converged fit is treated
# as a failure -- same as today -- rather than silently reporting a
# divergent/unreliable coefficient. This makes non-convergent fits fail fast.
_MAX_NEWTON_STEPS = 25


def fit_cox_with_fallback(
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    penalizers: list[float],
    l1_ratio: float,
    unpenalized_cols: list[str] | None = None,
    covariate_cols: list[str] | None = None,
) -> tuple[object | None, float, str]:
    require_lifelines()

    if covariate_cols is None:
        covariate_cols = [c for c in model_df.columns if c not in {duration_col, event_col}]
    unpenalized = set(unpenalized_cols or [])

    last_error = ""
    for penalizer in penalizers:
        try:
            if unpenalized:
                penalty_vec = np.array(
                    [0.0 if c in unpenalized else float(penalizer) for c in covariate_cols],
                    dtype=float,
                )
                pen_arg: float | np.ndarray = penalty_vec
            else:
                pen_arg = float(penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("error", category=ConvergenceWarning)
                model = CoxPHFitter(penalizer=pen_arg, l1_ratio=l1_ratio)
                model.fit(
                    model_df,
                    duration_col=duration_col,
                    event_col=event_col,
                    fit_options={"max_steps": _MAX_NEWTON_STEPS},
                )
            note = "fit_ok" if penalizer == 0 else f"fit_ok_penalizer_{penalizer:g}"
            return model, penalizer, note
        except (ConvergenceError, ConvergenceWarning, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)

    return None, float("nan"), f"fit_failed: {last_error}"


def duration_to_auc_units(duration: pd.Series, *, time_unit_days: int) -> np.ndarray:
    if time_unit_days <= 0:
        raise ValueError("time_unit_days must be positive.")
    duration_days = pd.to_numeric(duration, errors="coerce").to_numpy(dtype=float)
    return np.ceil(duration_days / float(time_unit_days))


def apply_auc_admin_censoring(
    event: np.ndarray,
    duration: np.ndarray,
    *,
    max_time_unit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Censor durations at a fixed AUC horizon for comparable finite-horizon metrics."""
    event = np.asarray(event, dtype=int).copy()
    duration = np.asarray(duration, dtype=float).copy()
    if max_time_unit is None:
        return event, duration
    if max_time_unit <= 0:
        raise ValueError("max_time_unit must be positive when provided.")

    within_horizon = event.astype(bool) & (duration > 0) & (duration <= float(max_time_unit))
    duration = np.where(duration <= float(max_time_unit), duration, float(max_time_unit))
    return within_horizon.astype(int), duration


def compute_ipcw_auc_t(
    eval_df: pd.DataFrame,
    risk_score: np.ndarray,
    *,
    duration_col: str,
    event_col: str,
    reference_df: pd.DataFrame,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
    quantiles: tuple[float, ...] = DEFAULT_AUC_QUANTILES,
    max_time_unit: int | None = None,
    fixed_horizons: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Compute IPCW cumulative/dynamic AUC(t)."""
    require_sksurv()

    empty_cols = [
        "horizon_quantile",
        "horizon_time_unit",
        "horizon_days",
        "auc_t",
        "n_eval",
        "n_eval_events",
        "admin_censor_time_unit",
        "note",
    ]

    train_duration = duration_to_auc_units(
        reference_df[duration_col],
        time_unit_days=time_unit_days,
    )
    train_event = reference_df[event_col].to_numpy(dtype=int)
    eval_duration = duration_to_auc_units(
        eval_df[duration_col],
        time_unit_days=time_unit_days,
    )
    eval_event = eval_df[event_col].to_numpy(dtype=int)
    risk_score = np.asarray(risk_score, dtype=float).reshape(-1)

    train_event, train_duration = apply_auc_admin_censoring(
        train_event,
        train_duration,
        max_time_unit=max_time_unit,
    )
    eval_event, eval_duration = apply_auc_admin_censoring(
        eval_event,
        eval_duration,
        max_time_unit=max_time_unit,
    )

    train_valid = np.isfinite(train_duration) & (train_duration > 0)
    eval_valid = np.isfinite(eval_duration) & (eval_duration > 0) & np.isfinite(risk_score)
    if train_valid.sum() == 0 or eval_valid.sum() == 0:
        return np.nan, pd.DataFrame(columns=empty_cols)

    train_surv = _make_survival_array(train_event[train_valid], train_duration[train_valid])
    eval_surv = _make_survival_array(eval_event[eval_valid], eval_duration[eval_valid])
    eval_risk = risk_score[eval_valid]

    if fixed_horizons is not None:
        horizon_times = np.asarray(fixed_horizons, dtype=float).reshape(-1)
        horizon_times = np.unique(horizon_times[horizon_times > 0])
        if max_time_unit is not None:
            horizon_times = horizon_times[horizon_times <= float(max_time_unit)]
        if len(horizon_times) == 0:
            return np.nan, pd.DataFrame(columns=empty_cols)
        horizon_quantiles: tuple[float, ...] = tuple([np.nan] * len(horizon_times))
    else:
        event_times = eval_duration[eval_valid & (eval_event == 1)]
        event_times = event_times[np.isfinite(event_times) & (event_times > 0)]
        if len(event_times) == 0:
            return np.nan, pd.DataFrame(columns=empty_cols)
        horizon_times = np.asarray(
            [int(val) for val in np.quantile(event_times, quantiles)],
            dtype=float,
        )
        horizon_quantiles = tuple(quantiles)

    rows = []
    for quantile, horizon in zip(horizon_quantiles, horizon_times):
        auc_t = np.nan
        note = ""
        if horizon <= 0:
            note = "non_positive_horizon"
        else:
            try:
                auc_values, _ = cumulative_dynamic_auc(
                    train_surv,
                    eval_surv,
                    eval_risk,
                    np.asarray([horizon], dtype=float),
                )
                auc_t = float(auc_values[0])
            except ValueError as exc:
                note = f"auc_failed: {exc}"
        rows.append(
            {
                "horizon_quantile": quantile,
                "horizon_time_unit": horizon,
                "horizon_days": horizon * float(time_unit_days),
                "auc_t": auc_t,
                "n_eval": int(eval_valid.sum()),
                "n_eval_events": int((eval_event[eval_valid] == 1).sum()),
                "admin_censor_time_unit": max_time_unit,
                "note": note,
            }
        )

    auc_df = pd.DataFrame(rows)
    if len(horizon_times) < 2 or horizon_times[-1] <= horizon_times[0]:
        return np.nan, auc_df

    mean_auc_times = np.arange(horizon_times[0], horizon_times[-1] + 1, dtype=float)
    mean_auc_times = mean_auc_times[mean_auc_times > 0]
    if len(mean_auc_times) == 0:
        return np.nan, auc_df

    try:
        _, mean_auc = cumulative_dynamic_auc(
            train_surv,
            eval_surv,
            eval_risk,
            mean_auc_times,
        )
    except ValueError:
        mean_auc = np.nan
    return float(mean_auc) if np.isfinite(mean_auc) else np.nan, auc_df


def build_coxnet_xy(
    model_df: pd.DataFrame,
    *,
    covariate_cols: list[str],
    duration_col: str,
    event_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = model_df[covariate_cols].to_numpy(dtype=float)
    y = Surv.from_arrays(
        event=model_df[event_col].astype(bool).to_numpy(),
        time=model_df[duration_col].astype(float).to_numpy(),
    )
    return X, y


def fit_coxnet_with_fallback(
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    penalizers: list[float],
    l1_ratio: float,
    covariate_cols: list[str],
    unpenalized_cols: list[str] | None = None,
    max_iter: int = DEFAULT_COXNET_MAX_ITER,
) -> tuple[object | None, float, str]:
    """Elastic-net Cox via sksurv's coordinate-descent CoxnetSurvivalAnalysis."""
    require_sksurv()
    X, y = build_coxnet_xy(
        model_df,
        covariate_cols=covariate_cols,
        duration_col=duration_col,
        event_col=event_col,
    )
    unpenalized = set(unpenalized_cols or [])
    penalty_factor = np.array(
        [0.0 if c in unpenalized else 1.0 for c in covariate_cols],
        dtype=float,
    )

    if float(penalty_factor.sum()) == 0.0 and len(covariate_cols) > 0:
        penalty_factor = np.ones(len(covariate_cols), dtype=float)
        penalizers = [DEFAULT_UNPENALIZED_ALPHA]

    last_error = ""
    for penalizer in penalizers:
        try:
            model = CoxnetSurvivalAnalysis(
                alphas=[float(penalizer)],
                l1_ratio=float(l1_ratio),
                penalty_factor=penalty_factor,
                max_iter=int(max_iter),
                fit_baseline_model=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)
            continue

        coefs = np.asarray(model.coef_, dtype=float)
        if coefs.ndim == 2:
            coefs = coefs[:, -1]
        coefs = coefs.reshape(-1)
        if coefs.size != len(covariate_cols):
            last_error = (
                f"coef_size_mismatch: expected {len(covariate_cols)} got {coefs.size}"
            )
            continue

        note = f"fit_ok_penalizer_{penalizer:g}"
        return model, float(penalizer), note

    return None, float("nan"), f"fit_failed: {last_error}"


def score_coxnet_model(
    model,
    model_df: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
) -> tuple[float, np.ndarray]:
    require_lifelines()
    X = model_df[covariate_cols].to_numpy(dtype=float)
    log_pred = np.asarray(model.predict(X)).reshape(-1)
    c_index = float(concordance_index(model_df[duration_col], -log_pred, model_df[event_col]))
    return c_index, log_pred


def coxnet_survival_at_horizons(
    model,
    train_mdf: pd.DataFrame,
    eval_mdf: pd.DataFrame,
    *,
    duration_col: str,
    event_col: str,
    covariate_cols: list[str],
    horizons: np.ndarray,
    time_unit_days: int = DEFAULT_AUC_TIME_UNIT_DAYS,
) -> np.ndarray:
    """Survival probabilities for each eval row at each horizon."""
    X_train = train_mdf[covariate_cols].to_numpy(dtype=float)
    X_eval = eval_mdf[covariate_cols].to_numpy(dtype=float)
    train_lp = np.asarray(model.predict(X_train)).reshape(-1)
    eval_lp = np.asarray(model.predict(X_eval)).reshape(-1)
    train_event = train_mdf[event_col].astype(int).to_numpy()
    train_duration = pd.to_numeric(train_mdf[duration_col], errors="coerce").to_numpy(dtype=float)
    train_duration_units = np.ceil(train_duration / float(time_unit_days))
    return breslow_survival_at_horizons(
        train_event=train_event,
        train_duration=train_duration_units,
        train_lp=train_lp,
        eval_lp=eval_lp,
        horizons=horizons,
    )


def coxnet_coefficients(model, covariate_cols: list[str]) -> pd.Series:
    coefs = np.asarray(model.coef_)
    if coefs.ndim == 2:
        coefs = coefs[:, -1]
    return pd.Series(coefs.reshape(-1), index=covariate_cols, name="coef")


def make_cv_splitter(
    train_val: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    event_col: str | None = None,
) -> tuple[StratifiedKFold | KFold, np.ndarray | None, str]:
    labels, label_name = choose_stratification_labels(
        train_val,
        min_count=n_folds,
        event_col=event_col,
    )
    if labels is not None:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed), labels, label_name
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed), None, "unstratified"


def summarize_fold_failures(fold_df: pd.DataFrame) -> str:
    """Render a per-(penalizer, l1_ratio, fold) note table for CV diagnostics."""
    if fold_df is None or fold_df.empty:
        return "  (no fold rows recorded)"
    cols = [
        "fold",
        "penalizer",
        "l1_ratio",
        "n_events_train",
        "n_events_val",
        "n_canonical_labs",
        "n_selected_features",
        "c_index_val",
        "note",
    ]
    available = [c for c in cols if c in fold_df.columns]
    notes = (
        fold_df["note"].fillna("").astype(str).value_counts().head(5).to_dict()
        if "note" in fold_df.columns
        else {}
    )
    note_summary = "\n".join(f"    {n:>4}x  {note!r}" for note, n in notes.items())
    table = fold_df[available].head(20).to_string(index=False)
    return (
        "  Most common per-fold notes:\n"
        f"{note_summary}\n"
        "  First 20 fold rows:\n"
        f"{table}"
    )
