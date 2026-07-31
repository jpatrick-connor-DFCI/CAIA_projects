"""Shared Cox model-fitting and evaluation primitives."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from survival_common.helper import (
    MIN_IPCW_TIMELINE_COVERAGE,
    MIN_IPCW_VALID_HORIZONS,
    _make_survival_array,
    breslow_survival_at_horizons,
    choose_stratification_labels,
    suppress_sksurv_warnings,
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
    """Compute IPCW cumulative/dynamic AUC(t) on an estimability-filtered timeline.

    ``fixed_horizons`` is the train/validation-derived requested timeline. The
    evaluation split is used only to decide which requested points are
    estimable, never to create replacement horizons. A mean is reported only
    when at least two unique horizons and at least half of the requested
    timeline are valid, and it is always sksurv's censoring-weighted integral
    over the valid horizons — never a substitute summary. Per-horizon output
    records the support diagnostics.

    The eligible timeline is scored in one ``cumulative_dynamic_auc`` call.
    If that batch fails, each horizon is retried independently so a single
    problematic boundary point cannot erase otherwise valid estimates.
    """
    require_sksurv()

    empty_cols = [
        "horizon_quantile",
        "horizon_time_unit",
        "horizon_days",
        "auc_t",
        "n_eval",
        "n_eval_events",
        "n_cases_by_horizon",
        "n_controls_by_horizon",
        "reference_min_time_unit",
        "reference_max_time_unit",
        "evaluation_min_time_unit",
        "evaluation_max_time_unit",
        "eligible_by_support",
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

    if fixed_horizons is not None:
        horizon_times = np.asarray(fixed_horizons, dtype=float).reshape(-1)
        horizon_times = np.unique(horizon_times[np.isfinite(horizon_times) & (horizon_times > 0)])
        horizon_quantiles: tuple[float, ...] = tuple([np.nan] * len(horizon_times))
    else:
        if eval_valid.sum() == 0:
            return np.nan, pd.DataFrame(columns=empty_cols)
        event_times = eval_duration[eval_valid & (eval_event == 1)]
        event_times = event_times[np.isfinite(event_times) & (event_times > 0)]
        if len(event_times) == 0:
            return np.nan, pd.DataFrame(columns=empty_cols)
        horizon_times = np.asarray(
            [int(val) for val in np.quantile(event_times, quantiles)],
            dtype=float,
        )
        horizon_quantiles = tuple(quantiles)

    if len(horizon_times) == 0:
        return np.nan, pd.DataFrame(columns=empty_cols)

    if train_valid.sum() == 0 or eval_valid.sum() == 0:
        reason = "no_valid_reference_rows" if train_valid.sum() == 0 else "no_valid_evaluation_rows"
        rows = []
        for quantile, horizon in zip(horizon_quantiles, horizon_times):
            rows.append({
                "horizon_quantile": quantile,
                "horizon_time_unit": horizon,
                "horizon_days": horizon * float(time_unit_days),
                "auc_t": np.nan,
                "n_eval": int(eval_valid.sum()),
                "n_eval_events": int(eval_event[eval_valid].sum()),
                "n_cases_by_horizon": 0,
                "n_controls_by_horizon": 0,
                "reference_min_time_unit": np.nan,
                "reference_max_time_unit": np.nan,
                "evaluation_min_time_unit": np.nan,
                "evaluation_max_time_unit": np.nan,
                "eligible_by_support": False,
                "admin_censor_time_unit": max_time_unit,
                "note": reason,
            })
        return np.nan, pd.DataFrame(rows, columns=empty_cols)

    train_surv = _make_survival_array(train_event[train_valid], train_duration[train_valid])
    eval_surv = _make_survival_array(eval_event[eval_valid], eval_duration[eval_valid])
    eval_risk = risk_score[eval_valid]
    train_duration_valid = train_duration[train_valid]
    eval_duration_valid = eval_duration[eval_valid]
    eval_event_valid = eval_event[eval_valid]
    train_min = float(np.min(train_duration_valid))
    train_max = float(np.max(train_duration_valid))
    eval_min = float(np.min(eval_duration_valid))
    eval_max = float(np.max(eval_duration_valid))

    rows = []
    for quantile, horizon in zip(horizon_quantiles, horizon_times):
        n_cases = int(((eval_event_valid == 1) & (eval_duration_valid <= horizon)).sum())
        n_controls = int((eval_duration_valid > horizon).sum())
        support_failures = []
        if max_time_unit is not None and horizon >= float(max_time_unit):
            support_failures.append("at_or_beyond_admin_censor")
        if horizon < eval_min or horizon >= eval_max:
            support_failures.append("outside_evaluation_followup")
        if horizon >= train_max:
            support_failures.append("outside_reference_followup")
        if n_cases == 0:
            support_failures.append("no_cases_by_horizon")
        if n_controls == 0:
            support_failures.append("no_controls_beyond_horizon")
        rows.append(
            {
                "horizon_quantile": quantile,
                "horizon_time_unit": horizon,
                "horizon_days": horizon * float(time_unit_days),
                "auc_t": np.nan,
                "n_eval": int(eval_valid.sum()),
                "n_eval_events": int((eval_event_valid == 1).sum()),
                "n_cases_by_horizon": n_cases,
                "n_controls_by_horizon": n_controls,
                "reference_min_time_unit": train_min,
                "reference_max_time_unit": train_max,
                "evaluation_min_time_unit": eval_min,
                "evaluation_max_time_unit": eval_max,
                "eligible_by_support": len(support_failures) == 0,
                "admin_censor_time_unit": max_time_unit,
                "note": ";".join(support_failures),
            }
        )

    auc_df = pd.DataFrame(rows)
    eligible_mask = auc_df["eligible_by_support"].to_numpy(dtype=bool)
    if not eligible_mask.any():
        return np.nan, auc_df

    eligible_times = np.unique(
        auc_df.loc[eligible_mask, "horizon_time_unit"].to_numpy(dtype=float)
    )
    mean_auc = np.nan
    try:
        with suppress_sksurv_warnings():
            auc_values, mean_auc = cumulative_dynamic_auc(
                train_surv,
                eval_surv,
                eval_risk,
                eligible_times,
            )
    except ValueError:
        # A batched failure can be caused by one boundary horizon. Retry each
        # requested point so an isolated failure does not erase earlier valid
        # AUC estimates and their diagnostics.
        auc_by_time = {}
        for horizon in eligible_times:
            try:
                with suppress_sksurv_warnings():
                    values, _ = cumulative_dynamic_auc(
                        train_surv,
                        eval_surv,
                        eval_risk,
                        np.asarray([horizon], dtype=float),
                    )
                auc_by_time[float(horizon)] = float(np.asarray(values).reshape(-1)[0])
            except ValueError as exc:
                horizon_mask = auc_df["horizon_time_unit"].eq(horizon).to_numpy()
                auc_df.loc[horizon_mask, "note"] = f"auc_failed: {exc}"
    else:
        auc_by_time = dict(zip(eligible_times, np.asarray(auc_values, dtype=float).reshape(-1)))

    auc_df["auc_t"] = auc_df["horizon_time_unit"].map(auc_by_time).astype(float)
    valid_mask = np.isfinite(auc_df["auc_t"].to_numpy(dtype=float))
    # sksurv can return NaN without raising when every case at a horizon carries
    # zero IPCW weight; say so rather than leaving an unexplained blank.
    unexplained_nan = eligible_mask & ~valid_mask & auc_df["note"].eq("").to_numpy()
    auc_df.loc[unexplained_nan, "note"] = "auc_nan_despite_support"

    valid_times = np.unique(
        auc_df.loc[valid_mask, "horizon_time_unit"].to_numpy(dtype=float)
    )
    n_requested = int(pd.Series(horizon_times).nunique())
    coverage = len(valid_times) / n_requested if n_requested else 0.0
    if (
        len(valid_times) < MIN_IPCW_VALID_HORIZONS
        or coverage < MIN_IPCW_TIMELINE_COVERAGE
    ):
        return np.nan, _append_note(auc_df, valid_mask, "insufficient_timeline_coverage_for_mean")

    if len(valid_times) < len(eligible_times) or not np.isfinite(mean_auc):
        # A failed/NaN horizon poisons sksurv's integral; re-integrate over the
        # finite points so the mean remains the censoring-weighted estimand.
        try:
            with suppress_sksurv_warnings():
                _, mean_auc = cumulative_dynamic_auc(
                    train_surv,
                    eval_surv,
                    eval_risk,
                    valid_times,
                )
        except ValueError:
            mean_auc = np.nan

    if not np.isfinite(mean_auc):
        return np.nan, _append_note(auc_df, valid_mask, "mean_auc_not_finite")
    return float(mean_auc), auc_df


def _append_note(auc_df: pd.DataFrame, mask: np.ndarray, suffix: str) -> pd.DataFrame:
    """Append ``suffix`` to the note of the masked rows, preserving any existing text."""
    auc_df.loc[mask, "note"] = auc_df.loc[mask, "note"].map(
        lambda value: suffix if not value else f"{value};{suffix}"
    )
    return auc_df


def summarize_auc_timeline(auc_df: pd.DataFrame) -> dict[str, float | int | str]:
    """Return compact requested/eligible/valid timeline diagnostics for metrics."""
    if auc_df.empty:
        return {
            "n_requested_auc_horizons": 0,
            "n_support_eligible_auc_horizons": 0,
            "n_valid_auc_horizons": 0,
            "auc_timeline_coverage": 0.0,
            "auc_valid_min_time_unit": np.nan,
            "auc_valid_max_time_unit": np.nan,
            # Covers both "nothing requested" and "the split had no evaluable
            # rows at all"; compute_ipcw_auc_t returns an empty frame for each.
            "auc_timeline_status": "no_evaluable_timeline",
        }
    requested = int(auc_df["horizon_time_unit"].nunique())
    eligible = auc_df["eligible_by_support"].fillna(False).to_numpy(dtype=bool)
    support_eligible = int(auc_df.loc[eligible, "horizon_time_unit"].nunique())
    valid = auc_df.loc[np.isfinite(auc_df["auc_t"]), "horizon_time_unit"].drop_duplicates()
    n_valid = int(len(valid))
    coverage = n_valid / requested if requested else 0.0
    if n_valid < MIN_IPCW_VALID_HORIZONS:
        status = "insufficient_valid_horizons"
    elif coverage < MIN_IPCW_TIMELINE_COVERAGE:
        status = "insufficient_timeline_coverage"
    else:
        status = "ok"
    return {
        "n_requested_auc_horizons": requested,
        "n_support_eligible_auc_horizons": support_eligible,
        "n_valid_auc_horizons": n_valid,
        "auc_timeline_coverage": float(coverage),
        "auc_valid_min_time_unit": float(valid.min()) if n_valid else np.nan,
        "auc_valid_max_time_unit": float(valid.max()) if n_valid else np.nan,
        "auc_timeline_status": status,
    }


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
            with suppress_sksurv_warnings():
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
