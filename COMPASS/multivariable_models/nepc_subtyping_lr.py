"""Logistic regression for incident NEPC: federated risk score vs Gleason vs TP53/PTEN/RB1.

Its notebook (10_LR_for_subtyping) was removed on 2026-11-17. The LR machinery
(CV AUC, bootstrap, calibration) is still used by ``ihc_lr`` (notebook 11). For each subcohort and
ADT landmark, three predictor sets are fit as separate unpenalized logistic
regressions of incident NEPC (first NEPC after ADT + landmark):

* ``risk``    -- the frozen federated (no-DFCI) platinum-trained XGBoost score,
                 as notebook 04 scored it on the local cohort; z-scored within
                 each fit's patients, so its odds ratio is per SD.
* ``gleason`` -- the Gleason score closest to ADT + landmark among those
                 available before it (ADT clock); odds ratio per point.
* ``trio``    -- TP53/PTEN/RB1 any-alteration indicators from the specimen
                 reported closest to, and strictly before, ADT + landmark. ``tp53`` / ``pten`` / ``rb1`` are the
                 single-gene fits.

Somatic reports must be available strictly before the landmark for both cases
and controls. The outcome remains a binary label over observed follow-up.

Two populations per subcohort x landmark:

* ``available`` -- each model on every patient with its own predictor observed.
* ``matched``   -- every model on the patients with the risk score, Gleason and
                   an eligible specimen, with paired-bootstrap delta-AUC against
                   ``risk`` so the AUCs are read on the same patients.

AUC is computed from out-of-fold predictions (stratified K-fold), with a
patient bootstrap of those predictions for the 95% CI. Odds ratios, Wald CIs
and likelihood-ratio p-values come from the fit on the whole population.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from COMPASS.federated_models import federated_clinical_comparison as fcc
from survival_common import federated_endpoint_transfer as fet

ID_COL = fcc.ID_COL                     # "person_id", as in transfer_scores.csv
LOCAL_ID_COL = fcc.LOCAL_ID_COL         # "DFCI_MRN", as in the local frame and MRN lists
EVENT_COL = fet.EVENT_COL               # "event"
RISK_COL = "risk_score"
RISK_Z = "RISK_Z"
GLEASON_FEATURE = fcc.GLEASON_FEATURE   # "GLEASON_SCORE"
HAS_SEQUENCING = fcc.HAS_SEQUENCING
TRIO_FEATURES = fcc.TRIO_FEATURES       # ("TP53_ANY", "PTEN_ANY", "RB1_ANY")

# Model -> the frame columns it needs. "risk" is z-scored into RISK_Z per fit.
MODEL_SPECS = {
    "risk": (RISK_COL,),
    "gleason": (GLEASON_FEATURE,),
    "trio": TRIO_FEATURES,
    **{gene.lower(): (f"{gene}_ANY",) for gene in fcc.TRIO_GENES},
}
REFERENCE_MODEL = "risk"
POPULATIONS = ("available", "matched")
MAX_ABS_COEF = 15.0                     # |beta| beyond this on a fitted term = separation

MODEL_COLUMNS = [
    "cohort", "landmark_days", "population", "model", "predictors", "n", "events",
    "lrt_stat", "lrt_df", "p_lrt", "auc", "auc_lo", "auc_hi",
    "auc_raw_score", "auc_raw_score_lo", "auc_raw_score_hi",
    "delta_auc_vs_risk", "delta_auc_lo", "delta_auc_hi", "converged", "n_failed_folds", "note",
]
COEF_COLUMNS = [
    "cohort", "landmark_days", "population", "model", "term", "n", "events",
    "coef", "se", "odds_ratio", "or_lo", "or_hi", "p_wald", "note",
]
COUNT_COLUMNS = [
    "cohort", "landmark_days", "n_cohort_list", "n_at_risk", "events",
    "n_gleason", "events_gleason", "n_sequenced", "events_sequenced",
    "n_trio_eligible", "events_trio_eligible",
    "n_matched", "events_matched",
]
# Per-patient out-of-fold predictions: the same CV fits the AUC is computed from.
PREDICTION_KEY_COLUMNS = [ID_COL, EVENT_COL, RISK_COL, "cv_fold", "oof_logit", "oof_prob"]
PREDICTION_COLUMNS = ["cohort", "landmark_days", "population", "model", *PREDICTION_KEY_COLUMNS]
CALIBRATION_COLUMNS = ["n", "events", "observed_rate", "mean_predicted", "brier",
                       "calibration_intercept", "calibration_slope"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_transfer_scores(path, *, family: str = "xgboost_cox", config: str = "both",
                         endpoint: str = "nepc") -> pd.DataFrame:
    """One bundle's patient-level scores from notebook 04's ``transfer_scores.csv``.

    The ``endpoint`` rows hold every patient at risk at each landmark, with
    ``event`` = that endpoint after the landmark. Returns ``person_id,
    landmark_days, risk_score, event``.
    """
    remediation = f"{path}: rerun notebook 04 (04_federated_transfer.ipynb) to write it."
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing transfer scores. {remediation}") from exc
    required = {"endpoint", "model_family", "config", "landmark_days", ID_COL, RISK_COL, EVENT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Transfer scores lack columns {sorted(missing)}. {remediation}")
    keep = (df["endpoint"].astype(str).eq(endpoint) & df["model_family"].astype(str).eq(family)
            & df["config"].astype(str).eq(config))
    if "evaluation_horizon_days" in df.columns:
        keep &= df["evaluation_horizon_days"].isna()
    out = df.loc[keep, [ID_COL, "landmark_days", RISK_COL, EVENT_COL]].copy()
    if out.empty:
        raise ValueError(
            f"No {family}/{config} scores for endpoint {endpoint!r}. {remediation}")
    out[ID_COL] = pd.to_numeric(out[ID_COL], errors="raise").astype(int)
    out["landmark_days"] = pd.to_numeric(out["landmark_days"], errors="raise").astype(int)
    out[EVENT_COL] = pd.to_numeric(out[EVENT_COL], errors="raise").astype(int)
    out[RISK_COL] = pd.to_numeric(out[RISK_COL], errors="coerce")
    dup = out.duplicated([ID_COL, "landmark_days"])
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} duplicate patient x landmark rows for {family}/{config}/{endpoint}; "
            "expected one score per patient per landmark.")
    return out.reset_index(drop=True)


def cohort_members(paths: dict) -> dict:
    """Cohort -> set of MRNs from its MRN list, or None (no restriction) for a None path."""
    members = {}
    for cohort, path in paths.items():
        if path is None:
            members[cohort] = None
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MRN list for cohort {cohort!r}: {path}. Build it in 01_preprocessing.ipynb.")
        mrns = pd.read_csv(path, usecols=[LOCAL_ID_COL])[LOCAL_ID_COL]
        members[cohort] = set(pd.to_numeric(mrns, errors="coerce").dropna().astype(int))
    return members


def build_lr_frame(scores: pd.DataFrame, gleason_features: pd.DataFrame,
                   somatic_features: pd.DataFrame) -> pd.DataFrame:
    """Join pre-landmark clinical features to scored patient x landmark rows.

    ``somatic_features`` must come from fcc.adt_anchored_somatic: it selects
    reports available strictly before each ADT landmark, with the same cutoff
    for cases and controls, and codes uncalled genes in eligible reports as 0.
    Patients without an eligible report retain missing trio values.
    """
    keys = [ID_COL, "landmark_days"]
    frame = scores[[*keys, RISK_COL, EVENT_COL]].merge(
        gleason_features[[*keys, GLEASON_FEATURE]],
        on=keys, how="left", validate="one_to_one")
    somatic = somatic_features[[*keys, fcc.ADT_HAS_SEQUENCING, *fcc.ADT_TRIO_FEATURES]].rename(
        columns={fcc.ADT_HAS_SEQUENCING: HAS_SEQUENCING,
                 **dict(zip(fcc.ADT_TRIO_FEATURES, TRIO_FEATURES))})
    frame = frame.merge(somatic, on=keys, how="left", validate="one_to_one")
    frame[HAS_SEQUENCING] = frame[HAS_SEQUENCING].eq(True)
    frame.loc[~frame[HAS_SEQUENCING], list(TRIO_FEATURES)] = np.nan
    return frame


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #
def _loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _irls(y: np.ndarray, A: np.ndarray, *, max_iter: int = 100, tol: float = 1e-8):
    """Newton steps with backtracking; returns (beta, inverse Hessian, converged)."""
    beta = np.zeros(A.shape[1])
    ybar = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    beta[0] = np.log(ybar / (1 - ybar))
    converged = False
    cov = None
    for _ in range(max_iter):
        p = expit(A @ beta)
        w = p * (1 - p)
        H = A.T @ (A * w[:, None])
        try:
            step = np.linalg.solve(H, A.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break
        # Test the undamped step: a tiny line-search step is not convergence. At the optimum float
        # rounding can hold the step just above tol forever, so a small step (1e-6) with a ~zero
        # score vector also counts; under separation the score vanishes but steps stay O(1).
        max_step = np.max(np.abs(step))
        score = np.max(np.abs(A.T @ (y - p)))
        if max_step < tol or (max_step < 1e-6 and score < 1e-6 * max(1.0, len(y))):
            beta = beta + step
            converged = True
            break
        eta = A @ beta
        loss = np.sum(np.logaddexp(0, eta) - y * eta)
        scale = 1.0
        for _ in range(50):
            candidate = beta + scale * step
            eta_new = A @ candidate
            loss_new = np.sum(np.logaddexp(0, eta_new) - y * eta_new)
            if np.isfinite(loss_new) and loss_new <= loss:
                beta = candidate
                break
            scale *= 0.5
        else:
            break
    p = expit(A @ beta)
    H = A.T @ (A * (p * (1 - p))[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full_like(H, np.nan)
    return beta, cov, converged


def fit_logistic(y, X: pd.DataFrame) -> dict:
    """Unpenalized logistic regression with Wald and likelihood-ratio tests.

    A predictor that is constant in ``X`` cannot be estimated; it is dropped
    from the fit and reported with NaN estimates. Non-convergence or a fitted
    |coefficient| above MAX_ABS_COEF is flagged as (quasi-)separation and its
    estimates are reported NaN, rather than as a meaningless huge odds ratio.

    Returns ``{"terms": DataFrame, "beta": {term: coef (0 for dropped)},
    "lrt_stat", "lrt_df", "p_lrt", "converged", "note"}``.
    """
    y = np.asarray(y, dtype=float)
    X = X.astype(float)
    names = list(X.columns)
    varying = [c for c in names if X[c].nunique(dropna=False) > 1]
    notes = []
    if len(varying) < len(names):
        notes.append("constant: " + ",".join(c for c in names if c not in varying))
    A = np.column_stack([np.ones(len(y)), X[varying].to_numpy()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beta, cov, converged = _irls(y, A)
    separated = (not converged) or bool(np.any(np.abs(beta[1:]) > MAX_ABS_COEF))
    if separated:
        notes.append("separation or non-convergence; estimates suppressed")

    ll = _loglik(y, expit(A @ beta))
    ll0 = _loglik(y, np.full(len(y), y.mean()))
    df = len(varying)
    lrt = max(2 * (ll - ll0), 0.0) if df and not separated else np.nan
    p_lrt = float(stats.chi2.sf(lrt, df)) if df and not separated else np.nan

    z95 = stats.norm.ppf(0.975)
    rows = []
    est = dict(zip(varying, beta[1:]))
    se = dict(zip(varying, np.sqrt(np.clip(np.diag(cov)[1:], 0, None))))
    for name in names:
        if name not in est or separated:
            rows.append({"term": name, "coef": np.nan, "se": np.nan, "odds_ratio": np.nan,
                         "or_lo": np.nan, "or_hi": np.nan, "p_wald": np.nan})
            continue
        b, s = float(est[name]), float(se[name])
        rows.append({"term": name, "coef": b, "se": s, "odds_ratio": np.exp(b),
                     "or_lo": np.exp(b - z95 * s), "or_hi": np.exp(b + z95 * s),
                     "p_wald": float(2 * stats.norm.sf(abs(b / s))) if s > 0 else np.nan})
    return {
        "terms": pd.DataFrame(rows),
        "beta": {"_intercept": float(beta[0]), **{n: float(est.get(n, 0.0)) for n in names}},
        "lrt_stat": lrt, "lrt_df": df, "p_lrt": p_lrt,
        "converged": bool(converged and not separated), "note": "; ".join(notes),
    }


def _linear_predictor(beta: dict, X: pd.DataFrame) -> np.ndarray:
    return beta["_intercept"] + X.astype(float).to_numpy() @ np.array([beta[c] for c in X.columns])


def cv_folds(y, *, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Stratified K-fold test-fold index per row (-1 when fewer than 2 folds are possible).

    Folds are capped by the minority class count. ``oof_predictions`` uses these
    folds, so a saved ``cv_fold`` names the fit that produced each prediction.
    """
    y = np.asarray(y, dtype=int)
    k = int(min(n_folds, y.sum(), len(y) - y.sum()))
    out = np.full(len(y), -1, dtype=int)
    if k < 2:
        return out
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splits.split(np.zeros(len(y)), y)):
        out[test] = fold
    return out


def oof_prediction_frame(pop: pd.DataFrame, y, oof, *, n_folds: int, seed: int,
                         tags: dict) -> pd.DataFrame:
    """Per-patient OOF rows: linear predictor, probability (expit) and CV fold.

    All-NaN predictions (failed or impossible CV) are kept as NaN rows so the
    patient set matches the fit population; they are dropped from calibration.
    """
    oof = np.asarray(oof, dtype=float)
    out = pd.DataFrame({ID_COL: pop[ID_COL].to_numpy(), EVENT_COL: np.asarray(y, dtype=int),
                        RISK_COL: pop[RISK_COL].to_numpy(dtype=float),
                        "cv_fold": cv_folds(y, n_folds=n_folds, seed=seed),
                        "oof_logit": oof, "oof_prob": expit(oof)})
    for key, value in reversed(list(tags.items())):
        out.insert(0, key, value)
    return out


def calibration_summary(predictions: pd.DataFrame, by) -> pd.DataFrame:
    """Brier score and logistic recalibration of OOF predictions, per group.

    ``calibration_intercept`` is calibration-in-the-large (the logit offset with
    the slope fixed at 1, 0 = mean prediction matches the event rate);
    ``calibration_slope`` is the coefficient of ``y ~ oof_logit`` (1 = ideal,
    <1 = predictions too extreme). Rows with a missing prediction are dropped.
    """
    by = list(by)
    rows = []
    for key, g in predictions.dropna(subset=["oof_logit"]).groupby(by, sort=False):
        y = g[EVENT_COL].to_numpy(dtype=float)
        lp = g["oof_logit"].to_numpy(dtype=float)
        p = expit(lp)
        slope_fit = fit_logistic(y, pd.DataFrame({"oof_logit": lp}))
        slope = slope_fit["terms"]["coef"].iloc[0]
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     "n": len(y), "events": int(y.sum()), "observed_rate": float(y.mean()),
                     "mean_predicted": float(p.mean()), "brier": float(np.mean((p - y) ** 2)),
                     "calibration_intercept": _offset_intercept(y, lp),
                     "calibration_slope": float(slope)})
    return pd.DataFrame(rows).reindex(columns=[*by, *CALIBRATION_COLUMNS])


def _offset_intercept(y: np.ndarray, offset: np.ndarray, *, max_iter: int = 100,
                      tol: float = 1e-10) -> float:
    """Intercept a in logit P(y) = a + offset (Newton steps on one parameter)."""
    if y.min() == y.max():
        return np.nan
    a = 0.0
    for _ in range(max_iter):
        p = expit(a + offset)
        step = np.sum(y - p) / max(np.sum(p * (1 - p)), 1e-12)
        a += step
        if abs(step) < tol:
            break
    return float(a)


def calibration_bins(predictions: pd.DataFrame, by, *, n_bins: int = 5) -> pd.DataFrame:
    """Observed event rate vs mean OOF probability per quantile bin of the prediction."""
    by = list(by)
    rows = []
    for key, g in predictions.dropna(subset=["oof_prob"]).groupby(by, sort=False):
        bins = pd.qcut(g["oof_prob"].rank(method="first"), q=min(n_bins, len(g)), labels=False)
        for b, gb in g.groupby(bins):
            rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                         "bin": int(b) + 1, "n": len(gb), "events": int(gb[EVENT_COL].sum()),
                         "mean_predicted": float(gb["oof_prob"].mean()),
                         "observed_rate": float(gb[EVENT_COL].mean())})
    return pd.DataFrame(rows).reindex(
        columns=[*by, "bin", "n", "events", "mean_predicted", "observed_rate"])


def oof_predictions(y, X: pd.DataFrame, *, n_folds: int = 5, seed: int = 0,
                    return_diagnostics: bool = False):
    """Out-of-fold linear predictors from stratified K-fold logistic fits.

    Folds are capped by the minority class count; with fewer than 2 of either
    class, returns all-NaN. Any failed fold suppresses the entire prediction
    vector so AUC and paired comparisons cannot silently use different subsets.
    With return_diagnostics, also returns the number of failed folds.
    """
    y = np.asarray(y, dtype=int)
    folds = cv_folds(y, n_folds=n_folds, seed=seed)
    out = np.full(len(y), np.nan)
    if folds.max() < 1:
        return (out, 0) if return_diagnostics else out
    failed = 0
    for fold in range(folds.max() + 1):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        fit = fit_logistic(y[train], X.iloc[train])
        if not fit["converged"]:
            failed += 1
            continue
        prediction = _linear_predictor(fit["beta"], X.iloc[test])
        if not np.isfinite(prediction).all():
            failed += 1
            continue
        out[test] = prediction
    if failed:
        out[:] = np.nan
    return (out, failed) if return_diagnostics else out


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    if not np.isfinite(score).all() or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def bootstrap_auc(y, scores: dict, *, n_bootstrap: int = 200, seed: int = 0) -> dict:
    """Patient-bootstrap AUC draws; every score shares the same resamples (paired)."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    draws = {k: np.full(n_bootstrap, np.nan) for k in scores}
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        for k, s in scores.items():
            draws[k][b] = _auc(y[idx], np.asarray(s)[idx])
    return draws


def _ci(draws: np.ndarray) -> tuple[float, float]:
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return np.nan, np.nan
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _design(pop: pd.DataFrame, model: str) -> pd.DataFrame:
    cols = list(MODEL_SPECS[model])
    X = pop[cols].astype(float)
    if model == REFERENCE_MODEL:
        sd = X[RISK_COL].std(ddof=0)
        X = pd.DataFrame({RISK_Z: (X[RISK_COL] - X[RISK_COL].mean()) / sd if sd > 0 else 0.0},
                         index=pop.index)
    return X


def _evaluate_population(pop: pd.DataFrame, models, tags: dict, *, min_events: int,
                         n_folds: int, n_bootstrap: int, seed: int, paired: bool,
                         predict_models=()):
    """Fit every model on ``pop`` (one shared population) or each on its own available cases.

    Returns ``(model_rows, coef_rows, prediction_frames)``; the last holds the
    per-patient OOF predictions of the fitted models named in ``predict_models``.
    """
    model_rows, coef_rows, oof, prediction_frames = [], [], {}, []
    pops, raw = {}, {}
    for model in models:
        sub = pop if paired else pop.dropna(subset=list(MODEL_SPECS[model]))
        pops[model] = sub
        y = sub[EVENT_COL].to_numpy(dtype=int)
        n, events = len(y), int(y.sum())
        base = {**tags, "model": model, "n": n, "events": events}
        if events < min_events or n - events < min_events:
            model_rows.append({**base, "predictors": "+".join(MODEL_SPECS[model]),
                               "note": f"fewer than {min_events} events or non-events; not fit"})
            continue
        X = _design(sub, model)
        fit = fit_logistic(y, X)
        oof[model], failed = oof_predictions(
            y, X, n_folds=n_folds, seed=seed, return_diagnostics=True)
        if model in predict_models:
            prediction_frames.append(oof_prediction_frame(
                sub, y, oof[model], n_folds=n_folds, seed=seed, tags={**tags, "model": model}))
        note = fit["note"]
        if not np.isfinite(oof[model]).all():
            cv_note = (f"{failed} failed CV folds; AUC and delta-AUC suppressed" if failed
                       else "fewer than 2 CV folds; AUC and delta-AUC suppressed")
            note = "; ".join(filter(None, [note, cv_note]))
        row = {**base, "predictors": "+".join(X.columns), "lrt_stat": fit["lrt_stat"],
               "lrt_df": fit["lrt_df"], "p_lrt": fit["p_lrt"],
               "auc": _auc(y, oof[model]), "converged": fit["converged"],
               "n_failed_folds": failed, "note": note}
        if model == REFERENCE_MODEL:
            # Untransformed frozen score: no fitting, no sign flip, so no CV is needed.
            raw[model] = sub[RISK_COL].to_numpy(dtype=float)
            row["auc_raw_score"] = _auc(y, raw[model])
        model_rows.append(row)
        coef_rows.extend({**base, **row, "note": fit["note"]}
                         for row in fit["terms"].to_dict("records"))

    if n_bootstrap and oof:
        if paired:
            y = pop[EVENT_COL].to_numpy(dtype=int)
            draws = bootstrap_auc(y, oof, n_bootstrap=n_bootstrap, seed=seed)
        else:
            draws = {}
            for model, score in oof.items():
                y = pops[model][EVENT_COL].to_numpy(dtype=int)
                draws.update(bootstrap_auc(y, {model: score}, n_bootstrap=n_bootstrap, seed=seed))
        for row in model_rows:
            m = row["model"]
            if m not in draws:
                continue
            row["auc_lo"], row["auc_hi"] = _ci(draws[m])
            if paired and m != REFERENCE_MODEL and REFERENCE_MODEL in draws:
                row["delta_auc_vs_risk"] = row["auc"] - next(
                    r["auc"] for r in model_rows if r["model"] == REFERENCE_MODEL)
                row["delta_auc_lo"], row["delta_auc_hi"] = _ci(draws[m] - draws[REFERENCE_MODEL])

    if n_bootstrap and raw:
        # Same seed and n as the OOF bootstrap, so the resamples are paired with it.
        for model, score in raw.items():
            y = pops[model][EVENT_COL].to_numpy(dtype=int)
            draws = bootstrap_auc(y, {model: score}, n_bootstrap=n_bootstrap, seed=seed)
            for row in model_rows:
                if row["model"] == model:
                    row["auc_raw_score_lo"], row["auc_raw_score_hi"] = _ci(draws[model])
    return model_rows, coef_rows, prediction_frames


def _counts(sub: pd.DataFrame, tags: dict, n_list) -> dict:
    ev = sub[EVENT_COL].eq(1)
    gleason = sub[GLEASON_FEATURE].notna()
    trio = sub[list(TRIO_FEATURES)].notna().all(axis=1)
    matched = sub[RISK_COL].notna() & gleason & trio
    return {**tags, "n_cohort_list": n_list, "n_at_risk": len(sub), "events": int(ev.sum()),
            "n_gleason": int(gleason.sum()), "events_gleason": int((gleason & ev).sum()),
            "n_sequenced": int(sub[HAS_SEQUENCING].sum()),
            "events_sequenced": int((sub[HAS_SEQUENCING] & ev).sum()),
            "n_trio_eligible": int(trio.sum()), "events_trio_eligible": int((trio & ev).sum()),
            "n_matched": int(matched.sum()), "events_matched": int((matched & ev).sum())}


def run_lr(frame: pd.DataFrame, members: dict, landmarks, *, models=tuple(MODEL_SPECS),
           min_events: int = 5, n_folds: int = 5, n_bootstrap: int = 200, seed: int = 0,
           return_predictions: bool = False, prediction_models=(REFERENCE_MODEL,),
           prediction_populations=("available",)):
    """Fit every model per cohort x landmark x population.

    ``members`` maps cohort -> MRN set (None = every scored patient). Returns
    ``(models, coefficients, counts)`` DataFrames; with ``return_predictions``,
    also a fourth: per-patient out-of-fold predictions (``PREDICTION_COLUMNS``)
    for ``prediction_models`` in ``prediction_populations``. Each patient's
    prediction comes from the CV fit that held that patient out, the same
    predictions the AUC uses; models below ``min_events`` contribute no rows.
    """
    model_rows, coef_rows, count_rows, prediction_frames = [], [], [], []
    for cohort, ids in members.items():
        for lm in landmarks:
            sub = frame[frame["landmark_days"].eq(int(lm))]
            if ids is not None:
                sub = sub[sub[ID_COL].isin(ids)]
            tags = {"cohort": cohort, "landmark_days": int(lm)}
            count_rows.append(_counts(sub, tags, None if ids is None else len(ids)))
            matched = sub.dropna(subset=[RISK_COL, GLEASON_FEATURE, *TRIO_FEATURES])
            for population, pop, paired in (("available", sub, False), ("matched", matched, True)):
                predict = (tuple(prediction_models)
                           if return_predictions and population in prediction_populations else ())
                m, c, p = _evaluate_population(
                    pop, models, {**tags, "population": population}, min_events=min_events,
                    n_folds=n_folds, n_bootstrap=n_bootstrap, seed=seed, paired=paired,
                    predict_models=predict)
                model_rows += m
                coef_rows += c
                prediction_frames += p
    out = (pd.DataFrame(model_rows).reindex(columns=MODEL_COLUMNS),
           pd.DataFrame(coef_rows).reindex(columns=COEF_COLUMNS),
           pd.DataFrame(count_rows).reindex(columns=COUNT_COLUMNS))
    if not return_predictions:
        return out
    predictions = (pd.concat(prediction_frames, ignore_index=True) if prediction_frames
                   else pd.DataFrame())
    return (*out, predictions.reindex(columns=PREDICTION_COLUMNS))
