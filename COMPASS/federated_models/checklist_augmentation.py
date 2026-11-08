"""Prespecified Figure 6 augmentation: nested, patient-grouped local development.

Frozen scores are never tuned. Cox models use training-fold median age, scaling,
ridge selection and baseline hazards. Their OOF scores are five-year net risks,
not pooled linear predictors. Evaluation uses the manuscript's day-scale
cumulative/dynamic AUC; censoring is estimated on the evaluation population.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from sklearn.model_selection import KFold, StratifiedKFold

from survival_common import federated_endpoint_transfer as fet
from COMPASS.multivariable_models.predictive_checklists.survival import (
    CensoringDistribution, horizon_targets,
)

GENES = ("TP53", "RB1", "PTEN")
ID, TIME, EVENT = "person_id", "duration_days", "event"
LABELS = {
    "clinical": "Age + Gleason", "age_labs": "Age + lab count",
    "augmented": "Age + Gleason + lab count", "clinical_no_tsg": "Age + Gleason",
    "L": "Frozen lab count", "B": "Frozen lab flag (M of N)", "gleason_binary": "Gleason ≥8",
    "gleason_score": "Lab count + Gleason ≥8", "gleason_ge_m": "≥M of N+1", "gleason_ge_m_plus1": "≥M+1 of N+1",
    "gleason_tsg_score": "Lab count + Gleason ≥8 + any TSG", "gleason_tsg_two_genes": "Lab count + Gleason ≥8 + ≥2 TSG",
}


@dataclass(frozen=True)
class Settings:
    schema_version: int = 1
    landmark: int = 90
    horizon: int = 1825
    seed: int = 20261107
    outer_folds: int = 5
    inner_folds: int = 3
    penalties: tuple = (0.01, 0.1, 1.0, 10.0)
    min_events: int = 5
    min_cases: int = 5
    min_controls: int = 10
    min_g: float = 0.05
    bootstrap: int = 1000
    bootstrap_mode: str = "refit"
    min_bootstrap_fraction: float = 0.9
    cross_endpoint_transfer: bool = True

    def __post_init__(self):
        if self.landmark not in (0, 90, 180) or self.horizon != 1825:
            raise ValueError("Use landmarks 0/90/180 with the fixed 1825-day horizon; day 90 is primary")
        if self.bootstrap_mode not in ("refit", "conditional"):
            raise ValueError("bootstrap_mode must be refit or conditional")
        if self.outer_folds < 2 or self.inner_folds < 2 or self.bootstrap < 0:
            raise ValueError("Invalid folds or bootstrap count")
        if not self.penalties or any(p <= 0 for p in self.penalties):
            raise ValueError("Ridge strengths must be positive")


def counts(d, cfg):
    return dict(n_patients=d[ID].nunique(), n_rows=len(d), n_events=int(d[EVENT].sum()),
                n_events_horizon=int((d[EVENT].eq(1) & d[TIME].le(cfg.horizon)).sum()),
                n_followed_beyond_horizon=int(d[TIME].gt(cfg.horizon).sum()))


def targets(d, cfg, training=None):
    source = d if training is None else training
    g = CensoringDistribution.fit(source[TIME], source[EVENT])
    return horizon_targets(d[TIME], d[EVENT], cfg.horizon, g, cfg.min_g)


def auc(d, score, cfg):
    if len(d) == 0 or not np.isfinite(score).all():
        return np.nan
    # Explicit positivity gate, in addition to the unchanged manuscript estimator.
    try:
        y, _, _ = targets(d, cfg)
        if not (y == 1).any() or not (y == 0).any():
            return np.nan
    except ValueError:
        return np.nan
    return fet.fixed_horizon_mean_auc(d[TIME], d[EVENT], score, [cfg.horizon])


def patient_folds(frame, n_folds, seed):
    """One assignment per original ID, including all copies in a bootstrap draw.

    Joint platinum/AVPC event strata when feasible; otherwise event-any strata,
    then unstratified grouped folds. The fallback is recorded, never hidden.
    """
    pt = frame.pivot_table(index=ID, columns="endpoint", values=EVENT, aggfunc="max").fillna(0)
    pt = pt.sort_index()
    if len(pt) < n_folds:
        raise ValueError(f"Only {len(pt)} distinct patients for {n_folds} folds")
    strata = pt.reindex(columns=["platinum", "avpc"], fill_value=0).astype(int).astype(str).agg("/".join, axis=1)
    method = "joint platinum/AVPC event"
    if strata.value_counts().min() < n_folds:
        strata = pt.max(axis=1).astype(int)
        method = "any event (joint strata too small)"
    if strata.value_counts().min() < n_folds:
        splitter = KFold(n_folds, shuffle=True, random_state=seed)
        method = "unstratified (event strata too small)"
    else:
        splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    result = pd.Series(index=pt.index, dtype=int, name="fold")
    for fold, (_, test) in enumerate(splitter.split(pt, strata)):
        result.iloc[test] = fold
    return result.astype(int), method


def design_models(genomic=False, gleason="gleason"):
    clinical = ["age", gleason, *GENES] if genomic else ["age", gleason]
    models = {"clinical": clinical, "age_labs": ["age", "L"], "augmented": [*clinical, "L"]}
    if genomic:
        models["clinical_no_tsg"] = ["age", gleason]
    return models


def fit_cox(train, columns, penalty, cfg):
    y, _, _ = targets(train, cfg)
    if (train[EVENT].sum() < cfg.min_events or (y == 1).sum() < cfg.min_cases
            or (y == 0).sum() < cfg.min_controls):
        raise ValueError("Insufficient training events / five-year cases / controls")
    x = train[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.drop(columns=["age"], errors="ignore").isna().any().any():
        raise ValueError("Unavailable pathology/genomics/lab scores cannot be imputed")
    medians = x.median()
    if medians.isna().any():
        raise ValueError("No training observations for a covariate")
    x = x.fillna(medians)
    center, scale = x.mean(), x.std(ddof=0)
    active = scale.index[scale > 1e-12].tolist()
    if not active:
        raise ValueError("All training covariates are constant")
    z = ((x[active] - center[active]) / scale[active]).copy()
    z[TIME], z[EVENT] = train[TIME].to_numpy(), train[EVENT].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = CoxPHFitter(penalizer=float(penalty), l1_ratio=0).fit(z, TIME, EVENT)
    return model, medians, center, scale, active


def predict_cox(fit, frame, cfg):
    model, medians, center, scale, active = fit
    x = frame[active].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.drop(columns=["age"], errors="ignore").isna().any().any():
        raise ValueError("Unavailable validation pathology/genomics/lab scores cannot be imputed")
    x = x.fillna(medians[["age"]] if "age" in medians else pd.Series(dtype=float))
    z = (x - center[active]) / scale[active]
    pred = 1 - model.predict_survival_function(z, times=[cfg.horizon]).iloc[0].to_numpy()
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite Cox prediction")
    return pred


def tune_cox(train, columns, cfg, seed):
    assignments, stratification = patient_folds(train, cfg.inner_folds, seed)
    fold = train[ID].map(assignments)
    scores, diagnostics = [], []
    for penalty in cfg.penalties:
        losses = []
        for k in range(cfg.inner_folds):
            tr, va = train.loc[fold.ne(k)], train.loc[fold.eq(k)]
            try:
                fit = fit_cox(tr, columns, penalty, cfg)
                risk = predict_cox(fit, va, cfg)
                y, weights, _ = targets(va, cfg, training=tr)
                if not (y == 1).any() or not (y == 0).any():
                    raise ValueError("Inner validation has no five-year cases or controls")
                loss = float(np.sum(weights * (np.maximum(y, 0) - risk) ** 2) / len(va))
                losses.append(loss)
                status = "ok"
            except (ValueError, ArithmeticError, ConvergenceError, ConvergenceWarning, np.linalg.LinAlgError) as error:
                losses.append(np.nan)
                status = str(error)
            diagnostics.append(dict(penalty=penalty, inner_fold=k, status=status,
                                    stratification=stratification))
        scores.append(np.mean(losses) if np.isfinite(losses).all() else np.inf)
    if not np.isfinite(scores).any():
        raise ValueError("All ridge candidates failed at least one inner fold")
    # Fixed order resolves ties, never held-out performance.
    penalty = cfg.penalties[int(np.argmin(scores))]
    return fit_cox(train, columns, penalty, cfg), penalty, diagnostics


def workload_thresholds(train_scores, flag_rate):
    """Attainable rates bracketing the training frozen flag rate, no outcome use."""
    values = np.asarray(train_scores, float)
    thresholds = np.r_[np.unique(values), np.inf]
    rates = np.array([(values >= t).mean() for t in thresholds])
    conservative = np.flatnonzero(rates <= flag_rate + 1e-12)[0]
    liberal = np.flatnonzero(rates >= flag_rate - 1e-12)
    liberal = liberal[-1] if len(liberal) else 0
    chosen = min((conservative, liberal), key=lambda i: (abs(rates[i] - flag_rate), rates[i]))
    return {"matched": thresholds[chosen], "lower_workload": thresholds[conservative],
            "upper_workload": thresholds[liberal]}


def checklist_size(d):
    """A cell carries its own frozen M/N; legacy day-90 frames use 2 of 3."""
    if ("frozen_M" in d) != ("frozen_N" in d):
        raise ValueError("Both frozen_M and frozen_N are required")
    m, n = 2, 3
    if "frozen_M" in d:
        if d[["frozen_M", "frozen_N"]].isna().any().any() or len(d[["frozen_M", "frozen_N"]].drop_duplicates()) != 1:
            raise ValueError("Do not mix frozen checklists or landmarks in one comparison")
        m, n = d[["frozen_M", "frozen_N"]].iloc[0]
    if not (1 <= m <= n and int(m) == m and int(n) == n):
        raise ValueError("Invalid frozen M-of-N checklist")
    return int(m), int(n)


def frozen_predictions(d, genomic):
    m, n = checklist_size(d)
    if (d[["gleason", "L", "B"]].isna().any().any() or not d.L.isin(range(n + 1)).all()
            or not d.B.eq(d.L.ge(m).astype(int)).all()):
        raise ValueError("Frozen scores require available Gleason, L=0..N and B=I(L≥M)")
    if genomic and not d[list(GENES)].isin([0, 1]).all().all():
        raise ValueError("Every gene must have an assessable 0/1 call; unknown is not negative")
    g = d.gleason.ge(8).astype(int)
    out = {"L": d.L.to_numpy(), "B": d.B.to_numpy(), "gleason_binary": g.to_numpy(),
           "gleason_score": (d.L + g).to_numpy(), "gleason_ge_m": (d.L + g).ge(m).astype(int).to_numpy(),
           "gleason_ge_m_plus1": (d.L + g).ge(m + 1).astype(int).to_numpy()}
    if genomic:
        count = d[list(GENES)].sum(axis=1)
        out.update(gleason_tsg_score=(d.L + g + count.ge(1)).to_numpy(),
                   gleason_tsg_two_genes=(d.L + g + count.ge(2)).to_numpy())
    return out


def cross_validate(frame, cfg, genomic=False, gleason="gleason", seed=None):
    """Long predictions, fit diagnostics and fold assignments. Failures stay NaN.

    All models in an endpoint retain the entire endpoint risk set. A failed
    fold invalidates its model estimate rather than improving it by exclusion.
    """
    seed = cfg.seed if seed is None else seed
    folds, stratification = patient_folds(frame, cfg.outer_folds, seed)
    frame = frame.copy().reset_index(drop=True)
    frame["fold"] = frame[ID].map(folds)
    rows, logs = [], []
    for endpoint, d in frame.groupby("endpoint", sort=False):
        d = d.copy()
        frozen = frozen_predictions(d, genomic)
        scores = {**frozen, **{m: np.full(len(d), np.nan) for m in design_models(genomic, gleason)
                              if endpoint != "nepc"}}
        if cfg.cross_endpoint_transfer and endpoint != "platinum":
            scores.update({m + "_platinum_transfer": np.full(len(d), np.nan)
                           for m in design_models(genomic, gleason)})
        flags = {}
        for m in ("B", "gleason_binary", "gleason_ge_m", "gleason_ge_m_plus1"):
            flags[(m, "fixed")] = scores[m]
        for k in range(cfg.outer_folds):
            held = d["fold"].eq(k).to_numpy()
            train = d.loc[~held]
            target_rate = train.B.mean()
            for m in ("L", "gleason_score", "gleason_tsg_score", "gleason_tsg_two_genes"):
                if m not in scores or train.empty:
                    continue
                for operating, cutoff in workload_thresholds(scores[m][~held], target_rate).items():
                    flags.setdefault((m, operating), np.full(len(d), np.nan))[held] = scores[m][held] >= cutoff
                    logs.append(dict(endpoint=endpoint, model=m, fold=k, status="ok", operating_point=operating,
                                     threshold=cutoff, target_flag_rate=target_rate, penalty=np.nan))
            tasks = [(m, cols, endpoint) for m, cols in design_models(genomic, gleason).items() if endpoint != "nepc"]
            if cfg.cross_endpoint_transfer and endpoint != "platinum":
                tasks += [(m + "_platinum_transfer", cols, "platinum") for m, cols in design_models(genomic, gleason).items()]
            for m, columns, training_endpoint in tasks:
                tr = frame.loc[frame.endpoint.eq(training_endpoint) & frame["fold"].ne(k)]
                log = dict(endpoint=endpoint, training_endpoint=training_endpoint, model=m, fold=k,
                           stratification=stratification, n_train=tr[ID].nunique(), n_test=int(held.sum()))
                try:
                    fit, penalty, inner = tune_cox(tr, columns, cfg, seed + k)
                    scores[m][held] = predict_cox(fit, d.loc[held], cfg)
                    train_risk = predict_cox(fit, tr, cfg)
                    for operating, cutoff in workload_thresholds(train_risk, tr.B.mean()).items():
                        flags.setdefault((m, operating), np.full(len(d), np.nan))[held] = scores[m][held] >= cutoff
                        logs.append(dict(**log, status="ok", penalty=penalty, operating_point=operating,
                                         threshold=cutoff, target_flag_rate=tr.B.mean()))
                    logs.extend({**log, **item, "phase": "inner"} for item in inner)
                except (ValueError, ArithmeticError, ConvergenceError, ConvergenceWarning, np.linalg.LinAlgError) as error:
                    logs.append(dict(**log, status=str(error), penalty=np.nan, phase="outer"))
        meta = [ID, "endpoint", TIME, EVENT, "fold", *[c for c in ("frozen_M", "frozen_N") if c in d]]
        for m, score in scores.items():
            block = d[meta].copy()
            block["model"], block["score"] = m, score
            block["score_kind"] = "frozen" if m in frozen else "local_net_risk"
            for (fm, operating), values in flags.items():
                if fm == m:
                    block["flag_" + operating] = values
            rows.append(block)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(logs), folds


def screening(d, flag, cfg):
    if len(d) == 0 or not np.isfinite(flag).all():
        raise ValueError("Missing held-out flags")
    y, w, _ = targets(d, cfg)
    positive = np.asarray(flag) > 0
    tp, fn = w[(y == 1) & positive].sum(), w[(y == 1) & ~positive].sum()
    fp, tn = w[(y == 0) & positive].sum(), w[(y == 0) & ~positive].sum()
    ratio = lambda a, b: float(a / b) if b > 0 else np.nan
    return dict(flag_rate=positive.mean(), sensitivity=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp),
                ppv=ratio(tp, tp + fp), npv=ratio(tn, tn + fn))


def evaluate_predictions(predictions, cfg):
    metrics, screen, calibration = [], [], []
    for (endpoint, model), d in predictions.groupby(["endpoint", "model"], sort=False):
        value = auc(d, d.score.to_numpy(), cfg)
        base = dict(endpoint=endpoint, model=model, **counts(d, cfg))
        metrics.append(dict(**base, auc=value, n_missing_predictions=int(d.score.isna().sum()),
                            status="ok" if np.isfinite(value) else "unsupported or failed folds"))
        for col in [c for c in d if c.startswith("flag_") and d[c].notna().any()]:
            try:
                values, status = screening(d, d[col].to_numpy(), cfg), "ok"
            except ValueError as error:
                values, status = {}, str(error)
            screen.append(dict(**base, operating_point=col[5:], **values, status=status))
        if d.score_kind.eq("local_net_risk").all() and d.score.notna().all():
            try:
                y, w, _ = targets(d, cfg)
                brier = np.sum(w * (np.maximum(y, 0) - d.score.to_numpy()) ** 2) / len(d)
                # Fixed probability bins; no fitting or selection on held-out outcomes.
                bins = pd.cut(d.score, [-np.inf, .05, .1, .2, .4, .6, .8, np.inf])
                for label in bins.cat.categories:
                    group = d.loc[bins.eq(label)]
                    risk, lo, hi = net_risk(group, cfg.horizon)
                    calibration.append(dict(**base, bin=str(label), n_bin=len(group), predicted=group.score.mean(),
                                            observed=risk, observed_lo=lo, observed_hi=hi, brier=brier))
            except ValueError:
                pass
    return pd.DataFrame(metrics), pd.DataFrame(screen), pd.DataFrame(calibration)


def contrasts(metrics):
    rows = []
    for endpoint, d in metrics.groupby("endpoint", sort=False):
        values = d.set_index("model").auc
        pairs = [("augmented", "clinical"), ("augmented", "age_labs"),
                 ("clinical", "clinical_no_tsg"), ("L", "B"),
                 ("gleason_score", "augmented"), ("gleason_tsg_score", "augmented"),
                 ("gleason_tsg_two_genes", "gleason_tsg_score"),
                 ("augmented_platinum_transfer", "clinical_platinum_transfer")]
        for model, reference in pairs:
            if model in values and reference in values:
                rows.append(dict(endpoint=endpoint, model=model, reference=reference, auc=values[model],
                                 reference_auc=values[reference], delta_auc=values[model] - values[reference]))
    return pd.DataFrame(rows)


def resample_patients(frame, rng):
    ids = frame[ID].drop_duplicates().to_numpy()
    draw = pd.DataFrame({ID: rng.choice(ids, len(ids), replace=True)})
    # Keep original IDs: copies of a sampled patient must share every fold.
    return draw.merge(frame, on=ID, how="left", validate="many_to_many")


def percentile_interval(values, cfg):
    values = np.asarray(values, float)
    good = np.isfinite(values)
    if good.sum() < max(10, int(np.ceil(cfg.bootstrap * cfg.min_bootstrap_fraction))):
        return np.nan, np.nan
    return tuple(np.quantile(values[good], [.025, .975]))


def analyze_cohort(frame, cfg, genomic=False, gleason="gleason", progress=None):
    predictions, fits, folds = cross_validate(frame, cfg, genomic, gleason)
    metrics, screen, calibration = evaluate_predictions(predictions, cfg)
    delta = contrasts(metrics)
    draws, screen_draws, delta_draws, failures = [], [], [], []
    rng = np.random.default_rng(cfg.seed)
    for b in range(cfg.bootstrap):
        try:
            if cfg.bootstrap_mode == "refit":
                sample = resample_patients(frame, rng)
                pred, _, _ = cross_validate(sample, cfg, genomic, gleason, cfg.seed + b + 1)
            else:
                pred = resample_patients(predictions, rng)
            bm, bs, _ = evaluate_predictions(pred, cfg)
            bd = contrasts(bm)
            draws.append(bm.assign(replicate=b))
            screen_draws.append(bs.assign(replicate=b))
            delta_draws.append(bd.assign(replicate=b))
            failures.append(dict(replicate=b, status="ok" if bm.auc.notna().all() else "some estimates unsupported",
                                 n_invalid=int(bm.auc.isna().sum())))
        except (ValueError, ArithmeticError, ConvergenceError, ConvergenceWarning, np.linalg.LinAlgError) as error:
            failures.append(dict(replicate=b, status=str(error), n_invalid=len(metrics)))
        if progress and (b + 1) % 25 == 0:
            progress(f"Bootstrap {b + 1}/{cfg.bootstrap}")
    def intervals(table, blocks, keys, values):
        boot = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        table = table.copy()
        for value in values:
            if value not in table:
                continue
            for col in (value + "_lo", value + "_hi", value + "_valid_bootstrap"):
                table[col] = np.nan
            for i, row in table.iterrows():
                sample = boot
                for key in keys:
                    if key in sample:
                        sample = sample.loc[sample[key].eq(row[key])]
                v = sample[value].to_numpy() if value in sample else []
                table.loc[i, [value + "_lo", value + "_hi"]] = (
                    percentile_interval(v, cfg) if np.isfinite(row[value]) else (np.nan, np.nan))
                table.loc[i, value + "_valid_bootstrap"] = np.isfinite(v).sum()
        table["interval_method"] = ("patient bootstrap repeating nested fitting" if cfg.bootstrap_mode == "refit"
                                    else "conditional on fitted OOF predictions")
        table["bootstrap_requested"] = cfg.bootstrap
        return table
    return dict(
        metrics=intervals(metrics, draws, ["endpoint", "model"], ["auc"]),
        contrasts=intervals(delta, delta_draws, ["endpoint", "model", "reference"], ["delta_auc"]),
        screening=intervals(screen, screen_draws, ["endpoint", "model", "operating_point"],
                            ["flag_rate", "sensitivity", "specificity", "ppv", "npv"]),
        calibration=calibration, predictions=predictions, fit_diagnostics=fits,
        folds=folds.reset_index(), bootstrap_diagnostics=pd.DataFrame(failures),
        bootstrap_metrics=pd.concat(draws, ignore_index=True) if draws else pd.DataFrame(),
    )


def net_risk(d, horizon):
    if d.empty or d[TIME].max() <= horizon:
        return np.nan, np.nan, np.nan
    km = KaplanMeierFitter().fit(d[TIME], d[EVENT])
    ci = km.confidence_interval_survival_function_
    pos = ci.index.searchsorted(horizon, side="right") - 1
    return float(1 - km.predict(horizon)), float(1 - ci.iloc[pos, 1]), float(1 - ci.iloc[pos, 0])


def cumulative_incidence(d, horizon):
    """Aalen–Johansen product integral, events before censorings at tied times.

    Death/event same day is treated as the endpoint and separately counted.
    Intervals are supplied by patient bootstrap, without jittering event dates.
    """
    if d.empty or "death_days" not in d:
        return np.nan
    death = d.death_days.to_numpy(float)
    t, e = d[TIME].to_numpy(float), d[EVENT].to_numpy(int)
    competing = np.isfinite(death) & (death > 0) & (death <= t) & ~((e == 1) & (death == t))
    duration = np.where(competing, death, t)
    code = np.where(competing, 2, e)
    if duration.max() <= horizon:
        return np.nan
    surv, cif = 1., 0.
    for time in np.unique(duration[(duration <= horizon) & (code > 0)]):
        risk = (duration >= time).sum()
        cif += surv * ((duration == time) & (code == 1)).sum() / risk
        surv *= 1 - ((duration == time) & (code > 0)).sum() / risk
    return cif


def four_group_risks(frame, cfg, death_available=False):
    rows, differences = [], []
    for endpoint, d in frame.groupby("endpoint", sort=False):
        d = d.copy()
        d["gleason_high"] = d.gleason.ge(8).astype(int)
        rng = np.random.default_rng(cfg.seed)
        point, boot = {}, {}
        for high in (0, 1):
            for flag in (0, 1):
                sub = d.loc[d.gleason_high.eq(high) & d.B.eq(flag)]
                risk, lo, hi = net_risk(sub, cfg.horizon)
                point[(high, flag)] = risk
                row = dict(endpoint=endpoint, gleason_high=high, flag=flag, **counts(sub, cfg),
                           risk=risk, risk_lo=lo, risk_hi=hi, estimand="net event risk (death censored)")
                rows.append(row)
                if death_available:
                    rows.append(dict(row, risk=cumulative_incidence(sub, cfg.horizon), risk_lo=np.nan, risk_hi=np.nan,
                                     estimand="cumulative incidence (death competing)"))
        for b in range(cfg.bootstrap):
            sample = resample_patients(d, rng)
            for high in (0, 1):
                values = []
                for flag in (0, 1):
                    sub = sample.loc[sample.gleason_high.eq(high) & sample.B.eq(flag)]
                    values.append(net_risk(sub, cfg.horizon)[0])
                    if death_available:
                        boot.setdefault((high, flag, "cif"), []).append(cumulative_incidence(sub, cfg.horizon))
                boot.setdefault((high, "difference"), []).append(values[1] - values[0])
        for row in rows:
            if row["endpoint"] == endpoint and row["estimand"].startswith("cumulative"):
                v = boot.get((row["gleason_high"], row["flag"], "cif"), [])
                row["risk_lo"], row["risk_hi"] = percentile_interval(v, cfg)
                row["valid_bootstrap"] = int(np.isfinite(v).sum())
        for high in (0, 1):
            v = boot.get((high, "difference"), [])
            lo, hi = percentile_interval(v, cfg)
            differences.append(dict(endpoint=endpoint, gleason_high=high,
                                    risk_difference=point[(high, 1)] - point[(high, 0)],
                                    risk_difference_lo=lo, risk_difference_hi=hi,
                                    valid_bootstrap=int(np.isfinite(v).sum()),
                                    estimand="within-stratum net risk difference; no interaction claim"))
    return pd.DataFrame(rows), pd.DataFrame(differences)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def write_analysis(result, directory):
    """Only aggregate files in aggregate/ are offered to R; patients stay protected."""
    directory = Path(directory)
    private = {"predictions", "folds"}
    for name, table in result.items():
        if not len(table.columns):
            continue
        target = directory / ("patient_level" if name in private else "aggregate") / f"{name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)


def parse_landmarks(value="90,0,180"):
    """Explicit selection shared by the notebook and R; never performance based."""
    try:
        values = tuple(int(x.strip()) for x in value.split(","))
    except (ValueError, AttributeError) as error:
        raise ValueError("COMPASS_AUGMENTATION_LANDMARKS must contain 0, 90 and/or 180") from error
    if not values or len(set(values)) != len(values) or not set(values) <= {0, 90, 180}:
        raise ValueError("COMPASS_AUGMENTATION_LANDMARKS must be distinct values from 0,90,180")
    return values


def landmark_directory(root, landmark):
    """Preserve the primary day-90 directory; keep other cells independent."""
    if landmark not in (0, 90, 180):
        raise ValueError("Unsupported landmark")
    return Path(root) if landmark == 90 else Path(root) / f"lm{landmark}_h1825"


def start_run(directory, cfg):
    """Retain old results, without allowing stale optional tables into a new run."""
    from datetime import datetime, timezone
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = [directory / name for name in ("aggregate", "patient_level", "analysis_config.json", "run_manifest.json")
                if (directory / name).exists()]
    if existing:
        archive = directory / "previous_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        for path in existing:
            path.rename(archive / path.name)
    write_json(directory / "run_manifest.json", dict(schema_version=1, status="running", settings=asdict(cfg)))


def frozen_reference(frame, cfg):
    """Full-cohort external transfer preserved independently of availability cohorts."""
    rows, screen = [], []
    for endpoint, d in frame.groupby("endpoint", sort=False):
        risks = {"L": d.L.to_numpy(float), "B": d.B.to_numpy(float)}
        boot = fet._bootstrap(d[TIME], d[EVENT], risks, cfg.bootstrap, cfg.seed,
                              lambda t, e, s: auc(pd.DataFrame({TIME: t, EVENT: e}), s, cfg))
        for model, score in risks.items():
            estimate = auc(d, score, cfg)
            lo, hi = percentile_interval(boot[model], cfg) if np.isfinite(estimate) else (np.nan, np.nan)
            rows.append(dict(endpoint=endpoint, model=model, **counts(d, cfg), auc=estimate,
                             auc_lo=lo, auc_hi=hi, auc_valid_bootstrap=int(np.isfinite(boot[model]).sum()),
                             n_complete_labs=int(d.complete_labs.sum()), n_evaluable=int(d.evaluable.sum()),
                             interval_method="patient bootstrap of frozen scores", status="frozen transfer"))
        try:
            screen.append(dict(endpoint=endpoint, model="B", **counts(d, cfg), **screening(d, d.B.to_numpy(), cfg)))
        except ValueError as error:
            screen.append(dict(endpoint=endpoint, model="B", **counts(d, cfg), status=str(error)))
    return pd.DataFrame(rows), pd.DataFrame(screen)


def run_analyses(frames, cfg, progress=print):
    """Prespecified cohorts/sensitivities; unavailable analyses receive explicit status."""
    results, status = {}, []
    primary = frames["gleason"]
    analyses = [("gleason", primary, False, "gleason"),
                ("gleason_grade_group", primary.loc[primary.grade_group.notna()], False, "grade_group"),
                ("genomic", frames["genomic"], True, "gleason"),
                ("genomic_adt", frames["genomic_adt"], True, "gleason")]
    if (~primary.complete_labs).any():
        analyses.insert(1, ("gleason_complete_labs", primary.loc[primary.complete_labs], False, "gleason"))
    else:
        status.append(dict(cohort="gleason_complete_labs", status="same as primary Gleason cohort; all labs observed"))
    genomic = frames["genomic"]
    pathogenic = genomic.loc[genomic[[g + "_pathogenic" for g in GENES]].notna().all(axis=1)].copy()
    for g in GENES:
        pathogenic[g] = pathogenic[g + "_pathogenic"]
    analyses.append(("genomic_pathogenic", pathogenic, True, "gleason"))
    for cohort, frame, genomic_model, representation in analyses:
        if frame.empty or frame[ID].nunique() < cfg.outer_folds:
            status.append(dict(cohort=cohort, status="unavailable: too few eligible patients or missing required inputs"))
            continue
        progress(f"{cohort}: {frame[ID].nunique()} patients; {cfg.bootstrap_mode} bootstrap, {cfg.bootstrap} draws")
        result = analyze_cohort(frame, cfg, genomic_model, representation, progress=progress)
        for name, table in result.items():
            if table.empty and not len(table.columns):
                continue
            table = table.assign(cohort=cohort, clock="sequencing" if cohort in ("genomic", "genomic_pathogenic") else "adt",
                                 landmark_days=cfg.landmark, horizon_days=cfg.horizon)
            results.setdefault(name, []).append(table)
        status.append(dict(cohort=cohort, status="completed; inspect missing predictions and bootstrap support"))
    output = {name: pd.concat(tables, ignore_index=True) for name, tables in results.items()}
    output["analysis_status"] = pd.DataFrame(status)
    output["full_transfer"], output["full_transfer_screening"] = frozen_reference(frames["full_checklist"], cfg)
    output["historical_transfer"], _ = frozen_reference(frames["historical_paired"], cfg)
    # Full-cohort reference above includes lab-imputed patients; this keeps complete-lab coverage visible.
    complete = frames["full_checklist"].loc[lambda d: d.complete_labs]
    output["full_transfer_complete_labs"], _ = frozen_reference(complete, cfg)
    death_available = bool(frames.get("death_source_available", False))
    output["four_groups"], output["within_gleason_differences"] = four_group_risks(primary, cfg, death_available)
    # Stable empty-table schemas let R show count-only/unavailable panels.
    defaults = {
        "metrics": ["cohort", "clock", "endpoint", "model", "auc", "auc_lo", "auc_hi", "n_patients", "n_events_horizon"],
        "contrasts": ["cohort", "endpoint", "model", "reference", "delta_auc", "delta_auc_lo", "delta_auc_hi"],
        "screening": ["cohort", "endpoint", "model", "operating_point", "flag_rate", "sensitivity", "specificity", "ppv", "npv"],
        "four_groups": ["endpoint", "gleason_high", "flag", "n_patients", "n_events_horizon", "risk", "risk_lo", "risk_hi", "estimand"],
    }
    for name, columns in defaults.items():
        if name not in output or output[name].empty:
            output[name] = pd.DataFrame(columns=columns)
    for name, table in output.items():
        output[name] = table.assign(landmark_days=cfg.landmark, horizon_days=cfg.horizon,
                                   analysis_role="primary" if cfg.landmark == 90 else "additional landmark")
    return output


def complete_run(directory, cfg):
    """Publication receipt written last. R accepts only this aggregate allowlist."""
    import hashlib
    directory = Path(directory)
    files = []
    for path in sorted((directory / "aggregate").glob("*.csv")):
        files.append(dict(name=path.name, md5=hashlib.md5(path.read_bytes()).hexdigest()))
    write_json(directory / "run_manifest.json", dict(schema_version=1, status="complete", settings=asdict(cfg),
                files=files, estimand="five-year net risk; competing-death CIF only in sensitivity tables",
                inference="final refitting bootstrap" if cfg.bootstrap_mode == "refit" and cfg.bootstrap >= 1000
                          else "diagnostic run; not final inference"))
