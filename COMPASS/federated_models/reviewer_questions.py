"""Aggregate preliminary reviewer analyses; patient-level inputs must stay protected."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.metrics import roc_auc_score

# The last band is closed so primary controls can be followed past every case in it.
BANDS = [(0, 180, '0–180'), (180, 365, '181–365'), (365, 730, '366–730'), (730, 1825, '731–1825')]
MIN_CONTROL_FOLLOWUP = 730
PRIMARY_CONTROLS = 'follow-up ≥ max(730 d, band end)'
ALL_CONTROLS = 'all event-free'


def control_followup_needed(hi):
    """Primary controls are event-free through 730 days and through the band's upper edge."""
    return max(MIN_CONTROL_FOLLOWUP, hi)


def lead_groups(frame, lo, hi, all_controls=False):
    """Return masks: incident cases in (lo, hi], and noncase controls.

    Primary controls are followed event-free through max(730, hi) days, so each control
    outlasts every case in its band. The sensitivity analysis uses all noncases.
    """
    if not np.isfinite(hi): raise ValueError('Lead bands must be closed so controls can outlast cases')
    t, e = frame.duration_days, frame.event.astype(bool)
    cases = e & (t > lo) & (t <= hi)
    followup = frame['control_followup_days'] if 'control_followup_days' in frame else t
    controls = ~e & ((followup >= control_followup_needed(hi)) | all_controls)
    return cases, controls


def km_risk(t, e, horizon):
    if not len(t) or np.max(t) < horizon:
        return np.nan
    return 1.0 - float(KaplanMeierFitter().fit(t, e).predict(horizon))


def ci_bootstrap(frame, func, n, rng):
    vals = []
    for _ in range(n):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        try:
            value = func(sample)
            if np.isfinite(value): vals.append(value)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass
    return tuple(np.quantile(vals, [.025, .975])) if len(vals) >= max(20, n // 2) else (np.nan, np.nan)


def lead_auc(frame, n_boot, seed, min_events=10):
    rng = np.random.default_rng(seed)
    rows = []
    for sensitivity in (False, True):
        for lo, hi, label in BANDS:
            cases, controls = lead_groups(frame, lo, hi, sensitivity)
            selected = frame.loc[cases | controls].copy()
            selected['case'] = cases.loc[selected.index].astype(int)
            ncase, ncontrol = int(cases.sum()), int(controls.sum())
            auc = low = high = np.nan
            if ncase >= min_events and ncontrol:
                auc = roc_auc_score(selected['case'], selected.risk_score)
                low, high = ci_bootstrap(selected, lambda d: roc_auc_score(d['case'], d.risk_score), n_boot, rng)
            rows.append(dict(band=label, sensitivity=ALL_CONTROLS if sensitivity else PRIMARY_CONTROLS,
                             control_followup_days=np.nan if sensitivity else control_followup_needed(hi),
                             cases=ncase, controls=ncontrol, auc=auc, lower=low, upper=high))
    return pd.DataFrame(rows)


def lead_summary(frame, checklist=False):
    d = frame.copy()
    if 'flagged' not in d:
        if checklist:
            d['flagged'] = d.flag.eq(1)
        else:
            # A threshold defined on the full landmark cohort; ties can produce >20% flagged.
            d['flagged'] = d.risk_score >= d.risk_score.quantile(.8)
    flagged = d[d.flagged]
    cases = d[d.event.eq(1)]
    fc = flagged[flagged.event.eq(1)]
    t = fc.duration_days
    enough = len(fc) >= 10
    return dict(n=len(d), events=len(cases), n_flagged=len(flagged), flagged_fraction=d.flagged.mean(),
                sensitivity=cases.flagged.mean() if len(cases) else np.nan,
                lead_median=t.median() if enough else np.nan, lead_q25=t.quantile(.25) if enough else np.nan, lead_q75=t.quantile(.75) if enough else np.nan,
                fraction_180=(t >= 180).mean() if enough else np.nan,
                fraction_365=(t >= 365).mean() if enough else np.nan,
                ppv_730_km=km_risk(flagged.duration_days, flagged.event, 730) if len(fc) >= 10 else np.nan)


def cox_unstandardized(d):
    if d.event.sum() < 10 or d.risk_score.std() == 0: return np.nan
    try:
        fit = CoxPHFitter().fit(d[['duration_days', 'event', 'risk_score']], 'duration_days', 'event')
        return float(fit.params_['risk_score'])
    except Exception:
        return np.nan


def pool_adjacent(groups, min_events=10):
    """Merge ordered groups until each has >= min_events events; a sparse tail joins the previous group."""
    pooled, pending = [], []
    for g in groups:
        pending.append(g)
        if sum(int(q.event.sum()) for q in pending) >= min_events:
            pooled.append(pd.concat(pending)); pending = []
    if pending:
        if pooled: pooled[-1] = pd.concat([pooled[-1], *pending])
        else: pooled = [pd.concat(pending)]
    return pooled


def risk_groups(frame, column, n_groups=10, min_events=10):
    """Quantile groups of `column`, pooled with neighbours so sparse low-risk groups are shown, not blanked."""
    bins = pd.qcut(frame[column], n_groups, duplicates='drop')
    return pool_adjacent([g for _, g in frame.groupby(bins, observed=True)], min_events)


def hr_per_sd(duration, event, risk, center, scale):
    """Cox HR per SD on a fixed reference scale, so HRs from different subsets are comparable."""
    out = {'hr_per_sd': np.nan, 'hr_per_sd_lo': np.nan, 'hr_per_sd_hi': np.nan, 'hr_per_sd_p': np.nan}
    event = np.asarray(event)
    if event.sum() < 2 or not scale > 0: return out
    z = (np.asarray(risk, dtype=float) - center) / scale
    if not np.std(z) > 0: return out
    try:
        s = CoxPHFitter().fit(pd.DataFrame({'T': np.asarray(duration, dtype=float), 'E': event, 'z': z}), 'T', 'E').summary.loc['z']
    except Exception:
        return out
    out.update(hr_per_sd=float(s['exp(coef)']), hr_per_sd_lo=float(s['exp(coef) lower 95%']),
               hr_per_sd_hi=float(s['exp(coef) upper 95%']), hr_per_sd_p=float(s['p']))
    return out


def draw_lead(table, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (ep, color) in enumerate([('avpc', '#2266a0'), ('nepc', '#d37925')]):
        d = table[(table.endpoint == ep) & (table.sensitivity == PRIMARY_CONTROLS)].set_index('band').reindex([b[2] for b in BANDS])
        x = np.arange(len(BANDS)) + (i-.5)*.12
        for xi, (_, row) in zip(x, d.iterrows()):
            if np.isfinite(row.auc):
                ax.scatter(xi, row.auc, color=color, label=ep.upper() if xi == x[0] else None)
                if np.isfinite(row.lower) and np.isfinite(row.upper): ax.plot([xi, xi], [row.lower, row.upper], color=color)
    ax.axhline(.5, color='grey', linestyle='--'); ax.set_xticks(range(len(BANDS)), [b[2] for b in BANDS]); ax.set_ylim(0, 1)
    ax.set(xlabel='Days from landmark to diagnosis', ylabel='Case–control AUC', title='Lead time: primary follow-up rule')
    handles, labels = ax.get_legend_handles_labels()
    if handles: ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
