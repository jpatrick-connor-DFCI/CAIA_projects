"""Federated platinum score on NEPC tiers / AVPC within metastatic and mCRPC subgroups (testing notebook 10).

Endpoints: NEPC pathology-tier sets t1, t1-t2, t1-t3; strict AVPC; AVPC or AVPC-adjacent (see
survival_common.federated_endpoint_transfer.ENDPOINT_SOURCES).

Subgroup status is judged at each landmark date (ADT anchor + landmark) or, as a retrospective
sensitivity, ever. Metrics come from `reviewer_extended.subgroup_metrics`; nothing is refit.
"""
import numpy as np
import pandas as pd
import reviewer_extended as ext
import reviewer_sources as src
import reviewer_specificity as spec

ENDPOINTS = ('nepc_t1', 'nepc_t1_t2', 'nepc_t1_t2_t3', 'avpc', 'avpc_or_adjacent')
FAMILIES = ('xgboost_cox', 'predictive_checklist')
PRIMARY_VARIABLES = ('full_cohort', 'metastatic_at_landmark', 'mcrpc_at_landmark')
EVER_VARIABLES = ('full_cohort', 'metastatic_ever', 'mcrpc_ever')
IDENTITY = ['endpoint', 'model_family', 'bundle', 'landmark_days', 'config', 'model_horizon_days']


def mcrpc_fields(mcrpc_labels):
    """One row per patient: is_mcrpc (boolean) and mcrpc_date (datetime) from the mCRPC labels."""
    missing = {'is_mcrpc', 'mcrpc_date'} - set(mcrpc_labels)
    if missing: raise ValueError(f'mCRPC labels missing {sorted(missing)}')
    d = src.one_patient(mcrpc_labels, ['is_mcrpc', 'mcrpc_date'], 'mCRPC labels')
    d['is_mcrpc'] = d.is_mcrpc.astype('boolean')
    d['mcrpc_date'] = ext.date(d.mcrpc_date)
    return d


def patient_frame(local, mcrpc_labels, avpc_labels):
    """anchor_date (longitudinal file), met_date + is_mcrpc + mcrpc_date (mCRPC labels), is_metastatic (AVPC labels).

    Patients without a label row keep missing values, which become the 'missing' subgroup level.
    """
    meta = src.one_patient(local, ['TREATMENT_ANCHOR_DATE'], 'longitudinal data').rename(
        columns={'TREATMENT_ANCHOR_DATE': 'anchor_date'})
    if 'met_date' not in mcrpc_labels: raise ValueError('mCRPC labels missing met_date')
    if 'is_metastatic' not in avpc_labels: raise ValueError('AVPC labels missing is_metastatic')
    mc = mcrpc_fields(mcrpc_labels).merge(src.one_patient(mcrpc_labels, ['met_date'], 'mCRPC labels'),
                                          on='person_id', validate='one_to_one')
    meta = meta.merge(mc, on='person_id', how='left', validate='one_to_one')
    meta = meta.merge(src.one_patient(avpc_labels, ['is_metastatic'], 'AVPC labels'),
                      on='person_id', how='left', validate='one_to_one')
    for col in ('anchor_date', 'met_date', 'mcrpc_date'):
        meta[col] = ext.date(meta[col])
    meta['is_metastatic'] = meta.is_metastatic.astype('boolean')
    meta['is_mcrpc'] = meta.is_mcrpc.astype('boolean')
    return meta


def mcrpc_status(mcrpc_date, is_mcrpc, reference):
    """mCRPC by `reference`, with the same levels as `metastatic_status`.

    yes: dated on or before reference; no: dated after, or is_mcrpc False;
    date unknown: is_mcrpc True without a date; missing: no label (or no reference).
    """
    idx = pd.Series(mcrpc_date).index
    when = pd.to_datetime(pd.Series(mcrpc_date), errors='coerce').reset_index(drop=True)
    ref = pd.to_datetime(pd.Series(reference), errors='coerce').reset_index(drop=True)
    flag = pd.Series(is_mcrpc).reset_index(drop=True).astype('boolean')
    yes_flag = flag.fillna(False).to_numpy(bool); no_flag = (~flag).fillna(False).to_numpy(bool)
    dated = when.notna().to_numpy(bool) & ~no_flag
    has_ref = ref.notna().to_numpy(bool)
    before = (when <= ref).fillna(False).to_numpy(bool)
    status = np.select([dated & has_ref & before, dated & has_ref, no_flag, ~dated & yes_flag],
                       ['yes', 'no', 'no', 'date unknown'], default='missing')
    return pd.Series(status, index=idx)


def landmark_subgroups(meta, landmark_days):
    """Subgroup columns for one landmark. `meta` needs anchor_date, met_date, is_metastatic, is_mcrpc, mcrpc_date."""
    d = meta[['person_id']].copy()
    landmark_date = meta.anchor_date + np.timedelta64(int(landmark_days), 'D')
    d['full_cohort'] = 'all'
    d['metastatic_at_landmark'] = spec.metastatic_status(meta.met_date, meta.is_metastatic, landmark_date).to_numpy()
    d['mcrpc_at_landmark'] = mcrpc_status(meta.mcrpc_date, meta.is_mcrpc, landmark_date).to_numpy()
    # Retrospective: uses the whole observed course, so it is a signal test, not a deployable subgroup.
    met_flag = meta.is_metastatic.astype('boolean')
    d['metastatic_ever'] = np.select([met_flag.fillna(False).to_numpy(bool) | meta.met_date.notna().to_numpy(bool),
                                      (~met_flag).fillna(False).to_numpy(bool)], ['yes', 'no'], default='missing')
    mc_flag = meta.is_mcrpc.astype('boolean')
    d['mcrpc_ever'] = np.select([mc_flag.fillna(False).to_numpy(bool), (~mc_flag).fillna(False).to_numpy(bool)],
                                ['yes', 'no'], default='missing')
    return d


def select_scores(scores, endpoints=ENDPOINTS, families=FAMILIES, config='both'):
    d = scores[scores.endpoint.isin(endpoints) & scores.model_family.isin(families) & scores.config.eq(config)].copy()
    if d.empty: raise ValueError(f'No scores for endpoints={endpoints}, families={families}, config={config!r}')
    d['model_horizon_days'] = pd.to_numeric(d.get('model_horizon_days'), errors='coerce')
    if d.duplicated([c for c in IDENTITY if c in d] + ['person_id']).any():
        raise ValueError('Duplicate patient/model identities in transfer scores')
    return d


def run(scores, meta, variables, n_boot=50, seed=0):
    """Subgroup metrics for every endpoint × family × bundle × landmark × horizon."""
    keys = [c for c in IDENTITY if c in scores]
    status = {lm: landmark_subgroups(meta, lm) for lm in sorted(scores.landmark_days.unique())}
    tables = []
    for key, g in scores.groupby(keys, dropna=False, sort=True):
        ident = dict(zip(keys, key))
        if g.person_id.duplicated().any(): raise ValueError(f'Duplicate patients for {ident}')
        d = g.merge(status[ident['landmark_days']], on='person_id', how='left', validate='one_to_one')
        for v in variables:
            d[v] = d[v].fillna('missing')
        table, _ = ext.subgroup_metrics(d, list(variables), n_boot=n_boot, seed=seed)
        tables.append(table.assign(**ident))
    return pd.concat(tables, ignore_index=True)


def counts(scores, meta):
    """Patients and events per subgroup level, for every endpoint × landmark frame (one model's frame each)."""
    rows = []
    frames = scores.drop_duplicates(['endpoint', 'landmark_days', 'person_id'])
    for (ep, lm), g in frames.groupby(['endpoint', 'landmark_days']):
        d = g.merge(landmark_subgroups(meta, lm), on='person_id', how='left', validate='one_to_one')
        for v in PRIMARY_VARIABLES[1:] + EVER_VARIABLES[1:]:
            c = d.assign(**{v: d[v].fillna('missing')}).groupby(v).agg(n=('person_id', 'size'), events=('event', 'sum'))
            rows.append(c.reset_index().rename(columns={v: 'level'}).assign(variable=v, endpoint=ep, landmark_days=lm))
    return pd.concat(rows, ignore_index=True)[['endpoint', 'landmark_days', 'variable', 'level', 'n', 'events']]
