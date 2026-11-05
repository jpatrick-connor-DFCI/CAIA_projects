"""CRPC specificity: endpoint definitions, out-of-fold models and paired uncertainty."""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from reviewer_questions import ci_bootstrap

ENDPOINTS = ('avpc', 'nepc', 'avpc_or_nepc')


def metastatic_status(met_date, is_metastatic, reference):
    """Metastatic by `reference`: a dated metastasis wins; an undated one is 'date unknown', not 'no'."""
    met = pd.to_datetime(pd.Series(met_date), errors='coerce').reset_index(drop=True)
    ref = pd.to_datetime(pd.Series(reference), errors='coerce').reset_index(drop=True)
    flag = pd.Series(is_metastatic).reset_index(drop=True).astype('boolean')
    known_yes = flag.fillna(False).to_numpy(bool); known_no = (~flag).fillna(False).to_numpy(bool)
    dated = met.notna().to_numpy(bool)
    status = np.select([dated & ref.notna().to_numpy(bool) & (met <= ref).to_numpy(bool),
                        dated & ref.notna().to_numpy(bool),
                        ~dated & known_no, ~dated & known_yes],
                       ['yes', 'no', 'no', 'date unknown'], default='missing')
    return pd.Series(status, index=pd.Series(met_date).index)


def define_crpc_outcome(frame, endpoint, horizon=730, after_landmark=True):
    """Return a complete, explicit cohort audit and binary case/control labels.

    endpoint 'avpc_or_nepc' is the primary composite (first of either phenotype); for
    the single-phenotype endpoints, other-phenotype-only patients are excluded, not
    called conventional CRPC. A case can predate CRPC (including C1); follow-up is
    measured from CRPC.
    """
    d = frame.copy()
    crpc = pd.to_datetime(d.crpc_date, errors='coerce')
    landmark = pd.to_datetime(d.landmark_date, errors='coerce')
    follow = pd.to_datetime(d.last_contact_date, errors='coerce')
    avpc, nepc = (pd.to_datetime(d[c], errors='coerce') for c in ('avpc_date', 'nepc_date'))
    if endpoint not in ENDPOINTS: raise ValueError(f'endpoint must be one of {ENDPOINTS}')
    requested = (avpc if endpoint == 'avpc' else nepc if endpoint == 'nepc'
                 else pd.concat([avpc, nepc], axis=1).min(axis=1))
    deadline = crpc + pd.to_timedelta(horizon, unit='D')
    eligible_crpc = crpc.notna() & landmark.notna()
    clock = (crpc > landmark) if after_landmark else (crpc <= landmark)
    case = eligible_crpc & clock & requested.notna() & (requested <= deadline) & (requested <= follow)
    any_phenotype = ((avpc.notna() & (avpc <= deadline) & (avpc <= follow)) |
                     (nepc.notna() & (nepc <= deadline) & (nepc <= follow)))
    control = eligible_crpc & clock & ~any_phenotype & (follow >= deadline)
    d['outcome'] = np.select([case, control], [1., 0.], default=np.nan)
    d['exclusion'] = np.select([~eligible_crpc, eligible_crpc & ~clock, case, control,
                                eligible_crpc & clock & any_phenotype],
                               ['missing CRPC/landmark', 'other CRPC timing', 'case', 'control',
                                'other phenotype'], default='insufficient follow-up')
    return d


def make_design(d):
    x = pd.DataFrame(index=d.index)
    x['age'] = pd.to_numeric(d.age, errors='coerce')
    psa = pd.to_numeric(d.psa_at_crpc, errors='coerce')
    x['log_psa'] = np.log1p(psa.where(psa >= 0))
    days = (pd.to_datetime(d.crpc_date) - pd.to_datetime(d.anchor_date)).dt.days
    x['log_adt_to_crpc'] = np.log1p(days.where(days >= 0))
    x['metastatic'] = metastatic_status(d.met_date, d.is_metastatic, d.crpc_date).to_numpy()
    g = pd.to_numeric(d.gleason_at_crpc, errors='coerce')
    x['gleason'] = np.where(g.isna(), 'missing', np.where(g >= 8, '8+', '<=7'))
    x['adt_intent'] = d.adt_intent.fillna('missing').astype(str)
    return x


def _pipeline(columns, include_score):
    nums = ['age','log_psa','log_adt_to_crpc'] + (['score'] if include_score else [])
    cats = ['metastatic','gleason','adt_intent']
    # Explicit missing indicators and regularization stabilize sparse preliminary models.
    transform = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                          ('scale', StandardScaler())]), nums),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats)])
    return Pipeline([('preprocess', transform), ('model', LogisticRegression(C=1.0, max_iter=2000))])


def fit_specificity(d, n_boot=1000, seed=0, min_events=10):
    d = d.dropna(subset=['outcome','risk_score']).copy()
    n1, n0 = int(d.outcome.sum()), int((1-d.outcome).sum())
    out = {'n':len(d),'events':n1,'controls':n0, 'score_auc':np.nan, 'clinical_auc':np.nan,
           'combined_auc':np.nan, 'delta_auc':np.nan,'delta_lower':np.nan,'delta_upper':np.nan,
           'adjusted_or_per_sd':np.nan,'or_lower':np.nan,'or_upper':np.nan,'lrt_p':np.nan,
           'note':''}
    if n1 < min_events or n0 < min_events:
        out['note']='not estimable: fewer than 10 cases or controls'; return out
    x = make_design(d).reset_index(drop=True); y = d.outcome.to_numpy(dtype=int)
    score = d.risk_score.to_numpy(float); sd = np.std(score)
    if not sd > 0: out['note']='not estimable: constant score'; return out
    x['score'] = (score-score.mean()) / sd
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p0=np.zeros(len(d)); p1=np.zeros(len(d))
    for tr, te in folds.split(x,y):
        for plus, pred in ((False,p0),(True,p1)):
            model=_pipeline(x.columns,plus).fit(x.iloc[tr],y[tr]); pred[te]=model.predict_proba(x.iloc[te])[:,1]
    out.update(score_auc=roc_auc_score(y,score), clinical_auc=roc_auc_score(y,p0),
               combined_auc=roc_auc_score(y,p1))
    out['delta_auc']=out['combined_auc']-out['clinical_auc']
    boot=pd.DataFrame({'y':y,'p0':p0,'p1':p1})
    out['delta_lower'], out['delta_upper'] = ci_bootstrap(boot, lambda z: roc_auc_score(z.y,z.p1)-roc_auc_score(z.y,z.p0), n_boot, np.random.default_rng(seed))
    # In-sample unpenalized inference is distinct from penalized OOF prediction.
    try:
        from scipy.optimize import minimize
        from scipy.special import expit
        transformed = _pipeline(x.columns,True).named_steps['preprocess'].fit_transform(x)
        if hasattr(transformed, 'toarray'): transformed=transformed.toarray()
        # Drop all-zero and linearly dependent columns to permit categorical reference levels.
        from scipy.linalg import qr
        design = np.column_stack([np.ones(len(y)), transformed])
        _, rr, piv = qr(design, mode='economic', pivoting=True)
        rank = np.linalg.matrix_rank(rr); keep=sorted(piv[:rank]); design=design[:,keep]
        # Locate the standardized score after numeric values (fourth numeric column).
        score_column=1+3
        if score_column not in keep: raise ValueError('score is linearly dependent')
        idx=keep.index(score_column)
        def fit_unpenalized(matrix):
            def loss(beta):
                eta=matrix @ beta
                return np.logaddexp(0,eta).sum()-y @ eta
            def grad(beta): return matrix.T @ (expit(matrix @ beta)-y)
            fitted=minimize(loss,np.zeros(matrix.shape[1]),jac=grad,method='BFGS',options={'maxiter':1000,'gtol':1e-7})
            if not fitted.success and np.linalg.norm(grad(fitted.x),ord=np.inf)>1e-4: raise ValueError('logistic optimization did not converge')
            probabilities=expit(matrix @ fitted.x)
            if np.max(np.abs(fitted.x))>30: raise ValueError('possible separation in logistic model')
            info=matrix.T @ ((probabilities*(1-probabilities))[:,None]*matrix)
            return fitted.x,np.linalg.inv(info),-loss(fitted.x)
        params,cov,ll_full=fit_unpenalized(design)
        _,_,ll_base=fit_unpenalized(np.delete(design,idx,axis=1))
        coef=params[idx]; se=np.sqrt(cov[idx,idx])
        out.update(adjusted_or_per_sd=np.exp(coef), or_lower=np.exp(coef-1.96*se),
                   or_upper=np.exp(coef+1.96*se), lrt_p=chi2.sf(max(0,2*(ll_full-ll_base)),1))
    except Exception as exc:
        out['note']=f'Unpenalized OR/LRT not estimable: {type(exc).__name__}: {exc}'
    return out
