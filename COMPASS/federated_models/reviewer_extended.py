"""Additional local reviewer analyses, with aggregate outputs only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from lifelines import CoxPHFitter
from reviewer_questions import (km_risk, ci_bootstrap, cox_unstandardized, pool_adjacent, risk_groups,
                                hr_per_sd, BANDS, PRIMARY_CONTROLS)
from survival_common import federated_endpoint_transfer as fet
from survival_common.federated_inference import predict_survival


def date(s):
    return pd.to_datetime(s, errors='coerce', format='mixed').dt.tz_localize(None)


def criterion_counts(labels):
    cols=[f'c{i}_met' for i in range(1,8)]
    missing=set(cols+['is_avpc'])-set(labels)
    if missing: raise ValueError(f'AVPC criterion columns missing: {sorted(missing)}')
    d=labels[labels.is_avpc.fillna(False).astype(bool)]
    flags=d[cols].fillna(False).astype(bool)
    rows=[dict(criterion=f'C{i}', cases=int(flags[f'c{i}_met'].sum()), total_cases=len(d)) for i in range(1,8)]
    rows.append(dict(criterion='C7 only', cases=int((flags.c7_met & flags.sum(axis=1).eq(1)).sum()),total_cases=len(d)))
    return pd.DataFrame(rows)


def milestones(d, endpoint):
    flagged=d[d.flagged & d.event.eq(1)].copy()
    for col in ['crpc_date','platinum_date',f'{endpoint}_date']:
        flagged[col+'_gap']=(date(flagged[col])-date(flagged.landmark_date)).dt.days
    flagged['crpc_to_endpoint']=(date(flagged[f'{endpoint}_date'])-date(flagged.crpc_date)).dt.days
    rows=[]
    for col in ['crpc_date_gap','platinum_date_gap',f'{endpoint}_date_gap','crpc_to_endpoint']:
        x=flagged[col].dropna()
        rows.append(dict(endpoint=endpoint, milestone=col, n=len(x), median_days=x.median() if len(x)>=10 else np.nan,
                         q25=x.quantile(.25) if len(x)>=10 else np.nan, q75=x.quantile(.75) if len(x)>=10 else np.nan))
    valid=flagged.crpc_date_gap.dropna()
    return pd.DataFrame(rows), {'endpoint':endpoint, 'flagged_cases':len(flagged), 'crpc_known':len(valid),
        'crpc_before_landmark':int(valid.lt(0).sum()), 'fraction_flagged_before_crpc':float(valid.gt(0).mean()) if len(valid)>=10 else np.nan}, flagged


def cumulative_flags(scores, endpoint, family, horizon=365):
    checklist='checklist' in family
    d=scores[(scores.endpoint==endpoint)&(scores.model_family==family)].copy()
    if checklist: d=d[d.model_horizon_days.eq(horizon)]
    if d.duplicated(['person_id','landmark_days']).any(): raise ValueError('Multiple score models in cumulative flag analysis')
    d['flagged'] = d.flag.eq(1) if checklist else d.groupby('landmark_days').risk_score.transform(lambda x:x>=x.quantile(.8))
    # Endpoint date is inferred from each observed event. Restrict to patients with
    # an observed eventual event at any landmark; flags count only pre-event rows.
    events=set(d.loc[d.event.eq(1),'person_id'])
    rows=[]
    for lm in (0,90,180):
        current=d[(d.landmark_days<=lm)&d.person_id.isin(events)]
        eligible=set(d.loc[(d.landmark_days<=lm)&d.event.eq(1),'person_id'])
        flagged=set(current.loc[current.flagged,'person_id']) & eligible
        rows.append(dict(endpoint=endpoint, model_family=family, through_landmark=lm,
                         eventual_cases=len(eligible), ever_flagged=len(flagged),
                         fraction=len(flagged)/len(eligible) if len(eligible)>=10 else np.nan))
    return pd.DataFrame(rows)


def breslow_at(t, e, lp, horizon):
    """Breslow cumulative baseline with fixed log hazard offset; no centering."""
    t=np.asarray(t,float); e=np.asarray(e,bool); lp=np.asarray(lp,float)
    times,index=np.unique(t,return_inverse=True)
    at_risk=np.bincount(index,weights=np.exp(np.clip(lp,-100,100)),minlength=len(times))[::-1].cumsum()[::-1]
    deaths=np.bincount(index,weights=e.astype(float),minlength=len(times))
    keep=(times<=horizon)&(deaths>0)
    return float((deaths[keep]/at_risk[keep]).sum())


def calibration_versions(d, model, horizons=(365,730,1095), n_boot=1000, seed=0, time_unit_days=1.0):
    if d.event.sum()<10: return pd.DataFrame(),pd.DataFrame()
    t=d.duration_days.to_numpy(float); e=d.event.to_numpy(int); lp=d.risk_score.to_numpy(float)
    slope=cox_unstandardized(d)
    transferred=1-predict_survival(lp,model,horizons,time_unit_days=time_unit_days).to_numpy()
    rows=[]; points=[]; rng=np.random.default_rng(seed)
    for j,h in enumerate(horizons):
        obs=km_risk(t,e,h)
        versions={'transferred':transferred[:,j],
                  'DFCI baseline updated':1-np.exp(-breslow_at(t,e,lp,h)*np.exp(np.clip(lp,-100,100))),
                  'DFCI slope and baseline updated':1-np.exp(-breslow_at(t,e,slope*lp,h)*np.exp(np.clip(slope*lp,-100,100))) if np.isfinite(slope) else np.full(len(lp),np.nan)}
        for name,p in versions.items():
            b=fet.ipcw_brier(t,e,p,h); ref=fet.ipcw_brier(t,e,np.full(len(t),obs),h)
            sampled=d.copy(); sampled['transferred_pred']=transferred[:,j]
            def _oe(z):
                zt=z.duration_days.to_numpy(float); ze=z.event.to_numpy(int); zl=z.risk_score.to_numpy(float)
                if name=='transferred': pred=z.transferred_pred.to_numpy(float)
                else:
                    beta=1.0 if name=='DFCI baseline updated' else cox_unstandardized(z)
                    if not np.isfinite(beta): return np.nan
                    v=beta*zl; pred=1-np.exp(-breslow_at(zt,ze,v,h)*np.exp(np.clip(v,-100,100)))
                expected=np.mean(pred)
                return km_risk(zt,ze,h)/expected if expected>0 else np.nan
            oe_lo,oe_hi=ci_bootstrap(sampled,_oe,n_boot,rng)
            rows.append(dict(version=name,horizon=h,n=len(t),events=int(e.sum()), observed=obs,
                expected=np.mean(p),observed_expected=obs/np.mean(p) if np.mean(p)>0 else np.nan,
                oe_lower=oe_lo,oe_upper=oe_hi,slope=slope,brier=b,scaled_brier=1-b/ref if ref>0 else np.nan))
            if h!=730 or not np.isfinite(p).all(): continue
            tmp=d.copy(); tmp['pred']=p
            # Deciles are pooled with neighbours to reach 10 events rather than blanked.
            for g in risk_groups(tmp,'pred'):
                o=km_risk(g.duration_days,g.event,h) if g.event.sum()>=10 else np.nan
                lo,hi=ci_bootstrap(g,lambda v:km_risk(v.duration_days,v.event,h),n_boot,rng) if np.isfinite(o) else (np.nan,np.nan)
                points.append(dict(version=name,group=f'{g.pred.min():.3f}–{g.pred.max():.3f}',n=len(g),events=int(g.event.sum()), predicted=g.pred.mean(),observed=o,lower=lo,upper=hi))
    return pd.DataFrame(rows),pd.DataFrame(points)


def checklist_calibration(d, n_boot=1000, seed=0, observed_only=False):
    """Observed KM risk by score total (pooled to 10 events), flag group and overall.

    observed_only reports observed risk without a predicted probability, for checklists
    whose exported calibrator does not apply (the generalized rule).
    """
    d=d.dropna(subset=['flag','model_horizon_days']+([] if observed_only else ['probability'])).copy()
    rows=[]; rng=np.random.default_rng(seed)
    for (bundle,horizon), part in d.groupby(['bundle','model_horizon_days']):
        for grouping in ('risk_score','flag','overall'):
            if grouping=='risk_score': groups=pool_adjacent([g for _,g in part.groupby('risk_score',sort=True)])
            elif grouping=='flag': groups=[g for _,g in part.groupby('flag')]
            else: groups=[part]
            for g in groups:
                h=float(horizon); obs=km_risk(g.duration_days,g.event,h) if g.event.sum()>=10 else np.nan
                lo,hi=ci_bootstrap(g,lambda z:km_risk(z.duration_days,z.event,h),n_boot,rng) if np.isfinite(obs) else (np.nan,np.nan)
                groupspec=(f'{g[grouping].min():g}–{g[grouping].max():g}' if grouping=='risk_score' else str(int(g.flag.iloc[0])) if grouping=='flag' else 'all')
                exp=np.nan if observed_only else g.probability.mean()
                rows.append(dict(bundle=bundle,horizon=h,grouping=grouping,group=groupspec,n=len(g),events=int(g.event.sum()), predicted=exp, observed=obs,lower=lo,upper=hi, observed_expected=obs/exp if exp>0 else np.nan))
    return pd.DataFrame(rows)


def subgroup_metrics(d, variables, n_boot=1000, seed=0, probabilities=None):
    rows=[]; interactions=[]; rng=np.random.default_rng(seed)
    # One full-cohort scale for every subgroup HR and the interaction test.
    center,sd=d.risk_score.mean(),d.risk_score.std(ddof=0)
    if not sd>0: return pd.DataFrame(),pd.DataFrame()
    for variable in variables:
        if variable not in d: continue
        for level,g in d.groupby(variable,dropna=False):
            row=dict(variable=variable,level=str(level),n=len(g),events=int(g.event.sum()))
            if g.event.sum()>=10 and len(g)>10:
                t,e,s=g.duration_days,g.event,g.risk_score
                row['c_index']=fet.harrell_c(t,e,s)
                row.update(hr_per_sd(t,e,s,center,sd))
                try: row['mean_auc']=fet.mean_auc(t,e,s,fet.auc_horizon_grid(t,e))
                except Exception: row['mean_auc']=np.nan
                row['c_lower'],row['c_upper']=ci_bootstrap(g,lambda z:fet.harrell_c(z.duration_days,z.event,z.risk_score),n_boot,rng)
                def _hr(z): return hr_per_sd(z.duration_days,z.event,z.risk_score,center,sd)['hr_per_sd']
                row['hr_per_sd_lo'],row['hr_per_sd_hi']=ci_bootstrap(g,_hr,n_boot,rng)
                def _auc(z): return fet.mean_auc(z.duration_days,z.event,z.risk_score,fet.auc_horizon_grid(z.duration_days,z.event))
                row['mean_auc_lower'],row['mean_auc_upper']=ci_bootstrap(g,_auc,n_boot,rng)
                if probabilities and probabilities in g:
                    o=km_risk(t,e,730); ex=g[probabilities].mean(); row['observed_expected']=o/ex if ex>0 else np.nan
                    def _oe(z):
                        expected=z[probabilities].mean()
                        return km_risk(z.duration_days,z.event,730)/expected if expected>0 else np.nan
                    row['oe_lower'],row['oe_upper']=ci_bootstrap(g,_oe,n_boot,rng)
            rows.append(row)
        # Likelihood-ratio test of all score × category interactions.
        part=d[[variable,'duration_days','event','risk_score']].dropna().copy()
        p=np.nan; note=''
        try:
            levels=pd.get_dummies(part[variable].astype(str),drop_first=True,dtype=float)
            if levels.empty or part.event.sum()<10: raise ValueError('insufficient levels/events')
            levels.columns=[f'group_{i}' for i in range(levels.shape[1])]
            base=pd.concat([part[['duration_days','event']].reset_index(drop=True), ((part.risk_score-center)/sd).reset_index(drop=True).rename('z'),levels.reset_index(drop=True)],axis=1)
            full=base.copy()
            for c in levels: full['z_'+c]=base.z*base[c]
            b=CoxPHFitter().fit(base,'duration_days','event'); f=CoxPHFitter().fit(full,'duration_days','event')
            p=chi2.sf(2*(f.log_likelihood_-b.log_likelihood_),len(levels.columns))
        except Exception as exc: note=f'{type(exc).__name__}: {exc}'
        interactions.append(dict(variable=variable,n=len(part),events=int(part.event.sum()),interaction_p=p,note=note))
    return pd.DataFrame(rows),pd.DataFrame(interactions)


def plot_forest(table, path, title, label_col='label'):
    d=table.dropna(subset=['hr_per_sd']).copy() if 'hr_per_sd' in table else pd.DataFrame(columns=['hr_per_sd',label_col])
    fig,ax=plt.subplots(figsize=(7,max(3,len(d)*.38+1.2)))
    if len(d):
        y=np.arange(len(d)); ax.scatter(d.hr_per_sd,y,color='#2266a0')
        for yi,(_,row) in zip(y,d.iterrows()):
            if np.isfinite(row.get('hr_per_sd_lo',np.nan)) and np.isfinite(row.get('hr_per_sd_hi',np.nan)):
                ax.plot([row.hr_per_sd_lo,row.hr_per_sd_hi],[yi,yi],color='#2266a0')
        ax.set_yticks(y,d[label_col]); ax.invert_yaxis(); ax.set_xscale('log')
    else: ax.text(.5,.5,'Not estimable',ha='center',transform=ax.transAxes)
    ax.axvline(1,color='grey',ls='--'); ax.set(xlabel='Hazard ratio per SD',title=title); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_calibration(table, path, title):
    fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],'--',color='grey')
    # Checklists are separated by horizon as well as bundle; their predictions are horizon-specific.
    keys=['version'] if 'version' in table else [c for c in ('bundle','horizon') if c in table]
    for name,d in table.groupby(keys):
        name=name if isinstance(name,tuple) else (name,)
        label=' · '.join(f'{int(v)} d' if k=='horizon' else str(v) for k,v in zip(keys,name))
        d=d.dropna(subset=['observed','predicted'])
        if d.empty: continue
        ax.scatter(d.predicted,d.observed,label=label)
        for _,r in d.iterrows():
            if np.isfinite(r.lower) and np.isfinite(r.upper): ax.plot([r.predicted]*2,[r.lower,r.upper],color='grey',alpha=.5)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted risk',ylabel='KM observed risk',title=title)
    if ax.get_legend_handles_labels()[0]: ax.legend(frameon=False,fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def crpc_anchor_audit(meta):
    """Events the CRPC-anchored landmark-0 frame cannot score: before or on the CRPC date."""
    crpc=meta.crpc_date.dt.normalize(); rows=[]
    for ep in ('avpc','nepc','platinum'):
        ev=meta[f'{ep}_date'].dt.normalize(); both=crpc.notna()&ev.notna()
        rows.append(dict(endpoint=ep, crpc_dated=int(crpc.notna().sum()), events_with_crpc=int(both.sum()),
                         before_crpc=int((both&(ev<crpc)).sum()), same_day_as_crpc=int((both&(ev==crpc)).sum()),
                         after_crpc=int((both&(ev>crpc)).sum())))
    return pd.DataFrame(rows)


def reanchor_local(local, meta):
    """CRPC clock (exploratory): scores use labs strictly before CRPC."""
    from reviewer_sources import ids
    d=ids(local).merge(meta[['person_id','crpc_date']],on='person_id',how='inner',validate='many_to_one')
    d=d[d.crpc_date.notna()].copy()
    original=date(d.TREATMENT_ANCHOR_DATE)
    d['AGE_AT_TREATMENTSTART']=pd.to_numeric(d.AGE_AT_TREATMENTSTART,errors='coerce')+(d.crpc_date-original).dt.days/365.25
    d['TREATMENT_ANCHOR_DATE']=d.crpc_date
    return d.drop(columns=['person_id','crpc_date'])


def score_comparison_endpoint(local, meta, endpoint, dates, bundle_stem, bundle, landmark=180, competing=None):
    """Use the transfer's exact feature build and incident-event gate for a new endpoint.

    dates and competing are patient/date frames; competing dates censor follow-up.
    """
    from reviewer_sources import ids
    tmp=ids(local).merge(meta[['person_id','anchor_date']],on='person_id',how='inner',validate='many_to_one')
    frame=dates[['person_id','event_date']].copy()
    if frame.person_id.duplicated().any(): raise ValueError('duplicate event dates')
    tmp=tmp.merge(frame,on='person_id',how='left',validate='many_to_one')
    if competing is not None:
        tmp=tmp.merge(competing[['person_id','competing_date']],on='person_id',how='left',validate='many_to_one')
        contact=date(tmp.LAST_CONTACT_DATE)
        tmp['LAST_CONTACT_DATE']=pd.concat([contact,date(tmp.competing_date)],axis=1).min(axis=1)
        tmp.loc[tmp.event_date>tmp.LAST_CONTACT_DATE,'event_date']=pd.NaT
    tmp['REVIEWER_EVENT_DATE']=date(tmp.event_date)
    tmp['REVIEWER_EVENT']=tmp.REVIEWER_EVENT_DATE.notna().astype(int)
    name='_reviewer_endpoint'
    fet.ENDPOINT_SOURCES[name]=('REVIEWER_EVENT','REVIEWER_EVENT_DATE')
    try:
        result=fet.evaluate_transfer(tmp.drop(columns=['person_id']),{bundle_stem:bundle},endpoints=(name,),landmarks=(landmark,),
             lab_mapping=fet.load_lab_mapping(fet.DEFAULT_LAB_MAPPING),n_bootstrap=0,verbose=False)
    finally:
        del fet.ENDPOINT_SOURCES[name]
    scores=result['scores'].copy(); scores['endpoint']=endpoint
    if not scores.empty: scores['person_id']=pd.to_numeric(scores.person_id,errors='coerce').astype('Int64').astype('string')
    return scores


def first_start(meds, therapy, after=None):
    """First start of one therapy; `after` (person_id-indexed dates) keeps starts on or after that date."""
    d=meds[meds.therapy.eq(therapy)][['person_id','med_date']]
    if after is not None:
        d=d.merge(after.rename('after_date'),left_on='person_id',right_index=True,how='inner')
        d=d[d.med_date>=d.after_date]
    return d.groupby('person_id',as_index=False).med_date.min().rename(columns={'med_date':'event_date'})


THERAPY_CLASS={'docetaxel':'taxane','cabazitaxel':'taxane'}


def first_new_therapy(meds, meta, landmark=180):
    """First systemic class whose first-ever start falls after the landmark.

    A class started at or before the landmark is not new, so continued or repeated
    administrations never count. Returns one row per patient and the same-day tie set.
    """
    d=meds.assign(therapy_class=meds.therapy.replace(THERAPY_CLASS))
    first=d.groupby(['person_id','therapy_class'],as_index=False).med_date.min()
    first=first.merge(meta[['person_id','anchor_date']],on='person_id',how='inner',validate='many_to_one')
    first=first[first.med_date>first.anchor_date+pd.to_timedelta(landmark,unit='D')]
    first=first[first.med_date.eq(first.groupby('person_id').med_date.transform('min'))]
    classes=first.groupby('person_id').therapy_class.nunique(); tied=set(classes[classes>1].index)
    return first[~first.person_id.isin(tied)][['person_id','therapy_class','med_date']].reset_index(drop=True), tied


def comparison_metrics(d, n_boot=1000, seed=0, center=None, scale=None):
    """C-index and HR per SD; pass one center/scale so HRs share a scale across endpoints and tables."""
    reference=d.drop_duplicates('person_id').risk_score
    center=reference.mean() if center is None else center
    scale=reference.std(ddof=0) if scale is None else scale
    rows=[]; rng=np.random.default_rng(seed)
    for ep,g in d.groupby('endpoint'):
        if g.person_id.duplicated().any(): raise ValueError(f'multiple scores for {ep}')
        row=dict(endpoint=ep,label=ep,n=len(g),events=int(g.event.sum()),c_index=np.nan,hr_per_sd=np.nan,hr_per_sd_lo=np.nan,hr_per_sd_hi=np.nan,c_lower=np.nan,c_upper=np.nan)
        if g.event.sum()>=10:
            row['c_index']=fet.harrell_c(g.duration_days,g.event,g.risk_score)
            row.update(hr_per_sd(g.duration_days,g.event,g.risk_score,center,scale))
            row['c_lower'],row['c_upper']=ci_bootstrap(g,lambda z:fet.harrell_c(z.duration_days,z.event,z.risk_score),n_boot,rng)
            def hr(x): return hr_per_sd(x.duration_days,x.event,x.risk_score,center,scale)['hr_per_sd']
            row['hr_per_sd_lo'],row['hr_per_sd_hi']=ci_bootstrap(g,hr,n_boot,rng)
        rows.append(row)
    return pd.DataFrame(rows)


def lead_death_audit(d):
    """Count potential event-free controls who died before the 730-day cutoff."""
    death=date(d.death_date); landmark=date(d.landmark_date)
    died=death.notna() & death.gt(landmark) & (death-landmark).dt.days.lt(730)
    excluded=d.event.eq(0)&(d.duration_days.lt(730)|died)
    return {'event_free_short_followup':int(excluded.sum()),
            'event_free_short_followup_died_before_730':int((excluded&died).sum()),
            'death_date_missing':int((excluded&death.isna()).sum())}


def crpc_date_agreement(crpc_labels, avpc_labels, patients):
    """Compare the mcrpc and AVPC-label CRPC dates within the cohort; aggregate counts only."""
    from reviewer_sources import one_patient
    for name,frame in (('CRPC',crpc_labels),('AVPC',avpc_labels)):
        if 'crpc_date' not in frame: raise ValueError(f'{name} labels missing crpc_date')
    m=one_patient(crpc_labels,['crpc_date'],'CRPC labels').merge(one_patient(avpc_labels,['crpc_date'],'AVPC labels'),
        on='person_id',how='outer',suffixes=('_mcrpc','_avpc'))
    cohort=pd.DataFrame({'person_id':pd.Series(patients).astype('string').drop_duplicates()})
    m=cohort.merge(m,on='person_id',how='left',validate='one_to_one')
    a,b=date(m.crpc_date_mcrpc),date(m.crpc_date_avpc); both=a.notna()&b.notna()
    gap=(a-b).dt.days[both]; diff=gap.abs()
    return pd.DataFrame([dict(cohort_patients=len(m), mcrpc_dated=int(a.notna().sum()), avpc_label_dated=int(b.notna().sum()),
        both_dated=int(both.sum()), mcrpc_only=int((a.notna()&b.isna()).sum()), avpc_label_only=int((a.isna()&b.notna()).sum()),
        identical=int(diff.eq(0).sum()), within_30_days=int(diff.le(30).sum()), within_90_days=int(diff.le(90).sum()),
        mcrpc_later=int(gap.gt(0).sum()), mcrpc_earlier=int(gap.lt(0).sum()),
        median_abs_difference_days=float(diff.median()) if len(diff) else np.nan)])


def main_figure(lead, comparisons, path):
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1.2,1]})
    for i,(ep,color) in enumerate([('avpc','#2266a0'),('nepc','#d37925')]):
        z=lead[(lead.endpoint==ep)&lead.sensitivity.eq(PRIMARY_CONTROLS)].set_index('band').reindex([v[2] for v in BANDS])
        x=np.arange(len(BANDS))+(i-.5)*.12
        a.scatter(x,z.auc,label=ep.upper(),color=color)
        for xi,(_,r) in zip(x,z.iterrows()):
            if np.isfinite(r.lower) and np.isfinite(r.upper): a.plot([xi,xi],[r.lower,r.upper],color=color)
    a.axhline(.5,ls='--',color='grey'); a.set_xticks(range(len(BANDS)),[v[2] for v in BANDS]); a.set(ylim=(0,1),xlabel='Days to diagnosis',ylabel='Case–control AUC',title='A  Lead time'); a.legend(frameon=False)
    z=comparisons.dropna(subset=['hr_per_sd']).copy(); y=np.arange(len(z))
    if len(z):
        b.scatter(z.hr_per_sd,y,color='#2266a0')
        for yi,(_,r) in zip(y,z.iterrows()):
            if np.isfinite(r.hr_per_sd_lo) and np.isfinite(r.hr_per_sd_hi): b.plot([r.hr_per_sd_lo,r.hr_per_sd_hi],[yi,yi],color='#2266a0')
        b.set_yticks(y,z.endpoint); b.invert_yaxis(); b.set_xscale('log')
    b.axvline(1,color='grey',ls='--'); b.set(xlabel='Hazard ratio per SD',title='B  Comparison endpoints')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
