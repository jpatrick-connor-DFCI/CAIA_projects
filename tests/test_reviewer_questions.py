import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'COMPASS' / 'federated_models'))
import reviewer_questions as rq
import reviewer_extended as ext
import reviewer_specificity as spec


def test_lead_bands_and_control_boundaries():
    df=pd.DataFrame({'duration_days':[1,180,181,365,366,730,731,729,730],
                     'event':[1,1,1,1,1,1,1,0,0], 'risk_score':range(9)})
    assert [int(rq.lead_groups(df,lo,hi)[0].sum()) for lo,hi,_ in rq.BANDS] == [2,2,2,1]
    assert rq.lead_groups(df,0,180)[1].tolist() == [False]*8+[True]
    assert rq.lead_groups(df,0,180,True)[1].sum() == 2
    # The last band is closed: its controls must outlast its latest possible case.
    long=pd.DataFrame({'duration_days':[1824,1825],'event':[0,0],'risk_score':[0.,1.]})
    assert rq.lead_groups(long,730,1825)[1].tolist()==[False,True]
    assert rq.lead_groups(long,0,180)[1].all()
    with pytest.raises(ValueError): rq.lead_groups(long,1825,np.inf)


def test_crpc_outcomes_exclude_competing_and_short_followup():
    df=pd.DataFrame({'crpc_date':['2020-01-01']*6,
        'landmark_date':['2019-12-01']*5+['2020-01-01'],
        'last_contact_date':['2023-01-01','2023-01-01','2020-12-01','2023-01-01','2023-01-01','2023-01-01'],
        'avpc_date':['2021-12-31',None,None,None,'2022-01-01','2020-01-01'],
        'nepc_date':[None,None,None,'2020-06-01',None,None]})
    out=spec.define_crpc_outcome(df,'avpc')
    assert out.outcome.iloc[0]==1 # CRPC + 730 is inclusive (leap year)
    assert out.outcome.iloc[1]==0
    assert out.exclusion.iloc[2]=='insufficient follow-up'
    assert out.exclusion.iloc[3]=='other phenotype'
    assert out.outcome.iloc[4]==0 # diagnosis after 730 days
    assert out.exclusion.iloc[5]=='other CRPC timing'
    composite=spec.define_crpc_outcome(df,'avpc_or_nepc')
    assert composite.outcome.iloc[3]==1 # NEPC is a composite case, not an exclusion
    assert composite.outcome.iloc[0]==1 and composite.outcome.iloc[1]==0
    assert not composite.exclusion.eq('other phenotype').any()
    with pytest.raises(ValueError): spec.define_crpc_outcome(df,'platinum')


def test_c7_only_and_breslow_offset():
    labels=pd.DataFrame({'is_avpc':[True,True,False],**{f'c{i}_met':[i==7,i in (1,7),True] for i in range(1,8)}})
    assert ext.criterion_counts(labels).set_index('criterion').loc['C7 only','cases']==1
    assert ext.breslow_at([1,2,3],[1,1,0],[0,0,0],2)==pytest.approx(1/3+1/2)
    assert ext.breslow_at([1,2,3],[1,1,0],[np.log(2)]*3,2)==pytest.approx((1/3+1/2)/2)


def test_crpc_specificity_runs_with_complete_and_missing_predictors():
    rng=np.random.default_rng(42); count=100
    d=pd.DataFrame({'outcome':np.tile([0.,1.],count//2), 'risk_score':rng.normal(size=count),
        'age':rng.normal(70,6,count),'psa_at_crpc':rng.lognormal(2,1,count),
        'anchor_date':pd.Timestamp('2018-01-01'), 'crpc_date':pd.Timestamp('2020-01-01'),
        'met_date':pd.Timestamp('2019-01-01'),'is_metastatic':True,'gleason_at_crpc':rng.choice([6,8,np.nan],count),
        'adt_intent':rng.choice(['METASTATIC','LOCALIZED_ADJUVANT'],count)})
    result=spec.fit_specificity(d,n_boot=40)
    assert np.isfinite(result['combined_auc']) and np.isfinite(result['delta_lower'])


def test_checklist_calibration_uses_horizon_and_event_floor():
    d=pd.DataFrame({'bundle':['direct']*30,'model_horizon_days':[365]*30,
         'risk_score':[0]*15+[1]*15,'flag':[0]*15+[1]*15,'probability':[.1]*15+[.4]*15,
         'duration_days':[50]*10+[500]*5+[100]*10+[600]*5,'event':[1]*10+[0]*5+[1]*10+[0]*5})
    tab=ext.checklist_calibration(d,n_boot=40)
    assert len(tab)==5
    assert tab.horizon.eq(365).all() and tab.observed.notna().all()


def test_comparison_transfer_builds_incident_endpoint_and_censors_competitor(monkeypatch):
    import importlib.util
    fixture_spec=importlib.util.spec_from_file_location('transfer_fixtures',ROOT/'tests/test_federated_endpoint_transfer.py')
    fixture=importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture)
    local=fixture._local(30)
    meta=pd.DataFrame({'person_id':pd.Series(range(1000,1030),dtype='string'), 'anchor_date':fixture.ANCHOR})
    dates=pd.DataFrame({'person_id':pd.Series(['1000','1001','1002'],dtype='string'),
                        'event_date':[fixture.ANCHOR+pd.Timedelta(days=d) for d in [10,300,400]]})
    other=pd.DataFrame({'person_id':pd.Series(['1002'],dtype='string'),
                        'competing_date':[fixture.ANCHOR+pd.Timedelta(days=200)]})
    bundle=fixture._bundle()
    for model in bundle['models']: model['landmark_days']=90
    monkeypatch.setattr(ext.fet,'load_lab_mapping',lambda _:fixture.MAPPING)
    z=ext.score_comparison_endpoint(local,meta,'test',dates,'test_bundle',bundle,landmark=90,competing=other)
    z=z[z.config.eq('both')].set_index('person_id')
    assert '1000' not in z.index
    assert z.loc['1001','event']==1 and z.loc['1001','duration_days']==210
    assert z.loc['1002','event']==0 and z.loc['1002','duration_days']==110


def test_calibration_updates_keep_absolute_predictions_between_zero_and_one():
    rng=np.random.default_rng(2); n=100
    d=pd.DataFrame({'duration_days':rng.uniform(30,1500,n),'event':rng.binomial(1,.6,n),
                    'risk_score':rng.normal(size=n)})
    model={'baseline_cumhaz':{'times':[0,365,730,1095],'cumhaz':[0,.1,.2,.3]}}
    summary,points=ext.calibration_versions(d,model,n_boot=20)
    assert len(summary)==9 and set(summary.version)=={'transferred','DFCI baseline updated','DFCI slope and baseline updated'}
    assert summary.expected.between(0,1).all()
    assert points.predicted.between(0,1).all()


def test_build_patient_frame_masks_backfilled_endpoints_and_pre_crpc_psa(monkeypatch):
    import reviewer_sources as src
    from types import SimpleNamespace
    local=pd.DataFrame({'DFCI_MRN':[1,1,2], 'TREATMENT_ANCHOR_DATE':['2020-01-01']*3,
      'AGE_AT_TREATMENTSTART':[70,70,72], 'LAST_CONTACT_DATE':['2024-01-01']*3,
      'PLATINUM_DATE':['2024-01-01']*3, 'PLATINUM':[0,0,0],
      'AVPC_DATE':['2024-01-01']*3, 'AVPC':[0,0,0], 'NEPC_DATE':['2024-01-01']*3,'NEPC':[0,0,0],
      'LAB_NAME':['PSA','PSA','PSA'], 'LAB_DATE':['2020-06-01','2022-01-01','2020-05-01'], 'LAB_VALUE':[5,999,10]})
    crpc=pd.DataFrame({'DFCI_MRN':[1,2],'crpc_date':['2021-01-01','2021-01-01'],
       'crpc_date_precision':['day','day'],'met_date':['2020-01-01',None],'lu_psma_date':[None,None]})
    avpc=pd.DataFrame({'DFCI_MRN':[1,2],**{f'c{i}_met':[False,False] for i in range(1,8)},'is_metastatic':[True,False]})
    intent=pd.DataFrame({'DFCI_MRN':[1,2],'ADT_INTENT':['METASTATIC','LOCALIZED_ADJUVANT']})
    death=pd.DataFrame({'DFCI_MRN':[1,2],'DEATH_DATE':[None,None]})
    monkeypatch.setattr(src.fcc,'load_clinical_sources',lambda *args:(pd.DataFrame(),pd.DataFrame(),[]))
    monkeypatch.setattr(src.fcc,'select_specimen',lambda *args:pd.DataFrame(index=pd.Index([1],name='DFCI_MRN')))
    monkeypatch.setattr(src.fcc.sg,'closest_available_by_landmark',lambda *args,**kwargs:pd.DataFrame({src.fcc.GLEASON_FEATURE:[8]},index=pd.Index([1],name='DFCI_MRN')))
    meta,_=src.build_patient_frame(local,crpc,avpc,intent,'gleason','somatic','manifest',death)
    meta=meta.set_index('person_id')
    assert meta.platinum_date.isna().all() and meta.avpc_date.isna().all()
    assert meta.loc['1','psa_at_crpc']==5 and meta.loc['1','gleason_at_crpc']==8
    assert meta.loc['1','sequenced']=='yes' and meta.loc['2','sequenced']=='no'
    assert bool(meta.loc['1','is_metastatic']) and not bool(meta.loc['2','is_metastatic'])
    with pytest.raises(ValueError,match='is_metastatic'):
        src.build_patient_frame(local,crpc,avpc.drop(columns='is_metastatic'),intent,'gleason','somatic','manifest',death)


def test_lead_controls_respect_early_death_without_changing_all_noncase_sensitivity():
    frame=pd.DataFrame({'duration_days':[900,900,500], 'control_followup_days':[400,900,500],
                        'event':[0,0,1], 'risk_score':[1.,2.,3.]})
    _,primary=rq.lead_groups(frame,365,730)
    _,sensitivity=rq.lead_groups(frame,365,730,all_controls=True)
    assert primary.tolist()==[False,True,False]
    assert sensitivity.tolist()==[True,True,False]


def test_metastatic_status_does_not_call_undated_metastasis_no():
    status=spec.metastatic_status(
        ['2019-01-01','2021-01-01',None,None,None,'2019-01-01'],
        pd.Series([True,True,False,True,None,True],dtype='object'),
        ['2020-01-01']*5+[None])
    assert status.tolist()==['yes','no','no','date unknown','missing','missing']


def test_first_new_therapy_needs_first_ever_start_after_landmark():
    t0=pd.Timestamp('2020-01-01'); day=lambda n:t0+pd.Timedelta(days=n)
    meds=pd.DataFrame({'person_id':pd.Series(['1','1','1','2','2','3','3','4'],dtype='string'),
        'therapy':['docetaxel','docetaxel','platinum','cabazitaxel','radium','docetaxel','cabazitaxel','arpi'],
        'med_date':[day(100),day(300),day(400),day(250),day(250),day(200),day(190),day(500)]})
    meta=pd.DataFrame({'person_id':pd.Series(['1','2','3','4'],dtype='string'),'anchor_date':t0})
    first,tied=ext.first_new_therapy(meds,meta,180)
    first=first.set_index('person_id')
    # Docetaxel started before the landmark, so its later doses are not a new class.
    assert first.loc['1','therapy_class']=='platinum' and first.loc['1','med_date']==day(400)
    assert tied=={'2'} and '2' not in first.index
    assert first.loc['3','therapy_class']=='taxane' and first.loc['3','med_date']==day(190)
    assert first.loc['4','therapy_class']=='arpi'
    after=pd.Series([day(250)],index=pd.Index(['1'],name='person_id'))
    post=ext.first_start(meds,'docetaxel',after=after)
    assert post.person_id.tolist()==['1'] and post.event_date.tolist()==[day(300)]
    assert set(ext.first_start(meds,'docetaxel').person_id)=={'1','3'}


def test_risk_groups_pool_sparse_deciles_instead_of_blanking():
    rng=np.random.default_rng(0); n=200
    frame=pd.DataFrame({'pred':rng.uniform(size=n)})
    frame['event']=(frame.pred>.75).astype(int)  # events only in the top quarter
    groups=rq.risk_groups(frame,'pred')
    assert sum(len(g) for g in groups)==n
    assert all(g.event.sum()>=10 for g in groups)
    assert len(groups)<10
    assert all(a.pred.max()<=b.pred.min() for a,b in zip(groups,groups[1:]))
    assert len(rq.pool_adjacent([frame.iloc[:5].assign(event=0)]))==1


def test_subgroup_and_comparison_hrs_share_the_full_cohort_scale():
    rng=np.random.default_rng(3); n=240
    group=np.repeat(['narrow','wide'],n//2)
    score=rng.normal(size=n)*np.where(group=='narrow',.3,1.5)
    duration=rng.exponential(600/np.exp(.8*score))+1
    d=pd.DataFrame({'person_id':pd.Series(range(n)).astype('string'),'group':group,'risk_score':score,
                    'duration_days':np.minimum(duration,1500),'event':(duration<1500).astype(int)})
    table,inter=ext.subgroup_metrics(d,['group'],n_boot=5)
    center,sd=d.risk_score.mean(),d.risk_score.std(ddof=0)
    for level,g in d.groupby('group'):
        row=table.set_index('level').loc[level]
        assert row.hr_per_sd==pytest.approx(rq.hr_per_sd(g.duration_days,g.event,g.risk_score,center,sd)['hr_per_sd'])
    assert np.isfinite(inter.interaction_p.iloc[0])
    both=pd.concat([d.assign(endpoint='a'),d[d.group.eq('narrow')].assign(endpoint='b')],ignore_index=True)
    comp=ext.comparison_metrics(both,n_boot=5,center=center,scale=sd).set_index('endpoint')
    narrow=table.set_index('level').loc['narrow']
    assert comp.loc['b','hr_per_sd']==pytest.approx(narrow.hr_per_sd)
    default=ext.comparison_metrics(both,n_boot=5).set_index('endpoint')
    assert default.loc['b','hr_per_sd']==pytest.approx(narrow.hr_per_sd)  # default = patient-level scale


def test_crpc_date_agreement_and_anchor_audit():
    crpc=pd.DataFrame({'DFCI_MRN':[1,2,3],'crpc_date':['2020-01-01','2020-03-01','2020-01-01']})
    avpc=pd.DataFrame({'DFCI_MRN':[1,2,4],'crpc_date':['2020-01-01','2020-01-01','2020-01-01']})
    row=ext.crpc_date_agreement(crpc,avpc,pd.Series(['1','2','3','4','5'])).iloc[0]
    assert row.cohort_patients==5 and row.mcrpc_dated==3 and row.avpc_label_dated==3 and row.both_dated==2
    assert row.mcrpc_only==1 and row.avpc_label_only==1 and row.identical==1
    assert row.within_30_days==1 and row.within_90_days==2 and row.mcrpc_later==1 and row.mcrpc_earlier==0
    with pytest.raises(ValueError): ext.crpc_date_agreement(crpc,avpc.drop(columns='crpc_date'),['1'])
    t=lambda v:pd.to_datetime(v,format='mixed')
    meta=pd.DataFrame({'crpc_date':t(['2020-01-01','2020-01-01','2020-01-01',None]),
        'avpc_date':t(['2019-06-01','2020-01-01 13:00','2021-01-01','2021-01-01']),
        'nepc_date':t([None,None,None,None]),'platinum_date':t([None,'2020-02-01',None,None])})
    audit=ext.crpc_anchor_audit(meta).set_index('endpoint')
    assert audit.loc['avpc',['before_crpc','same_day_as_crpc','after_crpc']].tolist()==[1,1,1]
    assert audit.loc['avpc','events_with_crpc']==3 and audit.loc['nepc','events_with_crpc']==0
    assert audit.loc['platinum','after_crpc']==1


def test_generalized_checklist_is_observed_only_and_cumulative_flags_use_horizon(tmp_path):
    d=pd.DataFrame({'bundle':['general']*30,'model_horizon_days':[365]*30,
         'risk_score':[0]*15+[1]*15,'flag':[0]*15+[1]*15,'probability':np.nan,
         'duration_days':[50]*10+[500]*5+[100]*10+[600]*5,'event':[1]*10+[0]*5+[1]*10+[0]*5})
    tab=ext.checklist_calibration(d,n_boot=5,observed_only=True)
    assert len(tab)==5 and tab.predicted.isna().all() and tab.observed_expected.isna().all()
    assert tab.observed.notna().all()
    ext.plot_calibration(tab[tab.grouping.eq('risk_score')],tmp_path/'general.pdf','observed only')
    direct=ext.checklist_calibration(d.assign(probability=.3,bundle='direct'),n_boot=5)
    both=pd.concat([direct,direct.assign(horizon=1825.)])
    ext.plot_calibration(both,tmp_path/'direct.pdf','direct')
    assert (tmp_path/'direct.pdf').exists()
    rows=[]
    for lm in (0,90,180):
        for h in (365,1825):
            rows.append(pd.DataFrame({'person_id':[str(i) for i in range(12)],'endpoint':'avpc','model_family':'guideline_checklist',
                'landmark_days':lm,'model_horizon_days':h,'flag':[1]*6+[0]*6,'event':1,'risk_score':0.}))
    c=ext.cumulative_flags(pd.concat(rows,ignore_index=True),'avpc','guideline_checklist')
    assert c.model_family.eq('guideline_checklist').all() and c.ever_flagged.eq(6).all()
