"""Build protected, patient-level reviewer inputs from existing DFCI sources."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import federated_clinical_comparison as fcc
from reviewer_extended import date


def read(path, columns=None):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Required input is missing: {path}')
    if path.suffix in ('.parquet','.pq'): return pd.read_parquet(path,columns=columns)
    return pd.read_csv(path,usecols=columns,low_memory=False)


def ids(frame):
    x=frame.copy()
    source='DFCI_MRN' if 'DFCI_MRN' in x else 'person_id'
    x['person_id']=pd.to_numeric(x[source],errors='coerce').astype('Int64').astype('string')
    return x.dropna(subset=['person_id'])


def one_patient(frame, cols, name):
    d=ids(frame)[['person_id',*cols]].copy()
    for c in cols:
        conflicts=d.groupby('person_id')[c].nunique(dropna=True)
        if (conflicts>1).any(): raise ValueError(f'{name}: conflicting {c} for {(conflicts>1).sum()} patients')
    return d.groupby('person_id',as_index=False,sort=False).first()


def build_patient_frame(local, crpc, avpc, intent, gleason_path, somatic_path, somatic_manifest, death, demographics=None):
    required=['TREATMENT_ANCHOR_DATE','AGE_AT_TREATMENTSTART','LAST_CONTACT_DATE','PLATINUM_DATE','PLATINUM','AVPC_DATE','AVPC','NEPC_DATE','NEPC']
    meta=one_patient(local,required,'longitudinal data').rename(columns={'TREATMENT_ANCHOR_DATE':'anchor_date','AGE_AT_TREATMENTSTART':'age','LAST_CONTACT_DATE':'last_contact_date','PLATINUM_DATE':'platinum_date','AVPC_DATE':'avpc_date','NEPC_DATE':'nepc_date'})
    for ep in ('platinum','avpc','nepc'):
        meta.loc[~pd.to_numeric(meta[ep.upper()],errors='coerce').eq(1),ep+'_date']=pd.NaT
    ccols=['crpc_date','crpc_date_precision','met_date','lu_psma_date']
    for c in ccols:
        if c not in crpc: raise ValueError(f'CRPC labels missing {c}')
    meta=meta.merge(one_patient(crpc,ccols,'CRPC labels'),on='person_id',how='left',validate='one_to_one')
    criteria=[f'c{i}_met' for i in range(1,8)]
    # is_metastatic separates non-metastatic patients from metastatic ones with no met_date.
    if 'is_metastatic' not in avpc: raise ValueError('AVPC labels missing is_metastatic')
    meta=meta.merge(one_patient(avpc,criteria+['is_metastatic'],'AVPC labels'),on='person_id',how='left',validate='one_to_one')
    meta=meta.merge(one_patient(intent,['ADT_INTENT'],'ADT intent').rename(columns={'ADT_INTENT':'adt_intent'}),on='person_id',how='left',validate='one_to_one')
    if death is not None:
        meta=meta.merge(one_patient(death,['DEATH_DATE'],'death cohort').rename(columns={'DEATH_DATE':'death_date'}),on='person_id',how='left',validate='one_to_one')
    else: meta['death_date']=pd.NaT
    if demographics is not None:
        columns=[c for c in ('race','ethnicity') if c in demographics]
        if not columns: raise ValueError('Demographics file must have race and/or ethnicity columns')
        meta=meta.merge(one_patient(demographics,columns,'demographics'),on='person_id',how='left',validate='one_to_one')
    for col in ['anchor_date','last_contact_date','platinum_date','avpc_date','nepc_date','crpc_date','met_date','lu_psma_date','death_date']:
        meta[col]=date(meta[col])
    # Follow-up cannot extend beyond a known death for CRPC control eligibility.
    meta['last_contact_date'] = pd.concat([meta.last_contact_date,meta.death_date],axis=1).min(axis=1)
    # PSA is last valid measurement on or before CRPC.
    labs=ids(local)
    labs=labs[labs.LAB_NAME.astype('string').str.upper().eq('PSA')][['person_id','LAB_DATE','LAB_VALUE']].copy()
    labs['LAB_DATE']=date(labs.LAB_DATE); labs['LAB_VALUE']=pd.to_numeric(labs.LAB_VALUE,errors='coerce')
    labs=labs.merge(meta[['person_id','crpc_date']],on='person_id',validate='many_to_one')
    labs=labs[(labs.LAB_DATE<=labs.crpc_date)&labs.LAB_VALUE.notna()&(labs.LAB_VALUE>=0)].sort_values('LAB_DATE').drop_duplicates('person_id',keep='last')
    meta=meta.merge(labs[['person_id','LAB_VALUE']].rename(columns={'LAB_VALUE':'psa_at_crpc'}),on='person_id',how='left',validate='one_to_one')
    gleason,somatic,features=fcc.load_clinical_sources(gleason_path,somatic_path,somatic_manifest)
    cut=meta[['person_id','crpc_date']].dropna().copy(); cut['DFCI_MRN']=pd.to_numeric(cut.person_id).astype(int); cut['_landmark_date']=cut.crpc_date
    selected=fcc.sg.closest_available_by_landmark(gleason,cut[['DFCI_MRN','_landmark_date']],date_col='gleason_date',available_date_col=fcc.sg.GLEASON_AVAILABLE_DATE,value_cols=[fcc.GLEASON_FEATURE])
    selected=selected.reset_index().rename(columns={fcc.GLEASON_FEATURE:'gleason_at_crpc'})
    meta=meta.merge(ids(selected)[['person_id','gleason_at_crpc']],on='person_id',how='left',validate='one_to_one')
    specimen=fcc.select_specimen(local,somatic,features)
    meta['sequenced']=meta.person_id.isin(pd.Index(specimen.index).astype('string')).map({True:'yes',False:'no'})
    return meta, gleason


def gleason_at_landmark(meta, gleason, lm):
    cut=meta[['person_id','anchor_date']].dropna().copy()
    cut['DFCI_MRN']=pd.to_numeric(cut.person_id).astype(int)
    cut['_landmark_date']=cut.anchor_date+pd.to_timedelta(lm-1,unit='D')
    selected=fcc.sg.closest_available_by_landmark(gleason,cut[['DFCI_MRN','_landmark_date']],date_col='gleason_date',available_date_col=fcc.sg.GLEASON_AVAILABLE_DATE,value_cols=[fcc.GLEASON_FEATURE])
    return ids(selected.reset_index()).rename(columns={fcc.GLEASON_FEATURE:'gleason_at_landmark'})[['person_id','gleason_at_landmark']]


def medication_dates(path, patients):
    """Scan only required columns for exact drug names; return every distinct dated administration.

    First starts (overall, after CRPC or after the landmark) are derived downstream.
    """
    wanted={'DOCETAXEL':'docetaxel','CABAZITAXEL':'cabazitaxel','RADIUM RA 223 DICHLORIDE':'radium',
            'CARBOPLATIN':'platinum','CISPLATIN':'platinum',
            'ABIRATERONE ACETATE':'arpi','ENZALUTAMIDE':'arpi','APALUTAMIDE':'arpi','DAROLUTAMIDE':'arpi'}
    path=Path(path); patient_set=set(pd.Series(patients).astype('string'))
    if path.suffix in ('.parquet','.pq'):
        import pyarrow.dataset as ds
        batches=ds.dataset(path,format='parquet').scanner(columns=['DFCI_MRN','NCI_PREFERRED_MED_NM','MED_START_DT'],batch_size=250000).to_batches()
        chunks=(b.to_pandas() for b in batches)
    else: chunks=pd.read_csv(path,usecols=['DFCI_MRN','NCI_PREFERRED_MED_NM','MED_START_DT'],chunksize=250000,low_memory=False)
    found=[]
    for chunk in chunks:
        chunk=ids(chunk); chunk=chunk[chunk.person_id.isin(patient_set)].copy()
        chunk['therapy']=chunk.NCI_PREFERRED_MED_NM.astype('string').str.upper().str.strip().map(wanted)
        chunk=chunk.dropna(subset=['therapy','MED_START_DT'])
        chunk['med_date']=date(chunk.MED_START_DT)
        found.append(chunk[['person_id','therapy','med_date']].dropna())
    if not found: return pd.DataFrame(columns=['person_id','therapy','med_date'])
    return pd.concat(found,ignore_index=True).drop_duplicates().sort_values('med_date').reset_index(drop=True)
