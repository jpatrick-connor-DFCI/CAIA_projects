import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'COMPASS' / 'federated_models'))
import subgroup_transfer as st


def _meta():
    anchor = pd.Timestamp('2020-01-01')
    return pd.DataFrame({
        'person_id': ['1', '2', '3', '4', '5'],
        'anchor_date': [anchor] * 5,
        # 1: metastatic before anchor; 2: at day 100; 3: undated metastatic; 4: non-metastatic; 5: no label
        'met_date': [anchor - pd.Timedelta(days=30), anchor + pd.Timedelta(days=100), pd.NaT, pd.NaT, pd.NaT],
        'is_metastatic': pd.array([True, True, True, False, pd.NA], dtype='boolean'),
        # mCRPC: 1 at day 50; 2 at day 150; 3 undated; 4 never; 5 no label
        'mcrpc_date': [anchor + pd.Timedelta(days=50), anchor + pd.Timedelta(days=150), pd.NaT, pd.NaT, pd.NaT],
        'is_mcrpc': pd.array([True, True, True, False, pd.NA], dtype='boolean'),
    })


def test_mcrpc_status_levels():
    m = _meta()
    ref = m.anchor_date + pd.Timedelta(days=90)
    assert st.mcrpc_status(m.mcrpc_date, m.is_mcrpc, ref).tolist() == ['yes', 'no', 'date unknown', 'no', 'missing']


def test_status_shifts_with_landmark():
    m = _meta()
    lm0, lm90, lm180 = (st.landmark_subgroups(m, lm) for lm in (0, 90, 180))
    assert lm0.metastatic_at_landmark.tolist() == ['yes', 'no', 'date unknown', 'no', 'missing']
    assert lm180.metastatic_at_landmark.tolist()[1] == 'yes'
    assert lm0.mcrpc_at_landmark.tolist()[:2] == ['no', 'no']
    assert lm90.mcrpc_at_landmark.tolist()[:2] == ['yes', 'no']
    assert lm180.mcrpc_at_landmark.tolist()[:2] == ['yes', 'yes']
    # 'Ever' ignores the landmark.
    assert lm0.metastatic_ever.tolist() == ['yes', 'yes', 'yes', 'no', 'missing']
    assert lm0.mcrpc_ever.tolist() == lm180.mcrpc_ever.tolist() == ['yes', 'yes', 'yes', 'no', 'missing']


def test_mcrpc_fields_validates_and_parses():
    with pytest.raises(ValueError): st.mcrpc_fields(pd.DataFrame({'DFCI_MRN': [1], 'is_mcrpc': [True]}))
    out = st.mcrpc_fields(pd.DataFrame({'DFCI_MRN': [1, 2], 'is_mcrpc': [True, False], 'mcrpc_date': ['2021-03-04', None]}))
    assert out.person_id.tolist() == ['1', '2'] and out.mcrpc_date.iloc[0] == pd.Timestamp('2021-03-04')


def test_patient_frame_joins_sources_and_keeps_unlabelled():
    local = pd.DataFrame({'DFCI_MRN': [1, 1, 2], 'TREATMENT_ANCHOR_DATE': ['2020-01-01'] * 2 + ['2020-06-01']})
    mcrpc = pd.DataFrame({'DFCI_MRN': [1], 'is_mcrpc': [True], 'mcrpc_date': ['2021-01-01'], 'met_date': ['2020-02-01']})
    avpc = pd.DataFrame({'DFCI_MRN': [1, 2], 'is_metastatic': [True, False]})
    m = st.patient_frame(local, mcrpc, avpc).set_index('person_id')
    assert m.loc['1', 'met_date'] == pd.Timestamp('2020-02-01') and bool(m.loc['1', 'is_mcrpc'])
    assert pd.isna(m.loc['2', 'is_mcrpc']) and not bool(m.loc['2', 'is_metastatic'])
    s = st.landmark_subgroups(m.reset_index(), 0).set_index('person_id')
    assert s.loc['2', 'mcrpc_at_landmark'] == 'missing' and s.loc['2', 'metastatic_at_landmark'] == 'no'
    with pytest.raises(ValueError): st.patient_frame(local, mcrpc.drop(columns='met_date'), avpc)


def _scores(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({'person_id': [str(i) for i in range(n)],
                         'risk_score': rng.normal(size=n),
                         'duration_days': rng.integers(30, 2000, n).astype(float),
                         'event': np.tile([1, 0], n // 2)})
    return pd.concat([base.assign(endpoint='avpc', model_family='xgboost_cox', bundle='x', landmark_days=lm,
                                  config='both', model_horizon_days=np.nan) for lm in (0, 90)], ignore_index=True)


def _big_meta(n=60):
    anchor = pd.Timestamp('2020-01-01')
    return pd.DataFrame({'person_id': [str(i) for i in range(n)], 'anchor_date': [anchor] * n,
                         'met_date': [anchor if i % 2 else pd.NaT for i in range(n)],
                         'is_metastatic': pd.array([bool(i % 2) for i in range(n)], dtype='boolean'),
                         'mcrpc_date': [anchor + pd.Timedelta(days=60) if i % 4 == 1 else pd.NaT for i in range(n)],
                         'is_mcrpc': pd.array([i % 4 == 1 for i in range(n)], dtype='boolean')})


def test_run_one_row_per_level_and_identity():
    s, m = _scores(), _big_meta()
    metrics, inter = st.run(st.select_scores(s), m, st.PRIMARY_VARIABLES, n_boot=3)
    lm0 = metrics[metrics.landmark_days.eq(0)]
    assert set(lm0.variable) == set(st.PRIMARY_VARIABLES)
    assert lm0.loc[lm0.variable.eq('full_cohort'), 'n'].item() == 60
    # Levels partition the frame for every variable and landmark.
    assert (metrics.groupby(['landmark_days', 'variable']).n.sum() == 60).all()
    assert metrics.loc[metrics.landmark_days.eq(90) & metrics.variable.eq('mcrpc_at_landmark') & metrics.level.eq('yes'), 'n'].item() == 15
    assert 'full_cohort' not in set(inter.variable)


def test_select_scores_rejects_duplicates():
    s = _scores()
    with pytest.raises(ValueError): st.select_scores(pd.concat([s, s.iloc[:1]]))
