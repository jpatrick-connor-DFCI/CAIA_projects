"""Patient-level high/low TF-IDF comparison, without resampling or continuous risk."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import mannwhitneyu

from COMPASS.multivariable_models import text_tfidf_association as tfa


def fixture():
    rng = np.random.default_rng(42)
    y = rng.uniform(size=(50, 4))
    y[:10, 0] = 0
    y[:, 1] = 1
    y[:, 2] = 0
    groups = pd.Series(["top"] * 20 + ["bottom"] * 20 + ["middle"] * 10)
    return sparse.csr_matrix(y), groups


def run(y, groups, **kwargs):
    return tfa.compare_tfidf_groups(y, pd.RangeIndex(y.shape[0]), list("abcd")[:y.shape[1]], groups, 90, **kwargs)


def test_weights_match_scipy_and_group_means():
    y, groups = fixture()
    result = run(y, groups)
    assert result['status'].eq('ok').all()
    h, l = y[:20].toarray(), y[20:40].toarray()
    test = mannwhitneyu(h, l, axis=0, method='asymptotic', use_continuity=True, alternative='two-sided')
    np.testing.assert_allclose(result.p_value, test.pvalue)
    np.testing.assert_allclose(result.auc_high_vs_low, test.statistic / 400)
    np.testing.assert_allclose(result.mean_tfidf_high, h.mean(axis=0))
    np.testing.assert_allclose(result.mean_tfidf_low, l.mean(axis=0))
    np.testing.assert_allclose(result.difference_high_minus_low, h.mean(axis=0) - l.mean(axis=0))
    assert result.loc[1:2, 'p_value'].eq(1).all()  # constant and all-zero terms are valid null comparisons
    assert result.loc[1:2, 'auc_high_vs_low'].eq(0.5).all()


def test_reversing_groups_flips_effect_but_keeps_two_sided_pvalue():
    y, groups = fixture()
    first = run(y, groups)
    reverse = run(y, groups.replace({'top': 'bottom', 'bottom': 'top'}))
    np.testing.assert_allclose(first.p_value, reverse.p_value)
    np.testing.assert_allclose(first.difference_high_minus_low, -reverse.difference_high_minus_low)
    np.testing.assert_allclose(first.auc_high_vs_low, 1 - reverse.auc_high_vs_low)


def test_middle_patients_are_excluded_and_input_order_and_blocks_do_not_matter():
    y, groups = fixture()
    first = run(y, groups)
    altered = y.toarray()
    altered[40:] = 1000
    other = run(sparse.csr_matrix(altered), groups.iloc[::-1], block_size=1)
    pd.testing.assert_frame_equal(first, other)
    assert first.n_high.eq(20).all() and first.n_low.eq(20).all()


def test_small_or_absent_groups_return_descriptions_without_inference():
    y, groups = fixture()
    small = run(y, groups, min_group_patients=25)
    assert small.p_value.isna().all()
    assert small.mean_tfidf_high.notna().all() and small.auc_high_vs_low.notna().all()
    absent = run(y, groups.replace({'bottom': 'middle'}))
    assert absent.n_low.eq(0).all()
    assert absent.mean_tfidf_low.isna().all() and absent.p_value.isna().all()
    assert absent.auc_high_vs_low.isna().all()
    empty = run(y[:, :0], groups)
    assert empty.empty and list(empty.columns) == tfa.RESULT_COLUMNS


def test_fdr_spans_all_terms_and_landmarks():
    from scipy.stats import false_discovery_control
    frames = [pd.DataFrame({'landmark': lm, 'p_value': [0.001 * (lm + 1), 0.05, np.nan]}) for lm in (0, 90)]
    result = tfa.adjust_fdr(frames)
    finite = result.dropna(subset=['p_value'])
    np.testing.assert_allclose(finite.q_value, false_discovery_control(finite.p_value))
    assert result.loc[result.p_value.isna(), 'q_value'].isna().all()


def test_missing_labels_and_duplicate_patients_are_rejected():
    y, groups = fixture()
    with pytest.raises(ValueError, match='Missing risk group'):
        run(y, groups.iloc[1:])
    with pytest.raises(ValueError, match='one row per patient'):
        run(y, pd.concat([groups, groups.iloc[:1]]))
