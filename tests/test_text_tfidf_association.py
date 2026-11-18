"""Numerical and patient-unit checks for continuous TF-IDF/risk inference."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import spearmanr

from COMPASS.multivariable_models import text_tfidf_association as tfa


SETTINGS = dict(n_permutations=199, n_bootstrap=79, seed=11, block_size=2)


def inputs():
    rng = np.random.default_rng(93)
    n = 120
    risk = pd.Series(np.linspace(-2, 2, n), index=pd.Index(np.arange(n), name="DFCI_MRN"))
    burden = rng.normal(size=n)
    y = np.column_stack([
        np.maximum(0, 0.2 + 0.08 * risk + 0.03 * burden + rng.normal(0, 0.06, n)),
        np.maximum(0, 0.2 - 0.07 * risk + rng.normal(0, 0.04, n)),
        np.ones(n), np.zeros(n),
    ])
    cov = pd.DataFrame({"burden": burden, "constant": 1.0}, index=risk.index)
    return sparse.csr_matrix(y), risk, cov


def run(y, risk, cov, **kwargs):
    return tfa.continuous_tfidf_association(
        y, risk.index, ["positive", "negative", "constant", "absent"][:y.shape[1]],
        risk, cov, 90, **(SETTINGS | kwargs))


def test_spearman_matches_scipy_and_patient_bootstrap_reference():
    y, risk, cov = inputs()
    result = run(y, risk, cov).query("method == 'spearman'").set_index("term")
    for j, term in enumerate(("positive", "negative")):
        values = y[:, j].toarray().ravel()
        observed = spearmanr(risk, values).statistic
        assert result.loc[term, "spearman_rho"] == pytest.approx(observed)
        rng = np.random.default_rng(SETTINGS["seed"] + 1)
        boot = []
        for _ in range(SETTINGS["n_bootstrap"]):
            rows = rng.integers(0, len(risk), size=len(risk))
            boot.append(spearmanr(risk.iloc[rows], values[rows]).statistic)
        np.testing.assert_allclose(result.loc[term, ["ci95_low", "ci95_high"]].astype(float),
                                   np.quantile(boot, [0.025, 0.975]))
        # Independent reference implementation shuffles raw scores, then ranks each pairing.
        rng = np.random.default_rng(SETTINGS["seed"])
        null = [spearmanr(rng.permutation(risk), values).statistic for _ in range(SETTINGS["n_permutations"])]
        expected = (1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-14)) / (SETTINGS["n_permutations"] + 1)
        assert result.loc[term, "p_value"] == pytest.approx(expected)
    assert result.loc["positive", "spearman_rho"] > 0.5
    assert result.loc["negative", "spearman_rho"] < -0.5
    assert result.loc[["constant", "absent"], "p_value"].isna().all()
    assert result.loc["constant", "status"] == "skipped: constant TF-IDF weight"


def test_adjusted_effect_and_hc3_match_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    y, risk, cov = inputs()
    result = run(y, risk, cov).query("method == 'ols_adjusted'").set_index("term")
    x = np.column_stack([np.ones(len(risk)), (risk - risk.mean()) / risk.std(ddof=0), cov["burden"]])
    for j, term in enumerate(("positive", "negative")):
        fit = sm.OLS(y[:, j].toarray().ravel(), x).fit(cov_type="HC3", use_t=True)
        assert result.loc[term, "beta_per_sd"] == pytest.approx(fit.params[1])
        assert result.loc[term, "p_value"] == pytest.approx(fit.pvalues[1])
        np.testing.assert_allclose(result.loc[term, ["ci95_low", "ci95_high"]].astype(float), fit.conf_int()[1])
        assert result.loc[term, "adjustment_covariates"] == "burden"


def test_block_size_alignment_and_positive_risk_scaling_do_not_change_results():
    y, risk, cov = inputs()
    original = run(y, risk, cov)
    other = tfa.continuous_tfidf_association(y, risk.index, ["positive", "negative", "constant", "absent"],
        risk.iloc[::-1], cov.iloc[::-1], 90, **(SETTINGS | {"block_size": 1}))
    pd.testing.assert_frame_equal(original, other, check_exact=False, atol=1e-12, rtol=1e-10)
    scaled = run(y, 10 + 3 * risk, cov)
    for column in ("spearman_rho", "beta_per_sd", "p_value", "ci95_low", "ci95_high"):
        np.testing.assert_allclose(original[column], scaled[column], atol=1e-12, equal_nan=True)


def test_invalid_scores_constant_risk_small_samples_and_empty_vocabulary():
    y, risk, cov = inputs()
    risk.iloc[:2] = [np.inf, np.nan]
    result = run(y, risk, cov)
    reference = run(y[2:], risk.iloc[2:], cov.iloc[2:])
    pd.testing.assert_frame_equal(result, reference)
    assert result["n_patients"].eq(118).all()
    constant = run(y, risk * 0 + 1, cov)
    assert constant["status"].str.contains("constant risk").all()
    assert constant["p_value"].isna().all()
    small = run(y[:10], risk.iloc[:10], cov.iloc[:10])
    assert small["status"].str.contains("too few patients").all()
    empty = run(y[:, :0], risk, cov)
    assert empty.empty and list(empty.columns) == tfa.RESULT_COLUMNS


def test_collinear_adjustment_does_not_suppress_spearman():
    y, risk, cov = inputs()
    cov["identical_to_risk"] = risk
    result = run(y, risk, cov)
    adjusted = result.query("method == 'ols_adjusted' and term == 'positive'").iloc[0]
    assert adjusted["status"] == "skipped: collinear adjusted design"
    assert pd.isna(adjusted["p_value"])
    assert result.query("method == 'spearman' and term == 'positive'")["status"].eq("ok").all()


def test_patient_covariates_use_only_supplied_reports_and_preserve_patient_order():
    reports = pd.DataFrame({"DFCI_MRN": [2, 2, 1], "NOTE_TYPE": ["Imaging", "Pathology", "Imaging"],
                            "specimen_site": ["prostate", "non-prostate", "unspecified"]})
    ids = pd.Index([1, 2])
    clinical = tfa.patient_covariates(reports, ids, scope="clinical_notes")
    np.testing.assert_allclose(clinical["log_report_count"], np.log1p([1, 2]))
    np.testing.assert_allclose(clinical["fraction_Pathology"], [0, 0.5])
    pathology = tfa.patient_covariates(reports, ids, scope="pathology")
    assert pathology["any_non_prostate"].tolist() == [0, 1]
    assert pathology["any_unspecified"].tolist() == [1, 0]


def test_fdr_pools_landmarks_within_method_and_keeps_skips_missing():
    from scipy.stats import false_discovery_control
    frames = [pd.DataFrame({"landmark": lm, "method": ["spearman"] * 3 + ["ols_adjusted"] * 3,
                            "p_value": [0.001 * (lm + 1), 0.05, np.nan, 0.2, 0.8, np.nan]}) for lm in (0, 1)]
    result = tfa.adjust_fdr(frames)
    for _, sub in result.groupby("method"):
        finite = sub.dropna(subset=["p_value"])
        np.testing.assert_allclose(finite["q_value"], false_discovery_control(finite["p_value"]))
    assert result.loc[result["p_value"].isna(), "q_value"].isna().all()


def test_duplicate_patients_and_missing_covariates_are_rejected():
    y, risk, cov = inputs()
    with pytest.raises(ValueError, match="one row per patient"):
        run(y, pd.concat([risk, risk.iloc[:1]]), cov)
    with pytest.raises(ValueError, match="Missing or nonfinite"):
        run(y, risk, cov.iloc[1:])
