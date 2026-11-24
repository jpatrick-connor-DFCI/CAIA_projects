import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.multivariable_models import nepc_subtyping_lr as nsl  # noqa: E402

TRIO = list(nsl.TRIO_FEATURES)


def _scores_csv(tmp_path, rows):
    base = {"endpoint": "nepc", "bundle": "no_DFCI", "model_family": "xgboost_cox",
            "config": "both", "training_arm": "platinum", "model_horizon_days": np.nan,
            "evaluation_horizon_days": np.nan, "duration_days": 100.0}
    df = pd.DataFrame([{**base, **r} for r in rows])
    path = tmp_path / "transfer_scores.csv"
    df.to_csv(path, index=False)
    return path


def _synthetic_frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    gleason = rng.integers(6, 11, n).astype(float)
    tp53 = rng.binomial(1, 0.3, n).astype(float)
    lp = -2.5 + 1.2 * risk + 0.3 * (gleason - 8) + 0.8 * tp53
    event = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    frame = pd.DataFrame({
        nsl.ID_COL: np.arange(n), "landmark_days": 0, nsl.EVENT_COL: event,
        nsl.RISK_COL: risk, nsl.GLEASON_FEATURE: gleason, nsl.HAS_SEQUENCING: True,
        "TP53_ANY": tp53,
        "PTEN_ANY": rng.binomial(1, 0.3, n).astype(float),
        "RB1_ANY": rng.binomial(1, 0.2, n).astype(float),
    })
    unseq = rng.random(n) < 0.3
    frame.loc[unseq, TRIO] = np.nan
    frame.loc[unseq, nsl.HAS_SEQUENCING] = False
    frame.loc[rng.random(n) < 0.1, nsl.GLEASON_FEATURE] = np.nan
    return frame


# --------------------------------------------------------------------------- #
def test_fit_matches_sklearn_unpenalized():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.binomial(1, 0.4, 500)})
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 0.8 * X["a"] - 0.6 * X["b"]))))
    fit = nsl.fit_logistic(y, X)
    ref = LogisticRegression(penalty=None, max_iter=1000, tol=1e-10).fit(X, y)
    np.testing.assert_allclose(fit["terms"]["coef"], ref.coef_[0], atol=1e-4)
    np.testing.assert_allclose(fit["beta"]["_intercept"], ref.intercept_[0], atol=1e-4)
    assert fit["converged"] and fit["lrt_df"] == 2
    assert (fit["terms"]["se"] > 0).all()
    t = fit["terms"].iloc[0]
    assert t["or_lo"] < t["odds_ratio"] < t["or_hi"]


def test_strong_effect_small_lrt_p_and_null_large_p():
    rng = np.random.default_rng(3)
    x = rng.normal(size=600)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.5 * x))))
    assert nsl.fit_logistic(y, pd.DataFrame({"x": x}))["p_lrt"] < 1e-10
    noise = rng.normal(size=600)
    y0 = rng.binomial(1, 0.3, 600)
    assert nsl.fit_logistic(y0, pd.DataFrame({"x": noise}))["p_lrt"] > 0.001


def test_constant_predictor_reported_nan():
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"x": rng.normal(size=200), "k": np.zeros(200)})
    y = rng.binomial(1, 0.3, 200)
    fit = nsl.fit_logistic(y, X)
    terms = fit["terms"].set_index("term")
    assert np.isnan(terms.loc["k", "odds_ratio"])
    assert np.isfinite(terms.loc["x", "odds_ratio"])
    assert "constant" in fit["note"] and fit["lrt_df"] == 1


def test_separation_flagged_not_crash():
    x = np.r_[np.zeros(50), np.ones(50)]
    y = x.astype(int)
    fit = nsl.fit_logistic(y, pd.DataFrame({"x": x}))
    assert not fit["converged"]
    assert "separation" in fit["note"]
    assert fit["terms"]["odds_ratio"].isna().all()
    assert np.isnan(fit["p_lrt"])


# --------------------------------------------------------------------------- #
def test_load_transfer_scores_filters(tmp_path):
    path = _scores_csv(tmp_path, [
        {"person_id": 1, "landmark_days": 0, "risk_score": 0.5, "event": 1},
        {"person_id": 2, "landmark_days": 0, "risk_score": 0.1, "event": 0},
        {"person_id": 1, "landmark_days": 0, "risk_score": 9, "event": 1, "config": "baseline"},
        {"person_id": 1, "landmark_days": 0, "risk_score": 9, "event": 1, "model_family": "cox"},
        {"person_id": 1, "landmark_days": 0, "risk_score": 9, "event": 1, "endpoint": "platinum"},
        {"person_id": 1, "landmark_days": 0, "risk_score": 9, "event": 1,
         "evaluation_horizon_days": 365},
    ])
    out = nsl.load_transfer_scores(path)
    assert list(out.columns) == [nsl.ID_COL, "landmark_days", nsl.RISK_COL, nsl.EVENT_COL]
    assert len(out) == 2 and set(out[nsl.RISK_COL]) == {0.5, 0.1}


def test_load_transfer_scores_raises(tmp_path):
    dup = _scores_csv(tmp_path, [
        {"person_id": 1, "landmark_days": 0, "risk_score": 0.5, "event": 1},
        {"person_id": 1, "landmark_days": 0, "risk_score": 0.6, "event": 1},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        nsl.load_transfer_scores(dup)
    with pytest.raises(ValueError, match="rerun notebook 04"):
        nsl.load_transfer_scores(dup, config="nope")
    with pytest.raises(FileNotFoundError, match="rerun notebook 04"):
        nsl.load_transfer_scores(tmp_path / "missing.csv")


def test_build_lr_frame_pre_landmark_reports_for_cases_and_controls():
    # Patient 1 has an older valid report and a closer, future report at day 1.
    # Patients 2 (case) and 3 (control) have only a report on the day-0 landmark.
    local = pd.DataFrame({"DFCI_MRN": [1, 2, 3, 4],
                          "TREATMENT_ANCHOR_DATE": "2020-01-01",
                          "AGE_AT_TREATMENTSTART": 65})
    reports = pd.DataFrame({
        "DFCI_MRN": [1, 1, 2, 3],
        nsl.fcc.sg.SEQUENCING_DATE: pd.to_datetime(
            ["2019-12-01", "2020-01-02", "2020-01-01", "2020-01-01"]),
        "TP53_ANY": [0, 1, 1, 1], "PTEN_ANY": [np.nan, 1, 0, 0],
        "RB1_ANY": [0, 1, 0, 0],
    })
    reports[nsl.fcc.sg.SOMATIC_AVAILABLE_DATE] = reports[nsl.fcc.sg.SEQUENCING_DATE]
    somatic = nsl.fcc.adt_anchored_somatic(
        local, [0, 90], somatic=reports, somatic_features=TRIO)
    scores = pd.DataFrame([
        {nsl.ID_COL: i, "landmark_days": lm, nsl.RISK_COL: i / 10,
         nsl.EVENT_COL: int(i in (1, 2))} for lm in (0, 90) for i in (1, 2, 3, 4)])
    gleason = scores[[nsl.ID_COL, "landmark_days"]].assign(GLEASON_SCORE=8.0)
    frame = nsl.build_lr_frame(scores, gleason, somatic)
    rows = frame.set_index([nsl.ID_COL, "landmark_days"])
    assert rows.loc[(1, 0), TRIO].tolist() == [0, 0, 0]
    assert rows.loc[(1, 90), TRIO].tolist() == [1, 1, 1]
    for i in (2, 3):
        assert not rows.loc[(i, 0), nsl.HAS_SEQUENCING]
        assert rows.loc[(i, 0), TRIO].isna().all()
        assert rows.loc[(i, 90), TRIO].tolist() == [1, 0, 0]
        assert rows.loc[(i, 0), nsl.GLEASON_FEATURE] == 8
    assert rows.loc[4, TRIO].isna().all().all()
    _, _, counts = nsl.run_lr(frame, {"all": None}, [0, 90], n_bootstrap=0)
    assert counts["n_sequenced"].tolist() == [1, 3]
    assert counts["n_matched"].tolist() == [1, 3]


def test_cohort_members(tmp_path):
    p = tmp_path / "met.csv"
    pd.DataFrame({"DFCI_MRN": [1, 2, 2, 5]}).to_csv(p, index=False)
    out = nsl.cohort_members({"all": None, "met": p})
    assert out == {"all": None, "met": {1, 2, 5}}
    with pytest.raises(FileNotFoundError):
        nsl.cohort_members({"x": tmp_path / "none.csv"})


# --------------------------------------------------------------------------- #
def test_run_lr_end_to_end():
    frame = _synthetic_frame()
    members = {"all": None, "half": set(range(0, 400, 2))}
    models, coefs, counts = nsl.run_lr(frame, members, [0], n_bootstrap=30)

    assert list(models.columns) == nsl.MODEL_COLUMNS
    assert list(coefs.columns) == nsl.COEF_COLUMNS
    assert list(counts.columns) == nsl.COUNT_COLUMNS
    c = counts.set_index("cohort")
    assert c.loc["all", "n_at_risk"] == 400 and c.loc["half", "n_at_risk"] == 200
    assert c.loc["half", "n_cohort_list"] == 200 and pd.isna(c.loc["all", "n_cohort_list"])

    m = models.set_index(["cohort", "population", "model"])
    risk = m.loc[("all", "available", "risk")]
    assert risk["n"] == 400 and risk["auc"] > 0.65 and risk["p_lrt"] < 1e-6
    assert risk["auc_lo"] < risk["auc"] < risk["auc_hi"]
    assert risk["predictors"] == nsl.RISK_Z
    assert m.loc[("all", "available", "trio"), "n"] == c.loc["all", "n_trio_eligible"]
    assert m.loc[("all", "available", "trio"), "lrt_df"] == 3

    matched = models[(models["cohort"] == "all") & (models["population"] == "matched")]
    assert matched["n"].nunique() == 1 and matched["n"].iloc[0] == c.loc["all", "n_matched"]
    assert np.isnan(matched.set_index("model").loc["risk", "delta_auc_vs_risk"])
    others = matched[matched["model"] != "risk"]
    assert others["delta_auc_vs_risk"].notna().all()
    assert (others["delta_auc_lo"] <= others["delta_auc_hi"]).all()
    assert models[models["population"] == "available"]["delta_auc_vs_risk"].isna().all()

    trio_terms = coefs[(coefs["cohort"] == "all") & (coefs["population"] == "available")
                       & (coefs["model"] == "trio")]["term"].tolist()
    assert trio_terms == TRIO


def test_run_lr_min_events_count_only():
    frame = _synthetic_frame()
    frame[nsl.EVENT_COL] = 0
    frame.loc[:2, nsl.EVENT_COL] = 1
    models, coefs, _ = nsl.run_lr(frame, {"all": None}, [0], min_events=5, n_bootstrap=10)
    assert models["auc"].isna().all()
    assert models["note"].str.contains("fewer than 5").all()
    assert coefs.empty


def test_run_lr_missing_landmark_gives_empty_counts():
    frame = _synthetic_frame()
    models, _, counts = nsl.run_lr(frame, {"all": None}, [90], n_bootstrap=0)
    assert counts.loc[0, "n_at_risk"] == 0
    assert models["auc"].isna().all()


@pytest.mark.parametrize("altered_events,altered_controls", [(2, 1), (3, 2)])
def test_rare_binary_predictor_has_finite_mle(altered_events, altered_controls):
    y = np.r_[np.ones(10), np.zeros(90)]
    x = np.zeros(100)
    x[:altered_events] = 1
    x[10:10 + altered_controls] = 1
    fit = nsl.fit_logistic(y, pd.DataFrame({"gene": x}))
    # Closed-form 2x2-table odds ratio and standard error; all cells positive.
    a, b = altered_events, altered_controls
    c, d = 10 - a, 90 - b
    term = fit["terms"].iloc[0]
    assert fit["converged"]
    np.testing.assert_allclose(term["coef"], np.log(a * d / (b * c)), atol=1e-7)
    np.testing.assert_allclose(term["se"], np.sqrt(1/a + 1/b + 1/c + 1/d), atol=1e-7)


def test_failed_cv_suppresses_auc_and_paired_comparison():
    frame = _synthetic_frame(n=100)
    frame[nsl.EVENT_COL] = np.r_[np.ones(10), np.zeros(90)]
    frame[nsl.GLEASON_FEATURE] = 8.0
    frame[nsl.HAS_SEQUENCING] = True
    frame[TRIO] = 0.0
    frame.loc[[0, 10, 11], "TP53_ANY"] = 1.0
    X = frame[["TP53_ANY"]]
    assert nsl.fit_logistic(frame[nsl.EVENT_COL], X)["converged"]
    predictions, failed = nsl.oof_predictions(
        frame[nsl.EVENT_COL], X, return_diagnostics=True)
    assert failed > 0 and np.isnan(predictions).all()
    models, _, _ = nsl.run_lr(frame, {"all": None}, [0],
                              models=("risk", "tp53"), n_bootstrap=20)
    gene = models[models["model"].eq("tp53")]
    assert gene["converged"].all()  # Full fit succeeds; CV does not.
    assert gene["n_failed_folds"].gt(0).all()
    assert gene["note"].str.contains("failed CV folds").all()
    assert gene[["auc", "auc_lo", "auc_hi", "delta_auc_vs_risk",
                 "delta_auc_lo", "delta_auc_hi"]].isna().all().all()
    assert models[models["model"].eq("risk")]["auc"].notna().all()


def test_auc_rejects_partial_predictions():
    y = np.array([0, 0, 1, 1])
    assert np.isnan(nsl._auc(y, np.array([0., np.nan, 1., 2.])))


# --------------------------------------------------------------------------- #
# Out-of-fold predictions
# --------------------------------------------------------------------------- #
def _reference_oof(y, X, n_folds=5, seed=0):
    """The pre-refactor loop, straight over StratifiedKFold."""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y, dtype=int)
    k = int(min(n_folds, y.sum(), len(y) - y.sum()))
    out = np.full(len(y), np.nan)
    for train, test in StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        fit = nsl.fit_logistic(y[train], X.iloc[train])
        out[test] = nsl._linear_predictor(fit["beta"], X.iloc[test])
    return out


def test_cv_folds_drive_oof_predictions_unchanged():
    frame = _synthetic_frame()
    X = nsl._design(frame, "risk")
    y = frame[nsl.EVENT_COL].to_numpy()
    np.testing.assert_allclose(nsl.oof_predictions(y, X, seed=4), _reference_oof(y, X, seed=4))
    folds = nsl.cv_folds(y, seed=4)
    assert set(folds) == set(range(5))
    assert (nsl.cv_folds(np.r_[1, np.zeros(20)]) == -1).all()


def test_run_lr_returns_oof_predictions_matching_auc():
    frame = pd.concat([_synthetic_frame(), _synthetic_frame(seed=2).assign(landmark_days=90)],
                      ignore_index=True)
    members = {"all": None, "half": set(range(200))}
    assert len(nsl.run_lr(frame, members, [0, 90], n_bootstrap=0)) == 3
    models, _, _, preds = nsl.run_lr(frame, members, [0, 90], n_bootstrap=0, return_predictions=True)
    assert list(preds.columns) == nsl.PREDICTION_COLUMNS
    assert set(preds["model"]) == {"risk"} and set(preds["population"]) == {"available"}
    np.testing.assert_allclose(preds["oof_prob"], 1 / (1 + np.exp(-preds["oof_logit"])))
    for (cohort, lm), g in preds.groupby(["cohort", "landmark_days"]):
        row = models.set_index(["cohort", "landmark_days", "population", "model"]).loc[
            (cohort, lm, "available", "risk")]
        assert len(g) == row["n"] == g[nsl.ID_COL].nunique()
        assert g[nsl.EVENT_COL].sum() == row["events"]
        assert nsl._auc(g[nsl.EVENT_COL].to_numpy(), g["oof_prob"].to_numpy()) == pytest.approx(row["auc"])
    assert preds.query("cohort == 'half'")[nsl.ID_COL].max() < 200

    _, _, _, more = nsl.run_lr(frame, {"all": None}, [0], n_bootstrap=0, return_predictions=True,
                               prediction_models=("risk", "gleason"),
                               prediction_populations=("available", "matched"))
    assert set(zip(more["population"], more["model"])) == {
        ("available", "risk"), ("available", "gleason"), ("matched", "risk"), ("matched", "gleason")}


def test_failed_cv_keeps_nan_prediction_rows_and_unfit_models_add_none():
    frame = _synthetic_frame(n=100)
    frame[nsl.EVENT_COL] = np.r_[np.ones(10), np.zeros(90)]
    frame[nsl.GLEASON_FEATURE] = 8.0
    frame[nsl.HAS_SEQUENCING] = True
    frame[TRIO] = 0.0
    frame.loc[[0, 10, 11], "TP53_ANY"] = 1.0
    _, _, _, preds = nsl.run_lr(frame, {"all": None}, [0], models=("risk", "tp53"), n_bootstrap=0,
                                return_predictions=True, prediction_models=("risk", "tp53"))
    gene = preds[preds["model"].eq("tp53")]
    assert len(gene) == 100 and gene["oof_prob"].isna().all()
    assert preds[preds["model"].eq("risk")]["oof_prob"].notna().all()

    frame[nsl.EVENT_COL] = np.r_[np.ones(3), np.zeros(97)]
    _, _, _, none = nsl.run_lr(frame, {"all": None}, [0], n_bootstrap=0, return_predictions=True)
    assert none.empty and list(none.columns) == nsl.PREDICTION_COLUMNS


def test_calibration_summary_and_bins():
    rng = np.random.default_rng(0)
    lp = rng.normal(-1, 1.2, 20000)
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    preds = pd.DataFrame({"g": "a", nsl.EVENT_COL: y, "oof_logit": lp, "oof_prob": 1 / (1 + np.exp(-lp))})
    preds = pd.concat([preds, preds.assign(g="b", oof_logit=2 * lp + 1, oof_prob=1 / (1 + np.exp(-(2 * lp + 1)))),
                       pd.DataFrame({"g": ["a"], nsl.EVENT_COL: [1], "oof_logit": [np.nan], "oof_prob": [np.nan]})])
    s = nsl.calibration_summary(preds, ["g"]).set_index("g")
    assert list(s.columns) == nsl.CALIBRATION_COLUMNS
    assert s.loc["a", "n"] == 20000
    assert s.loc["a", "calibration_slope"] == pytest.approx(1, abs=0.05)
    assert s.loc["a", "calibration_intercept"] == pytest.approx(0, abs=0.05)
    assert s.loc["b", "calibration_slope"] == pytest.approx(0.5, abs=0.03)  # too extreme
    assert s.loc["a", "brier"] < s.loc["b", "brier"]
    b = nsl.calibration_bins(preds, ["g"], n_bins=5)
    assert b.groupby("g")["n"].sum().to_dict() == {"a": 20000, "b": 20000}
    a = b[b["g"].eq("a")]
    assert a["mean_predicted"].is_monotonic_increasing
    np.testing.assert_allclose(a["observed_rate"], a["mean_predicted"], atol=0.03)


def test_raw_score_auc_on_risk_rows_only_and_independent_of_cv(monkeypatch):
    from sklearn.metrics import roc_auc_score
    frame = _synthetic_frame()
    models, _, _ = nsl.run_lr(frame, {"all": None}, [0], n_bootstrap=100)
    assert {"auc_raw_score", "auc_raw_score_lo", "auc_raw_score_hi"} <= set(models.columns)
    for population, g in models.groupby("population"):
        risk = g[g["model"].eq("risk")].iloc[0]
        assert risk["auc_raw_score_lo"] <= risk["auc_raw_score"] <= risk["auc_raw_score_hi"]
        assert g[g["model"].ne("risk")]["auc_raw_score"].isna().all()
    available = models.query("population == 'available' and model == 'risk'").iloc[0]
    assert available["auc_raw_score"] == pytest.approx(
        roc_auc_score(frame[nsl.EVENT_COL], frame[nsl.RISK_COL]))

    nan_oof = lambda y, X, **kw: (np.full(len(y), np.nan), 1)
    monkeypatch.setattr(nsl, "oof_predictions", nan_oof)
    failed, _, _ = nsl.run_lr(frame, {"all": None}, [0], models=("risk",), n_bootstrap=50)
    row = failed.query("population == 'available'").iloc[0]
    assert np.isnan(row["auc"]) and np.isnan(row["auc_lo"])
    assert row["auc_raw_score"] == pytest.approx(available["auc_raw_score"])
    assert np.isfinite([row["auc_raw_score_lo"], row["auc_raw_score_hi"]]).all()


def test_irls_converges_when_rounding_holds_step_above_tol():
    # Regression: this fold's Newton step stalled just above tol at the optimum and was reported
    # as non-converged, suppressing a valid OOF AUC.
    rng = np.random.default_rng(3)
    x = rng.normal(size=480)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2.6 + 1.6 * x)))).astype(float)
    A = np.column_stack([np.ones(len(y)), x])
    beta, _, converged = nsl._irls(y, A)
    assert converged and np.all(np.abs(beta) < 5)
    sep = (x > 0).astype(float)                       # perfect separation still flagged
    assert not nsl.fit_logistic(sep, pd.DataFrame({"x": x}))["converged"]
