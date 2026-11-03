"""Transferred federated score vs local Gleason / TP53-RB1-PTEN Cox (Figure 5 b-d)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.federated_models import federated_clinical_comparison as fcc  # noqa: E402

ANCHOR = pd.Timestamp("2018-06-01")
FEATURES = ["TP53_ANY", "RB1_ANY", "PTEN_ANY", "AR_ANY"]


def _day(n):
    return ANCHOR + pd.Timedelta(days=n)


def _local(ids):
    return pd.DataFrame({"DFCI_MRN": ids, "TREATMENT_ANCHOR_DATE": ANCHOR,
                         "AGE_AT_TREATMENTSTART": 70.0})


def _gleason(rows):
    return pd.DataFrame(rows, columns=["DFCI_MRN", "gleason_date", "_gleason_available_date",
                                       "GLEASON_SCORE"])


def _somatic(rows):
    return pd.DataFrame(rows, columns=["DFCI_MRN", "SEQUENCING_DATE", "_somatic_available_date",
                                       *FEATURES])


def test_clinical_data_must_be_available_before_the_landmark():
    gleason = _gleason([
        (1, _day(-400), _day(89), 9),     # available the day before: kept
        (2, _day(-400), _day(90), 9),     # available on the landmark: excluded
        (3, _day(-400), _day(-300), 6),   # old score, available early: kept
    ])
    out = fcc.build_clinical_features(_local([1, 2, 3]), [90], gleason=gleason)
    out = out.set_index("person_id")
    assert out.loc[1, "GLEASON_SCORE"] == 9
    assert np.isnan(out.loc[2, "GLEASON_SCORE"])
    assert out.loc[3, "GLEASON_SCORE"] == 6
    # No somatic data on the ADT clock: it comes only from the anchoring specimen.
    assert out[["HAS_SEQUENCING", "TP53_ANY", "SOMATIC_N_GENES_ALTERED"]].isna().all().all()
    assert (out["landmark_days"] == 90).all() and (out["AGE"] == 70.0).all()


def test_burden_counts_every_altered_panel_gene_and_uncalled_is_wild_type():
    somatic = _somatic([
        (1, _day(-50), _day(-50), 1, 0, np.nan, 1),
        # two same-day specimens are unioned
        (2, _day(-50), _day(-50), 0, 1, 0, 0),
        (2, _day(-50), _day(-50), 0, 0, 1, 0),
        # a specimen reported farther from ADT is not selected
        (3, _day(-900), _day(-900), 1, 1, 1, 1),
        (3, _day(-50), _day(-50), 0, 0, 0, 0),
    ])
    local = _local([1, 2, 3])
    specimen = fcc.select_specimen(local, somatic, FEATURES)
    out = fcc.build_clinical_features(fcc.sequencing_anchored_local(local, specimen), [0],
                                      gleason=_gleason([]), specimen=specimen,
                                      somatic_features=FEATURES)
    out = out.set_index("person_id")
    assert out.loc[1, "PTEN_ANY"] == 0
    assert out["SOMATIC_N_GENES_ALTERED"].to_dict() == {1: 2, 2: 2, 3: 0}
    assert (out.loc[2, ["RB1_ANY", "PTEN_ANY"]] == 1).all()


def _survival(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(np.exp(-0.8 * x)) * 1000
    c = rng.uniform(200, 2500, size=n)
    return pd.DataFrame({"x": x, "AGE": rng.normal(70, 8, size=n),
                         "duration_days": np.minimum(t, c), "event": (t <= c).astype(int)})


def test_out_of_fold_prediction_ignores_the_patients_own_outcome():
    frame = _survival()
    pred, k, failed, _ = fcc.out_of_fold_cox(frame, ["x", "AGE"], n_folds=5, seed=0)
    assert k == 5 and failed == 0
    assert pred.shape == (len(frame),) and np.isfinite(pred).all()
    # Changing patient 0's time (event unchanged, so folds are unchanged)
    # moves only the other folds' models, never patient 0's own prediction.
    moved = frame.copy()
    moved.loc[0, "duration_days"] = 1.0
    pred2, *_ = fcc.out_of_fold_cox(moved, ["x", "AGE"], n_folds=5, seed=0)
    assert pred2[0] == pred[0]
    assert not np.allclose(pred2, pred)


def _scores_and_clinical(n=150, n_events=None, seed=1):
    rng = np.random.default_rng(seed)
    gleason = rng.integers(6, 11, size=n).astype(float)
    t = rng.exponential(np.exp(-0.9 * (gleason - 8))) * 800
    c = rng.uniform(100, 3000, size=n)
    event = (t <= c).astype(int)
    if n_events is not None:
        event[:] = 0
        event[:n_events] = 1
    ids = np.arange(5000, 5000 + n)
    outcome = dict(person_id=ids, duration_days=np.minimum(t, c), event=event,
                   endpoint="nepc", landmark_days=0)
    rows = []
    for bundle, family, noise in (("fed_en", "elastic_net_cox", 1.0), ("fed_xgb", "xgboost_cox", 3.0)):
        rows.append(pd.DataFrame({**outcome, "bundle": bundle, "model_family": family, "config": "both",
                                  "risk_score": gleason + rng.normal(scale=noise, size=n)}))
        rows.append(pd.DataFrame({**outcome, "bundle": bundle, "model_family": family,
                                  "config": "baseline", "risk_score": rng.normal(size=n)}))
    scores = pd.concat(rows, ignore_index=True)
    seq = np.arange(n) % 2 == 0
    clinical = pd.DataFrame({
        "person_id": ids, "landmark_days": 0, "AGE": rng.normal(70, 8, size=n),
        "GLEASON_SCORE": np.where(np.arange(n) % 5 == 0, np.nan, gleason),
        "HAS_SEQUENCING": np.where(seq, True, np.nan),
        "TP53_ANY": np.where(seq, (gleason >= 9).astype(float), np.nan),
        "RB1_ANY": np.where(seq, rng.integers(0, 2, size=n), np.nan),
        "PTEN_ANY": np.where(seq, rng.integers(0, 2, size=n), np.nan),
        "SOMATIC_N_GENES_ALTERED": np.where(seq, rng.integers(0, 6, size=n), np.nan),
    })
    return scores, clinical


def test_comparison_is_paired_on_the_same_patients():
    scores, clinical = _scores_and_clinical()
    comp, coefs = fcc.compare_clinical_models(scores, clinical, n_bootstrap=60, seed=0)
    assert set(comp["subpopulation"]) == {"gleason", "trio", "trio_gleason"}
    assert len(comp) == 3 * 2 * 4  # subpopulation x bundle x model
    g = comp[comp["subpopulation"] == "gleason"]
    assert (g["n_patients"] == clinical["GLEASON_SCORE"].notna().sum()).all()
    t = comp[comp["subpopulation"] == "trio"]
    assert (t["n_patients"] == clinical["HAS_SEQUENCING"].notna().sum()).all()
    tg = comp[comp["subpopulation"] == "trio_gleason"]
    both = clinical["HAS_SEQUENCING"].notna() & clinical["GLEASON_SCORE"].notna()
    assert (tg["n_patients"] == both.sum()).all()
    for (_, bundle), d in comp.groupby(["subpopulation", "bundle"]):
        d = d.set_index("model")
        assert d["n_scored"].nunique() == 1
        assert np.isnan(d.loc["transferred_labs", "delta_c_vs_transferred"])
        for m in ("clinical_cox", "combined_cox"):
            delta = d.loc[m, "c_index"] - d.loc["transferred_labs", "c_index"]
            assert np.isclose(d.loc[m, "delta_c_vs_transferred"], delta)
            assert d.loc[m, "delta_lo"] <= delta <= d.loc[m, "delta_hi"]
            assert np.isfinite(d.loc[m, "c_index_apparent"])
            d_auc = d.loc[m, "mean_auc"] - d.loc["transferred_labs", "mean_auc"]
            assert np.isclose(d.loc[m, "delta_auc_vs_transferred"], d_auc)
            assert d.loc[m, "delta_auc_lo"] <= d_auc <= d.loc[m, "delta_auc_hi"]
            assert np.isfinite(d.loc[m, "mean_auc_apparent"])
        assert np.isnan(d.loc["transferred_labs", "delta_auc_vs_transferred"])
        for m in ("transferred_labs", "clinical_cox", "combined_cox"):
            assert d.loc[m, "mean_auc_lo"] <= d.loc[m, "mean_auc"] <= d.loc[m, "mean_auc_hi"]
        # Only the combined Cox carries the paired difference from the clinical Cox.
        assert d.drop(index="combined_cox")[["delta_c_vs_clinical", "delta_auc_vs_clinical"]].isna().all().all()
        comb = d.loc["combined_cox"]
        dc = comb["c_index"] - d.loc["clinical_cox", "c_index"]
        da = comb["mean_auc"] - d.loc["clinical_cox", "mean_auc"]
        assert np.isclose(comb["delta_c_vs_clinical"], dc)
        assert comb["delta_c_vs_clinical_lo"] <= dc <= comb["delta_c_vs_clinical_hi"]
        assert np.isclose(comb["delta_auc_vs_clinical"], da)
        assert comb["delta_auc_vs_clinical_lo"] <= da <= comb["delta_auc_vs_clinical_hi"]
    # The clinical Cox does not depend on the bundle; the combined Cox does.
    cox = comp[comp["model"] == "clinical_cox"]
    assert cox.groupby("subpopulation")["c_index"].nunique().eq(1).all()
    combined = comp[comp["model"] == "combined_cox"]
    assert combined.groupby("subpopulation")["c_index"].nunique().eq(2).all()
    # Gleason drives the simulated hazard; the burden is no longer a covariate.
    clin_coefs = coefs[coefs["model"] == "clinical_cox"]
    hr = clin_coefs.set_index(["subpopulation", "covariate"])["hr"]
    assert hr[("gleason", "GLEASON_SCORE")] > 1
    assert set(coefs["subpopulation"]) == {"gleason", "trio", "trio_gleason"}
    assert set(clin_coefs.loc[clin_coefs["subpopulation"] == "trio_gleason", "covariate"]) == {
        "GLEASON_SCORE", "TP53_ANY", "RB1_ANY", "PTEN_ANY", "AGE"}
    assert "SOMATIC_N_GENES_ALTERED" not in set(coefs["covariate"])
    # One combined fit per subpopulation x bundle, each with the transferred score.
    comb_coefs = coefs[coefs["model"] == "combined_cox"]
    score_rows = comb_coefs[comb_coefs["covariate"] == fcc.SCORE_COVARIATE]
    assert len(score_rows) == 3 * 2
    assert set(score_rows["bundle"]) == {"fed_en", "fed_xgb"}
    # The transferred full model already includes age, so the combined Cox does not add it.
    assert "AGE" not in set(comb_coefs["covariate"])
    comb_covs = comb_coefs.groupby("subpopulation")["covariate"].agg(set)
    assert comb_covs["gleason"] == {fcc.SCORE_COVARIATE, "GLEASON_SCORE"}
    assert comb_covs["trio"] == {fcc.SCORE_COVARIATE, "TP53_ANY", "RB1_ANY", "PTEN_ANY"}
    assert comb_covs["trio_gleason"] == {fcc.SCORE_COVARIATE, "GLEASON_SCORE", "TP53_ANY", "RB1_ANY", "PTEN_ANY"}
    assert clin_coefs["bundle"].isna().all()


def test_a_bundle_without_an_age_model_keeps_its_other_comparisons():
    """The landmark XGBoost packages export no age-only model; that must not empty the bundle's rows."""
    scores, clinical = _scores_and_clinical()
    full, _ = fcc.compare_clinical_models(scores, clinical, n_bootstrap=30, seed=0)
    no_age = scores[~((scores["bundle"] == "fed_xgb") & (scores["config"] == "baseline"))]
    comp, _ = fcc.compare_clinical_models(no_age, clinical, n_bootstrap=30, seed=0)
    assert len(comp) == len(full)
    keys = ["subpopulation", "bundle", "model"]
    full, comp = full.set_index(keys).sort_index(), comp.set_index(keys).sort_index()
    age = comp.xs(("fed_xgb", "transferred_age"), level=("bundle", "model"))
    assert age[["c_index", "mean_auc", "delta_c_vs_transferred", "mean_auc_lo"]].isna().all().all()
    kept = comp.index.get_level_values("model") != "transferred_age"
    pd.testing.assert_frame_equal(comp[kept], full[kept])
    assert comp.loc[kept, "n_scored"].gt(0).all()


def test_too_few_events_gives_a_count_only_row():
    scores, clinical = _scores_and_clinical(n_events=6)
    comp, coefs = fcc.compare_clinical_models(scores, clinical, n_bootstrap=20, min_events=10)
    assert comp["c_index"].isna().all()
    assert comp["mean_auc"].isna().all()
    assert comp["note"].str.contains("fewer than 10 events").all()
    assert (comp["n_patients"] > 0).all() and coefs.empty


def test_subpopulation_summary_strata():
    scores, clinical = _scores_and_clinical()
    summary = fcc.subpopulation_score_summary(scores, clinical)
    s = summary[summary["bundle"] == "fed_en"].set_index("stratum")
    assert s.loc["Full cohort", "n"] == 150
    assert s.loc["Gleason available", "n"] + s.loc["Gleason missing", "n"] == 150
    assert s.loc["Sequenced", "n"] + s.loc["Not sequenced", "n"] == 150
    assert s.loc["Gleason available", "spearman_rho"] > 0.5


def test_subpopulation_summary_sections_select_strata():
    scores, clinical = _scores_and_clinical()
    seq_only = fcc.subpopulation_score_summary(scores, clinical, sections=("sequencing",))
    strata = set(seq_only["stratum"])
    assert {"Sequenced", "Sequenced, Gleason available", "Any trio altered", "TP53 altered"} <= strata
    assert not strata & {"Full cohort", "Gleason available", "Gleason missing"}


def test_km_patients_keeps_labs_scores_and_each_clock_own_strata(tmp_path):
    scores, clinical = _scores_and_clinical()
    adt_clinical = clinical.assign(**{c: np.nan for c in (
        "HAS_SEQUENCING", "TP53_ANY", "RB1_ANY", "PTEN_ANY", "SOMATIC_N_GENES_ALTERED")})
    km = fcc.km_patients({"adt": (scores, adt_clinical), "sequencing": (scores, clinical)})
    assert list(km.columns) == fcc.KM_PATIENT_COLUMNS
    # One row per clock × bundle × patient; the baseline (age) score is dropped.
    assert len(km) == 2 * 2 * 150
    assert set(km["clock"]) == {"adt", "sequencing"}
    assert km.loc[km["clock"] == "adt", "TP53_ANY"].isna().all()
    seq = km[km["clock"] == "sequencing"]
    assert seq["HAS_SEQUENCING"].eq(True).sum() == 2 * 75
    en = scores[(scores["bundle"] == "fed_en") & (scores["config"] == "both")]
    assert np.allclose(km.loc[(km["clock"] == "adt") & (km["bundle"] == "fed_en"), "risk_score"],
                       en["risk_score"])
    fcc.write_outputs(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), tmp_path, km=km)
    assert len(pd.read_csv(tmp_path / fcc.KM_PATIENTS_FILE)) == len(km)


# --------------------------------------------------------------------------- #
# Figure 6 checklist tables
# --------------------------------------------------------------------------- #
def _checklist_tables():
    """200 cases/controls with known rule accuracy, 30 patients outside the cohort,
    and rows the Figure 6 filters must drop."""
    n = 230
    index = np.arange(n)
    case = index < 60
    flag = (index < 40) | ((index >= 165) & (index < 200))   # sensitivity 2/3, specificity 3/4
    tsg = (index < 30) | ((index >= 172) & (index < 200))    # sensitivity 1/2, specificity 4/5
    ids = 7000 + index
    scores = pd.DataFrame({
        "person_id": ids, "endpoint": "platinum", "bundle": "fed_checklist",
        "model_family": "predictive_checklist", "config": "both", "landmark_days": 180,
        "model_horizon_days": 365, "evaluation_horizon_days": 365, "risk_score": flag * 2 + 1,
        "flag": flag, "duration_days": np.where(case, 200.0, 2500.0), "event": 1})
    other = pd.concat([scores.assign(config="baseline"), scores.assign(landmark_days=90),
                       scores.assign(model_horizon_days=1825, evaluation_horizon_days=1825),
                       scores.assign(evaluation_horizon_days=1825),
                       scores.assign(model_family="xgboost_cox")], ignore_index=True)
    clinical = pd.DataFrame({
        "person_id": ids, "landmark_days": 180,
        "GLEASON_SCORE": np.where(index >= 215, np.nan, np.where(flag, 9.0, 7.0)),
        "HAS_SEQUENCING": np.where((index >= 200) & (index < 215), np.nan, True),
        "TP53_ANY": np.where(tsg, 1.0, 0.0), "RB1_ANY": 0.0, "PTEN_ANY": np.nan})
    return pd.concat([scores, other], ignore_index=True), clinical


def test_checklist_patients_keep_one_row_per_patient_of_the_figure_model(tmp_path):
    scores, clinical = _checklist_tables()
    patients = fcc.checklist_patients(scores)
    assert list(patients.columns) == fcc.CHECKLIST_PATIENT_COLUMNS
    assert len(patients) == patients["person_id"].nunique() == 230
    assert patients["landmark_days"].eq(180).all() and patients["model_horizon_days"].eq(365).all()
    assert set(patients["flag"]) == {0, 1}
    assert patients["flag"].eq(patients["risk_score"].ge(2).astype(int)).all()
    comparison = fcc.checklist_rule_comparison(scores, clinical, n_bootstrap=20)
    assert comparison["landmark_days"].eq(180).all() and comparison["horizon_days"].eq(365).all()
    assert fcc.checklist_patients(scores, landmark=180, horizon=1825)["model_horizon_days"].eq(1825).all()
    fcc.write_outputs(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), tmp_path, checklist=patients,
                      checklist_comparison=comparison)
    assert len(pd.read_csv(tmp_path / fcc.CHECKLIST_PATIENTS_FILE)) == 230
    assert len(pd.read_csv(tmp_path / fcc.CHECKLIST_COMPARISON_FILE)) == 3


def test_checklist_grid_stacks_each_landmark_horizon_cell_and_skips_missing_ones(tmp_path):
    scores, clinical = _checklist_tables()
    patients, comparison = fcc.checklist_grid(scores, scores, clinical, n_bootstrap=20,
                                              landmarks=(0, 180), horizons=(365, 1825))
    cells = patients.groupby(["landmark_days", "model_horizon_days"]).size()
    assert cells.to_dict() == {(180, 365): 230, (180, 1825): 230}
    assert set(zip(comparison["landmark_days"], comparison["horizon_days"])) == {(180, 365), (180, 1825)}
    fcc.write_checklist_grid(patients, comparison, tmp_path)
    grid = tmp_path / fcc.CHECKLIST_GRID_DIR
    assert len(pd.read_csv(grid / fcc.CHECKLIST_PATIENTS_FILE)) == 460
    assert len(pd.read_csv(grid / fcc.CHECKLIST_COMPARISON_FILE)) == 6
    with pytest.raises(ValueError, match="no checklist scores"):
        fcc.checklist_grid(scores, scores, clinical, landmarks=(0,), horizons=(365,))


def test_checklist_patients_require_the_figure_model():
    scores, _ = _checklist_tables()
    with pytest.raises(ValueError, match="no predictive_checklist scores"):
        fcc.checklist_patients(scores[scores["landmark_days"] == 90])


def test_checklist_rules_are_paired_balanced_accuracy_on_the_same_patients():
    scores, clinical = _checklist_tables()
    comparison = fcc.checklist_rule_comparison(scores, clinical, n_bootstrap=100)
    assert list(comparison.columns) == fcc.CHECKLIST_COMPARISON_COLUMNS
    rows = comparison.set_index("rule")
    assert list(rows.index) == ["checklist", "gleason", "tsg"]
    # Only sequenced patients with a Gleason score, and the same ones for every rule.
    assert rows["n_patients"].eq(200).all() and rows["n_events"].eq(200).all()
    assert rows["n_positive"].to_dict() == {"checklist": 75, "gleason": 75, "tsg": 58}
    # No censoring here, so AUC at the horizon is (sensitivity + specificity) / 2.
    assert np.isclose(rows.loc["checklist", "auc"], (2 / 3 + 3 / 4) / 2)
    assert np.isclose(rows.loc["tsg", "auc"], (1 / 2 + 4 / 5) / 2)
    # Gleason matches the flag patient for patient, so a paired delta is exactly zero.
    assert rows.loc["gleason", "delta_auc_vs_checklist"] == 0
    assert rows.loc["gleason", ["delta_auc_lo", "delta_auc_hi"]].eq(0).all()
    assert rows.loc["gleason", "auc_hi"] > rows.loc["gleason", "auc_lo"]
    expected = rows.loc["tsg", "auc"] - rows.loc["checklist", "auc"]
    assert np.isclose(rows.loc["tsg", "delta_auc_vs_checklist"], expected)
    assert rows.loc["tsg", "delta_auc_lo"] < expected < rows.loc["tsg", "delta_auc_hi"]


# --------------------------------------------------------------------------- #
# Sequencing-anchored trio arm
# --------------------------------------------------------------------------- #
def test_specimen_is_the_one_closest_to_adt():
    somatic = _somatic([
        (1, _day(-60), _day(-60), 1, 0, 0, 0),
        (1, _day(400), _day(400), 0, 1, 0, 0),   # farther from ADT: not selected
        (2, _day(200), _day(200), 0, 0, 1, 0),   # sequenced after ADT + 180
    ])
    specimen = fcc.select_specimen(_local([1, 2, 3]), somatic, FEATURES)
    assert set(specimen.index) == {1, 2}
    assert specimen.loc[1, "SEQUENCING_DATE"] == _day(-60)
    assert specimen.loc[1, "TP53_ANY"] == 1
    assert specimen.loc[2, "SEQUENCING_TO_ADT_DAYS"] == 200


def test_sequencing_anchored_local_moves_anchor_and_age():
    local = pd.concat([_local([1, 2, 3])] * 2, ignore_index=True)  # long frame
    specimen = pd.DataFrame(
        {"SEQUENCING_DATE": [_day(-365), _day(200)]},
        index=pd.Index([1, 2], name="DFCI_MRN"),
    )
    out = fcc.sequencing_anchored_local(local, specimen)
    assert sorted(out["DFCI_MRN"].unique()) == [1, 2]  # unsequenced dropped
    assert len(out) == 4
    by_id = out.groupby("DFCI_MRN").first()
    assert by_id.loc[1, "TREATMENT_ANCHOR_DATE"] == _day(-365)
    assert by_id.loc[2, "TREATMENT_ANCHOR_DATE"] == _day(200)
    assert np.isclose(by_id.loc[1, "AGE_AT_TREATMENTSTART"], 70.0 - 365 / fcc.fs.DAYS_PER_YEAR)
    assert np.isclose(by_id.loc[2, "AGE_AT_TREATMENTSTART"], 70.0 + 200 / fcc.fs.DAYS_PER_YEAR)
    # The input frame is not modified.
    assert (local["TREATMENT_ANCHOR_DATE"] == ANCHOR).all()


def test_trio_comes_from_the_anchoring_specimen_at_every_landmark():
    somatic = _somatic([
        (1, _day(-10), _day(-10), 1, 0, 0, 0),
        # reported after sequencing but before sequencing + 90: never used
        (1, _day(40), _day(40), 0, 1, 1, 1),
    ])
    local = _local([1, 2])
    specimen = fcc.select_specimen(local, somatic, FEATURES)
    seq_local = fcc.sequencing_anchored_local(local, specimen)
    gleason = _gleason([(1, _day(-200), _day(-200), 7)])
    out = fcc.build_clinical_features(seq_local, [0, 90], gleason=gleason, specimen=specimen,
                                      somatic_features=FEATURES)
    assert set(out["person_id"]) == {1}
    out = out.set_index("landmark_days")
    for lm in (0, 90):
        assert out.loc[lm, "TP53_ANY"] == 1
        assert out.loc[lm, "RB1_ANY"] == 0
        assert out.loc[lm, "SOMATIC_N_GENES_ALTERED"] == 1
        assert bool(out.loc[lm, "HAS_SEQUENCING"])
    assert np.isclose(out.loc[0, "AGE"], 70.0 - 10 / fcc.fs.DAYS_PER_YEAR)
    # Without the specimen there is no trio at all, never a later specimen.
    adt = fcc.build_clinical_features(local, [90], gleason=gleason)
    assert adt[["TP53_ANY", "RB1_ANY", "HAS_SEQUENCING"]].isna().all().all()


def test_compare_by_subpopulation_scores_each_arm_on_its_own_inputs():
    scores, clinical = _scores_and_clinical(seed=1)
    seq_scores, seq_clinical = _scores_and_clinical(seed=2)
    kwargs = dict(n_bootstrap=20, seed=0)
    arms = {"gleason": (scores, clinical), "trio": (seq_scores, seq_clinical),
            "trio_gleason": (seq_scores, seq_clinical)}
    comp, coefs = fcc.compare_by_subpopulation(arms, **kwargs)
    assert set(comp["subpopulation"]) == set(arms)
    assert set(coefs["subpopulation"]) == set(arms)
    for sub, (s, c) in arms.items():
        expected, _ = fcc.compare_clinical_models(
            s, c, subpopulations={sub: fcc.SUBPOPULATIONS[sub]}, **kwargs
        )
        got = comp[comp["subpopulation"] == sub].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, expected.reset_index(drop=True))



@pytest.mark.parametrize("value,expected", [
    (0.3, 0.25), (0.6, 0.5), (0.9, 1.0), (0.1, 0.0), (1.4, 1.0), (9.4, 9.0), (9.6, 10.0),
    (13.2, 15.0), (22.4, 20.0), (67.0, 65.0), (148.0, 150.0), (1234.0, 1200.0), (15250.0, 15000.0),
    (-0.6, -0.5), (-13.2, -15.0)])
def test_guideline_threshold_rounds_by_magnitude(value, expected):
    assert fcc.guideline_threshold(value) == expected


def test_guideline_threshold_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        fcc.guideline_threshold(float("nan"))


def _checklist_bundle():
    items = [{"feature": "age_at_anchor", "op": ">=", "threshold": 67.3, "group": "age"},
             {"feature": "ALB_mean", "op": "<=", "threshold": 3.84, "group": "lab"},
             {"feature": "PSA_max", "op": ">=", "threshold": 0.37, "group": "lab"},
             {"feature": "ALB_mean", "op": "is_missing", "threshold": None, "group": "lab"}]
    model = {"landmark_days": 180, "horizon_days": 365, "config": "both", "checklist": {
        "schema_version": 1, "binarizer": {"items": items},
        "model": {"selected_indices": [2, 0, 1], "threshold": 2}}}
    return {"model_family": "predictive_checklist", "models": [model]}


def test_guideline_bundle_rounds_thresholds_without_touching_the_original():
    bundle = _checklist_bundle()
    rounded = fcc.guideline_bundle(bundle)
    items = rounded["models"][0]["checklist"]["binarizer"]["items"]
    assert [i["threshold"] for i in items] == [65.0, 4.0, 0.25, None]
    assert bundle["models"][0]["checklist"]["binarizer"]["items"][0]["threshold"] == 67.3
    assert rounded["models"][0]["checklist"]["model"] == bundle["models"][0]["checklist"]["model"]
    with pytest.raises(ValueError, match="predictive_checklist"):
        fcc.guideline_bundle({"model_family": "xgboost_cox", "models": []})


def test_guideline_items_follow_selection_order_and_write_the_grid(tmp_path):
    bundle = _checklist_bundle()
    table = fcc.guideline_items(fcc.guideline_bundle(bundle), bundle)
    assert list(table.columns) == fcc.GUIDELINE_ITEM_COLUMNS
    assert table["feature"].tolist() == ["PSA_max", "age_at_anchor", "ALB_mean"]
    assert table["threshold_value"].tolist() == [0.25, 65.0, 4.0]
    assert table["threshold_exact"].tolist() == [0.37, 67.3, 3.84]
    assert set(table["M"]) == {2} and set(table["N"]) == {3} and table["position"].tolist() == [1, 2, 3]
    scores, clinical = _checklist_tables()
    patients, comparison = fcc.checklist_grid(scores, scores, clinical, n_bootstrap=20,
                                              landmarks=(180,), horizons=(365,))
    fcc.write_guideline_grid(patients, comparison, table, tmp_path)
    grid = tmp_path / fcc.GUIDELINE_GRID_DIR
    assert sorted(p.name for p in grid.iterdir()) == sorted(
        [fcc.CHECKLIST_PATIENTS_FILE, fcc.CHECKLIST_COMPARISON_FILE, fcc.GUIDELINE_ITEMS_FILE])
