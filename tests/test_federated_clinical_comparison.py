"""Transferred federated score vs local Gleason / TP53-RB1-PTEN Cox (Figure 5 b-c)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.survival_analysis import federated_clinical_comparison as fcc  # noqa: E402

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
    somatic = _somatic([
        (1, _day(80), _day(89), 1, 0, np.nan, 1),
        (2, _day(80), _day(90), 1, 1, 1, 1),
    ])
    out = fcc.build_clinical_features(_local([1, 2, 3]), [90], gleason=gleason,
                                      somatic=somatic, somatic_features=FEATURES)
    out = out.set_index("person_id")
    assert out.loc[1, "GLEASON_SCORE"] == 9
    assert np.isnan(out.loc[2, "GLEASON_SCORE"])
    assert out.loc[3, "GLEASON_SCORE"] == 6
    assert out.loc[1, "HAS_SEQUENCING"] is True
    assert out.loc[[2, 3], "HAS_SEQUENCING"].isna().all()
    assert out.loc[[2, 3], ["TP53_ANY", "SOMATIC_N_GENES_ALTERED"]].isna().all().all()
    assert (out["landmark_days"] == 90).all() and (out["AGE"] == 70.0).all()


def test_burden_counts_every_altered_panel_gene_and_uncalled_is_wild_type():
    somatic = _somatic([
        (1, _day(-50), _day(-40), 1, 0, np.nan, 1),
        # two same-day specimens are unioned
        (2, _day(-50), _day(-40), 0, 1, 0, 0),
        (2, _day(-50), _day(-40), 0, 0, 1, 0),
        # an older specimen is superseded by the latest one
        (3, _day(-900), _day(-890), 1, 1, 1, 1),
        (3, _day(-50), _day(-40), 0, 0, 0, 0),
    ])
    out = fcc.build_clinical_features(_local([1, 2, 3]), [0], gleason=_gleason([]),
                                      somatic=somatic, somatic_features=FEATURES)
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
    assert set(comp["subpopulation"]) == {"gleason", "trio"}
    assert len(comp) == 2 * 2 * 3  # subpopulation x bundle x model
    g = comp[comp["subpopulation"] == "gleason"]
    assert (g["n_patients"] == clinical["GLEASON_SCORE"].notna().sum()).all()
    t = comp[comp["subpopulation"] == "trio"]
    assert (t["n_patients"] == clinical["HAS_SEQUENCING"].notna().sum()).all()
    for (_, bundle), d in comp.groupby(["subpopulation", "bundle"]):
        d = d.set_index("model")
        assert d["n_scored"].nunique() == 1
        assert np.isnan(d.loc["transferred_labs", "delta_c_vs_transferred"])
        delta = d.loc["clinical_cox", "c_index"] - d.loc["transferred_labs", "c_index"]
        assert np.isclose(d.loc["clinical_cox", "delta_c_vs_transferred"], delta)
        assert d.loc["clinical_cox", "delta_lo"] <= delta <= d.loc["clinical_cox", "delta_hi"]
        assert np.isfinite(d.loc["clinical_cox", "c_index_apparent"])
        assert np.isnan(d.loc["transferred_labs", "delta_auc_vs_transferred"])
        d_auc = d.loc["clinical_cox", "mean_auc"] - d.loc["transferred_labs", "mean_auc"]
        assert np.isclose(d.loc["clinical_cox", "delta_auc_vs_transferred"], d_auc)
        assert d.loc["clinical_cox", "delta_auc_lo"] <= d_auc <= d.loc["clinical_cox", "delta_auc_hi"]
        for m in ("transferred_labs", "clinical_cox"):
            assert d.loc[m, "mean_auc_lo"] <= d.loc[m, "mean_auc"] <= d.loc[m, "mean_auc_hi"]
        assert np.isfinite(d.loc["clinical_cox", "mean_auc_apparent"])
    # The clinical Cox does not depend on the bundle.
    cox = comp[comp["model"] == "clinical_cox"]
    assert cox.groupby("subpopulation")["c_index"].nunique().eq(1).all()
    # Gleason drives the simulated hazard.
    hr = coefs.set_index(["subpopulation", "covariate"])["hr"]
    assert hr[("gleason", "GLEASON_SCORE")] > 1
    assert set(coefs["subpopulation"]) == {"gleason", "trio"}


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
    assert {"Sequenced", "Any trio altered", "TP53 altered"} <= strata
    assert not strata & {"Full cohort", "Gleason available", "Gleason missing"}


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
        # collected after sequencing but before sequencing + 90: never used
        (1, _day(40), _day(40), 0, 1, 1, 1),
    ])
    local = _local([1, 2])
    specimen = fcc.select_specimen(local, somatic, FEATURES)
    seq_local = fcc.sequencing_anchored_local(local, specimen)
    gleason = _gleason([(1, _day(-200), _day(-200), 7)])
    out = fcc.build_clinical_features(seq_local, [0, 90], gleason=gleason, somatic=somatic,
                                      somatic_features=FEATURES, specimen=specimen)
    assert set(out["person_id"]) == {1}
    out = out.set_index("landmark_days")
    for lm in (0, 90):
        assert out.loc[lm, "TP53_ANY"] == 1
        assert out.loc[lm, "RB1_ANY"] == 0
        assert out.loc[lm, "SOMATIC_N_GENES_ALTERED"] == 1
        assert bool(out.loc[lm, "HAS_SEQUENCING"])
    assert np.isclose(out.loc[0, "AGE"], 70.0 - 10 / fcc.fs.DAYS_PER_YEAR)
    # Without the specimen, the later specimen would have replaced it by +90.
    latest = fcc.build_clinical_features(seq_local, [90], gleason=gleason, somatic=somatic,
                                         somatic_features=FEATURES)
    assert latest.set_index("person_id").loc[1, "RB1_ANY"] == 1


def test_compare_by_subpopulation_scores_each_arm_on_its_own_inputs():
    scores, clinical = _scores_and_clinical(seed=1)
    seq_scores, seq_clinical = _scores_and_clinical(seed=2)
    kwargs = dict(n_bootstrap=20, seed=0)
    comp, coefs = fcc.compare_by_subpopulation(
        {"gleason": (scores, clinical), "trio": (seq_scores, seq_clinical)}, **kwargs
    )
    assert set(comp["subpopulation"]) == {"gleason", "trio"}
    assert set(coefs["subpopulation"]) == {"gleason", "trio"}
    for sub, (s, c) in (("gleason", (scores, clinical)), ("trio", (seq_scores, seq_clinical))):
        expected, _ = fcc.compare_clinical_models(
            s, c, subpopulations={sub: fcc.SUBPOPULATIONS[sub]}, **kwargs
        )
        got = comp[comp["subpopulation"] == sub].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, expected.reset_index(drop=True))
