"""Frozen labs checklist flag as an item of internal Gleason / TSG checklists, out of fold (reviewer, Figure 6)."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.federated_models import checklist_extension as ce  # noqa: E402
from COMPASS.federated_models import federated_clinical_comparison as fcc  # noqa: E402

LANDMARKS = (0, 90)
HORIZONS = (365,)
FROZEN_M, FROZEN_N = 2, 4
BUNDLE = "fed_checklist_guideline"


def _patients(n=400, seed=1):
    rng = np.random.default_rng(seed)
    count = rng.integers(0, FROZEN_N + 1, n)
    gleason = rng.choice([6.0, 7.0, 8.0, 9.0], n)
    gleason[rng.random(n) < 0.25] = np.nan
    sequenced = rng.random(n) < 0.6
    tp53 = (rng.random(n) < 0.3).astype(float)
    rb1 = (rng.random(n) < 0.1).astype(float)
    return pd.DataFrame({"person_id": 9000 + np.arange(n), "count": count, "gleason": gleason,
                         "sequenced": sequenced, "TP53_ANY": tp53, "RB1_ANY": rb1, "PTEN_ANY": np.nan})


def _scores(pt, seed=2):
    """Frozen checklist rows for platinum / AVPC / NEPC whose hazard rises with count, Gleason and TP53."""
    rng = np.random.default_rng(seed)
    rows = []
    for landmark, horizon, endpoint in product(LANDMARKS, HORIZONS, ("platinum", "avpc", "nepc")):
        lin = (0.6 * pt["count"] + 0.8 * pt["gleason"].fillna(7).ge(8) + 1.0 * pt["TP53_ANY"]).to_numpy()
        p = 1 / (1 + np.exp(-(lin - 3.0)))
        case = rng.random(len(pt)) < p
        duration = np.where(case, rng.uniform(20, 360, len(pt)), rng.uniform(400, 3000, len(pt)))
        event = np.where(case, 1, (rng.random(len(pt)) < 0.3).astype(int))
        rows.append(pd.DataFrame({
            "person_id": pt["person_id"], "endpoint": endpoint, "bundle": BUNDLE,
            "model_family": fcc.CHECKLIST_FAMILY, "config": "both", "landmark_days": landmark,
            "training_arm": "no_dfci", "model_horizon_days": horizon, "evaluation_horizon_days": horizon,
            "risk_score": pt["count"].to_numpy(), "duration_days": duration, "event": event,
            "probability": np.nan, "flag": (pt["count"] >= FROZEN_M).astype(int).to_numpy()}))
    scores = pd.concat(rows, ignore_index=True)
    decoys = pd.concat([scores.assign(config="baseline"), scores.assign(model_family="xgboost_cox")])
    return pd.concat([scores, decoys], ignore_index=True)


def _clinical(pt, sequencing):
    out = []
    for landmark in LANDMARKS:
        trio = pt[list(fcc.TRIO_FEATURES)]
        out.append(pd.DataFrame({
            "person_id": pt["person_id"], "landmark_days": landmark, "AGE": 70.0,
            "GLEASON_SCORE": pt["gleason"],
            "HAS_SEQUENCING": np.where(pt["sequenced"], 1.0, np.nan) if sequencing else np.nan,
            **{c: (trio[c].where(pt["sequenced"]) if sequencing else np.nan) for c in fcc.TRIO_FEATURES},
            "SOMATIC_N_GENES_ALTERED": np.nan}))
    return pd.concat(out, ignore_index=True)


def _frozen():
    return pd.DataFrame([{"bundle": BUNDLE, "landmark_days": lm, "horizon_days": h, "M": FROZEN_M, "N": FROZEN_N}
                         for lm, h in product(LANDMARKS, HORIZONS)])


def _sources(pt=None):
    pt = _patients() if pt is None else pt
    scores = _scores(pt)
    seq = _clinical(pt, True)
    return {"gleason": (scores, _clinical(pt, False)), "somatic": (scores, seq), "combined": (scores, seq)}


def _run(sources=None, folds=None, **kwargs):
    frames = ce.cohort_frames(sources or _sources(), LANDMARKS, HORIZONS)
    folds = ce.assign_folds(frames, 5, 0) if folds is None else folds
    patients, rules = ce.cross_fit(frames, folds, _frozen(), 5, **kwargs)
    return frames, folds, patients, rules


def test_frozen_rules_read_m_and_n_of_the_both_models():
    bundles = {"x_guideline": {"model_family": fcc.CHECKLIST_FAMILY, "models": [
        {"config": "both", "landmark_days": 90, "horizon_days": 1825,
         "checklist": {"model": {"threshold": 3, "selected_indices": [0, 4, 7, 9, 11]}}},
        {"config": "baseline", "landmark_days": 90, "horizon_days": 1825,
         "checklist": {"model": {"threshold": 1, "selected_indices": [0]}}}]},
        "cox": {"model_family": "xgboost_cox", "models": []}}
    frozen = ce.frozen_rules(bundles)
    assert frozen.to_dict("records") == [{"bundle": "x_guideline", "landmark_days": 90, "horizon_days": 1825,
                                          "M": 3, "N": 5}]


def test_cohort_frames_respect_membership_and_skip_missing_cells(capsys):
    pt = _patients()
    frames = ce.cohort_frames(_sources(pt), (*LANDMARKS, 180), HORIZONS)
    assert "skipping landmark 180" in capsys.readouterr().out
    n_gleason, n_seq = pt["gleason"].notna().sum(), pt["sequenced"].sum()
    n_both = (pt["gleason"].notna() & pt["sequenced"]).sum()
    per_cell = len(LANDMARKS) * len(HORIZONS) * 3
    assert len(frames["gleason"]) == n_gleason * per_cell
    assert len(frames["somatic"]) == n_seq * per_cell
    assert len(frames["combined"]) == n_both * per_cell
    assert frames["gleason"]["model_family"].eq(fcc.CHECKLIST_FAMILY).all()


def test_folds_are_patient_level_and_shared_by_every_cohort():
    frames, folds, patients, _ = _run()
    assert folds.index.is_unique and set(folds) == set(range(5))
    every = set().union(*(set(f["person_id"]) for f in frames.values()))
    assert set(folds.index) == every
    per_patient = patients.groupby("person_id")["fold"].nunique()
    assert per_patient.eq(1).all()
    sizes = folds.value_counts()
    assert sizes.max() - sizes.min() <= 3


def _training(n=300, seed=5):
    rng = np.random.default_rng(seed)
    labs = rng.integers(0, 2, n).astype(float)
    frame = pd.DataFrame({"GLEASON_SCORE": rng.choice([6, 7, 8, 9, 10], n).astype(float),
                          "TP53_ANY": rng.integers(0, 2, n), "RB1_ANY": rng.integers(0, 2, n),
                          "PTEN_ANY": np.nan})
    gleason, tsg = ce.candidate_items(frame)
    labels = rng.choice([-1, 0, 1], n, p=[0.1, 0.6, 0.3])
    weights = ce.balanced_weights(labels, np.where(labels >= 0, rng.uniform(1, 2, n), 0))
    return labs, gleason, tsg, labels, weights


def test_fit_rule_finds_the_brute_force_optimum_over_item_sets():
    labs, gleason, tsg, labels, weights = _training()
    assert weights[labels == 1].sum() == pytest.approx(0.5) and weights[labels == 0].sum() == pytest.approx(0.5)
    assert len(ce.TSG_OPTIONS) == 9 and ce.TSG_OPTIONS[:2] == ((), ("any",))
    for rule in ("checklist_gleason_tsg", "gleason_tsg", "checklist_tsg", "gleason"):
        spec = ce.RULES[rule]
        brute = np.inf
        for g in ((None, *ce.GLEASON_CUTOFFS) if spec.gleason else (None,)):
            for items in (ce.TSG_OPTIONS if spec.tsg else ((),)):
                parts = ([labs] if spec.labs else []) + ([gleason[g]] if g is not None else []) \
                    + [tsg[t] for t in items]
                for m in range(1, len(parts) + 1):
                    brute = min(brute, ce.balanced_error(np.sum(parts, axis=0) >= m, labels, weights))
        fit = ce.fit_rule(rule, labs, gleason, tsg, labels, weights)
        assert fit["train_balanced_error"] == pytest.approx(brute), rule
        assert fit["labs_item"] == spec.labs and fit["N"] >= 1
        assert fit["N"] == int(fit["labs_item"]) + (fit["gleason_cutoff"] is not None) \
            + len(ce.tsg_items(fit["tsg_items"]))
    fixed = ce.fit_rule("checklist", labs, gleason, tsg, labels, weights)
    assert (fixed["labs_item"], fixed["gleason_cutoff"], fixed["tsg_items"], fixed["M"], fixed["N"]) == \
        (True, None, "", 1, 1)
    assert ce.apply_rule(fixed, labs, gleason, tsg).tolist() == labs.tolist()


def test_labs_is_forced_in_and_an_empty_addition_reproduces_the_labs_flag():
    # the labs flag classifies perfectly, so nothing added can help and fewer items wins
    labels = np.array([1, 1, 0, 0, 0, 0])
    labs = labels.astype(float)
    frame = pd.DataFrame({"GLEASON_SCORE": [9.0, 6, 9, 6, 9, 6], "TP53_ANY": [1, 0, 1, 0, 1, 0],
                          "RB1_ANY": 0, "PTEN_ANY": 0})
    gleason, tsg = ce.candidate_items(frame)
    weights = ce.balanced_weights(labels, np.ones(6))
    fit = ce.fit_rule("checklist_gleason_tsg", labs, gleason, tsg, labels, weights)
    assert fit["train_balanced_error"] == 0
    assert (fit["labs_item"], fit["gleason_cutoff"], fit["tsg_items"], fit["M"], fit["N"]) == (True, None, "", 1, 1)
    # without the labs flag the internal checklist must keep at least one item
    internal = ce.fit_rule("gleason_tsg", labs, gleason, tsg, labels, weights)
    assert internal["labs_item"] is False and internal["N"] >= 1


def test_ties_prefer_fewer_items_then_the_figure_defaults():
    frame = pd.DataFrame({"GLEASON_SCORE": [9.0, 9.0, 6.0, 6.0], "TP53_ANY": [1, 1, 0, 0],
                          "RB1_ANY": [1, 1, 0, 0], "PTEN_ANY": [1, 1, 0, 0]})
    gleason, tsg = ce.candidate_items(frame)
    labels = np.array([1, 1, 0, 0])
    weights = ce.balanced_weights(labels, np.ones(4))
    zeros = np.zeros(4)
    fit = ce.fit_rule("gleason_tsg", zeros, gleason, tsg, labels, weights)
    assert fit["train_balanced_error"] == 0
    assert (fit["gleason_cutoff"], fit["tsg_items"], fit["N"], fit["M"]) == (8, "", 1, 1)
    fit = ce.fit_rule("tsg", zeros, gleason, tsg, labels, weights)
    assert (fit["tsg_items"], fit["N"]) == ("any", 1)


def test_out_of_fold_flags_ignore_the_patients_own_outcomes():
    pt = _patients()
    sources = _sources(pt)
    _, folds, base, _ = _run(sources)
    held = folds[folds.eq(2)].index
    scores = sources["gleason"][0].copy()
    hit = scores["person_id"].isin(held) & scores["endpoint"].eq("platinum")
    scores.loc[hit, "event"] = 1 - scores.loc[hit, "event"]
    scores.loc[hit, "duration_days"] = 3000 - scores.loc[hit, "duration_days"]
    changed = {c: (scores, clin) for c, (_, clin) in sources.items()}
    _, _, after, _ = _run(changed, folds)   # the strata use platinum, so hold the split fixed
    flag_cols = [c for c in base.columns if c.startswith("flag_")]
    key = ["cohort", "endpoint", "landmark_days", "person_id"]
    b = base[base["fold"].eq(2)].sort_values(key)[flag_cols].reset_index(drop=True)
    a = after[after["fold"].eq(2)].sort_values(key)[flag_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_checklist_rows_reproduce_the_frozen_flag_and_cohorts_have_their_rules():
    frames, _, patients, rules = _run()
    assert list(patients.columns) == ce.PATIENT_COLUMNS
    assert list(rules.columns) == ce.RULE_COLUMNS
    assert patients["flag_checklist"].eq(patients["risk_score"].ge(FROZEN_M).astype(float)).all()
    assert patients["flag_checklist"].eq(patients["flag"]).all()
    for cohort in ce.COHORTS:
        z = patients[patients["cohort"].eq(cohort)]
        for rule in ce.RULES:
            has = np.isfinite(z[f"flag_{rule}"]).all()
            assert has == (rule in ce.cohort_rules(cohort)), (cohort, rule)
    assert ce.cohort_rules("gleason") == ["checklist", "gleason_ge8", "gleason", "checklist_gleason"]
    assert ce.cohort_rules("somatic") == ["checklist", "tsg_any", "tsg", "checklist_tsg"]
    assert len(ce.cohort_rules("combined")) == len(ce.RULES) == 9
    ok = rules[rules["status"].eq("ok")]
    assert set(ok["fold"]) == {"0", "1", "2", "3", "4", "all"}
    g8 = ok[ok["rule"].eq("gleason_ge8")]
    assert g8["gleason_cutoff"].eq(8).all() and g8["M"].eq(1).all()
    labs_rules = ok[ok["rule"].str.startswith("checklist")]
    assert labs_rules["labs_item"].astype(bool).all()
    assert not ok[~ok["rule"].str.startswith("checklist")]["labs_item"].astype(bool).any()


def test_gate_failure_leaves_tuned_flags_missing():
    _, _, patients, rules = _run(min_cases=10_000)
    assert patients["flag_checklist_gleason"].isna().all()
    assert patients["flag_checklist"].notna().all() and patients["flag_gleason_ge8"].notna().sum() > 0
    assert rules["status"].str.startswith("too few").all()
    comparison = ce.extension_comparison(patients, rules, n_bootstrap=10)
    assert comparison["note"].eq("no patient with every out-of-fold flag").all()
    assert comparison["auc"].isna().all()


def test_comparison_end_to_end(tmp_path):
    comparison, rules, patients = ce.run_extension(_sources(), _frozen(), landmarks=LANDMARKS, horizons=HORIZONS,
                                                   n_bootstrap=20)
    assert list(comparison.columns) == ce.COMPARISON_COLUMNS
    n_cells = len(LANDMARKS) * len(HORIZONS) * 3
    assert len(comparison) == n_cells * sum(len(ce.cohort_rules(c)) for c in ce.COHORTS)
    base = comparison[comparison["rule"].eq("checklist")]
    assert base["delta_auc_vs_checklist"].eq(0).all()
    assert base["auc"].between(0.5, 1).all()
    tuned = comparison[comparison["rule"].eq("checklist_gleason_tsg")]
    assert tuned["refit_labs_item"].astype(bool).all() and tuned["refit_N"].between(1, 5).all()
    assert comparison["n_excluded"].eq(0).all() and comparison["n_failed_folds"].eq(0).all()
    # the synthetic hazard rises with Gleason and TP53, so adding them should help the labs checklist flag
    plat = comparison[comparison["endpoint"].eq("platinum") & comparison["cohort"].eq("combined")]
    assert plat.set_index(["landmark_days", "rule"])["auc"].unstack().eval(
        "checklist_gleason_tsg > checklist").all()
    out = ce.write_extension(tmp_path, comparison, rules, patients)
    assert sorted(p.name for p in out.iterdir()) == sorted([ce.COMPARISON_FILE, ce.RULES_FILE, ce.PATIENTS_FILE])
    assert len(pd.read_csv(out / ce.COMPARISON_FILE)) == len(comparison)
