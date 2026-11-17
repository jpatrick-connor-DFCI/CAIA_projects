import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.multivariable_models import ihc_lr as il  # noqa: E402

# ne_gate.NE_MARKER (ne-gate-v3), as a literal so the tests do not need the LCA repo.
NE_MARKER = (r"\b(?:synaptophysin|chromogranin(?:\s+a)?|cd\s?56|insm-?1|"
             r"neuron[\s-]+specific\s+enolase|nse)\b")
PATS = il.build_patterns(NE_MARKER)
ANCHOR = np.datetime64("2020-01-01")


def tag(text, proc="Prostate, needle biopsy"):
    return il.tag_note(text, proc, PATS)


# --------------------------------------------------------------------------- #
# Regex
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, markers", [
    ("Immunohistochemical stains for p63 and HMWCK show loss of basal cells; AMACR positive.",
     "AMACR;HMWCK;p63"),
    ("PIN4 cocktail supports the diagnosis.", "PIN4"),
    ("PSA (+), NKX3.1 positive.", "NKX3.1;PSA"),
    ("Tumor is synaptophysin positive, chromogranin negative and CD56 positive.",
     "CD56;chromogranin;synaptophysin"),
    ("Ki-67 proliferation index ~80%. Retained RB1 staining.", "Ki-67;RB1"),
])
def test_marker_mentions_count(text, markers):
    r = tag(text)
    assert r["ihc_any"] and r["qualifies"]
    assert r["ihc_markers"] == markers


@pytest.mark.parametrize("text", [
    "Serum PSA 4.2 ng/mL. Adenocarcinoma.",
    "PSA: 12 on 1/2020. Adenocarcinoma.",
    "PSMA PET avid lesion. Adenocarcinoma.",
    "No immunohistochemistry was performed.",
    "IHC pending; addendum to follow.",
    "Immunohistochemical stains are recommended if clinically indicated.",
    "TMPRSS2-ERG fusion reported by outside molecular testing.",
    "No synaptophysin stain was performed.",
    "Recommend synaptophysin and chromogranin stains.",
    "Pending immunohistochemistry.",
    "Awaiting synaptophysin staining.",
    "PSMA-positive lesion on PET imaging.",
    "PET shows a PSMA-positive lesion.",
])
def test_non_ihc_mentions_do_not_count(text):
    r = tag(text)
    assert not r["ihc_any"], r
    assert not r["qualifies"]


def test_generic_term_counts_and_negative_stain_counts():
    assert tag("Immunohistochemistry was performed on block A1.")["ihc_generic"]
    r = tag("No synaptophysin expression is seen.")
    assert r["ihc_markers"] == "synaptophysin"


def test_specimen_rule():
    met = tag("History of prostate cancer: metastatic prostatic adenocarcinoma. "
              "Synaptophysin positive.", "Lymph node, excision")
    assert met["specimen_site"] == "non-prostate" and met["prostatic_origin"] and met["qualifies"]

    lung = tag("Small cell carcinoma. Synaptophysin positive. Consistent with lung primary.",
               "Liver, biopsy")
    assert lung["non_prostate_primary"] and not lung["qualifies"]

    no_origin = tag("Neuroendocrine carcinoma; INSM1 positive.", "Liver, biopsy")
    assert no_origin["ihc_any"] and not no_origin["qualifies"]

    uro = tag("Urothelial carcinoma, high grade. GATA3 positive.", "Bladder and prostate")
    assert uro["specimen_site"] == "prostate" and uro["non_prostate_primary"] and not uro["qualifies"]

    # Benign urothelium, or a negated / historical other primary, is not a non-prostate primary.
    for text in ("Benign urothelial mucosa. AMACR positive.",
                 "Negative for urothelial carcinoma. AMACR positive.",
                 "History of melanoma. AMACR positive."):
        r = tag(text, "TURP")
        assert not r["non_prostate_primary"] and r["qualifies"], text

    both = tag("Metastatic carcinoma consistent with prostatic adenocarcinoma; urothelial carcinoma "
               "also considered. NKX3.1 positive.", "Bone, biopsy")
    assert both["prostatic_origin"] and both["qualifies"]

    neg_origin = tag("Not consistent with prostatic adenocarcinoma. GATA3 positive.", "Lymph node")
    assert not neg_origin["prostatic_origin"] and not neg_origin["qualifies"]


# --------------------------------------------------------------------------- #
# LLM merge
# --------------------------------------------------------------------------- #
def test_parse_note_uid():
    assert il.parse_note_uid("Pathology:12345") == ("Pathology", "12345")
    assert il.parse_note_uid("Pathology:12345#2") == ("Pathology", "12345#2")
    assert il.parse_note_uid("Imaging:9") == ("Imaging", "9")
    assert il.parse_note_uid(None) == (None, None)


def test_gated_rows_to_notes_keeps_pathology_and_unions_types():
    gated = {1: [{"note_uid": "Pathology:10", "evidence_type": "positive_ne_ihc"},
                 {"note_uid": "Pathology:10", "evidence_type": "nepc_diagnosis"},
                 {"note_uid": "Imaging:10", "evidence_type": "nepc_diagnosis"}],
             2: [{"note_uid": "Clinician:7", "evidence_type": "ne_differentiation"}]}
    out = il.gated_rows_to_notes(gated)
    assert out.to_dict("records") == [
        {"DFCI_MRN": 1, "RPT_ID": "10", "gated_evidence_types": "nepc_diagnosis;positive_ne_ihc"}]


def test_load_gated_ne_notes_uses_lca_loader_and_gate(tmp_path):
    calls = {}

    class NoteExtraction:
        @staticmethod
        def load_label_inputs(spec, output_dir):
            calls["args"] = (spec, output_dir)
            return {"findings": "FINDINGS", "notes": {(5, "Pathology:1"): {}},
                    "status": {5: "ok", 6: "failed"}}

    class Nepc:
        SPEC = "SPEC"

        @staticmethod
        def gate_findings(findings):
            calls["gated"] = findings
            return {5: [{"note_uid": "Pathology:1", "evidence_type": "positive_ne_ihc"}]}, {}

    out, status = il.load_gated_ne_notes({"note_extraction": NoteExtraction, "nepc": Nepc}, tmp_path)
    assert status == {5: "ok", 6: "failed"}
    assert calls == {"args": ("SPEC", tmp_path), "gated": "FINDINGS"}
    assert out.iloc[0].to_dict() == {"DFCI_MRN": 5, "RPT_ID": "1", "gated_evidence_types": "positive_ne_ihc"}


def test_merge_ne_positive_requires_qualifying_note():
    notes = pd.DataFrame({"DFCI_MRN": [1, 1, 2], "RPT_ID": ["a", "b", "c"],
                          "qualifies": [True, False, True]})
    gated = pd.DataFrame({"DFCI_MRN": [1, 1], "RPT_ID": ["a", "b"],
                          "gated_evidence_types": ["positive_ne_ihc", "nepc_diagnosis"]})
    out = il.merge_ne_positive(notes, gated, {1: "ok", 2: "ok"})
    assert out["ne_gated"].tolist() == [True, True, False]
    assert out["ne_positive"].tolist() == [True, False, False]


def test_load_pathology_notes(tmp_path):
    pl = pytest.importorskip("polars")
    path = tmp_path / "PATHOLOGY_NOTES.parquet"
    pl.DataFrame({"RPT_ID": [11, 12, 13], "DFCI_MRN": [1.0, 2.0, None],
                  "EVENT_DATE": ["2021-03-14T00:00:00-05:00", "2022-01-02", "2022-01-02"],
                  "PROC_DESC": ["Prostate", None, "x"], "RPT_TEXT": ["t", None, "u"],
                  "FILE": ["f", "f", "f"]}).write_parquet(path)
    out = il.load_pathology_notes(path, mrns={1, 2})
    assert out["RPT_ID"].tolist() == ["11", "12"]
    assert out["DFCI_MRN"].tolist() == [1, 2]
    assert out["note_date"].tolist() == [pd.Timestamp("2021-03-14"), pd.Timestamp("2022-01-02")]
    assert out["RPT_TEXT"].tolist() == ["t", ""]


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #
def _note(mrn, day, *, ihc=True, qualifies=True, gated=False):
    return {"DFCI_MRN": mrn, "RPT_ID": f"{mrn}-{day}", "note_date": ANCHOR + np.timedelta64(day, "D"),
            "ihc_any": ihc, "qualifies": qualifies and ihc, "ne_gated": gated,
            "ne_positive": gated and qualifies and ihc}


def _frame(notes, mrns, landmarks=(90,)):
    notes = pd.DataFrame(notes)
    anchors = pd.Series(pd.Timestamp(ANCHOR), index=pd.Index(mrns, name="DFCI_MRN"))
    notes = il.add_days_from_adt(notes, anchors)
    scores = pd.DataFrame([{"person_id": m, "landmark_days": lm, "risk_score": float(m)}
                           for lm in landmarks for m in mrns])
    return il.build_ihc_frame(scores, notes, landmarks, anchors=anchors,
                              review_status={m: "ok" for m in mrns}).set_index(["landmark_days", "person_id"])


def test_window_boundaries():
    f = _frame([_note(1, 90), _note(2, 91), _note(3, 3650), _note(4, 3651)], [1, 2, 3, 4, 5])
    assert f.loc[90, "ihc_future"].to_dict() == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}


def test_flags_prior_ne_and_lost_to_specimen():
    notes = [
        _note(1, 30, gated=True),                     # prior NE-positive
        _note(1, 200, gated=True),                    # also a future one
        _note(2, 200, gated=True, qualifies=False),   # gated IHC, fails specimen rule
        _note(3, 200, ihc=False, gated=True),         # gated report without IHC hit
        _note(4, 200, qualifies=False),               # IHC mention, wrong specimen
        _note(5, 200),                                # qualifying IHC, NE-negative
    ]
    f = _frame(notes, [1, 2, 3, 4, 5]).loc[90]
    assert f["prior_ne_pos"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0}
    assert f["ne_pos_future"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0}
    assert f["ihc_future"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 1}
    assert f["any_ihc_mention_future"].to_dict() == {1: 1, 2: 1, 3: 0, 4: 1, 5: 1}
    assert f["ne_gated_ihc_lost_to_specimen"].to_dict() == {1: 0, 2: 1, 3: 0, 4: 0, 5: 0}
    assert f["ne_gated_without_ihc_future"].to_dict() == {1: 0, 2: 0, 3: 1, 4: 0, 5: 0}

    sub = f.reset_index()
    assert set(il.outcome_population(sub, "ihc_any")["person_id"]) == {1, 2, 3, 4, 5}
    assert set(il.outcome_population(sub, "ne_pos_all")["person_id"]) == {2, 3, 4, 5}
    assert set(il.outcome_population(sub, "ne_pos_given_ihc")["person_id"]) == {5}


def test_landmark_changes_prior_vs_future():
    f = _frame([_note(1, 100, gated=True)], [1], landmarks=(0, 180))
    assert f.loc[(0, 1), "ne_pos_future"] == 1 and f.loc[(0, 1), "prior_ne_pos"] == 0
    assert f.loc[(180, 1), "ne_pos_future"] == 0 and f.loc[(180, 1), "prior_ne_pos"] == 1


def test_anchor_dates_rejects_conflicts():
    local = pd.DataFrame({"DFCI_MRN": [1, 1, 2], "TREATMENT_ANCHOR_DATE": ["2020-01-01"] * 2 + ["2021-05-05"]})
    assert il.anchor_dates(local).to_dict() == {1: pd.Timestamp("2020-01-01"), 2: pd.Timestamp("2021-05-05")}
    with pytest.raises(ValueError):
        il.anchor_dates(pd.DataFrame({"DFCI_MRN": [1, 1],
                                      "TREATMENT_ANCHOR_DATE": ["2020-01-01", "2020-02-01"]}))


# --------------------------------------------------------------------------- #
# LR driver
# --------------------------------------------------------------------------- #
def _synthetic(n=600, seed=3):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    ihc = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.0 * risk))))
    ne = ihc * rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 1.0 * risk))))
    prior = (rng.random(n) < 0.03).astype(int)
    return pd.DataFrame({
        "person_id": np.arange(n), "landmark_days": 0, "risk_score": risk * 3 + 10,
        "any_ihc_mention_future": ihc, "ihc_future": ihc, "ne_pos_future": ne * (1 - prior),
        "prior_ne_pos": prior, "ne_review_complete": True, "ne_review_status": "ok",
        "ne_gated_ihc_lost_to_specimen": 0, "ne_gated_without_ihc_future": 0,
    })


def test_run_ihc_lr_recovers_signal_and_counts():
    frame = _synthetic()
    frame.loc[0, "risk_score"] = np.nan
    models, coefs, counts = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=20)
    assert list(models.columns) == il.MODEL_COLUMNS and list(counts.columns) == il.COUNT_COLUMNS
    assert models["outcome"].tolist() == list(il.OUTCOMES)
    assert (models["auc"] > 0.6).all() and (models["auc_lo"] <= models["auc"]).all()
    assert (coefs["term"] == "RISK_Z").all() and (coefs["odds_ratio"] > 1).all()
    c = counts.iloc[0]
    assert c["n_at_risk"] == len(frame) - 1
    assert c["n_ne_pos_all"] == c["n_at_risk"] - c["n_prior_ne_pos"]
    assert c["n_ne_pos_given_ihc"] <= c["n_ihc_future"]
    ihc_row = models.set_index("outcome").loc["ne_pos_given_ihc"]
    assert ihc_row["n"] == c["n_ne_pos_given_ihc"] and ihc_row["events"] == c["events_ne_pos_given_ihc"]


def test_few_events_gives_count_only_row():
    frame = _synthetic(n=100)
    frame["ne_pos_future"] = 0
    frame.loc[:2, "ne_pos_future"] = 1
    frame["prior_ne_pos"] = 0
    models, coefs, _ = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0)
    ne = models[models["outcome"].ne("ihc_any")]
    assert ne["note"].str.contains("fewer than 5").all() and ne["auc"].isna().all()
    assert set(coefs["outcome"]) == {"ihc_any"}


def test_cohort_restriction():
    frame = _synthetic()
    _, _, counts = il.run_ihc_lr(frame, {"sub": set(range(100))}, (0,), n_bootstrap=0)
    assert counts.iloc[0]["n_at_risk"] == 100 and counts.iloc[0]["n_cohort_list"] == 100


def test_marker_frequency():
    notes = pd.DataFrame({"qualifies": [True, True, False], "ne_positive": [True, False, False],
                          "ihc_markers": ["synaptophysin;CD56", "synaptophysin", "p63"]})
    out = il.marker_frequency(notes).set_index("marker")
    assert out.loc["synaptophysin"].tolist() == [2, 1] and out.loc["CD56"].tolist() == [1, 1]
    assert "p63" not in out.index


@pytest.mark.parametrize("text", [
    "No synaptophysin expression is seen.",
    "No synaptophysin staining is seen.",
    "Synaptophysin stain was negative.",
    "Synaptophysin negative. Recommend chromogranin staining.",
    "PSMA-positive lesion on PET imaging. Synaptophysin is negative.",
    "Recommend synaptophysin staining. Synaptophysin stain is positive in the addendum.",
])
def test_completed_negative_stains_and_later_results_still_count(text):
    assert tag(text)["qualifies"]


def test_collision_suffix_is_rejected_instead_of_merged():
    with pytest.raises(ValueError, match="Ambiguous pathology"):
        il.gated_rows_to_notes({1: [
            {"note_uid": "Pathology:10#1", "evidence_type": "positive_ne_ihc"}]})
    notes = pd.DataFrame({"DFCI_MRN": [1, 1], "RPT_ID": ["10", "10"],
                          "note_date": pd.to_datetime(["2019-01-01", "2021-01-01"]),
                          "qualifies": [True, True]})
    gated = il.gated_rows_to_notes({1: [
        {"note_uid": "Pathology:10", "evidence_type": "positive_ne_ihc"}]})
    with pytest.raises(ValueError, match="ambiguous pathology"):
        il.merge_ne_positive(notes, gated, {1: "ok"})


def test_loader_rejects_ungated_collision_in_selected_evidence(tmp_path):
    from types import SimpleNamespace
    lca = {
        "note_extraction": SimpleNamespace(load_label_inputs=lambda *a: {
            "findings": None, "notes": {(1, "Pathology:10#1"): {}}, "status": {1: "ok"}}),
        "nepc": SimpleNamespace(SPEC=None, gate_findings=lambda *a: ({}, {})),
    }
    with pytest.raises(ValueError, match="Ambiguous pathology"):
        il.load_gated_ne_notes(lca, tmp_path, mrns={1})
    # An unrelated patient's collision does not block this cohort.
    out, status = il.load_gated_ne_notes(lca, tmp_path, mrns={2})
    assert out.empty and not status


def test_incomplete_review_is_unknown_and_excluded_only_from_ne_outcomes():
    mrns = list(range(1, 8))
    statuses = {1: "ok", 2: "no_trigger_notes", 3: "failed", 4: "not_run", 5: "partial:1/2"}
    # 6 and 7 are absent from the LCA cohort; 7 has no pathology notes either.
    notes = pd.DataFrame([_note(m, 100) for m in mrns[:-1]])
    gated = il.gated_rows_to_notes({5: [
        {"note_uid": "Pathology:5-100", "evidence_type": "positive_ne_ihc"}]})
    merged = il.merge_ne_positive(notes.drop(columns=["ne_positive", "ne_gated"]), gated, statuses)
    assert merged.set_index("DFCI_MRN").loc[[3, 4, 6], "ne_positive"].isna().all()
    assert not merged.set_index("DFCI_MRN").loc[2, "ne_positive"]
    assert merged.set_index("DFCI_MRN").loc[5, "ne_positive"]  # Known positive, review incomplete.
    scores = pd.DataFrame({"person_id": mrns, "landmark_days": 0, "risk_score": 0.5})
    anchors = pd.Series(pd.Timestamp(ANCHOR), index=mrns)
    frame = il.build_ihc_frame(scores, merged, [0], anchors=anchors, review_status=statuses)
    assert set(il.outcome_population(frame, "ihc_any")["person_id"]) == set(mrns)
    for outcome in ("ne_pos_all", "ne_pos_given_ihc"):
        assert set(il.outcome_population(frame, outcome)["person_id"]) == {1, 2}
    assert frame.loc[frame["person_id"].ge(3), "ne_pos_future"].isna().all()
    _, _, counts = il.run_ihc_lr(frame, {"all": None}, [0], n_bootstrap=0)
    assert counts.loc[0, "n_ne_review_excluded"] == 5
    assert counts.loc[0, "n_ne_review_complete"] == 2
    assert il.marker_frequency(merged.assign(ihc_markers="synaptophysin"))["ne_positive_notes"].sum() == 1


@pytest.mark.parametrize("missing_patient", [1, 2])
def test_missing_anchor_fails_even_without_notes(missing_patient):
    scores = pd.DataFrame({"person_id": [1, 2], "landmark_days": 0, "risk_score": 0.5})
    notes = pd.DataFrame([_note(1, 100, gated=True)])
    anchors = pd.Series(pd.Timestamp(ANCHOR), index=[3 - missing_patient])
    with pytest.raises(ValueError, match="lack an ADT anchor"):
        il.build_ihc_frame(scores, notes, [0], anchors=anchors,
                           review_status={1: "ok", 2: "ok"})


def test_run_ihc_lr_oof_predictions_per_outcome():
    frame = _synthetic()
    frame.loc[0, "risk_score"] = np.nan
    frame.loc[1:20, "ne_review_complete"] = False
    frame.loc[1:20, ["ne_pos_future", "prior_ne_pos"]] = np.nan
    assert len(il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0)) == 3
    models, _, _, preds = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0,
                                        return_predictions=True)
    assert list(preds.columns) == il.PREDICTION_COLUMNS
    m = models.set_index("outcome")
    for outcome, g in preds.groupby("outcome"):
        assert len(g) == m.loc[outcome, "n"] and g["event"].sum() == m.loc[outcome, "events"]
        assert il.nsl._auc(g["event"].to_numpy(), g["oof_prob"].to_numpy()) == pytest.approx(
            m.loc[outcome, "auc"])
        assert g["cv_fold"].between(0, 4).all() and g["oof_prob"].between(0, 1).all()
    assert 0 not in set(preds["person_id"])                                    # no risk score
    assert set(preds.query("outcome == 'ihc_any'")["person_id"]) >= set(range(1, 21))
    assert not set(preds.query("outcome != 'ihc_any'")["person_id"]) & set(range(1, 21))
    given = preds.query("outcome == 'ne_pos_given_ihc'")["person_id"]
    assert frame.set_index("person_id").loc[given, "ihc_future"].eq(1).all()

    frame["ne_pos_future"] = frame["ne_pos_future"] * 0                        # no NE events
    _, _, _, few = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0, return_predictions=True)
    assert set(few["outcome"]) == {"ihc_any"}


def test_raw_score_auc_per_outcome_independent_of_cv(monkeypatch):
    from sklearn.metrics import roc_auc_score
    frame = _synthetic()
    models, _, _, preds = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=100,
                                        return_predictions=True)
    assert list(models.columns) == il.MODEL_COLUMNS
    fit = models[models["auc"].notna()]
    assert len(fit)
    for _, row in fit.iterrows():
        g = preds[preds["outcome"].eq(row["outcome"])]
        assert row["auc_raw_score"] == pytest.approx(roc_auc_score(g["event"], g["risk_score"]))
        assert row["auc_raw_score_lo"] <= row["auc_raw_score"] <= row["auc_raw_score_hi"]

    monkeypatch.setattr(il.nsl, "oof_predictions", lambda y, X, **kw: (np.full(len(y), np.nan), 1))
    failed, _, _ = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=50)
    failed = failed.set_index("outcome").loc[fit["outcome"]]
    assert failed["auc"].isna().all() and failed["auc_lo"].isna().all()
    np.testing.assert_allclose(failed["auc_raw_score"], fit["auc_raw_score"])
    assert failed[["auc_raw_score_lo", "auc_raw_score_hi"]].notna().all().all()
