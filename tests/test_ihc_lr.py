import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.multivariable_models import ihc_lr as il  # noqa: E402
from COMPASS.prostate_ihc import paths as ihc_paths  # noqa: E402

ANCHOR = pd.Timestamp("2020-01-01")


def _day(d):
    return (ANCHOR + np.timedelta64(int(d), "D")).strftime("%Y-%m-%d")


def _tables(specs):
    """specs: (mrn, usid, day or None, kind) -> (specimen, long) CSV frames.

    kind: none (matched, no stain), stain (p63 only), ne (NE+, AR axis not stained), nepc (NE+, AR axis negative),
    arlow (AR axis negative, NE not stained), amphi (NE+, AR axis positive), unmatched.
    """
    srows, lrows = [], []
    for mrn, usid, day, kind in specs:
        matched = kind != "unmatched"
        ne, nepc = kind in ("ne", "nepc", "amphi"), kind == "nepc"
        ar_neg, ar_pos = kind in ("nepc", "arlow"), kind == "amphi"
        srows.append({
            "MRN_INT": mrn, "SAMPLE_ID": f"S{usid}", "UNIQUE_SAMPLE_ID": usid,
            "collection_date": _day(day if day is not None else 0),
            "path_matched": "true" if matched else "false",
            "ihc_phenotype": "x", "NEPC_ihc": str(nepc).lower(), "amphicrine_ihc": str(ar_pos).lower(),
            "NE_stained": str(ne).lower(), "NE_any_pos": str(ne).lower(),
            "ARaxis_any_pos": str(ar_pos).lower(), "ARaxis_any_neg": str(ar_neg).lower(), "NE_histology": "false",
            "Synaptophysin": "positive" if ne else None,
            "p63": "negative" if kind == "stain" else None,
            "NKX3.1": "negative" if ar_neg else ("positive" if ar_pos else None),
        })
        if matched and kind != "none":
            lrows.append({"UNIQUE_SAMPLE_ID": usid, "specimen_date": None if day is None else _day(day),
                          "target": "lesional", "marker_norm": "x"})
            lrows.append({"UNIQUE_SAMPLE_ID": usid, "specimen_date": _day(-5000), "target": "background",
                          "marker_norm": "x"})   # background rows never date the specimen
    long = pd.DataFrame(lrows, columns=["UNIQUE_SAMPLE_ID", "specimen_date", "target", "marker_norm"])
    return pd.DataFrame(srows), long


def _frame(specs, mrns, landmarks=(90,)):
    specimen, long = _tables(specs)
    spec = il.load_oncopanel_ihc(specimen, long)
    anchors = pd.Series(ANCHOR, index=pd.Index(mrns, name="DFCI_MRN"))
    scores = pd.DataFrame([{"person_id": m, "landmark_days": lm, "risk_score": float(m)}
                           for lm in landmarks for m in mrns])
    return il.build_ihc_frame(scores, spec, landmarks, anchors=anchors).set_index(["landmark_days", "person_id"])


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def test_load_oncopanel_ihc_flags_and_dates():
    specimen, long = _tables([(1, "a", 100, "nepc"), (2, "b", 200, "stain"), (3, "c", 300, "none"),
                              (4, "d", 400, "unmatched"), (5, "e", 500, "ne"), (6, "f", 600, "arlow"),
                              (7, "g", 700, "amphi")])
    out = il.load_oncopanel_ihc(specimen, long).set_index("UNIQUE_SAMPLE_ID")
    assert out["ihc_any"].to_dict() == {"a": True, "b": True, "c": False, "d": False, "e": True, "f": True, "g": True}
    assert out["ne_pos"].to_dict() == {"a": True, "b": False, "c": False, "d": False, "e": True, "f": False, "g": True}
    assert out["nepc_like"].to_dict() == {"a": True, "b": False, "c": False, "d": False, "e": False, "f": False,
                                          "g": False}
    # AR-low = AR axis negative and not positive, whatever NE shows: NEPC-like and AR-low-only, not amphicrine.
    assert out["ar_low"].to_dict() == {"a": True, "b": False, "c": False, "d": False, "e": False, "f": True, "g": False}
    assert out.loc["a", "specimen_date"] == pd.Timestamp(_day(100))      # lesional row, not background
    assert pd.isna(out.loc["c", "specimen_date"]) and pd.isna(out.loc["d", "specimen_date"])
    assert out.loc["d", "collection_date"] == pd.Timestamp(_day(400))


def test_load_oncopanel_ihc_rejects_bad_inputs():
    specimen, long = _tables([(1, "a", 100, "ne")])
    with pytest.raises(ValueError, match="duplicate"):
        il.load_oncopanel_ihc(pd.concat([specimen, specimen]), long)
    for col in ("NEPC_ihc", "ARaxis_any_pos", "ARaxis_any_neg"):
        with pytest.raises(ValueError, match="lacks columns"):
            il.load_oncopanel_ihc(specimen.drop(columns=col), long)
    s = specimen.copy(); s["NEPC_ihc"] = "true"        # NEPC-like but AR axis not negative
    with pytest.raises(ValueError, match="Inconsistent"):
        il.load_oncopanel_ihc(s, long)
    bad = long.copy(); bad.loc[bad["target"].eq("lesional"), "specimen_date"] = "not a date"
    with pytest.raises(ValueError, match="Unparseable"):
        il.load_oncopanel_ihc(specimen, bad)
    s = specimen.copy(); s["NE_any_pos"] = "maybe"
    with pytest.raises(ValueError, match="non-boolean"):
        il.load_oncopanel_ihc(s, long)
    with pytest.raises(FileNotFoundError):
        il.load_oncopanel_ihc("/nonexistent/spec.csv", long)


def test_load_oncopanel_ihc_reads_csv(tmp_path):
    specimen, long = _tables([(1, "a", 100, "ne")])
    specimen.to_csv(tmp_path / "s.csv", index=False); long.to_csv(tmp_path / "l.csv", index=False)
    # Pre-existing CSVs from the original scripts carry no version stamp; they are read as is.
    out = il.load_oncopanel_ihc(tmp_path / "s.csv", tmp_path / "l.csv")
    assert out["ne_pos"].tolist() == [True] and out["DFCI_MRN"].tolist() == [1]


def test_anchor_dates_rejects_conflicts():
    local = pd.DataFrame({"DFCI_MRN": [1, 1, 2], "TREATMENT_ANCHOR_DATE": ["2020-01-01"] * 2 + ["2021-05-05"]})
    assert il.anchor_dates(local).to_dict() == {1: pd.Timestamp("2020-01-01"), 2: pd.Timestamp("2021-05-05")}
    with pytest.raises(ValueError):
        il.anchor_dates(pd.DataFrame({"DFCI_MRN": [1, 1],
                                      "TREATMENT_ANCHOR_DATE": ["2020-01-01", "2020-02-01"]}))


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #
def test_window_boundaries():
    f = _frame([(1, "a", 90, "stain"), (2, "b", 91, "stain"), (3, "c", 3650, "stain"),
                (4, "d", 3651, "stain")], [1, 2, 3, 4, 5])
    assert f.loc[90, "ihc_future"].to_dict() == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}


def test_flags_prior_ne_and_attrition():
    specs = [
        (1, "a", 30, "ne"),           # prior NE-positive
        (1, "b", 200, "nepc"),        # also a future NEPC-like one
        (2, "c", 200, "nepc"),
        (3, "d", 200, "ne"),
        (4, "e", 200, "unmatched"),   # never an event
        (5, "f", None, "ne"),         # stained but undated
        (6, "g", 200, "none"),        # matched, no stain
    ]
    f = _frame(specs, [1, 2, 3, 4, 5, 6]).loc[90]
    assert f["prior_ne_pos"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    assert f["ihc_future"].to_dict() == {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}
    assert f["ne_pos_future"].to_dict() == {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}
    assert f["nepc_like_future"].to_dict() == {1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0}
    assert f["unmatched_specimen_future"].to_dict() == {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0}
    assert f["undated_ihc_specimen"].to_dict() == {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0}

    sub = f.reset_index()
    assert set(il.outcome_population(sub, "ihc_any")["person_id"]) == {1, 2, 3, 4, 5, 6}
    assert set(il.outcome_population(sub, "ne_pos")["person_id"]) == {2, 3, 4, 5, 6}
    assert set(il.outcome_population(sub, "nepc_like")["person_id"]) == {2, 3, 4, 5, 6}
    with pytest.raises(ValueError):
        il.outcome_population(sub, "ne_pos_given_ihc")


def test_ar_low_flags_and_prior_exclusion():
    specs = [
        (1, "a", 30, "arlow"),        # prior AR-low -> out of the ar_low population, still in the NE ones
        (1, "b", 200, "arlow"),
        (2, "c", 200, "nepc"),        # NEPC-like is AR-low
        (3, "d", 200, "amphi"),       # AR axis positive: not AR-low
        (4, "e", 200, "arlow"),
        (5, "f", 30, "ne"),           # prior NE+ does not exclude from ar_low
        (5, "g", 200, "arlow"),
    ]
    f = _frame(specs, [1, 2, 3, 4, 5, 6]).loc[90]
    assert f["prior_ar_low"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    assert f["ar_low_future"].to_dict() == {1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0}
    assert f["ne_pos_future"].to_dict() == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}
    sub = f.reset_index()
    assert set(il.outcome_population(sub, "ar_low")["person_id"]) == {2, 3, 4, 5, 6}
    assert set(il.outcome_population(sub, "ne_pos")["person_id"]) == {1, 2, 3, 4, 6}


def test_landmark_changes_prior_vs_future():
    f = _frame([(1, "a", 100, "ne")], [1], landmarks=(0, 180))
    assert f.loc[(0, 1), "ne_pos_future"] == 1 and f.loc[(0, 1), "prior_ne_pos"] == 0
    assert f.loc[(180, 1), "ne_pos_future"] == 0 and f.loc[(180, 1), "prior_ne_pos"] == 1


def test_specimen_of_unscored_patient_is_ignored():
    f = _frame([(1, "a", 100, "ne"), (99, "z", 100, "ne")], [1])
    assert f.index.get_level_values("person_id").tolist() == [1]


@pytest.mark.parametrize("missing_patient", [True])
def test_missing_anchor_fails_even_without_specimens(missing_patient):
    specimen, long = _tables([(1, "a", 100, "ne")])
    spec = il.load_oncopanel_ihc(specimen, long)
    scores = pd.DataFrame({"person_id": [1, 2], "landmark_days": 0, "risk_score": [0.1, 0.2]})
    anchors = pd.Series({1: ANCHOR})
    with pytest.raises(ValueError, match="lack an ADT anchor"):
        il.build_ihc_frame(scores, spec, (0,), anchors=anchors)


# --------------------------------------------------------------------------- #
# LR driver
# --------------------------------------------------------------------------- #
def _synthetic(n=600, seed=3):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    ihc = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.0 * risk))))
    ne = ihc * rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.0 * risk))))
    nepc = ne * rng.binomial(1, 1 / (1 + np.exp(-(0.0 + 1.0 * risk))))
    prior = (rng.random(n) < 0.03).astype(int)
    ar_low = np.maximum(nepc, ihc * rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 1.0 * risk)))))
    prior_ar = (rng.random(n) < 0.05).astype(int)
    return pd.DataFrame({
        "person_id": np.arange(n), "landmark_days": 0, "risk_score": risk * 3 + 10,
        "ihc_future": ihc, "ne_pos_future": ne * (1 - prior), "nepc_like_future": nepc * (1 - prior),
        "prior_ne_pos": prior, "ar_low_future": ar_low * (1 - prior_ar), "prior_ar_low": prior_ar,
        "unmatched_specimen_future": 0, "undated_ihc_specimen": 0,
    })


def test_run_ihc_lr_recovers_signal_and_counts():
    frame = _synthetic()
    frame.loc[0, "risk_score"] = np.nan
    models, coefs, counts = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=20)
    assert list(models.columns) == il.MODEL_COLUMNS and list(counts.columns) == il.COUNT_COLUMNS
    assert models["outcome"].tolist() == list(il.OUTCOMES)
    assert (models["auc"] > 0.6).all() and (models["auc_lo"] <= models["auc"]).all()
    assert (models["auc_raw_score"] > 0.6).all()
    assert (coefs["term"] == "RISK_Z").all() and (coefs["odds_ratio"] > 1).all()
    c = counts.iloc[0]
    assert c["n_at_risk"] == len(frame) - 1
    assert c["n_ne_pos"] == c["n_nepc_like"] == c["n_at_risk"] - c["n_prior_ne_pos"]
    assert c["events_nepc_like"] <= c["events_ne_pos"] <= c["n_ihc_future"]
    row = models.set_index("outcome").loc["nepc_like"]
    assert row["n"] == c["n_nepc_like"] and row["events"] == c["events_nepc_like"]
    assert c["n_ar_low"] == c["n_at_risk"] - c["n_prior_ar_low"]
    assert c["events_ar_low"] == c["n_ar_low_future"]
    row = models.set_index("outcome").loc["ar_low"]
    assert row["n"] == c["n_ar_low"] and row["events"] == c["events_ar_low"]


def test_few_events_gives_count_only_row():
    frame = _synthetic(n=100)
    frame["nepc_like_future"] = 0
    frame.loc[:2, "nepc_like_future"] = 1
    frame["prior_ne_pos"] = 0
    frame["ar_low_future"] = 0
    models, coefs, _ = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0)
    for outcome in ("nepc_like", "ar_low"):
        row = models.set_index("outcome").loc[outcome]
        assert "fewer than 5" in row["note"] and pd.isna(row["auc"])
        assert outcome not in set(coefs["outcome"])


def test_cohort_restriction():
    frame = _synthetic()
    _, _, counts = il.run_ihc_lr(frame, {"sub": set(range(100))}, (0,), n_bootstrap=0)
    assert counts.iloc[0]["n_at_risk"] == 100 and counts.iloc[0]["n_cohort_list"] == 100


def test_run_ihc_lr_oof_predictions_per_outcome():
    frame = _synthetic()
    models, _, _, preds = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0, return_predictions=True)
    assert list(preds.columns) == il.PREDICTION_COLUMNS
    for outcome, col in il.OUTCOMES.items():
        p = preds[preds["outcome"].eq(outcome)]
        pop = il.outcome_population(frame, outcome)
        assert len(p) == len(pop) and p["event"].sum() == pop[col].sum()
        assert p["oof_prob"].between(0, 1).all()


def test_marker_frequency():
    specimen, _ = _tables([(1, "a", 100, "nepc"), (2, "b", 100, "stain"), (3, "c", 100, "unmatched")])
    specimen.loc[2, "p63"] = "positive"            # unmatched: ignored
    out = il.marker_frequency(specimen).set_index("marker")
    assert out.loc["Synaptophysin"].tolist() == [1, 1, 1]
    assert out.loc["p63"].tolist() == [1, 0, 0]
    assert out.loc["NKX3.1"].tolist() == [1, 0, 1]


def test_load_oncopanel_patients(tmp_path):
    p = tmp_path / ihc_paths.ONCOPANEL_SPECIMEN_FILE
    pd.DataFrame({"MRN_INT": [3, 3, 7, None], "UNIQUE_SAMPLE_ID": ["a", "b", "c", "d"],
                  "path_matched": [True, False, True, False]}).to_csv(p, index=False)
    assert il.load_oncopanel_patients(p) == {3, 7}          # matched or not; one row per specimen
    with pytest.raises(FileNotFoundError, match="step-2"):
        il.load_oncopanel_patients(tmp_path / "missing.csv")
    pd.DataFrame({"MRN_INT": []}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="No patients"):
        il.load_oncopanel_patients(p)
