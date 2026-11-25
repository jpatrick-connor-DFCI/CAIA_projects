import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.multivariable_models import ihc_lr as il  # noqa: E402

from COMPASS.prostate_ihc import paths as ihc_paths  # noqa: E402
from tests.test_prostate_ihc_annotations import ANCHOR, tables  # noqa: E402


def _specimens(specs):
    """specs: (mrn, part, day or None, kind[, relevant]); kinds as in test_prostate_ihc_annotations.KINDS."""
    return il.load_ihc_specimens(*tables(specs))


def _frame(specs, mrns, landmarks=(90,)):
    spec = _specimens(specs)
    anchors = pd.Series(ANCHOR, index=pd.Index(mrns, name="DFCI_MRN"))
    scores = pd.DataFrame([{"person_id": m, "landmark_days": lm, "risk_score": float(m)}
                           for lm in landmarks for m in mrns])
    return il.build_ihc_frame(scores, spec, landmarks, anchors=anchors).set_index(["landmark_days", "person_id"])


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def test_load_ihc_specimens_flags():
    out = _specimens([(1, "A", 100, "nepc"), (2, "A", 200, "arlow"), (3, "A", 300, "ki67")]).set_index("DFCI_MRN")
    assert out[["ihc_any", "ne_pos", "nepc_like", "ar_low"]].astype(int).values.tolist() == [
        [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]]
    with pytest.raises(ValueError, match="both"):
        il.load_ihc_specimens(tables([(1, "A", 100, "ne")])[0])


def test_load_ihc_specimens_from_intermediate_dir(tmp_path):
    pytest.importorskip("pyarrow")
    long, spec_all = tables([(1, "A", 100, "nepc"), (2, "B", 200, "arlow", False)])
    pd.DataFrame({"MRN_INT": [1, 2]}).to_parquet(tmp_path / ihc_paths.COHORT_FILE)
    long.to_parquet(tmp_path / ihc_paths.LONG_FILE)
    spec_all.to_parquet(tmp_path / ihc_paths.SPECIMEN_ALL_FILE)
    from_disk = il.load_ihc_specimens(intermediate_dir=tmp_path)
    from_frames = il.load_ihc_specimens(long, spec_all)
    pd.testing.assert_frame_equal(from_disk, from_frames)
    assert il.LOCAL_ID_COL in from_disk.columns and "MRN_INT" not in from_disk.columns
    assert from_disk[il.SPECIMEN_ID_COL].tolist() == ["1|BS20-1|A", "2|BS20-2|B"]


@pytest.mark.parametrize("kind", ["disc_ne", "disc_ar"])
def test_rerun_style_dir_gives_same_specimens(tmp_path, kind):
    """A dir whose spec_all counts discordant as both (a rerun of the extraction) loads to the same flags."""
    pytest.importorskip("pyarrow")
    long, spec_all = tables([(1, "A", 100, kind)], "both")
    pd.DataFrame({"MRN_INT": [1]}).to_parquet(tmp_path / ihc_paths.COHORT_FILE)
    long.to_parquet(tmp_path / ihc_paths.LONG_FILE)
    spec_all.to_parquet(tmp_path / ihc_paths.SPECIMEN_ALL_FILE)
    pd.testing.assert_frame_equal(il.load_ihc_specimens(intermediate_dir=tmp_path),
                                  _specimens([(1, "A", 100, kind)]))


def test_audit_columns_present():
    spec = il.add_days_from_adt(_specimens([(1, "A", 100, "nepc"), (2, "A", None, "ne")]),
                                pd.Series(ANCHOR, index=pd.Index([1, 2], name="DFCI_MRN")))
    assert all(c in spec.columns for c in il.AUDIT_COLUMNS)
    days = spec.set_index(il.LOCAL_ID_COL)["days_from_adt"]
    assert days[1] == 100 and pd.isna(days[2])


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
    f = _frame([(1, "A", 90, "stain"), (2, "A", 91, "stain"), (3, "A", 3650, "stain"),
                (4, "A", 3651, "stain")], [1, 2, 3, 4, 5])
    assert f.loc[90, "ihc_future"].to_dict() == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}


def test_flags_prior_rules_and_attrition():
    specs = [
        (1, "A", 30, "ne"),             # prior NE-positive
        (1, "B", 200, "nepc"),          # also a future NEPC-like one
        (2, "A", 200, "nepc"),
        (3, "A", 200, "ne"),
        (4, "A", 200, "nepc", False),   # not prostate-relevant: never an event
        (5, "A", None, "ne"),           # stained but undated
        (6, "A", 30, "arlow"),          # prior AR-low
        (6, "B", 200, "arlow"),
        (7, "A", 200, "arlow"),
    ]
    f = _frame(specs, [1, 2, 3, 4, 5, 6, 7]).loc[90]
    col = lambda c: {k: int(v) for k, v in f[c].to_dict().items()}  # noqa: E731
    assert col("prior_ne_pos") == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    assert col("prior_ar_low") == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0}
    assert col("ihc_future") == {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1}
    assert col("ne_pos_future") == {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}
    assert col("nepc_like_future") == {1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    assert col("ar_low_future") == {1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1}
    assert col("undated_ihc_specimen") == {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}

    sub = f.reset_index()
    pop = lambda o: set(il.outcome_population(sub, o)["person_id"])  # noqa: E731
    assert pop("ihc_any") == {1, 2, 3, 4, 5, 6, 7}
    assert pop("ne_pos") == pop("nepc_like") == {2, 3, 4, 5, 6, 7}
    assert pop("ar_low") == {1, 2, 3, 4, 5, 7}
    with pytest.raises(ValueError):
        il.outcome_population(sub, "ne_pos_given_ihc")


def test_ar_low_and_ne_priors_are_independent():
    specs = [
        (1, "A", 30, "amphi"),          # prior NE+ but AR-axis positive: not AR-low
        (1, "B", 200, "arlow"),
        (2, "A", 30, "arlow"),          # prior AR-low, not NE+
        (2, "B", 200, "ne"),
        (3, "A", 30, "nepc"),           # prior NEPC-like is both
        (3, "B", 200, "nepc"),
    ]
    f = _frame(specs, [1, 2, 3]).loc[90].reset_index()
    pop = lambda o: set(il.outcome_population(f, o)["person_id"])  # noqa: E731
    assert pop("ne_pos") == pop("nepc_like") == {2}
    assert pop("ar_low") == {1}
    ev = f.set_index("person_id")
    assert ev.loc[1, "ar_low_future"] == 1 and ev.loc[1, "ne_pos_future"] == 0
    assert ev.loc[2, "ne_pos_future"] == 1 and ev.loc[2, "ar_low_future"] == 0


@pytest.mark.parametrize("kind, expected", [
    # (ihc, ne_pos, nepc_like, ar_low) futures
    ("nepc", (1, 1, 1, 1)),
    ("amphi", (1, 1, 0, 0)),
    ("arlow", (1, 0, 0, 1)),
    ("ar_mixed", (1, 0, 0, 0)),
    ("disc_ne", (1, 1, 1, 1)),      # discordant NE marker is NE+ (NKX3.1 negative -> NEPC-like)
    ("disc_ar", (1, 0, 0, 0)),      # discordant AR marker is neither positive nor negative
    ("ki67", (1, 0, 0, 0)),
    ("lost", (1, 0, 0, 0)),
])
def test_specimen_kind_to_future_flags(kind, expected):
    f = _frame([(1, "A", 200, kind)], [1]).loc[(90, 1)]
    assert tuple(int(f[c]) for c in il.SPECIMEN_FLAGS) == expected


def test_discordant_ne_prior_excludes_from_ne_outcomes():
    specs = [(1, "A", 30, "disc_ne"), (1, "B", 200, "nepc"), (2, "A", 200, "nepc")]
    f = _frame(specs, [1, 2]).loc[90].reset_index()
    pop = lambda o: set(il.outcome_population(f, o)["person_id"])  # noqa: E731
    assert pop("ne_pos") == pop("nepc_like") == {2}         # discordant Syn is a prior NE+
    assert pop("ar_low") == {2}                              # and the NKX3.1-negative part is prior AR-low


def test_nonrelevant_specimens_never_count():
    specs = [
        (1, "A", 30, "nepc", False),    # non-relevant prior: no exclusion
        (1, "B", 200, "nepc"),
        (2, "A", None, "ne", False),    # non-relevant undated: not counted
        (3, "A", 200, "arlow", False),
    ]
    f = _frame(specs, [1, 2, 3]).loc[90]
    assert f["prior_ne_pos"].sum() == f["prior_ar_low"].sum() == 0
    assert f["undated_ihc_specimen"].sum() == 0
    assert f.loc[1, list(il.SPECIMEN_FLAGS)].tolist() == [1, 1, 1, 1]
    assert f.loc[3, list(il.SPECIMEN_FLAGS)].sum() == 0


@pytest.mark.parametrize("day, prior, future", [(89, 1, 0), (90, 1, 0), (91, 0, 1), (3650, 0, 1), (3651, 0, 0)])
def test_ar_low_window_boundaries(day, prior, future):
    f = _frame([(1, "A", day, "arlow")], [1]).loc[(90, 1)]
    assert (f["prior_ar_low"], f["ar_low_future"]) == (prior, future)


def test_landmark_changes_prior_vs_future():
    f = _frame([(1, "A", 100, "ne")], [1], landmarks=(0, 180))
    assert f.loc[(0, 1), "ne_pos_future"] == 1 and f.loc[(0, 1), "prior_ne_pos"] == 0
    assert f.loc[(180, 1), "ne_pos_future"] == 0 and f.loc[(180, 1), "prior_ne_pos"] == 1


def test_specimen_of_unscored_patient_is_ignored():
    f = _frame([(1, "A", 100, "ne"), (99, "A", 100, "ne")], [1])
    assert f.index.get_level_values("person_id").tolist() == [1]


def test_missing_anchor_fails_even_without_specimens():
    spec = _specimens([(1, "A", 100, "ne")])
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
    arlow = ihc * rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.0 * risk))))
    prior = (rng.random(n) < 0.03).astype(int)
    prior_ar = (rng.random(n) < 0.03).astype(int)
    return pd.DataFrame({
        "person_id": np.arange(n), "landmark_days": 0, "risk_score": risk * 3 + 10,
        "ihc_future": ihc, "ne_pos_future": ne * (1 - prior), "nepc_like_future": nepc * (1 - prior),
        "ar_low_future": arlow * (1 - prior_ar), "prior_ne_pos": prior, "prior_ar_low": prior_ar,
        "undated_ihc_specimen": 0,
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
    assert c["n_ar_low"] == c["n_at_risk"] - c["n_prior_ar_low"]
    for outcome in ("nepc_like", "ar_low"):
        row = models.set_index("outcome").loc[outcome]
        assert row["n"] == c[f"n_{outcome}"] and row["events"] == c[f"events_{outcome}"]


def test_few_events_gives_count_only_row():
    frame = _synthetic(n=100)
    frame["nepc_like_future"] = 0
    frame.loc[:2, "nepc_like_future"] = 1
    frame["prior_ne_pos"] = 0
    models, coefs, _ = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0)
    nepc = models.set_index("outcome").loc["nepc_like"]
    assert "fewer than 5" in nepc["note"] and pd.isna(nepc["auc"])
    assert "nepc_like" not in set(coefs["outcome"])


def test_few_ar_low_events_gives_count_only_row():
    frame = _synthetic(n=100)
    frame["ar_low_future"] = 0
    frame.loc[:3, "ar_low_future"] = 1
    models, coefs, counts = il.run_ihc_lr(frame, {"all": None}, (0,), n_bootstrap=0)
    row = models.set_index("outcome").loc["ar_low"]
    assert "fewer than 5" in row["note"] and pd.isna(row["auc_raw_score"])
    assert "ar_low" not in set(coefs["outcome"])
    assert counts.iloc[0]["events_ar_low"] == int(il.outcome_population(frame, "ar_low")["ar_low_future"].sum())


def test_end_to_end_specimens_to_lr():
    rng = np.random.default_rng(7)
    n = 400
    risk = rng.normal(size=n)
    specs = []
    for m in range(n):
        if rng.random() < 1 / (1 + np.exp(-(-0.3 + 1.5 * risk[m]))):
            kind = rng.choice(["nepc", "arlow", "ne"]) if risk[m] > 0 else rng.choice(["stain", "amphi", "ki67"])
            specs.append((m, "A", int(rng.integers(91, 3650)), str(kind)))
    spec = _specimens(specs)
    anchors = pd.Series(ANCHOR, index=pd.Index(range(n), name="DFCI_MRN"))
    scores = pd.DataFrame({"person_id": range(n), "landmark_days": 90, "risk_score": risk})
    frame = il.build_ihc_frame(scores, spec, (90,), anchors=anchors)
    models, _, counts = il.run_ihc_lr(frame, {"all": None}, (90,), n_bootstrap=0)
    m = models.set_index("outcome")
    assert list(m.index) == list(il.OUTCOMES)
    assert (m["auc_raw_score"] > 0.6).all()
    c = counts.iloc[0]
    kinds = pd.Series([k for *_, k in specs])
    assert c["n_ihc_future"] == len(specs)
    assert c["n_ar_low_future"] == int(kinds.isin(["nepc", "arlow"]).sum())
    assert c["n_ne_pos_future"] == int(kinds.isin(["nepc", "ne", "amphi"]).sum())
    assert c["n_nepc_like_future"] == int(kinds.eq("nepc").sum())


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


def test_marker_frequency_relevant_only():
    long, spec_all = tables([(1, "A", 100, "nepc"), (2, "A", 100, "stain"), (3, "A", 100, "stain", False)])
    spec = il.load_ihc_specimens(long, spec_all)
    out = il.marker_frequency(il.ann.marker_calls(long), spec).set_index("marker")
    assert out.loc["p63", "specimens_stained"] == 1
    assert out.loc["NKX3.1"].tolist() == [1, 0, 1, 0, 1]
