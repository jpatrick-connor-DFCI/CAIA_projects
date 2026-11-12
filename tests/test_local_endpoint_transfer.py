"""Metastatic-vs-full platinum XGBoost transfer comparison (synthetic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from COMPASS.multivariable_models import local_endpoint_transfer as let


def _outcome(n=80, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(100, 1500, n)
    ev = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame({"id": np.arange(n), "duration": t, "event": ev})


def _risk_file(tmp_path, name, ids, risk, endpoint="platinum", extra_test=True):
    rows = [{"landmark_day": 0, "endpoint": endpoint, "DFCI_MRN": i, "dataset": "cv_oof",
             "risk_score": r} for i, r in zip(ids, risk)]
    if extra_test:  # held-out test rows must never be read
        rows += [{"landmark_day": 0, "endpoint": endpoint, "DFCI_MRN": 10_000 + i,
                  "dataset": "test", "risk_score": 1.0} for i in range(5)]
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_oof_scores_ignores_test_rows(tmp_path):
    p = _risk_file(tmp_path, "r.csv", range(10), np.arange(10.0))
    s = let.load_oof_scores(p, "platinum")
    assert len(s) == 10 and not any(i >= 10_000 for i in s.index)


def test_compare_cuts_arms_to_shared_patients_and_ranks_perfectly():
    out = _outcome()
    perfect = pd.Series(-out["duration"].to_numpy(), index=out["id"])  # earlier = riskier
    noise = pd.Series(np.random.default_rng(1).normal(size=len(out)), index=out["id"])
    a = perfect.iloc[:60]            # A covers fewer patients than B
    res = let.compare(out, {"A": a, "B": noise}, ref="B", n_bootstrap=20, seed=0)
    assert set(res["n"]) == {60} and set(res["events"]) == {int(out.iloc[:60]["event"].sum())}
    assert res.set_index("arm").loc["A", "c_index"] == 1.0
    assert res.set_index("arm").loc["A", "delta_c_vs_ref"] > 0


def test_paired_delta_of_score_against_itself_is_zero():
    out = _outcome()
    s = pd.Series(np.random.default_rng(2).normal(size=len(out)), index=out["id"])
    res = let.compare(out, {"A": s, "B": s.copy()}, ref="B", n_bootstrap=20, seed=0)
    r = res.set_index("arm").loc["A"]
    assert r["delta_c_vs_ref"] == 0 and r["delta_c_vs_ref_lo"] == 0 and r["delta_c_vs_ref_hi"] == 0


def test_delta_vs_baseline_computed_when_baseline_arm_present():
    out = _outcome()
    s = pd.Series(-out["duration"].to_numpy(), index=out["id"])
    base = pd.Series(np.random.default_rng(3).normal(size=len(out)), index=out["id"])
    res = let.compare(out, {"A": s, "A_baseline": base}, n_bootstrap=0).set_index("arm")
    assert res.loc["A", "delta_c_vs_baseline"] > 0
    assert np.isnan(res.loc["A_baseline", "delta_c_vs_baseline"])


@pytest.mark.parametrize("invalid", ["missing_file", "missing_dataset", "test_only", "wrong_endpoint", "nan", "inf"])
def test_load_oof_scores_rejects_missing_or_incomplete_coverage(tmp_path, invalid):
    p = _risk_file(tmp_path, "stale.csv", range(10), np.arange(10.0))
    df = pd.read_csv(p)
    if invalid == "missing_file":
        p.unlink()
    else:
        if invalid == "missing_dataset":
            df = df.drop(columns="dataset")
        elif invalid == "test_only":
            df = df.loc[df["dataset"].eq("test")]
        elif invalid == "wrong_endpoint":
            df["endpoint"] = "nepc"
        else:
            df.loc[0, "risk_score"] = np.nan if invalid == "nan" else np.inf
        df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="refit.*OUT_OF_FOLD_RISKS=True.*FORCE_RERUN=True"):
        let.load_oof_scores(p, "platinum")


def test_optional_federated_coverage_does_not_change_local_comparisons(tmp_path, monkeypatch):
    out = _outcome()
    monkeypatch.setattr(let, "endpoint_outcomes", lambda *args: out)
    rng = np.random.default_rng(8)
    met_paths, all_paths = {}, {}
    for config in ("both", "baseline"):
        met_paths[(0, config)] = _risk_file(
            tmp_path, f"met_{config}.csv", range(60), rng.normal(size=60))
        all_paths[(0, config)] = _risk_file(
            tmp_path, f"all_{config}.csv", range(80), rng.normal(size=80))
    # Even F's baseline has a different coverage set from its full model.
    fed_rows = [
        {"model_family": "xgboost_cox", "landmark_days": 0, "endpoint": "nepc",
         "config": config, let.fs.ID_COL: i, "risk_score": float(rng.normal())}
        for config, count in (("both", 40), ("baseline", 30)) for i in range(count)
    ]
    fed_path = tmp_path / "federated.csv"
    pd.DataFrame(fed_rows).to_csv(fed_path, index=False)
    kwargs = dict(landmarks=(0,), endpoints=("nepc",), met_paths=met_paths,
                  all_paths=all_paths, n_bootstrap=20, seed=0)
    without = let.evaluate_metastatic_transfer(pd.DataFrame(), {}, **kwargs)
    with_fed = let.evaluate_metastatic_transfer(
        pd.DataFrame(), {}, federated_scores_path=fed_path, **kwargs)
    pd.testing.assert_frame_equal(
        without.reset_index(drop=True),
        with_fed.loc[with_fed["comparison"].eq("local")].reset_index(drop=True))
    met = without.loc[without["population"].eq("metastatic")]
    full = without.loc[without["population"].eq("full")]
    assert set(met["n"]) == {60}
    assert set(full["n"]) == {80}
    assert set(full["arm"]) == {"C", "C_baseline"}
    federated = with_fed.loc[with_fed["comparison"].eq("federated")]
    assert set(federated["n"]) == {30}
    assert set(federated.loc[federated["population"].eq("full"), "arm"]) == {
        "C", "C_baseline", "F", "F_baseline"}


def test_evaluation_fails_instead_of_silently_omitting_primary_comparison(tmp_path, monkeypatch):
    out = _outcome()
    monkeypatch.setattr(let, "endpoint_outcomes", lambda *args: out)
    met = _risk_file(tmp_path, "met.csv", range(20), np.arange(20.0))
    full = _risk_file(tmp_path, "full.csv", range(20, 40), np.arange(20.0))
    with pytest.raises(ValueError, match="No shared local OOF patients for nepc at landmark 0"):
        let.evaluate_metastatic_transfer(
            pd.DataFrame(), {}, landmarks=(0,), endpoints=("nepc",),
            met_paths={(0, "both"): met}, all_paths={(0, "both"): full}, n_bootstrap=0)
