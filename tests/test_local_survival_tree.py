"""Local DFCI shallow survival tree: vendoring, items, cross-fitting hygiene, evaluation and bundle round trip."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _survival_tree_synthetic import HGB_FEATURE, MAPPING, synthetic_local  # noqa: E402

from COMPASS.multivariable_models.survival_tree import local_tree as lt  # noqa: E402
from survival_common import federated_endpoint_transfer as fet  # noqa: E402
from survival_common import federated_inference as fi  # noqa: E402

VENDOR = Path(lt.__file__).resolve().parent / "_vendor"
RHINO_APP = (Path(__file__).resolve().parents[2] / "caia-project-compass" / "rhino_scripts"
             / "feature_set_survival_tree")
CFG = lt.TreeConfig(landmarks=(0, 90), n_bootstrap=20)


@pytest.fixture(scope="module")
def run():
    return lt.run(synthetic_local(n=300, seed=1), MAPPING, CFG, verbose=False)


# --------------------------------------------------------------------- #
# Vendoring
# --------------------------------------------------------------------- #
def test_vendored_modules_match_manifest():
    manifest = json.loads((VENDOR / "VENDORED.json").read_text())
    assert set(manifest["files"]) == {p.name for p in VENDOR.glob("*.py")}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.skipif(not RHINO_APP.is_dir(), reason="caia-project-compass checkout not beside PROFILE-testing")
def test_vendored_modules_are_byte_identical_to_the_federated_app():
    for name in json.loads((VENDOR / "VENDORED.json").read_text())["files"]:
        assert (VENDOR / name).read_bytes() == (RHINO_APP / name).read_bytes(), (
            f"{name} differs from rhino_scripts; re-vendor it and update VENDORED.json")


# --------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------- #
def test_items_are_rounded_deciles_of_the_fit_rows_and_missing_is_false(run):
    from federated_checklist import guideline_threshold

    wide = run["frames"][("platinum", 0)]
    rows = wide.iloc[: len(wide) // 2]
    space = lt.item_space(rows, CFG)
    values = rows[HGB_FEATURE].dropna().to_numpy()
    expected = {guideline_threshold(np.quantile(values, q)) for q in CFG.quantiles}
    X_all = space.binarizer.transform(rows)
    got = {it["threshold"] for it in space.items if it["feature"] == HGB_FEATURE}
    dropped = {t for t in expected if t not in got}
    # a threshold is missing only when both of its items are constant on the fit rows
    for t in dropped:
        assert (values <= t).all() or (values >= t).all() or not (values >= t).any()
    assert got <= expected and got
    X = space.transform(rows)
    assert ((X.sum(0) > 0) & (X.sum(0) < len(rows))).all(), "constant item kept"
    # no identifier, outcome or missingness item; every item is age or <lab>__<stat>
    assert all(lt.ccore.feature_group(it["feature"], lt.AGE_COL) for it in space.items)
    assert all(it["op"] in ("<=", ">=") for it in space.items)
    psa = [i for i, it in enumerate(space.items) if it["feature"].startswith("Prostate") and it["op"] == ">="]
    missing = rows[space.items[psa[0]]["feature"]].isna().to_numpy()
    assert missing.any() and not X_all[missing][:, psa].any(), "a missing PSA made an item true"
    allowed = space.allowed("baseline")
    assert allowed.any() and all(space.items[i]["feature"] == lt.AGE_COL for i in np.flatnonzero(allowed))


def test_baseline_trees_split_on_age_only(run):
    for (_, config, _), ft in run["models"].items():
        feats = {ft.tree.items[n.split]["feature"] for n in ft.tree.nodes.values() if n.split is not None}
        if config == "baseline":
            assert feats <= {lt.AGE_COL}


# --------------------------------------------------------------------- #
# Cross-fitting hygiene
# --------------------------------------------------------------------- #
def test_fold_trees_are_fit_on_other_folds_platinum_rows_only(run):
    folds = run["folds"]
    plat = run["frames"][("platinum", 0)]
    for k in (0, 3):
        train = plat[plat[lt.ID_COL].map(folds) != k]
        refit, _ = lt.fit_selected(train, CFG, seed=CFG.seed)
        for config, ft in refit.items():
            assert ft.tree.signature() == run["models"][(0, config, k)].tree.signature()
    pts = run["patients"]
    assert (pts["fold"].astype(int).to_numpy() == pts[lt.ID_COL].map(folds).to_numpy()).all()


def test_transfer_labels_never_reach_the_trees():
    local = synthetic_local(n=300, seed=1)
    shuffled = local.copy()
    rng = np.random.default_rng(9)
    for col in ("NEPC", "AVPC"):
        per_pt = shuffled.drop_duplicates("DFCI_MRN")[["DFCI_MRN", col, f"{col}_DATE"]]
        perm = per_pt.iloc[rng.permutation(len(per_pt))][[col, f"{col}_DATE"]].to_numpy()
        per_pt = per_pt.assign(**{col: perm[:, 0], f"{col}_DATE": perm[:, 1]})
        shuffled = shuffled.drop(columns=[col, f"{col}_DATE"]).merge(per_pt, on="DFCI_MRN")
    cfg = lt.TreeConfig(landmarks=(0,), n_bootstrap=0)
    a = lt.cross_fit(lt.build_frames(local, MAPPING, cfg, verbose=False)[0], cfg, verbose=False)
    b = lt.cross_fit(lt.build_frames(shuffled, MAPPING, cfg, verbose=False)[0], cfg, verbose=False)
    pd.testing.assert_frame_equal(a["selection"], b["selection"])
    for key in a["models"]:
        assert a["models"][key].tree.signature() == b["models"][key].tree.signature()
    pa = a["patients"].query("endpoint == 'platinum'").reset_index(drop=True)
    pb = b["patients"].query("endpoint == 'platinum'").reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb)


def test_score_does_not_depend_on_the_endpoint(run):
    pts = run["patients"].query("landmark_days == 0 and config == 'both'")
    wide = pts.pivot(index=lt.ID_COL, columns="endpoint", values="risk_score").dropna()
    assert len(wide) > 100
    np.testing.assert_array_equal(wide["platinum"], wide["nepc"])
    np.testing.assert_array_equal(wide["platinum"], wide["avpc"])


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #
def test_metrics_equal_fet_primitives_on_out_of_fold_scores(run):
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    m = run["metrics"].set_index(["endpoint", "landmark_days", "config"])
    assert set(m.index.get_level_values(0)) == {"platinum", "nepc", "avpc"}
    for endpoint in ("platinum", "nepc"):
        z = run["patients"].query("endpoint == @endpoint and landmark_days == 0")
        both = z[z["config"] == "both"].set_index(lt.ID_COL)
        base = z[z["config"] == "baseline"].set_index(lt.ID_COL).loc[both.index]
        d = both[lt.DURATION_COL].to_numpy(float)
        e = both[lt.EVENT_COL].to_numpy(int)
        row = m.loc[(endpoint, 0, "both")]
        assert row["c_index"] == pytest.approx(fet.harrell_c(d, e, both["risk_score"].to_numpy()))
        assert row["delta_c_vs_baseline"] == pytest.approx(
            row["c_index"] - fet.harrell_c(d, e, base["risk_score"].to_numpy()))
        # every grid week: sksurv AUC of the fold tree's leaf-KM failure at that week
        grid = fet.auc_horizon_grid(d, e)
        units = fet.to_auc_units(d)
        y = Surv.from_arrays(event=e.astype(bool), time=units)
        auc = run["auc_t"].query("endpoint == @endpoint and landmark_days == 0 and config == 'both'")
        for h, got in zip(auc["horizon_time_unit"], auc["auc"]):
            fail = np.array([1.0 - lt.stc.step_at(run["models"][(0, "both", int(k))].curves[int(lf)], [h])[0]
                             for k, lf in zip(both["fold"], both["leaf"])])
            ref = cumulative_dynamic_auc(y, y, fail, np.array([float(h)]))[0][0]
            assert got == pytest.approx(ref, abs=1e-12)
        assert row["c_index_lo"] <= row["c_index"] <= row["c_index_hi"]
        assert row["delta_auc_lo"] <= row["delta_auc_vs_baseline"] <= row["delta_auc_hi"]
    # hemoglobin drives every endpoint in the generator
    for endpoint in ("platinum", "nepc", "avpc"):
        assert m.loc[(endpoint, 0, "both"), "delta_auc_vs_baseline"] > 0.05


def test_leaf_tables_are_sane(run):
    lv = run["leaves"]
    for h in CFG.horizons_days:
        assert (lv[f"S_{h}d_lower"] <= lv[f"S_{h}d"] + 1e-12).all()
        assert (lv[f"S_{h}d"] <= lv[f"S_{h}d_upper"] + 1e-12).all()
    assert (lv["rate"] > 0).all()
    sizes = lv.groupby(["landmark_days", "fold", "config"])["n"].sum()
    sel = run["selection"].set_index(["landmark_days", "fold", "config"])["n_fit"]
    pd.testing.assert_series_equal(sizes.sort_index(), sel.sort_index(), check_names=False, check_dtype=False)


def test_select_prefers_auc_then_fewer_leaves_then_larger_frac():
    base = {"cv_mean": 0.6, "n_valid_folds": 5, "n_valid_auc_t_folds": 5, "all_folds_valid": True,
            "integrated_brier_cv_mean": 0.2}
    rows = [{**base, "max_depth": 3, "min_leaf_frac": 0.02, "mean_auc_t_cv_mean": 0.70, "n_leaves_train_val": 6},
            {**base, "max_depth": 2, "min_leaf_frac": 0.05, "mean_auc_t_cv_mean": 0.70, "n_leaves_train_val": 3},
            {**base, "max_depth": 2, "min_leaf_frac": 0.10, "mean_auc_t_cv_mean": 0.70, "n_leaves_train_val": 3},
            {**base, "max_depth": 1, "min_leaf_frac": 0.10, "mean_auc_t_cv_mean": 0.69, "n_leaves_train_val": 2}]
    best = lt.select(rows, 5)
    assert (best["max_depth"], best["min_leaf_frac"]) == (2, 0.10)
    rows[0]["all_folds_valid"] = rows[1]["all_folds_valid"] = rows[2]["all_folds_valid"] = False
    assert lt.select(rows, 5)["max_depth"] == 1
    rows[3]["n_valid_auc_t_folds"] = 4
    assert lt.select(rows, 5)["selection_key"] == "cv_mean"


# --------------------------------------------------------------------- #
# Bundle round trip
# --------------------------------------------------------------------- #
def test_refit_bundle_scores_identically_through_federated_inference(run, tmp_path):
    path = tmp_path / "tree.json"
    lt.write_outputs(run, tmp_path)
    (tmp_path / lt.BUNDLE_FILE).rename(path)
    bundle = fi.load_bundle(path)
    assert {(m["landmark_days"], m["config"]) for m in bundle["models"]} == set(run["refits"])
    for model in bundle["models"]:
        ft = run["refits"][(model["landmark_days"], model["config"])]
        for endpoint in ("platinum", "avpc"):
            wide = run["frames"][(endpoint, model["landmark_days"])]
            leaf, rate = ft.predict(wide)
            pred = fi.predict_tree(wide, model, horizons_days=(365,))
            np.testing.assert_array_equal(pred["leaf"].to_numpy(), leaf)
            np.testing.assert_allclose(pred["rate"].to_numpy(), rate, rtol=0, atol=0)
            km = {lf: lt.stc.step_at(ft.curves[lf], [365 / 7])[0] for lf in ft.curves}
            np.testing.assert_allclose(pred["survival_365d"], [km[int(x)] for x in leaf], atol=0)
    # the transfer wiring scores the local bundle like any federated tree
    res = fet.evaluate_transfer(synthetic_local(n=300, seed=1), {"local": bundle}, endpoints=("nepc",),
                                lab_mapping=MAPPING, landmarks=(0,), n_bootstrap=0, verbose=False)
    row = res["metrics"].set_index("config").loc["both"]
    wide = run["frames"][("nepc", 0)]
    _, rate = run["refits"][(0, "both")].predict(wide)
    assert row["c_index"] == pytest.approx(fet.harrell_c(wide[lt.DURATION_COL], wide[lt.EVENT_COL], rate))
    assert np.isfinite(row["mean_auc"]) and np.isfinite(row["ibs"])
