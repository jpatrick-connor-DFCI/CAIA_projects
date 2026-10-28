"""Transporting federated platinum bundles onto local NEPC / AVPC endpoints."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from survival_common import federated_endpoint_transfer as fet
from survival_common import federated_scoring as fs

ANCHOR = pd.Timestamp("2018-06-01")
HGB_LOCAL = "Hemoglobin"
HGB_OMOP = "Hemoglobin [Mass/volume] in Blood"
HGB_FEATURE = "Hemoglobin__Mass_volume__in_Blood__mean"
MAPPING = {HGB_LOCAL: HGB_OMOP, "Testosterone": "Testosterone [Mass/volume] in Serum or Plasma"}


def _local(n=60, seed=0) -> pd.DataFrame:
    """Stage-2 schema. Low hemoglobin -> earlier NEPC, so a model that scores
    low Hgb as high risk should be concordant with NEPC."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        hgb = float(rng.normal(12, 2))
        nepc = pid % 2 == 0
        nepc_day = int(np.clip(2000 - 120 * (15 - hgb) ** 2, 200, 3000))
        base = {
            "DFCI_MRN": 1000 + pid,
            "AGE_AT_TREATMENTSTART": 65.0,
            "DIAGNOSIS_DATE": ANCHOR - pd.Timedelta(days=100),
            "TREATMENT_ANCHOR_DATE": ANCHOR,
            "LAST_CONTACT_DATE": ANCHOR + pd.Timedelta(days=3200),
            "PLATINUM": 0,
            # back-filled with last contact for non-events, as upstream does
            "PLATINUM_DATE": ANCHOR + pd.Timedelta(days=3200),
            "NEPC": int(nepc),
            "NEPC_DATE": ANCHOR + pd.Timedelta(days=nepc_day) if nepc else pd.NaT,
            "AVPC": 0,
            "AVPC_DATE": pd.NaT,
        }
        for day in (-30, -10):
            rows.append({**base, "LAB_DATE": ANCHOR + pd.Timedelta(days=day),
                         "LAB_NAME": HGB_LOCAL, "LAB_VALUE": hgb})
        rows.append({**base, "LAB_DATE": ANCHOR - pd.Timedelta(days=5),
                     "LAB_NAME": "Unmapped thing", "LAB_VALUE": 1.0})
    return pd.DataFrame(rows)


def _bundle(center=12.0) -> dict:
    def model(config, cols, beta):
        return {
            "landmark_days": 0, "config": config,
            "covariate_cols": cols, "coefficients": beta,
            "preprocessing": {
                "covariate_cols": cols, "missing_indicator_cols": [],
                "impute_means": {c: (center if c != "age" else 65.0) for c in cols},
                "centers": {c: (center if c != "age" else 65.0) for c in cols},
                "scales": {c: (2.0 if c != "age" else 8.0) for c in cols},
                "source_cols": {"age": fs.AGE_COL},
            },
        }
    return {
        "format": "caia-federated-model-bundle", "format_version": 1,
        "model_family": "elastic_net_cox", "analysis_label": "adt",
        "endpoint": "platinum",
        "models": [
            model("both", [HGB_FEATURE, "age"], [-1.0, 0.0]),
            model("baseline", ["age"], [0.1]),
        ],
    }


def test_endpoint_date_fills_the_event_slot_and_backfill_is_masked():
    local = _local()
    nepc, _ = fet.to_federated_long(local, "nepc", MAPPING)
    plat, stats = fet.to_federated_long(local, "platinum", MAPPING)

    first = nepc.drop_duplicates(fs.ID_COL).set_index(fs.ID_COL)
    assert first.loc[1000, fs.PLATINUM_DATE_COL] == local.loc[0, "NEPC_DATE"]
    assert pd.isna(first.loc[1001, fs.PLATINUM_DATE_COL])
    # PLATINUM==0 everywhere: the back-filled date must not become an event
    assert plat[fs.PLATINUM_DATE_COL].isna().all()
    assert stats["n_patients_event"] == 0


def test_lab_names_are_mapped_and_unmapped_rows_blanked():
    long_df, stats = fet.to_federated_long(_local(), "nepc", MAPPING)
    assert HGB_OMOP in set(long_df[fs.LAB_NAME_COL].dropna())
    assert stats["n_rows_unmapped_lab"] == 60
    assert long_df.loc[long_df[fs.LAB_NAME_COL].isna(), fs.LAB_VALUE_COL].isna().all()
    wide, _ = fs.build_landmark_frame(long_df, 0)
    assert HGB_FEATURE in wide.columns
    assert len(wide) == 60


def test_prevalent_events_are_dropped_at_the_landmark():
    local = _local()
    local.loc[local["DFCI_MRN"] == 1000, "NEPC_DATE"] = ANCHOR + pd.Timedelta(days=50)
    long_df, _ = fet.to_federated_long(local, "nepc", MAPPING)
    wide0, _ = fs.build_landmark_frame(long_df, 0)
    wide90, _ = fs.build_landmark_frame(long_df, 90)
    assert 1000 in set(wide0[fs.ID_COL])
    assert 1000 not in set(wide90[fs.ID_COL])


def test_transported_score_is_concordant_and_beats_age_only():
    res = fet.evaluate_transfer(
        _local(), {"enet": _bundle()}, endpoints=("nepc", "avpc"),
        lab_mapping=MAPPING, n_bootstrap=50, verbose=False,
        horizons_days=(1000, 2000),
    )
    m = res["metrics"].set_index(["endpoint", "config"])
    both = m.loc[("nepc", "both")]
    assert both["n_events"] == 30
    # censored patients carry random Hgb, so C is well above 0.5 but not near 1
    assert both["c_index"] > 0.65
    assert both["c_index_lo"] <= both["c_index"] <= both["c_index_hi"]
    # age is constant, so the age-only baseline sits at 0.5
    assert both["c_index_baseline"] == pytest.approx(0.5)
    assert both["delta_c_vs_baseline"] > 0.15
    assert both["mean_auc_lo"] <= both["mean_auc"] <= both["mean_auc_hi"]
    assert both["delta_auc_vs_baseline"] == pytest.approx(
        both["mean_auc"] - both["mean_auc_baseline"])
    assert both["delta_auc_vs_baseline"] > 0.1
    assert both["delta_auc_lo"] <= both["delta_auc_vs_baseline"] <= both["delta_auc_hi"]
    assert both["hr_per_sd"] > 1
    # AVPC has no events: metrics are NaN, not an exception
    assert np.isnan(m.loc[("avpc", "both"), "c_index"])

    scores = res["scores"]
    per_ep = scores[scores["config"] == "both"].pivot(
        index=fs.ID_COL, columns="endpoint", values="risk_score"
    )
    # features do not depend on the endpoint, so neither does the score
    np.testing.assert_allclose(per_ep["nepc"], per_ep["avpc"])


def test_baseline_fallback_pairs_a_bundle_without_an_age_model_to_a_donor():
    nobase = _bundle()
    nobase["models"] = [m for m in nobase["models"] if m["config"] == "both"]
    kw = dict(endpoints=("nepc",), lab_mapping=MAPPING, n_bootstrap=30,
              verbose=False, horizons_days=(1000, 2000))
    bundles = {"enet": _bundle(), "nobase": nobase}

    m = fet.evaluate_transfer(_local(), bundles, **kw)["metrics"].set_index(["bundle", "config"])
    assert np.isnan(m.loc[("nobase", "both"), "delta_auc_vs_baseline"])

    m = fet.evaluate_transfer(_local(), bundles, baseline_fallback={"nobase": "enet"},
                              **kw)["metrics"].set_index(["bundle", "config"])
    donor, borrowed = m.loc[("enet", "both")], m.loc[("nobase", "both")]
    assert donor["baseline_bundle"] == "enet"
    assert borrowed["baseline_bundle"] == "enet"
    # same full model, same patients, same resamples -> identical paired deltas
    for col in ("c_index_baseline", "delta_c_vs_baseline", "delta_c_lo",
                "mean_auc_baseline", "delta_auc_vs_baseline", "delta_auc_hi"):
        assert borrowed[col] == pytest.approx(donor[col])

    with pytest.raises(ValueError, match="unknown bundles"):
        fet.evaluate_transfer(_local(), bundles, baseline_fallback={"nobase": "xgb"}, **kw)


def test_unit_check_flags_a_tenfold_mismatch():
    long_df, _ = fet.to_federated_long(_local(), "nepc", MAPPING)
    wide, _ = fs.build_landmark_frame(long_df, 0)
    ok = fet.unit_check(wide, _bundle(center=12.0)["models"][0])
    bad = fet.unit_check(wide, _bundle(center=120.0)["models"][0])
    assert not ok["unit_flag"].any()
    assert bad.set_index("covariate").loc[HGB_FEATURE, "unit_flag"]


def test_cli_writes_all_outputs(tmp_path):
    data = tmp_path / "local.csv"
    _local().to_csv(data, index=False)
    bundle = tmp_path / "enet_adt.json"
    bundle.write_text(json.dumps(_bundle()))
    mapping = tmp_path / "map.csv"
    pd.DataFrame({
        "collapsed_measurement": list(MAPPING),
        "omop_measurement_name": list(MAPPING.values()),
        "mapping_status": "mapped",
    }).to_csv(mapping, index=False)

    out = tmp_path / "out"
    rc = fet.main([
        "--bundle", str(bundle), "--data", str(data), "--output-dir", str(out),
        "--lab-mapping", str(mapping), "--n-bootstrap", "20",
    ])
    assert rc == 0
    for name in ("scores", "metrics", "coverage", "unit_check"):
        assert (out / f"transfer_{name}.csv").exists()
    run = json.loads((out / "transfer_run.json").read_text())
    assert run["endpoints"] == ["nepc", "avpc", "platinum"]


def test_missing_endpoint_columns_fail_loudly(tmp_path):
    data = tmp_path / "local.csv"
    _local().drop(columns=["AVPC", "AVPC_DATE"]).to_csv(data, index=False)
    with pytest.raises(ValueError, match="AVPC"):
        fet.read_local_frame(data, ["avpc"])


def test_checklist_transfer_selects_frozen_model_at_each_horizon(tmp_path):
    from COMPASS.survival_analysis.predictive_checklists.evaluation import ProbabilityCalibrator
    from COMPASS.survival_analysis.predictive_checklists.features import FeatureBinarizer
    from COMPASS.survival_analysis.predictive_checklists.optimizer import ChecklistModel
    from COMPASS.survival_analysis.predictive_checklists.inference import predict_bundle
    from survival_common.federated_inference import predict_checklist, select_model

    local = _local(n=24)
    long_df, _ = fet.to_federated_long(local, "nepc", MAPPING)
    wide, _ = fs.build_landmark_frame(long_df, 0)
    binarizer = FeatureBinarizer(age_col=fs.AGE_COL, id_col=fs.ID_COL).fit(wide)
    candidates = [i for i, item in enumerate(binarizer.items)
                  if item["feature"] == HGB_FEATURE]
    assert len(candidates) >= 2
    models = []
    for horizon, index in zip((1000, 2000), candidates[:2]):
        payload = {
            "schema_version": 1, "config": {"id_col": fs.ID_COL},
            "binarizer": binarizer.to_dict(),
            "model": ChecklistModel([index], 1, len(binarizer.items), {}).to_dict(),
            "calibrator": ProbabilityCalibrator(
                "logistic", {"intercept": -2.0, "slope": 3.0}).to_dict(),
        }
        models.append({"landmark_days": 0, "horizon_days": horizon,
                       "config": "both", "checklist": payload,
                       "preprocessing": {"covariate_cols": binarizer.feature_columns}})
        path = tmp_path / f"checklist_{horizon}.json"
        path.write_text(json.dumps(payload))
        ref = predict_bundle(path, wide)
        pred = predict_checklist(wide, models[-1])
        np.testing.assert_array_equal(pred["score"], ref["score"])
        np.testing.assert_allclose(pred["probability"], ref["probability"])

    bundle = {"format": "caia-federated-model-bundle", "format_version": 1,
              "model_family": "predictive_checklist", "analysis_label": "adt",
              "models": models}
    with pytest.raises(ValueError, match="ambiguous"):
        select_model(bundle, 0)
    assert select_model(bundle, 0, horizon_days=2000) is models[1]
    res = fet.evaluate_transfer(local, {"checklist": bundle}, endpoints=("nepc",),
                                lab_mapping=MAPPING, horizons_days=(1000, 2000),
                                n_bootstrap=0, verbose=False)
    scores = res["scores"]
    assert set(scores["model_horizon_days"]) == {1000, 2000}
    assert set(scores["evaluation_horizon_days"]) == {1000, 2000}
    assert "brier" in res["metrics"]
    for horizon in (1000, 2000):
        rows = scores[scores["model_horizon_days"] == horizon]
        assert len(rows) == len(wide)
        expected = predict_checklist(wide, select_model(bundle, 0, horizon_days=horizon))
        np.testing.assert_array_equal(rows["risk_score"], expected["score"])
        np.testing.assert_allclose(rows["probability"], expected["probability"])
        np.testing.assert_array_equal(rows["flag"], expected["flag"])


def test_v3_deephit_transfer_keeps_arms_and_pairs_landmark_age_baseline():
    def weights(input_dim, bias):
        hidden, window = 2, 300
        shapes = {
            "gru.weight_ih_l0": (6, input_dim), "gru.weight_hh_l0": (6, 2),
            "gru.bias_ih_l0": (6,), "gru.bias_hh_l0": (6,),
            "head.1.weight": (2, 2), "head.1.bias": (2,),
            "head.4.weight": (window + 1, 2), "head.4.bias": (window + 1,),
        }
        state = {}
        for name, shape in shapes.items():
            arr = np.zeros(shape, dtype=np.float32)
            if name == "head.4.bias":
                arr[0] = bias
            state[name] = {"shape": list(shape), "values": arr.ravel().tolist()}
        return state

    def entry(config, arm, bias):
        is_age = config == "baseline"
        return {"landmark_days": 0, "config": config, "training_arm": arm,
                "architecture": {"input_dim": 1 if is_age else 4,
                                 "hidden_dim": 2, "dropout": 0.0,
                                 "n_events": 1, "max_pred_window": 300},
                "preprocessing": ({"input_kind": "age", "covariate_cols": ["age"],
                                   "centers": {"age": 65.0}, "scales": {"age": 10.0},
                                   "source_cols": {"age": fs.AGE_COL}, "time_unit_days": 7}
                                  if is_age else
                                  {"input_kind": "labs", "lab_names": [HGB_OMOP],
                                   "clip_lo": [None], "clip_hi": [None],
                                   "mean": [12.0], "std": [2.0],
                                   "max_observed_time": 8, "time_unit_days": 7,
                                   "include_age": True, "age_mean": 65.0,
                                   "age_std": 10.0, "age_source_col": fs.AGE_COL}),
                "state_dict": weights(1 if is_age else 4, bias)}

    bundle = {"format": "caia-federated-model-bundle", "format_version": 3,
              "model_family": "dynamic_deephit", "analysis_label": "adt",
              "models": [entry("both", "landmark", 0.5),
                         entry("both", "dynamic", 1.0),
                         entry("baseline", "landmark", 0.0)]}
    res = fet.evaluate_transfer(_local(n=8), {"dh": bundle}, endpoints=("nepc",),
                                lab_mapping=MAPPING, horizons_days=(1000,),
                                n_bootstrap=0, verbose=False)
    scores = res["scores"]
    # one row per model, as in the federated runs: CIF_total, no evaluation horizon
    assert len(scores) == 3 * 8
    assert set(scores["training_arm"]) == {"landmark", "dynamic"}
    assert scores["evaluation_horizon_days"].isna().all()
    assert scores["risk_score"].notna().all()
    m = res["metrics"]
    assert len(m) == 3
    assert {"ibs", "n_auc_horizons_requested", "n_auc_horizons_valid"} <= set(m)
    assert m[m["training_arm"] == "dynamic"]["c_index_baseline"].notna().all()


def _survival(n=300, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.exponential(700 * np.exp(-0.7 * x))
    c = rng.uniform(100, 3000, n)
    return np.minimum(t, c).clip(1), (t <= c).astype(int), x


def test_grid_is_the_federated_week_grid():
    from survival_common.helper import compute_horizon_grid

    duration, event, _ = _survival()
    grid = fet.auc_horizon_grid(duration, event)
    expected = compute_horizon_grid(
        pd.DataFrame({"d": duration, "e": event}), duration_col="d", event_col="e",
        admin_censor_days=fs.DEFAULT_MAX_FOLLOWUP_DAYS)
    np.testing.assert_array_equal(grid, expected)
    assert 2 <= len(grid) <= 25 and grid.max() < 522
    weeks = np.ceil(duration[event == 1] / 7)
    assert grid.min() >= np.floor(np.quantile(weeks, 0.25))
    assert grid.max() <= np.ceil(np.quantile(weeks, 0.75))
    assert fet.auc_horizon_grid(duration, np.zeros_like(event)).size == 0


def test_valid_horizons_apply_the_support_filter_and_coverage_gates():
    grid = np.arange(10, 20, dtype=float)
    # weeks 1..15 of follow-up: 10..14 are strictly inside, 5/10 = coverage 0.5
    np.testing.assert_array_equal(
        fet.valid_horizons(np.arange(1, 16) * 7, grid), np.arange(10, 15))
    # 4/10 < 0.5 -> nothing is scored
    assert fet.valid_horizons(np.arange(1, 15) * 7, grid).size == 0
    # one horizon < MIN_IPCW_VALID_HORIZONS
    assert fet.valid_horizons(np.arange(1, 12) * 7, np.array([10.0])).size == 0

    duration, event, x = _survival()
    narrow = np.full(len(duration), 100.0)
    assert np.isnan(fet.mean_auc(narrow, event, x, fet.auc_horizon_grid(duration, event)))
    summary, auc_df = fet.ipcw_metrics(narrow, event, x, fet.auc_horizon_grid(duration, event))
    assert np.isnan(summary["mean_auc"]) and auc_df.empty


def test_auc_matches_sksurv_on_week_units():
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    duration, event, x = _survival()
    grid = fet.auc_horizon_grid(duration, event)
    valid = fet.valid_horizons(duration, grid)
    y = Surv.from_arrays(event=event.astype(bool), time=np.ceil(duration / 7))
    auc, mean = cumulative_dynamic_auc(y, y, x, valid)
    assert fet.mean_auc(duration, event, x, grid) == pytest.approx(mean)
    summary, auc_df = fet.ipcw_metrics(duration, event, x, grid)
    assert summary["mean_auc"] == pytest.approx(mean)
    np.testing.assert_allclose(auc_df["auc"], auc)
    np.testing.assert_array_equal(auc_df["horizon_days"], valid * 7)
    assert summary["n_auc_horizons_valid"] == len(valid)


def test_deephit_per_horizon_mean_reduces_to_sksurv_for_a_constant_score():
    duration, event, x = _survival()
    grid = fet.auc_horizon_grid(duration, event)
    # the same score at every week: per-horizon calls + KM weighting == one call
    cif_h = np.repeat(x[:, None], len(grid), axis=1)
    assert fet.mean_auc(duration, event, cif_h, grid) == pytest.approx(
        fet.mean_auc(duration, event, x, grid))
    # a score that changes by week matches the rhino DeepHit controller's
    # per-horizon calls and Kaplan-Meier-weighted mean
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.nonparametric import SurvivalFunctionEstimator
    from sksurv.util import Surv

    rng = np.random.default_rng(3)
    varying = cif_h + rng.normal(scale=0.5, size=cif_h.shape)
    valid = fet.valid_horizons(duration, grid)
    y = Surv.from_arrays(event=event.astype(bool), time=np.ceil(duration / 7))
    auc_t = np.array([
        cumulative_dynamic_auc(y, y, varying[:, list(grid).index(h)], np.array([h]))[0][0]
        for h in valid])
    s = SurvivalFunctionEstimator().fit(y).predict_proba(valid)
    d = -np.diff(np.r_[1.0, s])
    assert fet.mean_auc(duration, event, varying, grid) == pytest.approx(
        (auc_t * d).sum() / (1 - s[-1]))
    summary, auc_df = fet.ipcw_metrics(duration, event, x, grid,
                                       auc_score=np.clip(1 / (1 + np.exp(-cif_h)), 0, 1))
    assert np.isfinite(summary["ibs"]) and auc_df["brier"].notna().all()

