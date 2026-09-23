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
    assert both["hr_per_sd"] > 1
    # AVPC has no events: metrics are NaN, not an exception
    assert np.isnan(m.loc[("avpc", "both"), "c_index"])

    scores = res["scores"]
    per_ep = scores[scores["config"] == "both"].pivot(
        index=fs.ID_COL, columns="endpoint", values="risk_score"
    )
    # features do not depend on the endpoint, so neither does the score
    np.testing.assert_allclose(per_ep["nepc"], per_ep["avpc"])


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
