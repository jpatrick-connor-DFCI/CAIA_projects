"""Leakage, full-cohort retention, and source provenance for checklists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis.predictive_checklists.data import load_landmark, resolved_nepc_mask
from COMPASS.survival_analysis.predictive_checklists.features import FeatureBinarizer, schema_feature_columns


def _write_inputs(tmp_path: Path, *, endpoint="platinum", landmark=0, manifest_changes=None, frame=None):
    folder = tmp_path / f"prediction_inputs_adt_{endpoint}"
    folder.mkdir(exist_ok=True)
    manifest = {
        "data": "/cluster/COMPASS/longitudinal_prediction_data_adt.csv",
        "anchor_col": "none", "endpoint": endpoint, "require_nepc": endpoint == "nepc",
        "landmark_days": [landmark], "split_mode": "shared", "restrict_to_mrns": None,
        "exclude_mrns": None, "time_unit_days": 7, "auc_time_unit_days": 7,
        "auc_max_time_units": 260, "auc_timeline_schema_version": 3,
        "auc_horizons_by_landmark": {str(landmark): {endpoint: [52, 104, 156]}},
        "n_patients_by_landmark": {str(landmark): 6}, "max_followup_days": 3650,
    }
    manifest.update(manifest_changes or {})
    (folder / "build_manifest.json").write_text(json.dumps(manifest))
    if frame is None:
        frame = pd.DataFrame({
            "DFCI_MRN": ["0001", "0002", "0003", "0004", "0005", "0006"],
            "split": ["train", "train", "valid", "valid", "test", "test"],
            "AGE_AT_TREATMENTSTART": [60, 61, 62, 63, 64, 65],
            "t_platinum": [20, 600, 300, 800, 90, 900], "PLATINUM": [1, 0, 1, 0, 1, 0],
            "t_nepc": [100, 600, 400, 800, 190, 900], "NEPC": [1, 0, 1, 0, 1, 0],
            "PSA__last": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan],
            "NEPC_LABEL_SOURCE": ["adjudicated", "failed", "reviewed", "not_run", "complete", "missing_from_labels"],
        })
    frame.to_csv(folder / f"aggregated_landmark{landmark}.csv", index=False)
    frame[["DFCI_MRN", "split"]].to_csv(folder / f"split_assignments_landmark{landmark}.csv", index=False)
    return folder, frame


def test_loader_preserves_ids_all_patients_splits_and_landmark_durations(tmp_path):
    folder, frame = _write_inputs(tmp_path, landmark=180)
    loaded = load_landmark(folder, "platinum", 180)
    assert len(loaded.frame) == len(frame)
    assert loaded.frame["DFCI_MRN"].tolist() == frame["DFCI_MRN"].tolist()
    assert loaded.frame["split"].tolist() == frame["split"].tolist()
    assert loaded.frame["t_platinum"].tolist() == frame["t_platinum"].tolist()
    assert loaded.frame["PSA__last"].isna().sum() == 2
    assert loaded.audit["auc_horizons_days"] == [364, 728, 1092]
    assert loaded.audit["resolved_nepc_n"] == 3
    assert loaded.audit["stored_split_assignments_checked"]
    assert "0001" not in json.dumps(loaded.audit)


@pytest.mark.parametrize("changes,match", [
    ({"endpoint": "avpc"}, "Rebuild endpoint-specific"),
    ({"data": "/cluster/longitudinal_prediction_data.csv"}, "Full ADT"),
    ({"arm": "arpi"}, "non-ADT"),
    ({"restrict_to_mrns": "/cluster/metastatic.csv"}, "beyond the full ADT"),
    ({"exclude_mrns": "/cluster/pre_adt_castrate_mrns.csv"}, "additional patient exclusions"),
    ({"cohort": "localized"}, "full ADT cohort"),
    ({"anchor_col": "t_first_treatment"}, "already be anchor-relative"),
    ({"split_mode": "independent"}, "shared patient splits"),
    ({"auc_timeline_schema_version": 2}, "schema version 3"),
    ({"require_nepc": True}, "require_nepc disagrees"),
])
def test_loader_rejects_wrong_cohort_or_stale_provenance(tmp_path, changes, match):
    folder, _ = _write_inputs(tmp_path, manifest_changes=changes)
    with pytest.raises(ValueError, match=match):
        load_landmark(folder, "platinum", 0)


def test_loader_allows_explicit_adt_source_and_canonical_base_cohort(tmp_path):
    folder, _ = _write_inputs(tmp_path, manifest_changes={
        "data": "synthetic.csv", "arm": "adt",
        "restrict_to_mrns": "/cluster/prostate_adt_survival_cohort_adt.csv",
    })
    assert len(load_landmark(folder, "platinum", 0).frame) == 6


def test_loader_does_not_apply_other_endpoint_incident_gate(tmp_path):
    folder, frame = _write_inputs(tmp_path, endpoint="nepc")
    frame.loc[0, "t_platinum"] = -100
    frame.to_csv(folder / "aggregated_landmark0.csv", index=False)
    assert len(load_landmark(folder, "nepc", 0).frame) == 6


@pytest.mark.parametrize("corruption", ["duplicate_id", "split", "duration", "event"])
def test_loader_rejects_invalid_patient_or_outcome_rows_without_dropping(tmp_path, corruption):
    folder, frame = _write_inputs(tmp_path)
    if corruption == "duplicate_id":
        frame.loc[1, "DFCI_MRN"] = frame.loc[0, "DFCI_MRN"]
    elif corruption == "split":
        frame.loc[0, "split"] = "test"
    elif corruption == "duration":
        frame.loc[0, "t_platinum"] = 0
    else:
        frame.loc[0, "PLATINUM"] = 2
    frame.to_csv(folder / "aggregated_landmark0.csv", index=False)
    with pytest.raises(ValueError):
        load_landmark(folder, "platinum", 0)


def test_resolved_mask_is_conservative_without_affecting_primary_rows():
    frame = pd.DataFrame({"NEPC_LABEL_SOURCE": ["Reviewed", " failed ", None, "unknown", "complete"]})
    assert resolved_nepc_mask(frame).tolist() == [True, False, False, False, True]
    assert not resolved_nepc_mask(pd.DataFrame(index=range(3))).any()


def _training_features():
    return pd.DataFrame({
        "DFCI_MRN": ["1", "2", "3", "4"], "split": ["train"] * 4,
        "AGE_AT_TREATMENTSTART": [50, 60, 70, 80],
        "PSA__last": [1.0, 2.0, 3.0, np.nan], "PSA__mean": [1.0, 2.5, 4.0, np.nan],
        "PSA__n_observations": [1, 3, 4, np.nan],
        "Unseen__last": [np.nan] * 4,
        "t_nepc": [100, 200, 300, 400], "NEPC": [1, 0, 1, 0],
        "NEPC__last": [0, 0, 1, 1], "NEPC_LABEL_SOURCE": ["complete"] * 4,
        "TP53_SNV": [1, 0, 0, 1], "unknown_metadata": [1, 2, 3, 4],
    })


def test_binarizer_uses_training_thresholds_and_imputation_only():
    train = _training_features()
    fitted = FeatureBinarizer().fit(train)
    snapshot = fitted.to_dict()
    held_out = train.copy()
    held_out["PSA__last"] = [10000, np.nan, -10000, np.nan]
    held_out["Unseen__last"] = [1, 2, 3, 4]
    matrix = fitted.transform(held_out)
    assert matrix.shape == (4, len(fitted.items))
    assert fitted.medians["PSA__last"] == 2
    assert fitted.to_dict() == snapshot
    assert "Unseen__last" not in fitted.feature_columns
    j = next(i for i, item in enumerate(fitted.items)
             if item["feature"] == "PSA__last" and item["op"] == "is_missing")
    assert matrix[:, j].tolist() == [0, 1, 0, 1]
    k = next(i for i, item in enumerate(fitted.items)
             if item["feature"] == "PSA__last" and item["op"] == "<=" and item["threshold"] == 2)
    assert matrix[:, k].tolist() == [0, 1, 1, 1]


def test_feature_whitelist_lab_groups_and_roundtrip():
    frame = _training_features()
    fitted = FeatureBinarizer().fit(frame)
    assert set(fitted.feature_columns) == {"AGE_AT_TREATMENTSTART", "PSA__last", "PSA__mean", "PSA__n_observations"}
    assert {item["group"] for item in fitted.items if item["feature"].startswith("PSA__")} == {"PSA"}
    restored = FeatureBinarizer.from_dict(json.loads(json.dumps(fitted.to_dict())))
    np.testing.assert_array_equal(restored.transform(frame), fitted.transform(frame))
    assert set(np.unique(restored.transform(frame))) == {0, 1}


def test_schema_whitelist_is_value_independent_and_excludes_outcomes():
    frame = _training_features()
    columns = schema_feature_columns(frame)
    assert set(columns) == {
        "AGE_AT_TREATMENTSTART", "PSA__last", "PSA__mean",
        "PSA__n_observations", "Unseen__last",
    }
    assert schema_feature_columns(frame.iloc[:0]) == columns
    altered = frame.copy()
    altered["PSA__last"] = np.nan
    altered["Unseen__last"] = 999
    assert schema_feature_columns(altered) == columns


def test_fold_local_coverage_ignores_held_out_long_labs():
    train = _training_features()
    long_rows = pd.DataFrame({
        "DFCI_MRN": ["1", "2", "3", "4", "100", "101", "102", "103"],
        "LAB_NAME": ["Other"] * 4 + ["PSA"] * 4,
        "LAB_VALUE": [1, 2, 3, 4, 10, 20, 30, 40],
    })
    fitted = FeatureBinarizer().fit(train, lab_long=long_rows)
    assert fitted.feature_columns == ["AGE_AT_TREATMENTSTART"]


def test_binarizer_keeps_completely_missing_patient_and_evaluation_column():
    frame = _training_features()
    fitted = FeatureBinarizer().fit(frame)
    held_out = pd.DataFrame({"DFCI_MRN": ["new_patient"]})
    binary = fitted.transform(held_out)
    assert binary.shape[0] == 1
    assert np.isfinite(binary).all()


@pytest.mark.parametrize("endpoint", ["platinum", "nepc"])
def test_builder_manifest_preserves_selected_endpoint_after_horizon_loop(tmp_path, endpoint):
    """Real builder regression: iterating all endpoint horizons cannot rename the cohort."""
    from COMPASS.data_preprocessing import build_prediction_inputs as bpi

    rows = []
    for i in range(40):
        for offset in (-20, -10):
            rows.append({
                "DFCI_MRN": i + 1, "AGE_AT_TREATMENTSTART": 50 + i,
                "FIRST_RECORD_DATE": "2010-01-01", "FIRST_TREATMENT": 1,
                "t_first_treatment": 0, "t_last_contact": 1000,
                "t_death": 1000, "DEATH": 0,
                "PLATINUM": int(i % 3 == 0), "t_platinum": 100 + i * 10 if i % 3 == 0 else 1000,
                "NEPC": int(i % 4 == 0), "t_nepc": 200 + i * 10 if i % 4 == 0 else 1000,
                "AVPC": int(i % 5 == 0), "t_avpc": 300 + i * 10 if i % 5 == 0 else 1000,
                "LAB_NAME": "PSA", "LAB_VALUE": i + offset / 100,
                "t_lab": offset, "RAW_TEST_CODE": "PSA",
            })
    source = tmp_path / "longitudinal_prediction_data_adt.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "build"
    args = argparse.Namespace(
        id_col="DFCI_MRN", age_col="AGE_AT_TREATMENTSTART", data=str(source),
        restrict_to_mrns=None, exclude_mrns=None, anchor_col="none", min_psa_count=0,
        exclude_parpi=False, landmark_days=[0], output_dir=str(output), seed=42,
        test_frac=0.2, val_frac=0.2, min_patient_coverage=0.2, time_unit_days=7,
        auc_quantiles=[0.1, 0.5, 0.9], auc_max_time_units=260, max_followup_days=3650,
        build_longitudinal=False, endpoint=endpoint, split_mode="shared",
    )
    bpi.main(args)
    manifest = json.loads((output / "build_manifest.json").read_text())
    assert manifest["endpoint"] == endpoint
    assert manifest["require_nepc"] == (endpoint == "nepc")
    assert set(manifest["auc_horizons_by_landmark"]["0"]) == {"platinum", "nepc", "avpc"}
    loaded = load_landmark(output, endpoint, 0)
    assert len(loaded.frame) == 40
