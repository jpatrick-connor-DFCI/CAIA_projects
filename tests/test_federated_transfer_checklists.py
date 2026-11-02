"""Sequencing clock, inherited splits, held-out isolation and portable exports."""
import json

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import federated_transfer_checklists as ftc
from COMPASS.survival_analysis.predictive_checklists.inference import predict_bundle


def sources(tmp_path, endpoints=("avpc",), landmarks=(0, 90)):
    rng = np.random.default_rng(16)
    n = 180
    split = np.tile(["train", "train", "valid", "test", "train", "test"], n // 6)
    status = np.tile([1, 1, 0, 1, 0, 0], n // 6)
    # Both cases and controls in development/test, at both horizons; no early censoring.
    duration = np.where(status, rng.choice([200., 310., 1000., 1600.], n), 3000.)
    trio = rng.integers(0, 2, size=(n, 3))
    patients = pd.DataFrame({ftc.ID: np.arange(1, n+1), "clock": "sequencing",
        "model_family": "xgboost_cox", "bundle": "frozen_no_dfci", "landmark_days": 0,
        "risk_score": status + rng.normal(0, .6, n), "GLEASON_SCORE": rng.integers(6, 11, n),
        "TP53_ANY": trio[:, 0], "RB1_ANY": trio[:, 1], "PTEN_ANY": trio[:, 2],
        "HAS_SEQUENCING": True, "event": status, "duration_days": duration,
        "AGE_AT_TREATMENTSTART": rng.integers(40, 80, n), "PSA__mean": rng.normal(size=n)})
    frames, inputs = [], {}
    for ep in endpoints:
        folder = tmp_path / f"inputs_{ep}"
        folder.mkdir(parents=True)
        inputs[ep] = folder
        pd.DataFrame({"DFCI_MRN": patients[ftc.ID], "split": split}).to_csv(
            folder / "split_assignments.csv", index=False)
        manifest = dict(data="SYNTHETIC/longitudinal_prediction_data_adt.csv", synthetic=True,
            arm="adt", cohort="all", exclusion="none", anchor_col="none", endpoint=ep,
            landmark_days=list(landmarks), restrict_to_mrns=None, exclude_mrns=None,
            split_mode="shared", time_unit_days=7, auc_time_unit_days=7, auc_max_time_units=522,
            auc_timeline_schema_version=3,
            auc_horizons_by_landmark={str(lm): {ep: [52, 260]} for lm in landmarks})
        (folder / "build_manifest.json").write_text(json.dumps(manifest))
        for lm in landmarks:
            frames.append(patients.assign(endpoint=ep, landmark_days=lm, duration_days=duration-lm))
    table = pd.concat(frames, ignore_index=True)
    path = tmp_path / "transfer_km_patients.csv"
    table.to_csv(path, index=False)
    return path, inputs, table


def config(**overrides):
    return ftc.transfer_config(endpoints=("avpc",), landmarks=(0, 90), max_items=(3,),
        n_folds=2, min_events=2, min_controls=2, bootstrap_samples=5,
        solver_time_limit=10, **overrides)


def test_sequencing_clock_and_existing_splits_only(tmp_path):
    path, inputs, table = sources(tmp_path)
    table.loc[table.person_id.eq(1), "GLEASON_SCORE"] = np.nan
    table.loc[table.person_id.eq(2), "HAS_SEQUENCING"] = False
    table.loc[table.person_id.eq(3), "person_id"] = 9999  # No existing split: audited, never resplit.
    pd.concat([table, table.assign(clock="adt", risk_score=1e9)]).to_csv(path, index=False)
    cells, source = ftc.prepare_cells(path, inputs, config())
    assert len(cells) == 2
    assignments = []
    for _, ep, lm, data, identity in cells:
        assert set(data.frame) == {ftc.ID, "split", "t_avpc", "AVPC", *ftc.FEATURES}
        assert len(data.frame) == 177 and data.audit["n_without_base_split"] == 1
        assert data.frame[ftc.SCORE].max() < 1e9
        assert identity["clock"] == "sequencing"
        assert data.manifest["prediction_time_origin"] == "sequencing date"
        assert data.frame.t_avpc.max() == 3000-lm
        assignments.append(data.frame.set_index(ftc.ID).split)
    pd.testing.assert_series_equal(*assignments)
    assert source["avpc"]["source_manifest"]["synthetic"]


@pytest.mark.parametrize("issue", ["duplicate", "bad_mutation", "bad_duration", "split", "clock"])
def test_bad_inputs_fail_before_any_fit(tmp_path, issue):
    path, inputs, table = sources(tmp_path)
    if issue == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif issue == "bad_mutation":
        table.loc[0, "TP53_ANY"] = 2
    elif issue == "bad_duration":
        table.loc[0, "duration_days"] = 0
    elif issue == "split":
        split_path = inputs["avpc"] / "split_assignments.csv"
        split = pd.read_csv(split_path)
        split.loc[0, "split"] = "unknown"
        split.to_csv(split_path, index=False)
    else:
        table["clock"] = "adt"
    table.to_csv(path, index=False)
    out = tmp_path / "run"
    with pytest.raises(ValueError):
        ftc.run_transfer_checklists(path, inputs, out, config=config())
    assert not list(out.rglob("model.json"))


def test_fit_export_resume_and_heldout_isolation(tmp_path, monkeypatch):
    path, inputs, table = sources(tmp_path, landmarks=(0,))
    # Notebook landmarks come from pandas.unique(), hence NumPy integer scalars.
    settings = ftc.transfer_config(endpoints=("avpc",), landmarks=(np.int64(0),), max_items=(3,),
        n_folds=2, min_events=2, min_controls=2, bootstrap_samples=5, solver_time_limit=10)
    out = tmp_path / "run"
    metrics = ftc.run_transfer_checklists(path, inputs, out, config=settings)
    assert len(metrics) == 2 and set(metrics.horizon_days) == {365, 1825}
    assert metrics.status.eq("ok").all()
    assert metrics.dataset.eq("test").all()
    assert metrics.cohort.eq(ftc.POPULATION).all()
    data = ftc.prepare_cells(path, inputs, settings)[0][0][3]
    test = data.frame.loc[data.frame.split.eq("test")]
    bundles = {}
    for horizon in (365, 1825):
        folder = out / "xgboost_cox/avpc/landmark_0" / ftc.POPULATION / f"horizon_{horizon}"
        bundles[horizon] = json.loads((folder / "model.json").read_text())
        assert set(bundles[horizon]["binarizer"]["feature_columns"]) == set(ftc.FEATURES)
        saved = pd.read_csv(folder / "predictions.csv").query("dataset == 'test'")
        scored = predict_bundle(folder / "model.json", test)
        np.testing.assert_allclose(saved.probability, scored.probability)
        np.testing.assert_array_equal(saved.score, scored.score)
        assert "sequencing date" in (folder / "checklist.txt").read_text()
    for name in ftc.TABLES:
        assert set(pd.read_csv(out / name).horizon_days) == {365, 1825}
    cells = pd.read_csv(out / "checklist_cells.csv").set_index("horizon_days")
    assert cells.loc[365, "n_events_test"] < cells.loc[1825, "n_events_test"]
    for horizon in (365, 1825):
        assert cells.loc[horizon, "n_events_test"] == metrics.set_index("horizon_days").loc[horizon, "n_cases"]
    with monkeypatch.context() as patch:
        patch.setattr(ftc.runner, "fit_horizon", lambda *a, **kw: pytest.fail("Refit on resume"))
        ftc.run_transfer_checklists(path, inputs, out, config=settings, resume=True)
    with pytest.raises(FileExistsError):
        ftc.run_transfer_checklists(path, inputs, out, config=settings)
    # Change only test predictors AND outcomes; no fitted state may change.
    test_ids = test.person_id.astype(int)
    mask = table.person_id.isin(test_ids)
    table.loc[mask, "risk_score"] = 1e8
    table.loc[mask, "event"] = 1-table.loc[mask, "event"]
    table.loc[mask, "duration_days"] = 2500
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Cannot resume"):
        ftc.run_transfer_checklists(path, inputs, out, config=settings, resume=True)
    changed = tmp_path / "changed"
    ftc.run_transfer_checklists(path, inputs, changed, config=settings)
    for horizon in (365, 1825):
        folder = changed / "xgboost_cox/avpc/landmark_0" / ftc.POPULATION / f"horizon_{horizon}"
        bundle = json.loads((folder / "model.json").read_text())
        for key in ("binarizer", "calibrator", "censoring", "selected_max_items", "cv_balanced_error"):
            assert bundle[key] == bundles[horizon][key]
        assert bundle["model"]["selected_indices"] == bundles[horizon]["model"]["selected_indices"]


def test_empty_test_and_unsupported_horizons_are_exported(tmp_path):
    path, inputs, table = sources(tmp_path, landmarks=(0,))
    settings = ftc.transfer_config(endpoints=("avpc",), landmarks=(0,))
    split_path = inputs["avpc"] / "split_assignments.csv"
    split = pd.read_csv(split_path)
    split.loc[split.split.eq("test"), "split"] = "train"
    split.to_csv(split_path, index=False)
    out = tmp_path / "empty"
    results = ftc.run_transfer_checklists(path, inputs, out, config=settings)
    assert results.status.eq("underpowered").all()
    assert set(results.horizon_days) == {365, 1825}
    assert pd.read_csv(out / "checklist_cells.csv").n_test.eq(0).all()
    assert not list(out.rglob("model.json"))
