"""Exercise the patient-extraction notebook with synthetic data only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from survival_common import federated_endpoint_transfer as fet
from survival_common import federated_scoring as fs


NOTEBOOK = Path(__file__).resolve().parents[1] / (
    "COMPASS/notebooks/patient_extraction/01_high_vs_low_risk_transfer_pxs.ipynb"
)


def _run_cells(env, *indices):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for i in indices:
        exec(compile("".join(cells[i]["source"]), f"{NOTEBOOK}:cell_{i}", "exec"), env)


@pytest.fixture
def extraction(tmp_path):
    scores = pd.DataFrame([
        {"model_family": "xgboost_cox", "config": "both", "endpoint": "platinum",
         "landmark_days": lm, fs.ID_COL: 100001 + i, "risk_score": float(i)}
        for lm in (0, 90, 180) for i in range(20)
    ])
    env = dict(np=np, pd=pd, fet=fet, fs=fs, SCORES_PATH=tmp_path / "scores.csv",
               DATA_PATH=tmp_path / "local.csv", PATHOLOGY_PATH=tmp_path / "notes.parquet",
               OUTPUT_DIR=tmp_path / "out", MODEL_FAMILY="xgboost_cox",
               SCORE_ENDPOINT="platinum", LANDMARKS=(0, 90, 180), DECILE=0.1,
               ENDPOINTS=("avpc", "platinum", "nepc"))
    return env, scores


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "bad"])
def test_invalid_score_is_not_scored_at_one_landmark(extraction, invalid):
    env, scores = extraction
    scores["risk_score"] = scores["risk_score"].astype(object)
    scores.loc[(scores["landmark_days"] == 90) & (scores[fs.ID_COL] == 100001), "risk_score"] = invalid
    scores.to_csv(env["SCORES_PATH"], index=False)
    _run_cells(env, 4, 5)
    patient = env["cohort"].set_index("DFCI_MRN").loc[100001]
    assert patient["bottom_decile_lm0"]
    assert patient["decile_group_lm90"] == "not scored"
    assert not patient["bottom_decile_lm90"] and not patient["top_decile_lm90"]
    assert pd.isna(patient["risk_score_lm90"]) and pd.isna(patient["risk_percentile_lm90"])
    assert env["cutoffs"].loc[90, "n_scored"] == 19
    assert env["cutoffs"].loc[90, "n_not_scored"] == 1
    assert env["cutoffs"].loc[90, "bottom_cutoff"] == pytest.approx(2.8)
    assert env["cutoffs"].loc[90, "top_cutoff"] == pytest.approx(17.2)


@pytest.mark.parametrize("values, message", [
    ([1.0] * 20, "cutoffs coincide"),
    ([0.0] + [1.0] * 18 + [2.0], "cutoffs coincide"),
    ([np.nan] * 20, "no finite numeric risk scores"),
])
def test_unusable_landmark_stops_extraction(extraction, values, message):
    env, scores = extraction
    scores.loc[scores["landmark_days"] == 90, "risk_score"] = values
    scores.to_csv(env["SCORES_PATH"], index=False)
    with pytest.raises(ValueError, match=f"Landmark 90:.*{message}"):
        _run_cells(env, 4, 5)
    assert not env["OUTPUT_DIR"].exists()


def test_extract_writes_manifest_and_multiline_reports(extraction, capsys):
    env, scores = extraction
    scores.to_csv(env["SCORES_PATH"], index=False)
    ids = list(range(100001, 100021))
    local = pd.DataFrame({col: [0] * 20 for col in fet.LOCAL_COLUMNS})
    local["DFCI_MRN"] = ids
    local["TREATMENT_ANCHOR_DATE"] = "2020-01-01"
    for indicator, date in fet.ENDPOINT_SOURCES.values():
        local[indicator] = [0] * 19 + [1]
        local[date] = "2020-12-31"
    local.to_csv(env["DATA_PATH"], index=False)
    text = "synthetic\ttext\nsecond line"
    pl.DataFrame({"DFCI_MRN": ids, "EVENT_DATE": ["2021-01-01"] * 20,
                  "RPT_ID": list(range(20)), "RPT_TEXT": [text] * 20}).write_parquet(env["PATHOLOGY_PATH"])
    _run_cells(env, 4, 5, 7, 9, 11)
    manifest = pd.read_csv(env["OUTPUT_DIR"] / "patient_manifest.csv")
    reports = pd.read_csv(env["OUTPUT_DIR"] / "pathology_reports.tsv", sep="\t")
    assert set(manifest["DFCI_MRN"]) == {100001, 100002, 100019, 100020}
    assert len(reports) == manifest["n_pathology_notes"].sum() == 4
    assert reports["note_text"].eq(text).all()
    assert reports["note_days_from_adt_start"].eq(366).all()
    assert manifest.loc[~manifest["platinum_event"], "days_adt_to_platinum"].isna().all()
    output = capsys.readouterr().out
    assert "Manifest ready: 4 patients" in output
    assert text not in output
    assert not any(str(patient_id) in output for patient_id in ids)
