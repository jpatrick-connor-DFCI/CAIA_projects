"""Data-boundary and persistent-cache regressions for the figure migration."""
import json
import math
from pathlib import Path

import polars as pl
import pytest

from COMPASS.survival_analysis import prepare_figure_data as prep
from COMPASS.survival_analysis.prepare_metastatic_figure_labels import build_labels


def input_frame():
    return pl.DataFrame({
        "DFCI_MRN": ["001"] * 6 + ["002", "003", "004", "005"],
        "LAB_NAME": ["PSA", "Prostate specific Ag serum", "PSA", "PSA", "PSA", "Sodium",
                     "Testosterone", None, "PSA", "PSA"],
        "LAB_VALUE": [1., 9., 3., -1., 7., 140., 2., None, 0., 4.],
        "LAB_DATE": ["2020-01-01", "2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-21",
                     "2020-02-01", None, None, "2020-01-01"],
        "t_lab": [-1., 0., 0., 180., 1826.25, 10., -180., None, -1826.25, -365.25],
        "PLATINUM": [1.] * 6 + [0., 0., None, 0.],
        "DIAGNOSIS_DATE": ["2019-01-01"] * 10,
        "TREATMENT_ANCHOR_DATE": ["2020-01-01"] * 10,
        "UNUSED_FEATURE": list(range(10)),
    })


def test_patient_spans_bins_and_log_before_averaging(tmp_path):
    path = tmp_path / "source.csv"
    input_frame().write_csv(path)
    tables = prep.prepare_arm(path, ["PSA", "Testosterone"])
    patients = tables["patients"]
    assert patients["DFCI_MRN"].to_list() == ["001", "002", "003", "004", "005"]
    assert patients["record_span_days"].to_list() == [20., 0., None, -math.inf, 0.]
    assert patients["lab_rows"].to_list() == [6, 1, None, 1, 1]
    assert patients["t_dx_to_anchor"].to_list() == [365.] * 5
    assert "UNUSED_FEATURE" not in patients.columns
    bins = tables["patient_bins"].filter(pl.col("DFCI_MRN") == "001").sort("t_mid")
    assert bins["value_raw"].to_list() == [1., 6., -1., 7.]
    assert bins["value_log"].to_list() == pytest.approx([math.log(2), (math.log(10)+math.log(4))/2, None, math.log(8)])
    assert bins["n_raw"].to_list() == [1, 2, 1, 1]
    assert bins["n_log"].to_list() == [1, 2, 0, 1]
    assert bins["t_mid"][0] < 0 < bins["t_mid"][1]
    assert tables["canonical"].height == 8
    counts = tables["coverage_patient"].sort("DFCI_MRN")
    assert counts["DFCI_MRN"].to_list() == ["001", "002", "004", "005"]
    assert counts["n_pre_180"].to_list() == [1, 1, 0, 0]
    assert tables["coverage_bins"].height == 4


def test_all_boundary_times_match_left_closed_bins():
    edges = prep.bin_edges()
    values = [-400., -365.25, -360., -180., -0.01, 0., 180., 1826.25, 1826.26]
    result = pl.DataFrame({"t_rel": values}).select(prep.bin_expr(edges))["bin_id"].to_list()
    assert result == [None, 0, 1, 2, 2, 3, 4, len(edges)-2, None]


def make_config(tmp_path, scope="all"):
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    input_frame().write_csv(root / "longitudinal_prediction_data_adt.csv")
    for endpoint in ["", "_nepc"]:
        (root / "survival_analysis" / f"prediction_inputs_adt{endpoint}").mkdir(parents=True)
    return dict(data_root=str(root), cache_root=str(tmp_path / "cache"),
                cohorts=["adt"], endpoints=["platinum", "nepc"], scope=scope,
                labs=["PSA", "Testosterone"], classifier_path=str(root / "classifier"),
                federated_path=str(root / "fed.csv"))


def test_cache_reuse_invalidation_and_force(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    first = prep.prepare(config)
    key = first["arms"]["adt"]["key"]
    original = prep.prepare_arm
    monkeypatch.setattr(prep, "prepare_arm", lambda *a: pytest.fail("Unchanged CSV was rescanned"))
    assert prep.prepare(config) == first
    # A new endpoint result changes only that cell; no longitudinal rescan.
    result = Path(config["data_root"]) / "survival_analysis" / "local_runs_adt"
    result.mkdir()
    (result / "result.csv").write_text("feature,hr\na,2\n")
    changed = prep.prepare(config)
    assert changed["cells"]["adt__platinum"] != first["cells"]["adt__platinum"]
    assert changed["cells"]["adt__nepc"] == first["cells"]["adt__nepc"]
    assert changed["arms"]["adt"]["key"] == key
    monkeypatch.setattr(prep, "prepare_arm", original)
    forced = prep.prepare(dict(config, force=True))
    assert forced["cells"] != changed["cells"]
    assert prep.prepare(config) == forced
    # Truncated cache output is not a valid hit.
    output = Path(forced["arms"]["adt"]["directory"]) / "patient_bins.parquet"
    output.write_bytes(b"PAR1")
    repaired = prep.prepare(config)
    assert pl.read_parquet(output).height > 0
    assert repaired["arms"]["adt"]["key"] != forced["arms"]["adt"]["key"]


def test_federated_scope_needs_no_patient_data(tmp_path, monkeypatch):
    config = make_config(tmp_path, "federated")
    config["federated"] = True
    monkeypatch.setattr(prep, "prepare_arm", lambda *a: pytest.fail("Federated scope read longitudinal data"))
    result = prep.prepare(config)
    assert result["arms"] == {} and result["cells"] == {}
    assert result["federated"]
    assert [item["path"] for item in result["federated_sources"]] == [config["federated_path"]]


def test_arpi_preparation_does_not_require_adt(tmp_path):
    config = make_config(tmp_path)
    root = Path(config["data_root"])
    (root / "longitudinal_prediction_data_adt.csv").rename(root / "longitudinal_prediction_data.csv")
    for suffix in ["", "_nepc"]:
        (root / "survival_analysis" / f"prediction_inputs_arpi{suffix}").mkdir()
    result = prep.prepare(dict(config, cohorts="arpi", endpoints="nepc"))
    assert set(result["arms"]) == {"arpi"}
    assert set(result["cells"]) == {"arpi__nepc"}
    assert pl.read_parquet(Path(result["arms"]["arpi"]["directory"]) / "canonical.parquet").is_empty()


def test_bad_arm_does_not_discard_other_arms_or_federation(tmp_path):
    config = make_config(tmp_path)
    root = Path(config["data_root"])
    (root / "longitudinal_prediction_data.csv").write_text("wrong,column\n1,2\n")
    result = prep.prepare(dict(config, cohorts=["adt", "arpi"]))
    assert "adt__platinum" in result["cells"] and "adt__nepc" in result["cells"]
    assert "arpi__platinum" in result["errors"] and "arpi__nepc" in result["errors"]
    assert result["federated"]


def test_preparation_notebook_runs_all_python_cells(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setenv("COMPASS_DATA_ROOT", config["data_root"])
    monkeypatch.setenv("COMPASS_FIGURE_DATA_ROOT", config["cache_root"])
    monkeypatch.setenv("COMPASS_METASTATIC_LABEL_SUPPLEMENT", "false")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    notebook = json.loads((root / "COMPASS/survival_analysis/04_prep_figure_data.ipynb").read_text())
    state = {"display": lambda frame: None}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"notebook-cell-{i}", "exec"), state)
    assert set(state["manifest"]["cells"]) == {"adt__platinum", "adt__nepc"}
    assert Path(config["cache_root"], "preparation_config.json").exists()
    assert "PSA" in state["figure_data"].canonical_lab_names()
    assert state["rows"] and all(row["rows"] >= 0 for row in state["rows"])


def test_overview_counts_use_endpoint_inputs_and_keep_missing_labels(tmp_path):
    config = make_config(tmp_path)
    config["forest_cohorts"] = ["adt", "adt_noprecastrate"]
    root = Path(config["data_root"])
    for suffix, event, values in [("", "PLATINUM", [1, 0, 1]), ("_nepc", "NEPC", [1, 0])]:
        pl.DataFrame({"DFCI_MRN": [str(i) for i in range(len(values))], event: values}).write_csv(
            root / "survival_analysis" / f"prediction_inputs_adt{suffix}" / "aggregated_landmark180.csv")
    mrns = root / "mrn_lists"
    mrns.mkdir()
    pl.DataFrame({"DFCI_MRN": ["001", "2.0", "3"], "ADT_INTENT": ["METASTATIC", "LOCALIZED_ADJUVANT", "UNKNOWN"]}).write_csv(
        mrns / "adt_intent_labels_model_cohort.csv")
    pl.DataFrame({"DFCI_MRN": ["1", "002.0"], "LLM_METASTATIC": ["true", "false"]}).write_csv(
        mrns / "llm_met_labels_model_cohort.csv")
    result = prep.prepare_cohort_overview(config)
    d = pl.read_parquet(Path(result["directory"]) / "incidence.parquet")
    assert d["n_patients"].to_list() == [3, 2, None, None]
    assert d["n_events"].to_list() == [2, 1, None, None]
    labels = pl.read_parquet(Path(result["directory"]) / "label_overlap.parquet")
    assert labels["n"].sum() == 3
    assert labels.filter(pl.col("adt_label") == "Unlabelled")["llm_label"].to_list() == ["Unlabelled"]
    assert prep.prepare_cohort_overview(config) == result
    pl.DataFrame({"DFCI_MRN": ["1", "1"], "LLM_METASTATIC": ["true", "false"]}).write_csv(
        mrns / "llm_met_labels_model_cohort.csv")
    with pytest.raises(ValueError, match="conflicting"):
        prep.prepare_cohort_overview(config)


def test_labels_use_analysis_anchor_and_retain_unlabelled_patients():
    intent = pl.DataFrame({"DFCI_MRN": [1, 2], "ADT_FIRST_DATE": ["2020-01-01"]*2,
                           "ADT_INTENT": ["METASTATIC", "LOCALIZED_ADJUVANT"]})
    anchors = pl.DataFrame({"DFCI_MRN": [1, 2, 3],
                            "TREATMENT_ANCHOR_DATE": ["2020-03-01", None, "2020-01-01"]})
    notes = pl.DataFrame({"DFCI_MRN": [1, 2, 3], "EVENT_DATE": ["2020-02-01"]*3,
                         "DERIVED_STAGE_MERGED": [4, 4, 3]})
    llm = pl.DataFrame({"DFCI_MRN": [3], "has_metastatic_disease": [True]})
    result = build_labels(intent, notes, llm, anchors).sort("DFCI_MRN")
    assert result["ANCHOR_DELTA_DAYS"].to_list() == [60, None, None]
    assert result["REGEX_LABEL"].to_list() == ["Metastatic", None, None]
    assert result["REGEX_MAX_ANY"].to_list() == ["Metastatic", "Metastatic", "Local"]
    assert result["ADT_LABEL"].to_list() == ["Metastatic", "Local", None]
    assert result["LLM_LABEL"].to_list() == [None, None, "Metastatic"]
