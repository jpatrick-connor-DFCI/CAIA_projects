"""ADT-intent-stratified Stage-3 and model orchestration."""

from __future__ import annotations

import polars as pl
import pytest

from COMPASS.survival_analysis import compass_pipeline as cp


def test_make_adt_intent_endpoint_runs_are_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_adt_intent_endpoint_runs(
        strata=("metastatic", "localized"),
        endpoints=("platinum", "nepc", "avpc"),
    )

    assert len(runs) == 6
    assert {(r["adt_intent"], r["endpoint"]) for r in runs} == {
        (intent, endpoint)
        for intent in ("METASTATIC", "LOCALIZED_ADJUVANT")
        for endpoint in ("platinum", "nepc", "avpc")
    }
    metastatic_platinum = next(
        r for r in runs
        if r["adt_intent"] == "METASTATIC" and r["endpoint"] == "platinum"
    )
    localized_nepc = next(
        r for r in runs
        if r["adt_intent"] == "LOCALIZED_ADJUVANT" and r["endpoint"] == "nepc"
    )
    assert metastatic_platinum["inputs_dir"].name == "prediction_inputs_adt_metastatic"
    assert metastatic_platinum["output_dir"].name == "local_runs_adt_metastatic"
    assert localized_nepc["inputs_dir"].name == "prediction_inputs_adt_localized_nepc"
    assert localized_nepc["output_dir"].name == "local_runs_adt_localized_nepc"
    assert localized_nepc["restrict_to_mrns"] == (
        tmp_path / "mrn_lists" / "adt_localized_mrns.csv"
    )
    assert all(r["input_csv"].name == "longitudinal_prediction_data_adt.csv" for r in runs)
    assert all(r["retrospective_stratification"] is True for r in runs)


def test_adt_intent_runs_reject_composite_endpoint():
    with pytest.raises(ValueError, match="platinum, nepc, and avpc"):
        cp.make_adt_intent_endpoint_runs(endpoints=("avpc_nepc",))


def test_build_adt_intent_mrn_lists_and_endpoint_counts(monkeypatch, tmp_path):
    meds_path = tmp_path / "medications.csv"
    cohort_path = tmp_path / "prostate_adt_survival_cohort_adt.csv"

    # MRN 1 has a short, completed, single ADT course. MRN 2 has definitive
    # escalation and is therefore metastatic even though its ADT course is short.
    # MRN 3 is antiandrogen-only: it has no depot span but must remain in the
    # conservative unresolved/metastatic group rather than fail the cohort guard.
    pl.DataFrame(
        {
            "DFCI_MRN": [1, 1, 2, 2, 2, 3],
            "NCI_PREFERRED_MED_NM": [
                "LEUPROLIDE ACETATE",
                "LEUPROLIDE ACETATE",
                "LEUPROLIDE ACETATE",
                "LEUPROLIDE ACETATE",
                "DOCETAXEL",
                "BICALUTAMIDE",
            ],
            "MED_START_DT": [
                "2015-01-01",
                "2015-04-01",
                "2015-01-01",
                "2015-04-01",
                "2016-01-01",
                "2015-01-01",
            ],
        }
    ).write_csv(meds_path)
    pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3],
            "FOLLOW_UP_END_DATE": ["2020-01-01"] * 3,
            "PLATINUM": [0, 1, 0],
            "TT_PLATINUM": [1826, 500, 1826],
            "NEPC": [0, 1, 0],
            "TT_NEPC": [1826, 700, 1826],
            "AVPC": [0, 1, 0],
            "TT_AVPC": [1826, 600, 1826],
        }
    ).write_csv(cohort_path)
    monkeypatch.setattr(
        cp,
        "scan_source",
        lambda path: pl.scan_csv(path, infer_schema_length=0),
    )

    outputs = cp.build_adt_intent_mrn_lists(
        medications_path=meds_path,
        base_cohort_csv=cohort_path,
        data_root=tmp_path,
    )

    localized = pl.read_csv(outputs["localized"])
    metastatic = pl.read_csv(outputs["metastatic"])
    assert localized["DFCI_MRN"].to_list() == [1]
    assert metastatic["DFCI_MRN"].to_list() == [2, 3]
    assert localized["ADT_INTENT"].to_list() == ["LOCALIZED_ADJUVANT"]
    assert metastatic["ADT_INTENT"].to_list() == ["METASTATIC", "METASTATIC"]

    counts = pl.read_csv(outputs["counts"])
    assert counts.height == 6
    localized_platinum = counts.filter(
        (pl.col("stratum") == "localized") & (pl.col("endpoint") == "platinum")
    ).row(0, named=True)
    metastatic_avpc = counts.filter(
        (pl.col("stratum") == "metastatic") & (pl.col("endpoint") == "avpc")
    ).row(0, named=True)
    assert localized_platinum["n_incident_events"] == 0
    assert metastatic_avpc["n_incident_events"] == 1
