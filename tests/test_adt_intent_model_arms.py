"""ADT-intent-stratified Stage-3 and model orchestration."""

from __future__ import annotations

import polars as pl

from COMPASS.survival_analysis import compass_pipeline as cp


def test_cohort_endpoint_runs_cross_every_cohort(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum", "nepc", "avpc", "avpc_nepc"),
        cohorts=("all", "metastatic", "localized"),
    )

    # Cohort and endpoint are orthogonal axes: every cohort runs every
    # endpoint, including the composite avpc_nepc that the retired
    # ADT-intent factory refused.
    assert len(runs) == 12
    assert {(r["cohort"], r["endpoint"]) for r in runs} == {
        (cohort, endpoint)
        for cohort in ("all", "metastatic", "localized")
        for endpoint in ("platinum", "nepc", "avpc", "avpc_nepc")
    }
    assert len({cp.run_key(r) for r in runs}) == 12


def test_cohort_runs_keep_the_established_path_convention(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum", "nepc", "avpc", "avpc_nepc"),
        cohorts=("all", "metastatic", "localized"),
    )
    by_key = {(r["cohort"], r["endpoint"]): r for r in runs}

    # Directory names are unchanged from the pre-collapse ADT-intent
    # notebooks, so existing output trees stay addressable.
    metastatic_platinum = by_key[("metastatic", "platinum")]
    assert metastatic_platinum["inputs_dir"].name == "prediction_inputs_adt_metastatic"
    assert metastatic_platinum["output_dir"].name == "local_runs_adt_metastatic"

    localized_nepc = by_key[("localized", "nepc")]
    assert localized_nepc["inputs_dir"].name == "prediction_inputs_adt_localized_nepc"
    assert localized_nepc["output_dir"].name == "local_runs_adt_localized_nepc"
    assert localized_nepc["restrict_to_mrns"] == (
        tmp_path / "mrn_lists" / "adt_localized_mrns.csv"
    )

    unrestricted = by_key[("all", "platinum")]
    assert unrestricted["inputs_dir"].name == "prediction_inputs_adt"
    assert unrestricted["output_dir"].name == "local_runs_adt"

    assert all(r["input_csv"].name == "longitudinal_prediction_data_adt.csv" for r in runs)
    assert {
        r["cohort"]: r["retrospective_stratification"] for r in runs
    } == {"all": False, "metastatic": True, "localized": True}
    assert {r["cohort"]: r["adt_intent"] for r in runs} == {
        "all": None,
        "metastatic": "METASTATIC",
        "localized": "LOCALIZED_ADJUVANT",
    }


def test_stage2_runs_collapse_to_one_run_per_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum", "nepc", "avpc", "avpc_nepc"),
        cohorts=("all", "metastatic", "localized"),
    )

    # Lab preprocessing is endpoint- and cohort-independent, so it must
    # see one run per treatment anchor rather than the full 12-way cross.
    stage2 = cp.stage2_runs(runs)
    assert [r["anchor"] for r in stage2] == ["adt"]


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
    # 2 strata x 4 endpoints: the preliminary count table now covers the
    # composite avpc_nepc endpoint alongside the three original ones.
    assert counts.height == 8
    assert set(counts["endpoint"].to_list()) == {
        "platinum",
        "nepc",
        "avpc",
        "avpc_nepc",
    }
    localized_platinum = counts.filter(
        (pl.col("stratum") == "localized") & (pl.col("endpoint") == "platinum")
    ).row(0, named=True)
    metastatic_avpc = counts.filter(
        (pl.col("stratum") == "metastatic") & (pl.col("endpoint") == "avpc")
    ).row(0, named=True)
    assert localized_platinum["n_incident_events"] == 0
    assert metastatic_avpc["n_incident_events"] == 1
