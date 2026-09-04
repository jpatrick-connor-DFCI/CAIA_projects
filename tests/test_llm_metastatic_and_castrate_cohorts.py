"""LLM-metastatic cohorts and the pre-ADT castrate exclusion axis."""

from __future__ import annotations

import polars as pl
import pytest

from COMPASS.survival_analysis import compass_pipeline as cp


def _write_cohort(root, mrns):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prostate_adt_survival_cohort_adt.csv"
    pl.DataFrame({"DFCI_MRN": list(mrns)}).write_csv(path)
    return path


def _write_met_labels(path, rows):
    """rows: (mrn, has_metastatic_disease, first_metastasis_date, label_source).

    Mirrors the met_diagnosis pipeline's LABEL_COLUMNS subset this reads.
    """
    pl.DataFrame(
        {
            "DFCI_MRN": [r[0] for r in rows],
            "has_metastatic_disease": [r[1] for r in rows],
            "first_metastasis_date": [r[2] for r in rows],
            "label_source": [r[3] for r in rows],
        },
        schema={
            "DFCI_MRN": pl.Int64,
            "has_metastatic_disease": pl.Boolean,
            "first_metastasis_date": pl.String,
            "label_source": pl.String,
        },
    ).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# LLM-metastatic strata
# ---------------------------------------------------------------------------

def test_the_boolean_label_splits_the_cohort_into_two_strata(tmp_path):
    _write_cohort(tmp_path, [1, 2, 3])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [
            (1, True, "2019-04-02", "adjudicated"),
            # Adjudicated negative: the note evidence existed and was rejected.
            (2, False, None, "adjudicated"),
            # Auto-negative: no trigger-bearing evidence anywhere in the record.
            # The upstream task materializes these, so it is a real label.
            (3, False, None, "auto_negative_no_evidence"),
        ],
    )

    out = cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    met = set(pl.read_csv(out["llm_metastatic"])["DFCI_MRN"].to_list())
    non = set(pl.read_csv(out["llm_nonmetastatic"])["DFCI_MRN"].to_list())
    assert met == {1}
    assert non == {2, 3}
    # The two strata partition the labelled cohort exactly.
    assert met | non == {1, 2, 3}
    assert met & non == set()


def test_the_metastasis_date_is_carried_into_the_audit_file(tmp_path):
    """Carrying the date is what makes a baseline-only variant cuttable later
    without re-running the LLM."""
    _write_cohort(tmp_path, [1])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [(1, True, "2019-04-02", "adjudicated")],
    )

    out = cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    audit = pl.read_csv(out["labels"])
    assert audit["first_metastasis_date"].to_list() == ["2019-04-02"]
    assert audit["label_source"].to_list() == ["adjudicated"]


def test_patients_outside_the_adt_cohort_are_ignored(tmp_path):
    _write_cohort(tmp_path, [1])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [
            (1, False, None, "adjudicated"),
            (999, True, "2020-01-01", "adjudicated"),
        ],
    )

    out = cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    assert set(pl.read_csv(out["labels"])["DFCI_MRN"].to_list()) == {1}
    assert pl.read_csv(out["llm_metastatic"]).height == 0


def test_an_unlabelled_cohort_patient_is_dropped_not_called_negative(tmp_path, capsys):
    """The upstream task writes its own auto-negatives, so a missing patient
    means the labels file does not cover this cohort -- NOT that they are
    negative. Folding them into llm_nonmetastatic would assert a negative the
    LLM never made, so they are dropped from both strata instead."""
    _write_cohort(tmp_path, [1, 2])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [(1, True, "2019-04-02", "adjudicated")],
    )

    out = cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    met = set(pl.read_csv(out["llm_metastatic"])["DFCI_MRN"].to_list())
    non = set(pl.read_csv(out["llm_nonmetastatic"])["DFCI_MRN"].to_list())
    assert met == {1}
    # Patient 2 is unlabelled, so they appear in NEITHER stratum.
    assert non == set()


def test_the_dropped_count_and_coverage_fraction_are_printed(tmp_path, capsys):
    """Dropping is the default, so the shortfall has to stay visible: the LLM
    strata cover fewer patients than the ADT-intent strata, and their Ns are
    therefore not directly comparable."""
    _write_cohort(tmp_path, [1, 2, 3, 4])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [
            (1, True, "2019-04-02", "adjudicated"),
            (2, False, None, "adjudicated"),
            (3, False, None, "auto_negative_no_evidence"),
        ],
    )

    cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "1 of 4" in out
    assert "25.0%" in out


def test_a_coverage_gap_can_be_made_fatal(tmp_path):
    _write_cohort(tmp_path, [1, 2])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [(1, True, "2019-04-02", "adjudicated")],
    )

    with pytest.raises(ValueError, match="mrns-limited"):
        cp.build_llm_met_mrn_lists(
            met_labels_path=labels,
            data_root=tmp_path,
            require_full_cohort_coverage=True,
        )


def test_a_null_verdict_is_not_treated_as_a_negative(tmp_path, capsys):
    _write_cohort(tmp_path, [1, 2])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [
            (1, True, "2019-04-02", "adjudicated"),
            (2, None, None, "adjudicated"),
        ],
    )

    out = cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    assert set(pl.read_csv(out["llm_metastatic"])["DFCI_MRN"].to_list()) == {1}
    assert pl.read_csv(out["llm_nonmetastatic"]).height == 0
    assert "null has_metastatic_disease" in capsys.readouterr().out


def test_a_duplicated_patient_collapses_to_one_positive_row(tmp_path, capsys):
    """One row per patient upstream, but a resumed run could append. Row order
    must not decide the split."""
    _write_cohort(tmp_path, [1])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [
            (1, False, None, "auto_negative_no_evidence"),
            (1, True, "2019-04-02", "adjudicated"),
        ],
    )

    out = cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    assert set(pl.read_csv(out["llm_metastatic"])["DFCI_MRN"].to_list()) == {1}
    assert pl.read_csv(out["llm_nonmetastatic"]).height == 0
    assert pl.read_csv(out["labels"]).height == 1
    assert "duplicate MRN" in capsys.readouterr().out


def test_an_empty_stratum_still_writes_both_lists(tmp_path, capsys):
    """The strata are a partition with no unclassifiable bucket, so one side
    being empty is a real result. Failing would also withhold the other side."""
    _write_cohort(tmp_path, [1])
    labels = _write_met_labels(
        tmp_path / "met_dx_labels.parquet",
        [(1, True, "2019-04-02", "adjudicated")],
    )

    out = cp.build_llm_met_mrn_lists(met_labels_path=labels, data_root=tmp_path)

    assert set(pl.read_csv(out["llm_metastatic"])["DFCI_MRN"].to_list()) == {1}
    assert out["llm_nonmetastatic"].exists()
    assert pl.read_csv(out["llm_nonmetastatic"]).height == 0
    assert "WARNING" in capsys.readouterr().out


def test_a_missing_labels_file_names_the_upstream_pipeline(tmp_path):
    _write_cohort(tmp_path, [1])
    with pytest.raises(FileNotFoundError, match="build_met_dx_labels"):
        cp.build_llm_met_mrn_lists(
            met_labels_path=tmp_path / "absent.parquet", data_root=tmp_path
        )


def test_a_labels_file_with_the_wrong_schema_names_the_missing_column(tmp_path):
    _write_cohort(tmp_path, [1])
    path = tmp_path / "met_dx_labels.parquet"
    pl.DataFrame({"DFCI_MRN": [1], "stage_group": ["IV"]}).write_parquet(path)

    with pytest.raises(ValueError, match="has_metastatic_disease"):
        cp.build_llm_met_mrn_lists(met_labels_path=path, data_root=tmp_path)


# ---------------------------------------------------------------------------
# Pre-ADT castrate exclusion
# ---------------------------------------------------------------------------

def _write_longitudinal(root, rows):
    """rows: (mrn, lab_name, lab_value, t_lab)."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "longitudinal_prediction_data_adt.csv"
    pl.DataFrame(
        {
            "DFCI_MRN": [r[0] for r in rows],
            "LAB_NAME": [r[1] for r in rows],
            "LAB_VALUE": [r[2] for r in rows],
            "t_lab": [r[3] for r in rows],
        }
    ).write_csv(path)
    return path


def test_only_pre_anchor_castrate_testosterone_is_excluded(tmp_path):
    data = _write_longitudinal(
        tmp_path,
        [
            (1, "Testosterone", 20.0, -30.0),   # castrate BEFORE ADT -> excluded
            (2, "Testosterone", 400.0, -30.0),  # eugonadal before ADT -> kept
            (3, "Testosterone", 20.0, 30.0),    # castrate AFTER ADT: the expected
                                                # treatment effect -> kept
            (4, "PSA", 0.1, -30.0),             # a different lab -> kept
            (5, "Testosterone", 20.0, 0.0),     # same day as the anchor may already
                                                # reflect the first dose -> kept
        ],
    )

    path = cp.build_pre_adt_castrate_mrn_list(
        longitudinal_csv=data, data_root=tmp_path
    )

    assert set(pl.read_csv(path)["DFCI_MRN"].to_list()) == {1}


def test_a_single_castrate_result_excludes_despite_eugonadal_ones(tmp_path):
    data = _write_longitudinal(
        tmp_path,
        [
            (1, "Testosterone", 500.0, -90.0),
            (1, "Testosterone", 10.0, -30.0),
            (1, "Testosterone", 450.0, -10.0),
        ],
    )

    path = cp.build_pre_adt_castrate_mrn_list(
        longitudinal_csv=data, data_root=tmp_path
    )

    flagged = pl.read_csv(path)
    assert flagged["DFCI_MRN"].to_list() == [1]
    assert flagged["MIN_PRE_ADT_TESTOSTERONE"].to_list() == [10.0]
    assert flagged["N_CASTRATE_RESULTS_PRE_ADT"].to_list() == [1]


def test_the_castrate_threshold_is_strict(tmp_path):
    data = _write_longitudinal(
        tmp_path,
        [
            (1, "Testosterone", 49.9, -10.0),
            (2, "Testosterone", 50.0, -10.0),
        ],
    )

    path = cp.build_pre_adt_castrate_mrn_list(
        longitudinal_csv=data, data_root=tmp_path
    )

    assert set(pl.read_csv(path)["DFCI_MRN"].to_list()) == {1}


# ---------------------------------------------------------------------------
# The run cross
# ---------------------------------------------------------------------------

def test_exclusion_is_orthogonal_to_cohort_and_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum", "nepc", "avpc"),
        cohorts=cp.DEFAULT_COHORTS,
        exclusions=cp.DEFAULT_EXCLUSIONS,
    )

    assert len(runs) == 5 * 3 * 2
    assert {(r["cohort"], r["endpoint"], r["exclusion"]) for r in runs} == {
        (cohort, endpoint, exclusion)
        for cohort in cp.DEFAULT_COHORTS
        for endpoint in ("platinum", "nepc", "avpc")
        for exclusion in cp.DEFAULT_EXCLUSIONS
    }
    # Every cell needs its own tree, or one run overwrites another's outputs.
    assert len({r["output_dir"] for r in runs}) == len(runs)
    assert len({r["inputs_dir"] for r in runs}) == len(runs)


def test_excluded_runs_carry_the_exclusion_list_and_keep_their_cohort_list(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum",),
        cohorts=("llm_metastatic",),
        exclusions=("none", "pre_adt_castrate"),
    )
    by_exclusion = {r["exclusion"]: r for r in runs}

    # The exclusion is a SEPARATE list: it must not displace the cohort's own
    # restrict list, or the excluded arm would silently run on every patient.
    cohort_list = tmp_path / "mrn_lists" / "adt_llm_metastatic_mrns.csv"
    assert by_exclusion["none"]["restrict_to_mrns"] == cohort_list
    assert by_exclusion["pre_adt_castrate"]["restrict_to_mrns"] == cohort_list

    assert by_exclusion["none"]["exclude_mrns"] is None
    assert by_exclusion["pre_adt_castrate"]["exclude_mrns"] == (
        tmp_path / "mrn_lists" / "pre_adt_castrate_mrns.csv"
    )

    assert by_exclusion["pre_adt_castrate"]["label"] == "adt_llm_metastatic_noprecastrate"
    assert (
        by_exclusion["pre_adt_castrate"]["output_dir"].name
        == "local_runs_adt_llm_metastatic_noprecastrate"
    )


def test_legacy_cross_labels_are_unchanged(monkeypatch, tmp_path):
    """Existing output trees stay addressable: the exclusion axis contributes an
    empty suffix at "none", so the pre-exclusion cross is reproduced exactly."""
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum", "nepc"),
        cohorts=("all", "metastatic", "localized"),
        exclusions=("none",),
    )

    assert {r["label"] for r in runs} == {"adt", "adt_metastatic", "adt_localized"}
    assert all(r["exclude_mrns"] is None for r in runs)


def test_stage2_never_runs_on_a_restricted_cohort(monkeypatch, tmp_path):
    """preprocess_labs() passes restrict_to_mrns as --survival-cohort-csv, so a
    restricted run would build the anchor's shared lab table from that cohort
    alone."""
    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum",),
        cohorts=cp.DEFAULT_COHORTS,
        exclusions=cp.DEFAULT_EXCLUSIONS,
    )

    stage2 = cp.stage2_runs(runs)
    assert len(stage2) == 1
    assert stage2[0]["cohort"] == "all"
    assert stage2[0]["exclusion"] == "none"
    assert stage2[0]["exclude_mrns"] is None

    # Order must not decide it.
    assert cp.stage2_runs(list(reversed(runs)))[0]["cohort"] == "all"

    # And with no unrestricted run present, it must fail loudly rather than
    # silently preprocessing on a subset.
    restricted_only = [r for r in runs if r["cohort"] != "all"]
    with pytest.raises(ValueError, match="Stage 2"):
        cp.stage2_runs(restricted_only)


def test_the_r_figure_pipeline_knows_every_python_run_label(monkeypatch, tmp_path):
    """The R pipeline enumerates run labels independently (SUPPORTED_COHORTS) and
    derives the arm from them. If the two registries drift, a cohort builds in
    Python and then dies in the figure pass."""
    import re

    monkeypatch.setattr(cp, "_PROFILE_OUTPUT_ROOT", tmp_path)

    r_source = (
        cp.SURVIVAL_DIR / "COMPASS_generate_figures_pipeline.R"
    ).read_text()

    def r_vector(name):
        body = re.search(rf"^{name} <- c\((.*?)\)$", r_source, re.S | re.M).group(1)
        return re.findall(r'"([^"]*)"', body)

    arms = r_vector("COHORT_ARMS")
    subsets = r_vector("COHORT_SUBSET_SUFFIXES")
    exclusions = r_vector("COHORT_EXCLUSION_SUFFIXES")
    supported = {
        f"{arm}{subset}{exclusion}"
        for arm in arms
        for subset in subsets
        for exclusion in exclusions
    }

    runs = cp.make_endpoint_runs(
        ["adt"],
        endpoints=("platinum",),
        cohorts=cp.DEFAULT_COHORTS,
        exclusions=cp.DEFAULT_EXCLUSIONS,
    )
    missing = {r["label"] for r in runs} - supported
    assert not missing, f"R SUPPORTED_COHORTS is missing run labels: {sorted(missing)}"
