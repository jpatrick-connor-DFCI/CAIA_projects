"""Localized-vs-metastatic comparison module.

This replaces the static source-text checks that guarded notebook 10. The
analysis is now a module, so the tests exercise the real computation on
synthetic result trees instead of grepping notebook JSON.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import adt_intent_comparison as aic
from COMPASS.survival_analysis import compass_pipeline as cp
from COMPASS.survival_analysis import cox_aggregated as ca


def _tree(root, cohort, endpoint, kind):
    label = f"adt{cp.COHORT_SPECS[cohort]['label_suffix']}"
    prefix = "prediction_inputs" if kind == "inputs" else "local_runs"
    return (
        root / "survival_analysis"
        / f"{prefix}_{label}{cp.endpoint_suffix(endpoint)}"
    )


def _write_inputs(root, cohort, endpoint, landmark, *, mrns, events, durations=None):
    spec = ca.ENDPOINTS[endpoint]
    path = _tree(root, cohort, endpoint, "inputs")
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "DFCI_MRN": list(mrns),
        spec["event_col"]: list(events),
        spec["duration_col"]: list(durations if durations is not None else [365] * len(mrns)),
    })
    frame.to_csv(path / f"aggregated_landmark{landmark}.csv", index=False)


def _write_univariate(root, cohort, endpoint, landmark, *, features, coefs, ses):
    path = _tree(root, cohort, endpoint, "results") / "cox" / f"landmark_{landmark}" / "both"
    path.mkdir(parents=True, exist_ok=True)
    coefs = np.asarray(coefs, dtype=float)
    ses = np.asarray(ses, dtype=float)
    pd.DataFrame({
        "endpoint": endpoint,
        "feature": list(features),
        "lab_name": [f.split("_")[0] for f in features],
        "coef_feature": coefs,
        "hazard_ratio_per_sd": np.exp(coefs),
        "ci_lower": np.exp(coefs - 1.96 * ses),
        "ci_upper": np.exp(coefs + 1.96 * ses),
        "p_value": np.full(len(coefs), 0.01),
        "q_value": np.full(len(coefs), 0.02),
    }).to_csv(path / "cox_agg_univariate_nobs_adjusted.csv", index=False)


def test_cohort_counts_flag_sparse_event_cells(tmp_path):
    # 2 events is far below the sparse threshold; 100 is comfortably above it.
    _write_inputs(tmp_path, "localized", "platinum", 0,
                  mrns=range(40), events=[1] * 2 + [0] * 38)
    _write_inputs(tmp_path, "metastatic", "platinum", 0,
                  mrns=range(1000, 1200), events=[1] * 100 + [0] * 100)

    counts, cohort_ids = aic.collect_cohort_counts(
        endpoints=("platinum",), landmarks=(0,), data_root=tmp_path
    )

    by_cohort = counts.set_index("cohort")
    assert by_cohort.loc["localized", "n_events"] == 2
    assert bool(by_cohort.loc["localized", "sparse_events"]) is True
    assert by_cohort.loc["metastatic", "n_events"] == 100
    assert bool(by_cohort.loc["metastatic", "sparse_events"]) is False
    assert set(counts["status"]) == {"ok"}
    assert set(cohort_ids) == {
        ("localized", "platinum", 0),
        ("metastatic", "platinum", 0),
    }


def test_missing_input_tree_is_recorded_not_raised(tmp_path):
    _write_inputs(tmp_path, "localized", "platinum", 0, mrns=range(10), events=[0] * 10)

    counts, _ = aic.collect_cohort_counts(
        endpoints=("platinum",), landmarks=(0,), data_root=tmp_path
    )

    metastatic = counts.loc[counts["cohort"].eq("metastatic")].iloc[0]
    assert metastatic["status"] == "missing"


def test_overlapping_cohorts_are_rejected(tmp_path):
    # The two intent strata are built from disjoint MRN lists; an overlap
    # means the stratification broke upstream, so the build must not proceed.
    shared = range(50)
    _write_inputs(tmp_path, "localized", "platinum", 0, mrns=shared, events=[0] * 50)
    _write_inputs(tmp_path, "metastatic", "platinum", 0, mrns=shared, events=[1] * 50)

    with pytest.raises(AssertionError, match="cohorts overlap"):
        aic.build_comparison(
            endpoints=("platinum",), landmarks=(0,),
            data_root=tmp_path, output_dir=tmp_path / "out",
        )


def test_heterogeneity_matches_a_hand_computed_wald_test(tmp_path):
    # One feature, known coefficients and standard errors, so the z statistic
    # and its two-sided p-value can be checked in closed form.
    coef_loc, se_loc = 0.20, 0.10
    coef_met, se_met = 0.80, 0.10
    _write_univariate(tmp_path, "localized", "platinum", 0,
                      features=["LAB1_mean"], coefs=[coef_loc], ses=[se_loc])
    _write_univariate(tmp_path, "metastatic", "platinum", 0,
                      features=["LAB1_mean"], coefs=[coef_met], ses=[se_met])

    comparison, summary = aic.compare_univariate(
        endpoints=("platinum",), landmarks=(0,), data_root=tmp_path
    )

    row = comparison.iloc[0]
    delta = coef_met - coef_loc
    z = delta / math.sqrt(se_loc**2 + se_met**2)
    assert row["delta_log_hr_met_minus_loc"] == pytest.approx(delta, abs=1e-6)
    assert row["hr_ratio_met_vs_loc"] == pytest.approx(math.exp(delta), rel=1e-6)
    assert row["z_heterogeneity"] == pytest.approx(z, rel=1e-3)
    assert row["p_heterogeneity"] == pytest.approx(
        math.erfc(abs(z) / math.sqrt(2)), rel=1e-3
    )
    # A single test per endpoint/landmark: BH leaves it unchanged.
    assert row["q_heterogeneity"] == pytest.approx(row["p_heterogeneity"], rel=1e-6)
    assert bool(row["same_direction"]) is True
    assert summary.iloc[0]["n_features_both"] == 1


def test_heterogeneity_is_bh_adjusted_within_each_endpoint_landmark(tmp_path):
    features = [f"LAB{i}_mean" for i in range(10)]
    _write_univariate(tmp_path, "localized", "platinum", 0,
                      features=features, coefs=[0.0] * 10, ses=[0.1] * 10)
    _write_univariate(tmp_path, "metastatic", "platinum", 0,
                      features=features,
                      coefs=np.linspace(0.0, 0.9, 10), ses=[0.1] * 10)

    comparison, _ = aic.compare_univariate(
        endpoints=("platinum",), landmarks=(0,), data_root=tmp_path
    )

    ranked = comparison.sort_values("p_heterogeneity")
    p = ranked["p_heterogeneity"].to_numpy()
    q = ranked["q_heterogeneity"].to_numpy()
    expected = np.minimum.accumulate(
        (p * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1].clip(max=1.0)
    assert q == pytest.approx(expected, rel=1e-9)
    assert (q >= p - 1e-12).all()


def test_features_present_in_only_one_cohort_survive_the_join(tmp_path):
    _write_univariate(tmp_path, "localized", "platinum", 0,
                      features=["SHARED_mean", "LOC_ONLY_mean"],
                      coefs=[0.1, 0.2], ses=[0.1, 0.1])
    _write_univariate(tmp_path, "metastatic", "platinum", 0,
                      features=["SHARED_mean", "MET_ONLY_mean"],
                      coefs=[0.3, 0.4], ses=[0.1, 0.1])

    comparison, summary = aic.compare_univariate(
        endpoints=("platinum",), landmarks=(0,), data_root=tmp_path
    )

    assert set(comparison["feature"]) == {"SHARED_mean", "LOC_ONLY_mean", "MET_ONLY_mean"}
    # Only the shared feature can carry a heterogeneity test.
    tested = comparison.dropna(subset=["z_heterogeneity"])
    assert set(tested["feature"]) == {"SHARED_mean"}
    assert summary.iloc[0]["n_features_both"] == 1


def test_build_comparison_writes_every_table(tmp_path):
    _write_inputs(tmp_path, "localized", "platinum", 0,
                  mrns=range(30), events=[1] * 5 + [0] * 25)
    _write_inputs(tmp_path, "metastatic", "platinum", 0,
                  mrns=range(1000, 1060), events=[1] * 20 + [0] * 40)
    _write_univariate(tmp_path, "localized", "platinum", 0,
                      features=["LAB1_mean"], coefs=[0.1], ses=[0.1])
    _write_univariate(tmp_path, "metastatic", "platinum", 0,
                      features=["LAB1_mean"], coefs=[0.5], ses=[0.1])

    written = aic.build_comparison(
        endpoints=("platinum",), landmarks=(0,),
        data_root=tmp_path, output_dir=tmp_path / "out",
    )

    assert set(written) == {
        "cohort_counts", "overlap", "association",
        "association_summary", "performance",
    }
    for path in written.values():
        assert path.exists(), path
    assert written["association"].name == aic.ASSOCIATION_FILENAME
    assert written["performance"].name == aic.PERFORMANCE_FILENAME

    association = pd.read_csv(written["association"])
    for column in (
        "delta_log_hr_met_minus_loc", "hr_ratio_met_vs_loc",
        "z_heterogeneity", "p_heterogeneity", "q_heterogeneity",
    ):
        assert column in association.columns

    # An empty performance frame is still written: the R side distinguishes
    # "no comparable runs" from "this stage never ran" by the file's presence.
    assert written["performance"].exists()


def test_module_covers_every_pipeline_endpoint():
    # The retired notebook hard-coded three endpoints and silently omitted
    # all modeled endpoints; the module takes its default from the pipeline registry.
    assert aic.DEFAULT_ENDPOINTS == tuple(ca.ENDPOINTS)
    assert aic.DEFAULT_ENDPOINTS == ("platinum", "nepc", "avpc")


def test_comparison_module_is_read_only():
    source = (
        aic.__file__
        and __import__("pathlib").Path(aic.__file__).read_text()
    )
    for forbidden in (
        "build_prediction_inputs(",
        "build_adt_intent_mrn_lists(",
        "run_univariate(",
        "run_multivariate(",
        "preprocess_labs(",
        "compile_cohort(",
    ):
        assert forbidden not in source
