"""Static wiring checks for the localized-vs-metastatic comparison notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "COMPASS"
    / "survival_analysis"
    / "10_adt_intent_comparison.ipynb"
)


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def test_comparison_notebook_covers_both_strata_and_three_endpoints():
    source = _source()
    assert '"localized"' in source
    assert '"metastatic"' in source
    assert 'ENDPOINTS = ("platinum", "nepc", "avpc")' in source


def test_comparison_notebook_is_read_only():
    source = _source()
    for forbidden in (
        "build_prediction_inputs(",
        "build_adt_intent_mrn_lists(",
        "run_univariate(",
        "run_multivariate(",
    ):
        assert forbidden not in source


def test_comparison_includes_effect_heterogeneity_and_performance_deltas():
    source = _source()
    assert "p_heterogeneity" in source
    assert "q_heterogeneity" in source
    assert "hr_ratio_met_vs_loc" in source
    assert "delta_c_index_met_minus_loc" not in source  # generated dynamically
    assert 'f"delta_{metric}_met_minus_loc"' in source

