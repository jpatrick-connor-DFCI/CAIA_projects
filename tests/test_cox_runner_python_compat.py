"""Compatibility checks for shared Cox runner filename handling."""

from survival_common.cox_runners import _csv_stem


def test_csv_stem_removes_only_a_trailing_csv_suffix():
    assert _csv_stem("cox_agg_horizon_grid.csv") == "cox_agg_horizon_grid"
    assert _csv_stem("report.csv.backup") == "report.csv.backup"
    assert _csv_stem("report") == "report"
