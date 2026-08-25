"""Selection-metric consistency test for Finding 4.

``tune_multivariable_model``'s CV loop (cox_models.py:816-834) previously
selected the winning hyperparameter setting by sorting on ``cv_mean`` --
the mean of ``c_index_val``, lifelines' plain ``concordance_index`` on the
uncapped, unweighted duration/event pair (cox_engine.py:539-551, no
``auc_max_time_units`` cap, no IPCW weighting). The metric ultimately
reported for the winning model is IPCW AUC(t) (``mean_auc_t_cv_mean``),
horizon-capped and censoring-weighted -- a materially different statistic.
A candidate could win on uncapped Harrell C while another candidate on the
same grid scores higher on the metric that is actually published.

The fix makes ``mean_auc_t_cv_mean`` the primary sort key whenever at least
one candidate has it estimable across every fold, falling back to
``cv_mean`` only when no candidate does (e.g. all folds abstain via the
IPCW coverage guard). This test constructs a grid where the two metrics
disagree on the winner and asserts selection now follows the reported
metric.
"""

from __future__ import annotations

import pandas as pd
import pytest

from survival_common.cox_models import select_best_cv_row


def _select_best_row(cv_df: pd.DataFrame, n_folds: int = 5) -> dict:
    """The real selection rule, imported rather than reproduced.

    This test used to reimplement the rule verbatim, which meant a change to
    cox_models.py would make the two diverge silently instead of failing here.
    """
    return select_best_cv_row(cv_df, n_folds=n_folds)


def _disagreeing_grid() -> pd.DataFrame:
    # Candidate A has the best uncapped Harrell C but the worse IPCW mean;
    # candidate B is the reverse -- a genuine, not-tied disagreement between
    # the two metrics.
    return pd.DataFrame(
        {
            "penalizer": [0.01, 0.1],
            "l1_ratio": [0.5, 0.5],
            "cv_mean": [0.75, 0.70],  # A wins on uncapped Harrell C
            "mean_auc_t_cv_mean": [0.60, 0.68],  # B wins on the reported metric
            "n_valid_auc_t_folds": [5, 5],
            "n_valid_folds": [5, 5],
            "all_folds_valid": [True, True],
        }
    )


class TestSelectionFollowsReportedMetric:
    def test_selection_prefers_higher_mean_auc_t_over_higher_uncapped_c_index(self):
        cv_df = _disagreeing_grid()
        best_row = _select_best_row(cv_df)
        # Post-fix: B wins, since it has the higher IPCW mean_auc_t (the
        # reported metric), even though A has the higher uncapped cv_mean.
        assert best_row["penalizer"] == pytest.approx(0.1)
        assert best_row["mean_auc_t_cv_mean"] == pytest.approx(0.68)

    def test_falls_back_to_cv_mean_when_no_candidate_has_full_auc_t_coverage(self):
        cv_df = _disagreeing_grid()
        cv_df["n_valid_auc_t_folds"] = [3, 2]  # neither candidate clears n_folds
        best_row = _select_best_row(cv_df)
        # No candidate has an estimable mean_auc_t across every fold, so
        # selection falls back to cv_mean -- A wins here instead.
        assert best_row["penalizer"] == pytest.approx(0.01)
        assert best_row["cv_mean"] == pytest.approx(0.75)
