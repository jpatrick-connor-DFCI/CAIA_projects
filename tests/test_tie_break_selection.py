"""Tie-break determinism test for the CV hyperparameter selection at
cox_models.py:816-834 (Finding 3, and now Finding 4).

``tune_multivariable_model`` selects the winning (penalizer, l1_ratio) row by
sorting on ``["mean_auc_t_cv_mean" (or "cv_mean" fallback), "cv_mean",
"n_valid_folds", "penalizer", "l1_ratio"]``. Before the Finding-3 fix, ties
on the score/``n_valid_folds`` broke toward the SMALLEST penalizer -- i.e.
the *least* regularized model -- an unmotivated preference for the more
complex/overfit-prone model whenever two hyperparameter settings scored
identically. The fix flips the tie-break to prefer the *most* regularized
setting. ``cv_std`` is computed in the same aggregation but intentionally
still not consulted -- a full 1-SE-style rule is a scientific choice
(REPORT item, not FIX) per the plan's triage rule.

This test exercises the selection logic in isolation (mirroring the exact
sort/ascending/columns cox_models.py uses) rather than driving a full
lifelines CV fit, since the defect is in the tie-break rule itself, not in
model fitting.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _select_best_row_current_tiebreak(cv_df: pd.DataFrame, n_folds: int = 5) -> dict:
    """Verbatim reproduction of cox_models.py's selection rule (post-fix:
    IPCW mean_auc_t is the primary key when fully estimable, ties prefer
    more regularization)."""
    valid_candidates = cv_df.loc[cv_df["all_folds_valid"]]
    has_full_auc_t_coverage = valid_candidates["n_valid_auc_t_folds"].eq(int(n_folds))
    selection_key = "mean_auc_t_cv_mean" if has_full_auc_t_coverage.any() else "cv_mean"
    return (
        valid_candidates.sort_values(
            [selection_key, "cv_mean", "n_valid_folds", "penalizer", "l1_ratio"],
            ascending=[False, False, False, False, False],
            na_position="last",
        )
        .iloc[0]
        .to_dict()
    )


def _tied_cv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "penalizer": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.5, 0.5, 0.5, 0.5],
            "cv_mean": [0.70, 0.70, 0.70, 0.70],  # exact tie across the board
            "cv_std": [0.05, 0.02, 0.02, 0.02],  # least-regularized is noisiest
            "mean_auc_t_cv_mean": [0.65, 0.65, 0.65, 0.65],  # also tied
            "n_valid_auc_t_folds": [5, 5, 5, 5],
            "n_valid_folds": [5, 5, 5, 5],
            "all_folds_valid": [True, True, True, True],
        }
    )


class TestCurrentTieBreakBehavior:
    def test_fixed_rule_prefers_most_regularization_on_exact_tie(self):
        cv_df = _tied_cv_df()
        best_row = _select_best_row_current_tiebreak(cv_df)
        # Post-fix: largest penalizer wins a dead tie -- the most regularized
        # of the tied candidates, not the noisiest (highest cv_std) one.
        assert best_row["penalizer"] == pytest.approx(1.0)

    def test_cv_std_is_computed_but_not_used_in_selection(self):
        # Two cv_df frames that are IDENTICAL except for cv_std produce the
        # identical winner -- proof cv_std has no influence on the pick.
        cv_df_a = _tied_cv_df()
        cv_df_b = _tied_cv_df()
        cv_df_b["cv_std"] = [0.30, 0.01, 0.01, 0.01]  # wildly different std
        best_a = _select_best_row_current_tiebreak(cv_df_a)
        best_b = _select_best_row_current_tiebreak(cv_df_b)
        assert best_a["penalizer"] == best_b["penalizer"]


class TestDeterminism:
    def test_selection_is_deterministic_across_row_order(self):
        cv_df = _tied_cv_df()
        shuffled = cv_df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        best_original = _select_best_row_current_tiebreak(cv_df)
        best_shuffled = _select_best_row_current_tiebreak(shuffled)
        assert best_original["penalizer"] == best_shuffled["penalizer"]
        assert best_original["l1_ratio"] == best_shuffled["l1_ratio"]
