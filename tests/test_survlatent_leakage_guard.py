"""Finding 7: survlatent_ode.py has no assert_no_test_leakage call.

Every other arm (Cox via cox_aggregated.py:426, XGBoost via
multivariate_analysis.py's run_xgboost(), Dynamic-DeepHit via
helper.iter_stratified_folds's assert_disjoint_folds inside CV) calls one of
the two hard leakage guards in
survival_common/helper.py before touching test data. survlatent_ode.py's
``main()`` loads train/valid/test purely by the ``split`` column
(:479-481, ``load_split(df, "train"/"valid"/"test", ...)``) and never calls
either guard -- so if the person-period landmark builder upstream ever
produced an overlapping split (a real risk, per README invariant #1: "Models
use the aggregated table's split and never re-split"), this arm is the one
place in the codebase that would not catch it.

This test exercises ``load_split`` + the guard directly (the guard call this
finding adds cannot itself require the external survlatent_ode repo or torch,
so it is fully unit-testable without either). It documents both: (a) that
load_split by itself does nothing to prevent overlap, and (b) that wiring
assert_no_test_leakage in after the three load_split calls (the fix) does
catch it. Per the plan's Stage 7 order, this fix "cannot change results" --
it only adds a guard, it does not alter any metric computation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVLATENT_DIR = REPO_ROOT / "COMPASS" / "survival_analysis" / "multivariate_longitudinal"
if str(SURVLATENT_DIR) not in sys.path:
    sys.path.insert(0, str(SURVLATENT_DIR))

from survival_common.helper import assert_no_test_leakage
from survlatent_ode import load_split  # noqa: E402


def _person_period_df(overlap: bool) -> pd.DataFrame:
    rows = []
    # Clean train/valid MRNs.
    for mrn in range(1, 6):
        rows.append({"DFCI_MRN": mrn, "TIME": 0, "split": "train"})
    for mrn in range(6, 9):
        rows.append({"DFCI_MRN": mrn, "TIME": 0, "split": "valid"})
    # Test MRNs: overlap with train if requested (simulating an upstream
    # split-assignment bug the guard exists to catch).
    test_mrns = [1, 9, 10] if overlap else [9, 10, 11]
    for mrn in test_mrns:
        rows.append({"DFCI_MRN": mrn, "TIME": 0, "split": "test"})
    return pd.DataFrame(rows)


class TestLoadSplitAloneDoesNotGuardAgainstLeakage:
    def test_load_split_returns_overlapping_rows_silently(self):
        df = _person_period_df(overlap=True)
        train = load_split(df, "train", id_col="DFCI_MRN", time_col="TIME")
        test = load_split(df, "test", id_col="DFCI_MRN", time_col="TIME")
        # MRN 1 appears in both -- load_split has no opinion about this.
        assert set(train["DFCI_MRN"]) & set(test["DFCI_MRN"]) == {1}


class TestGuardWiredAfterLoadSplitCatchesOverlap:
    """This is the shape of the fix: call assert_no_test_leakage right after
    the three load_split calls in survlatent_ode.main() (around :479-481).
    """

    def test_guard_raises_on_train_test_overlap(self):
        df = _person_period_df(overlap=True)
        train = load_split(df, "train", id_col="DFCI_MRN", time_col="TIME")
        test = load_split(df, "test", id_col="DFCI_MRN", time_col="TIME")
        with pytest.raises(RuntimeError, match="leakage"):
            assert_no_test_leakage(
                test_mrns=test["DFCI_MRN"],
                train_mrns=train["DFCI_MRN"],
                context="survlatent_ode.main",
            )

    def test_guard_passes_on_clean_splits(self):
        df = _person_period_df(overlap=False)
        train = load_split(df, "train", id_col="DFCI_MRN", time_col="TIME")
        valid = load_split(df, "valid", id_col="DFCI_MRN", time_col="TIME")
        test = load_split(df, "test", id_col="DFCI_MRN", time_col="TIME")
        assert_no_test_leakage(
            test_mrns=test["DFCI_MRN"],
            train_mrns=train["DFCI_MRN"],
            context="survlatent_ode.main",
        )
        assert_no_test_leakage(
            test_mrns=test["DFCI_MRN"],
            train_mrns=valid["DFCI_MRN"],
            context="survlatent_ode.main",
        )
