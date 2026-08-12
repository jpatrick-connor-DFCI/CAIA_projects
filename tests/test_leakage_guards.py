"""Tests for the two hard leakage guards in survival_common/helper.py.

Covers the guards' basic contract (raise on overlap, pass when disjoint) and,
critically, the float-MRN dtype hazard: helper.py:692-693 does
``{str(m) for m in ...}``, and ``str(12345.0) == "12345.0"`` while
``str(12345) == "12345"``. A merge that introduces a NaN into an otherwise-
integer MRN column silently upcasts the whole column to float in pandas, so
the *same* patient ends up stringified two different ways depending on which
side of the merge produced the value. That means a genuine train/test overlap
can silently fail to compare equal and the guard passes when it should raise.
"""

from __future__ import annotations

import pytest

from survival_common.helper import assert_disjoint_folds, assert_no_test_leakage


class TestAssertNoTestLeakageBaseline:
    def test_raises_on_overlap(self):
        with pytest.raises(RuntimeError, match="leakage"):
            assert_no_test_leakage(test_mrns=[1, 2, 3], train_mrns=[3, 4, 5])

    def test_passes_when_disjoint(self):
        assert_no_test_leakage(test_mrns=[1, 2, 3], train_mrns=[4, 5, 6])

    def test_passes_on_empty_sets(self):
        assert_no_test_leakage(test_mrns=[], train_mrns=[])


class TestAssertDisjointFoldsBaseline:
    def test_raises_on_overlap(self):
        with pytest.raises(RuntimeError, match="overlap"):
            assert_disjoint_folds(fold_train_mrns=[1, 2], fold_val_mrns=[2, 3], fold=1)

    def test_passes_when_disjoint(self):
        assert_disjoint_folds(fold_train_mrns=[1, 2], fold_val_mrns=[3, 4], fold=1)


class TestFloatMrnDtypeHazard:
    """Documents the guard's known blind spot -- not a fix, just a red pin.

    This test currently PASSES (i.e. the guard fails to catch the leak) and
    is intentionally written to demonstrate that, not to assert desired
    behavior. It is filed as evidence for the REPORT item (leakage-guard
    dtype hazard), not one of the three authorized FIX findings, so no source
    change accompanies it.
    """

    def test_float_and_int_mrn_do_not_collide_as_strings(self):
        # Same patient, MRN 12345, but the test-side value survived a merge
        # that introduced a NaN elsewhere in the MRN column and got upcast to
        # float64 by pandas, while the train-side value stayed a clean int.
        test_mrns = [12345.0, 67890]
        train_mrns = [12345, 11111]

        # The guard SHOULD raise (12345 appears on both sides) but does not,
        # because str(12345.0) == "12345.0" != "12345" == str(12345).
        assert_no_test_leakage(test_mrns=test_mrns, train_mrns=train_mrns)

        # Prove directly that this is a string-coercion collision, not a
        # coincidence of this particular guard's control flow.
        assert {str(m) for m in test_mrns} != {str(m) for m in [12345, 67890]}

    def test_float_and_int_mrn_do_collide_after_int_coercion(self):
        # The fix-shaped comparison (coerce through int/round-trip) DOES catch
        # the same overlap -- included so the report's proposed remediation
        # is falsifiable, not just asserted in prose.
        test_mrns = [12345.0, 67890]
        train_mrns = [12345, 11111]
        test_set = {str(int(float(m))) for m in test_mrns}
        train_set = {str(int(float(m))) for m in train_mrns}
        assert test_set & train_set == {"12345"}
