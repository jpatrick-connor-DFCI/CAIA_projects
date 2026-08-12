"""Tests for the two hard leakage guards in survival_common/helper.py.

Covers the guards' basic contract (raise on overlap, pass when disjoint) and
the float-MRN dtype hazard: a merge that introduces a NaN into an otherwise-
integer MRN column silently upcasts the whole column to float in pandas, so
the *same* patient can be stringified two different ways (``"12345"`` vs.
``"12345.0"``) depending on which side of the merge produced the value.
``_normalize_mrn`` routes int-valued floats through ``int()`` first so both
forms collapse to the same key before the set comparison.
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
    """Proves the guard now catches leakage across an int/float MRN dtype
    split, instead of silently passing on it.
    """

    def test_float_and_int_mrn_now_collide_and_raise(self):
        # Same patient, MRN 12345, but the test-side value survived a merge
        # that introduced a NaN elsewhere in the MRN column and got upcast to
        # float64 by pandas, while the train-side value stayed a clean int.
        test_mrns = [12345.0, 67890]
        train_mrns = [12345, 11111]

        with pytest.raises(RuntimeError, match="leakage"):
            assert_no_test_leakage(test_mrns=test_mrns, train_mrns=train_mrns)

    def test_disjoint_folds_also_catches_the_dtype_split(self):
        with pytest.raises(RuntimeError, match="overlap"):
            assert_disjoint_folds(
                fold_train_mrns=[12345, 11111],
                fold_val_mrns=[12345.0, 67890],
                fold=1,
            )

    def test_genuinely_fractional_ids_still_compare_distinctly(self):
        # Non-integer-valued floats aren't a dtype-upcast artifact -- they
        # should not be coerced into colliding with an unrelated integer ID.
        assert_no_test_leakage(test_mrns=[12345.5], train_mrns=[12345])
