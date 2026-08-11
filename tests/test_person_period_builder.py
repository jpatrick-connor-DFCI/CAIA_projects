"""Tests for the person-period ("wide", binned-time) builder in
survival_common/cohort.py: select_longitudinal_labs, fit_clip_bounds,
apply_clip_bounds, build_person_period_wide.

Grounded in a synthetic raw frame run through the real build_landmark_merged
+ build_pre_treatment_lab_long pipeline (anchor_col=None, COMPASS's clock),
so the TIME axis is exercised end-to-end rather than mocked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.cohort import (
    apply_clip_bounds,
    build_landmark_merged,
    build_person_period_wide,
    build_pre_treatment_lab_long,
    fit_clip_bounds,
    select_longitudinal_labs,
)

TIME_UNIT_DAYS = 7


def _patient_rows(
    mrn: int,
    *,
    lab_times: list[float],
    lab_values: list[float],
    lab_name: str = "PSA",
    platinum: int = 0,
    t_platinum: float = np.nan,
    death: int = 0,
    t_death: float = np.nan,
    t_last_contact: float = 400.0,
) -> list[dict]:
    rows = []
    t_death_val = t_death if death else t_last_contact
    for t_lab, val in zip(lab_times, lab_values):
        rows.append(
            {
                "DFCI_MRN": mrn,
                "AGE_AT_TREATMENTSTART": 65.0,
                "FIRST_RECORD_DATE": "2015-01-01",
                "FIRST_TREATMENT": 1,
                "t_first_treatment": 0.0,
                "t_platinum": t_platinum,
                "PLATINUM": platinum,
                "t_death": t_death_val,
                "DEATH": death,
                "t_last_contact": t_last_contact,
                "LAB_NAME": lab_name,
                "LAB_VALUE": val,
                "t_lab": t_lab,
            }
        )
    return rows


def _cohort(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _build(df: pd.DataFrame, *, landmark_day: int = 0, train_mrns=None, canonical_labs=None,
           min_coverage=0.5, max_labs=None, outlier_lo=0.005, outlier_hi=0.995):
    _, _, merged = build_landmark_merged(
        df,
        landmark_offset_days=landmark_day,
        anchor_col=None,
        require_first_treatment=False,
    )
    lab_long = build_pre_treatment_lab_long(
        df,
        cohort_index=merged.index,
        landmark_offset_days=landmark_day,
        anchor_col=None,
    )
    if train_mrns is None:
        train_mrns = set(merged.index)
    merged = merged.copy()
    merged["split"] = "train"
    wide, extras, selected_labs, bounds = build_person_period_wide(
        lab_long,
        merged,
        landmark_day=landmark_day,
        train_mrns=train_mrns,
        canonical_labs=canonical_labs,
        time_unit_days=TIME_UNIT_DAYS,
        min_coverage=min_coverage,
        max_labs=max_labs,
        outlier_lo=outlier_lo,
        outlier_hi=outlier_hi,
    )
    return wide, extras, selected_labs, bounds, merged


def test_time_axis_pure_offset_no_anchor():
    # Patient has labs at t_lab = -21, -14, -7 relative to time 0; at
    # landmark_day=0 the earliest lab (t_lab=-21) is 3 bins back
    # (floor(-21/7) = -3), so landmark_time = 3 and the earliest TIME = 0.
    # landmark_time itself is not a retained output column (patient_targets
    # recomputes it as TIME.max() per patient), so this test checks TIME
    # directly, matching that downstream convention.
    rows = _patient_rows(
        1, lab_times=[-21.0, -14.0, -7.0], lab_values=[1.0, 2.0, 3.0],
        platinum=1, t_platinum=100.0,
    )
    df = _cohort(rows)
    wide, extras, selected_labs, bounds, merged = _build(df)
    patient_rows = wide.loc[wide["DFCI_MRN"] == 1].sort_values("TIME")
    assert patient_rows["TIME"].min() == 0
    assert patient_rows["TIME"].max() == 3
    # bins: -21->-3(+3=0), -14->-2(+3=1), -7->-1(+3=2), plus landmark row at 3
    assert list(patient_rows["TIME"]) == [0, 1, 2, 3]


def test_landmark_row_is_synthetic_and_all_nan_for_labs():
    # variability > 1 requires more than one distinct LAB_VALUE among train
    # rows for a lab to be selectable, so this patient needs >=2 lab rows.
    rows = _patient_rows(
        1, lab_times=[-14.0, -7.0], lab_values=[5.0, 6.0], platinum=1, t_platinum=100.0,
    )
    df = _cohort(rows)
    wide, extras, selected_labs, bounds, merged = _build(df)
    patient_rows = wide.loc[wide["DFCI_MRN"] == 1]
    landmark_time = patient_rows["TIME"].max()
    landmark_row = patient_rows.loc[patient_rows["TIME"] == landmark_time]
    assert len(landmark_row) == 1
    for lab in selected_labs:
        assert landmark_row[lab].isna().all()


def test_patient_with_no_pre_landmark_labs_gets_landmark_only_row():
    # Patient 2 has no PSA labs at all before the landmark; patient 1 does,
    # so PSA is still selectable via coverage on patient 1.
    rows = _patient_rows(
        1, lab_times=[-21.0, -14.0, -7.0], lab_values=[1.0, 2.0, 3.0],
        platinum=1, t_platinum=100.0,
    ) + _patient_rows(
        2, lab_times=[], lab_values=[], platinum=0, t_platinum=np.nan,
        t_last_contact=200.0,
    )
    # patient 2 needs at least one row to survive make_outcome_df's cohort
    # filter (t_platinum/t_death notna); give it a death-derived fallback via
    # t_last_contact and no labs at all -- but build_feature_matrix/lab_long
    # requires the LAB_NAME/LAB_VALUE columns to exist, so add an out-of-window
    # (post landmark) lab that won't count as pre-landmark.
    rows.append(
        {
            "DFCI_MRN": 2, "AGE_AT_TREATMENTSTART": 70.0,
            "FIRST_RECORD_DATE": "2015-01-01", "FIRST_TREATMENT": 1,
            "t_first_treatment": 0.0, "t_platinum": np.nan, "PLATINUM": 0,
            "t_death": 200.0, "DEATH": 0, "t_last_contact": 200.0,
            "LAB_NAME": "PSA", "LAB_VALUE": 9.0, "t_lab": 50.0,
        }
    )
    df = _cohort(rows)
    wide, extras, selected_labs, bounds, merged = _build(
        df, min_coverage=0.0,
    )
    assert extras["n_patients_without_selected_labs"] == 1
    patient_2_rows = wide.loc[wide["DFCI_MRN"] == 2]
    assert len(patient_2_rows) == 1
    for lab in selected_labs:
        assert patient_2_rows[lab].isna().all()


def test_pre_event_rows_are_filtered():
    # Platinum event at t=10 with time_unit_days=7 -> event bin threshold is
    # small; a lab row landing at/after the event bin must be dropped.
    rows = _patient_rows(
        1,
        lab_times=[-21.0, -14.0, -7.0, -1.0],
        lab_values=[1.0, 2.0, 3.0, 4.0],
        platinum=1,
        t_platinum=3.0,
    )
    df = _cohort(rows)
    wide, extras, selected_labs, bounds, merged = _build(df)
    patient_rows = wide.loc[wide["DFCI_MRN"] == 1]
    event_time = patient_rows["t_platinum"].iloc[0]
    assert (patient_rows["TIME"] < event_time).all()


def test_one_row_per_patient_time():
    rows = _patient_rows(
        1, lab_times=[-21.0, -14.0, -7.0], lab_values=[1.0, 2.0, 3.0],
        platinum=1, t_platinum=100.0,
    )
    df = _cohort(rows)
    wide, extras, selected_labs, bounds, merged = _build(df)
    dupes = wide.duplicated(subset=["DFCI_MRN", "TIME"]).sum()
    assert dupes == 0


def test_max_landmark_time_from_train_only():
    # Patient 1 (test split, deep history) should not inflate max_landmark_time
    # since it's computed on TRAIN only.
    rows_train = _patient_rows(
        1, lab_times=[-14.0, -7.0], lab_values=[1.0, 1.5], platinum=1, t_platinum=100.0,
    )
    rows_test = _patient_rows(
        2, lab_times=[-70.0, -63.0, -56.0, -49.0, -42.0, -35.0, -28.0, -21.0, -14.0, -7.0],
        lab_values=list(range(10)), platinum=1, t_platinum=200.0,
    )
    df = _cohort(rows_train + rows_test)
    _, _, merged = build_landmark_merged(
        df, landmark_offset_days=0, anchor_col=None, require_first_treatment=False,
    )
    lab_long = build_pre_treatment_lab_long(
        df, cohort_index=merged.index, landmark_offset_days=0, anchor_col=None,
    )
    merged = merged.copy()
    merged["split"] = pd.Series({1: "train", 2: "test"})
    wide, extras, selected_labs, bounds = build_person_period_wide(
        lab_long, merged, landmark_day=0, train_mrns={1}, canonical_labs=None,
        time_unit_days=TIME_UNIT_DAYS, min_coverage=0.0, max_labs=None,
        outlier_lo=0.005, outlier_hi=0.995,
    )
    # patient 1 (train) has landmark_time=2; patient 2 (test) has a much
    # deeper history (landmark_time=10), which must NOT leak into the
    # TRAIN-only max_landmark_time.
    patient_2_landmark = wide.loc[wide["DFCI_MRN"] == 2, "TIME"].max()
    assert patient_2_landmark > extras["max_landmark_time"]
    assert extras["max_landmark_time"] == 2


class TestSelectLongitudinalLabs:
    def _labs(self):
        return pd.DataFrame(
            {
                "DFCI_MRN": [1, 1, 2, 2, 3],
                "LAB_NAME": ["PSA", "HGB", "PSA", "HGB", "PSA"],
                "LAB_VALUE": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )

    def test_canonical_labs_intersected_with_present(self):
        selected = select_longitudinal_labs(
            self._labs(), train_mrns={1, 2, 3}, min_coverage=0.5, max_labs=None,
            canonical_labs=["PSA", "ALK"],
        )
        assert selected == ["PSA"]

    def test_coverage_based_selection_train_only(self):
        selected = select_longitudinal_labs(
            self._labs(), train_mrns={1, 2, 3}, min_coverage=0.5, max_labs=None,
            canonical_labs=None,
        )
        # PSA covers all 3 train patients but each has only 1 distinct value
        # here (no variability across patients per-lab is fine; variability
        # is computed within LAB_NAME across all rows).
        assert "PSA" in selected

    def test_empty_train_raises(self):
        with pytest.raises(ValueError, match="Training set is empty"):
            select_longitudinal_labs(
                self._labs(), train_mrns=set(), min_coverage=0.5, max_labs=None,
                canonical_labs=None,
            )

    def test_no_canonical_labs_present_raises(self):
        with pytest.raises(ValueError, match="No canonical labs"):
            select_longitudinal_labs(
                self._labs(), train_mrns={1, 2, 3}, min_coverage=0.5, max_labs=None,
                canonical_labs=["ALK"],
            )


class TestClipBounds:
    def _agg(self):
        train_vals = list(np.linspace(0, 100, 20))
        return pd.DataFrame(
            {
                "DFCI_MRN": [1] * 20,
                "TIME": list(range(20)),
                "LAB_NAME": ["PSA"] * 20,
                "LAB_VALUE": train_vals,
            }
        )

    def test_bounds_unmoved_by_test_outlier(self):
        agg = self._agg()
        bounds = fit_clip_bounds(
            agg, train_mrns={1}, labs=["PSA"], q_lo=0.05, q_hi=0.95
        )
        lo, hi = bounds["PSA"]
        # An extreme value from a TEST patient (id=2) must not move the
        # TRAIN-fit bounds.
        test_row = pd.DataFrame(
            {"DFCI_MRN": [2], "TIME": [0], "LAB_NAME": ["PSA"], "LAB_VALUE": [1e6]}
        )
        bounds_with_outlier = fit_clip_bounds(
            pd.concat([agg, test_row], ignore_index=True),
            train_mrns={1}, labs=["PSA"], q_lo=0.05, q_hi=0.95,
        )
        assert bounds_with_outlier["PSA"] == (lo, hi)

    def test_apply_clip_bounds_clips_values(self):
        agg = self._agg()
        bounds = {"PSA": (10.0, 90.0)}
        clipped = apply_clip_bounds(agg, bounds)
        assert clipped["LAB_VALUE"].min() >= 10.0
        assert clipped["LAB_VALUE"].max() <= 90.0

    def test_apply_clip_bounds_noop_when_empty(self):
        agg = self._agg()
        result = apply_clip_bounds(agg, {})
        pd.testing.assert_frame_equal(result, agg)

    def test_fit_clip_bounds_skips_sparse_lab(self):
        agg = pd.DataFrame(
            {
                "DFCI_MRN": [1, 1],
                "TIME": [0, 1],
                "LAB_NAME": ["RARE", "RARE"],
                "LAB_VALUE": [1.0, 2.0],
            }
        )
        bounds = fit_clip_bounds(agg, train_mrns={1}, labs=["RARE"], q_lo=0.05, q_hi=0.95)
        assert "RARE" not in bounds
