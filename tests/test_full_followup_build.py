"""The full-follow-up longitudinal build (--longitudinal-full-followup).

Two properties matter, and they pull in opposite directions:

1. The full-follow-up frame must actually *retain* post-landmark observations --
   otherwise the dynamic arm has nothing extra to learn from and silently
   degenerates to the landmark model.
2. Turning the flag on must not perturb the landmark frame by so much as a row.
   The landmark files are what every published number came from, and the two
   builds share a code path.

The landmark filter and the pre-event filter are different filters; only the
first is relaxed. A lab drawn after platinum start must never appear in either
frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.cohort import (
    build_landmark_merged,
    build_person_period_wide,
    build_pre_treatment_lab_long,
)

ID_COL = "DFCI_MRN"
LANDMARK = 90
TIME_UNIT_DAYS = 7


@pytest.fixture
def raw() -> pd.DataFrame:
    """Labs straddling the landmark; patient 1 starts platinum at day 300."""
    platinum = {1: (1, 300.0), 2: (0, np.nan), 3: (0, np.nan)}
    rows = []
    for mrn in (1, 2, 3):
        plat, t_plat = platinum[mrn]
        for t in (-60.0, -10.0, 30.0, 120.0, 200.0, 400.0):
            for lab_name, value in (("PSA", 5.0 + t / 100), ("ALP", 90.0)):
                rows.append(
                    {
                        ID_COL: mrn,
                        "AGE_AT_TREATMENTSTART": 65.0,
                        "FIRST_RECORD_DATE": "2015-01-01",
                        "FIRST_TREATMENT": 1,
                        "t_first_treatment": 0.0,
                        "t_platinum": t_plat,
                        "PLATINUM": plat,
                        "t_death": 500.0,
                        "DEATH": 0,
                        "t_last_contact": 500.0,
                        "LAB_NAME": lab_name,
                        "LAB_VALUE": value,
                        "t_lab": t,
                    }
                )
    return pd.DataFrame(rows)


def _merged(raw: pd.DataFrame) -> pd.DataFrame:
    _, _, merged = build_landmark_merged(
        raw,
        landmark_offset_days=LANDMARK,
        anchor_col=None,
        require_first_treatment=False,
    )
    merged = merged.copy()
    merged["split"] = "train"
    return merged


def _lab_long(raw: pd.DataFrame, merged: pd.DataFrame, *, full: bool):
    return build_pre_treatment_lab_long(
        raw,
        cohort_index=merged.index,
        landmark_offset_days=LANDMARK,
        anchor_col=None,
        include_post_landmark=full,
    )


def _wide(raw: pd.DataFrame, *, full: bool):
    merged = _merged(raw)
    lab_long = _lab_long(raw, merged, full=full)
    wide, extras, selected, _bounds = build_person_period_wide(
        lab_long,
        merged,
        landmark_day=LANDMARK,
        train_mrns=set(merged.index),
        canonical_labs=["PSA", "ALP"],
        time_unit_days=TIME_UNIT_DAYS,
        min_coverage=0.0,
        max_labs=10,
        outlier_lo=0.005,
        outlier_hi=0.995,
        include_post_landmark=full,
    )
    return wide, extras, selected


class TestLabLongWindow:
    def test_landmark_build_drops_post_landmark_labs(self, raw):
        out = _lab_long(raw, _merged(raw), full=False)
        assert (out["t_lab"] < LANDMARK).all()

    def test_full_build_retains_post_landmark_labs(self, raw):
        merged = _merged(raw)
        out = _lab_long(raw, merged, full=True)
        assert (out["t_lab"] >= LANDMARK).any(), "no post-landmark labs retained"
        assert len(out) > len(_lab_long(raw, merged, full=False))

    def test_full_build_is_a_superset(self, raw):
        merged = _merged(raw)
        narrow = _lab_long(raw, merged, full=False)
        full = _lab_long(raw, merged, full=True)
        key = [ID_COL, "LAB_NAME", "t_lab"]
        joined = narrow.merge(full, on=key, how="left", indicator=True)
        assert (joined["_merge"] == "both").all()


class TestPersonPeriodVariants:
    def test_landmark_frame_is_unchanged_by_the_flag(self, raw):
        """Building the full variant must not perturb the landmark variant.

        Both are produced from the same cohort in one invocation, so a shared
        mutable (clip bounds, canonical labs, the landmark_time series) leaking
        between them would show up here.
        """
        a, _, _ = _wide(raw, full=False)
        _full, _, _ = _wide(raw, full=True)
        b, _, _ = _wide(raw, full=False)
        pd.testing.assert_frame_equal(a, b)

    def test_full_frame_has_post_landmark_rows(self, raw):
        wide, _, _ = _wide(raw, full=True)
        assert (wide["TIME"] > wide["landmark_time"]).any(), (
            "full-follow-up frame carries no post-landmark timepoints; the "
            "dynamic arm would have nothing extra to predict from"
        )

    def test_landmark_frame_does_not_carry_landmark_time(self, raw):
        """The landmark frame's schema must not change. See cohort.py's note:
        there TIME ends at the landmark, so max(TIME) recovers it."""
        wide, _, _ = _wide(raw, full=False)
        assert "landmark_time" not in wide.columns

    def test_landmark_frame_has_no_post_landmark_rows(self, raw):
        """In the landmark frame the anchor IS each patient's last row."""
        wide, _, _ = _wide(raw, full=False)
        anchors = wide.groupby(ID_COL)["TIME"].max()
        full, _, _ = _wide(raw, full=True)
        true_anchors = full.groupby(ID_COL)["landmark_time"].first()
        pd.testing.assert_series_equal(
            anchors.astype(float).rename("landmark_time"),
            true_anchors.astype(float).loc[anchors.index],
        )

    def test_full_frame_still_drops_post_event_rows(self, raw):
        """Relaxing the LANDMARK filter must not relax the PRE-EVENT filter.

        Patient 1 starts platinum at day 300. A lab drawn after that must not
        reach the model under either flag -- that is outcome leakage, not
        legitimate follow-up. The builder's pre-event filter is what enforces
        this, and it is deliberately kept in the full-follow-up path.
        """
        wide, _, _ = _wide(raw, full=True)
        p1 = wide.loc[wide[ID_COL] == 1]
        assert not p1.empty
        # t_platinum is rebased onto the same landmark_time-origin bin clock as
        # TIME, so the filter is directly checkable in those units.
        assert (p1["TIME"] < p1["t_platinum"]).all(), (
            "a post-platinum observation leaked into the full-follow-up frame"
        )
        # Anti-vacuity: patient 1's day-400 lab is genuinely post-platinum, so
        # something must actually have been dropped.
        assert p1["TIME"].max() < wide["TIME"].max()

    def test_the_shared_prefix_is_identical_across_variants(self, raw):
        """The full frame must EXTEND the landmark frame, not redraw it.

        Every row at or before the landmark must carry the same values under
        both flags -- same TIME clock, same normalized lab values. If the flag
        shifted the axis or refit the clip bounds over the wider window, the
        two would diverge here and the dynamic arm's landmark slice would stop
        being comparable to the landmark arm.
        """
        narrow, _, _ = _wide(raw, full=False)
        full, _, _ = _wide(raw, full=True)
        prefix = full.loc[full["TIME"] <= full["landmark_time"]].drop(
            columns=["landmark_time"]
        )
        a = narrow.sort_values([ID_COL, "TIME"]).reset_index(drop=True)
        b = prefix.sort_values([ID_COL, "TIME"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b[a.columns])

    def test_landmark_time_is_never_negative(self, raw):
        wide, _, _ = _wide(raw, full=True)
        assert (wide["landmark_time"] >= 0).all()

    def test_extras_record_the_flag(self, raw):
        _, extras_full, _ = _wide(raw, full=True)
        _, extras_narrow, _ = _wide(raw, full=False)
        assert extras_full["include_post_landmark"] is True
        assert extras_narrow["include_post_landmark"] is False

    def test_no_duplicate_patient_timepoints(self, raw):
        """The synthetic landmark anchor must not collide with a real observation."""
        wide, _, _ = _wide(raw, full=True)
        assert not wide.duplicated(subset=[ID_COL, "TIME"]).any()

    def test_every_patient_survives_the_full_build(self, raw):
        """Patients with only post-landmark labs must not be dropped."""
        narrow, _, _ = _wide(raw, full=False)
        full, _, _ = _wide(raw, full=True)
        assert set(full[ID_COL]) >= set(narrow[ID_COL])


class TestFilenames:
    def test_variants_get_distinct_filenames(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "COMPASS" / "data_preprocessing"))
        from build_prediction_inputs import (
            longitudinal_filename,
            longitudinal_manifest_filename,
        )

        assert longitudinal_filename(90) == "longitudinal_landmark90.csv"
        assert (
            longitudinal_filename(90, full_followup=True)
            == "longitudinal_full_landmark90.csv"
        )
        # The two must never collide, or one build would overwrite the other.
        assert longitudinal_filename(90) != longitudinal_filename(90, full_followup=True)
        assert longitudinal_manifest_filename(
            90, full_followup=True
        ) == "longitudinal_full_landmark90_manifest.json"
