"""Tests for the dynamic (per-timepoint) target builder.

The load-bearing property is the equivalence in
:class:`TestLandmarkSliceEquivalence`: restricted to ``TIME == landmark_time``,
``patient_targets_dynamic`` must reproduce ``patient_targets`` exactly. That is
what lets the dynamic arm's landmark slice be compared to the Cox/XGBoost arms
on the same footing, and it is the cheapest guard against the target rebasing
drifting out from under the loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.longitudinal_targets import (
    patient_targets,
    patient_targets_dynamic,
)

ID_COL = "DFCI_MRN"
TIME_COL = "TIME"
MAX_WINDOW = 52


def _person_period(
    mrn: int,
    *,
    times: list[int],
    t_platinum: float,
    platinum: int,
    t_death: float,
    death: int,
) -> pd.DataFrame:
    """A person-period frame shaped like build_person_period_wide's output."""
    return pd.DataFrame(
        {
            ID_COL: mrn,
            TIME_COL: times,
            "PSA": np.linspace(1.0, 2.0, len(times)),
            "PLATINUM": platinum,
            "DEATH": death,
            "t_platinum": t_platinum,
            "t_death": t_death,
        }
    )


def _cohort() -> pd.DataFrame:
    return pd.concat(
        [
            # Platinum event 10 bins after a landmark at TIME=3.
            _person_period(
                1, times=[0, 1, 2, 3], t_platinum=13.0, platinum=1, t_death=40.0, death=0
            ),
            # Censored: no event, followed to bin 30.
            _person_period(
                2, times=[0, 1, 2], t_platinum=30.0, platinum=0, t_death=30.0, death=0
            ),
            # Death before platinum -- competing config should label it 2.
            _person_period(
                3, times=[0, 1, 2, 3, 4], t_platinum=25.0, platinum=0, t_death=9.0, death=1
            ),
        ],
        ignore_index=True,
    )


def _dyn(df, **kw):
    return patient_targets_dynamic(
        df,
        id_col=ID_COL,
        time_col=TIME_COL,
        max_pred_window=kw.pop("max_pred_window", MAX_WINDOW),
        **kw,
    )


def _static(df, **kw):
    return patient_targets(
        df,
        id_col=ID_COL,
        time_col=TIME_COL,
        max_pred_window=kw.pop("max_pred_window", MAX_WINDOW),
        **kw,
    )


PLATINUM_ONLY = dict(event_cols=["PLATINUM"], time_cols=["t_platinum"])
COMPETING = dict(event_cols=["PLATINUM", "DEATH"], time_cols=["t_platinum", "t_death"])


class TestLandmarkSliceEquivalence:
    """The dynamic builder must agree with the landmark builder at the landmark."""

    @pytest.mark.parametrize("config", [PLATINUM_ONLY, COMPETING], ids=["platinum", "competing"])
    def test_landmark_slice_matches_patient_targets(self, config):
        df = _cohort()
        static = _static(df, **config)
        dynamic = _dyn(df, **config)

        # Slice the dynamic frame to each patient's own landmark_time.
        at_landmark = dynamic.reset_index()
        at_landmark = at_landmark.loc[
            at_landmark[TIME_COL] == at_landmark["landmark_time"]
        ].set_index(ID_COL)

        assert set(at_landmark.index) == set(static.index)
        shared = ["landmark_time", "duration", "duration_bin", "label"]
        pd.testing.assert_frame_equal(
            at_landmark.loc[static.index, shared],
            static[shared],
            check_dtype=False,
        )


class TestLandmarkTimeSource:
    """landmark_time must be read off the frame, not inferred from max(TIME).

    The full-follow-up inputs retain post-landmark observations, so a patient's
    last row is no longer their landmark. Deriving it as max(TIME) would point
    the landmark slice at each patient's final follow-up visit instead -- which
    does not crash, and produces a plausible-looking but wrong headline metric.
    """

    def _with_landmark_col(self, landmark: int) -> pd.DataFrame:
        # Observations run past the landmark, as --longitudinal-full-followup builds them.
        df = _person_period(
            1, times=[0, 1, 2, 3, 4, 5], t_platinum=20.0, platinum=1, t_death=40.0, death=0
        )
        df["landmark_time"] = landmark
        return df

    def test_landmark_time_comes_from_the_column(self):
        dynamic = _dyn(self._with_landmark_col(2), **PLATINUM_ONLY)
        assert (dynamic["landmark_time"] == 2.0).all()

    def test_landmark_slice_is_not_the_last_observation(self):
        df = self._with_landmark_col(2)
        dynamic = _dyn(df, **PLATINUM_ONLY).reset_index()
        at_landmark = dynamic.loc[dynamic[TIME_COL] == dynamic["landmark_time"]]
        assert len(at_landmark) == 1
        assert float(at_landmark[TIME_COL].iloc[0]) == 2.0
        # max(TIME) is 5; the bug would have selected that row instead.
        assert float(at_landmark[TIME_COL].iloc[0]) != float(dynamic[TIME_COL].max())

    def test_falls_back_to_max_time_without_the_column(self):
        """Landmark-truncated frames have no landmark_time column; max(TIME) is correct there."""
        df = _person_period(
            1, times=[0, 1, 2, 3], t_platinum=13.0, platinum=1, t_death=40.0, death=0
        )
        assert "landmark_time" not in df.columns
        dynamic = _dyn(df, **PLATINUM_ONLY)
        assert (dynamic["landmark_time"] == 3.0).all()


class TestResidualDurations:
    def test_duration_is_residual_from_prediction_time(self):
        df = _cohort()
        dynamic = _dyn(df, **PLATINUM_ONLY).reset_index()
        patient = dynamic.loc[dynamic[ID_COL] == 1].sort_values(TIME_COL)
        # t_platinum = 13; at TIME=t the residual duration is 13 - t.
        expected = 13.0 - patient[TIME_COL].to_numpy(dtype=float)
        np.testing.assert_allclose(patient["duration"].to_numpy(dtype=float), expected)

    def test_duration_strictly_decreases_along_the_sequence(self):
        df = _cohort()
        dynamic = _dyn(df, **PLATINUM_ONLY).reset_index()
        for _, patient in dynamic.groupby(ID_COL):
            durations = patient.sort_values(TIME_COL)["duration"].to_numpy(dtype=float)
            assert np.all(np.diff(durations) < 0), "residual time must shrink"

    def test_all_surviving_rows_are_at_risk(self):
        dynamic = _dyn(_cohort(), **PLATINUM_ONLY)
        assert dynamic["at_risk"].all()

    def test_one_row_per_patient_time(self):
        dynamic = _dyn(_cohort(), **PLATINUM_ONLY)
        assert not dynamic.index.duplicated().any()

    def test_rows_at_or_after_the_event_are_dropped(self):
        # Event at bin 2; prediction times 0 and 1 are at risk, 2 and 3 are not.
        df = _person_period(
            9, times=[0, 1, 2, 3], t_platinum=2.0, platinum=1, t_death=40.0, death=0
        )
        dynamic = _dyn(df, **PLATINUM_ONLY).reset_index()
        assert sorted(dynamic[TIME_COL].tolist()) == [0.0, 1.0]


class TestCompetingSemantics:
    def test_death_before_platinum_is_labeled_cause_two(self):
        dynamic = _dyn(_cohort(), **COMPETING).reset_index()
        patient = dynamic.loc[dynamic[ID_COL] == 3]
        assert (patient["label"] == 2).all()

    def test_platinum_wins_an_exact_tie(self):
        # Fixed cause ordering: argmin ties break toward the first listed cause.
        df = _person_period(
            7, times=[0, 1], t_platinum=5.0, platinum=1, t_death=5.0, death=1
        )
        dynamic = _dyn(df, **COMPETING)
        assert (dynamic["label"] == 1).all()

    def test_censored_patient_is_label_zero(self):
        dynamic = _dyn(_cohort(), **COMPETING).reset_index()
        patient = dynamic.loc[dynamic[ID_COL] == 2]
        assert (patient["label"] == 0).all()


class TestWindowCensoring:
    def test_event_beyond_window_is_censored_at_the_window(self):
        df = _person_period(
            5, times=[0, 1], t_platinum=500.0, platinum=1, t_death=600.0, death=0
        )
        dynamic = _dyn(df, max_pred_window=10, **PLATINUM_ONLY)
        assert (dynamic["label"] == 0).all()
        assert (dynamic["duration"] == 10.0).all()
        # The pre-censoring diagnostics retain the true values.
        assert (dynamic["uncensored_label"] == 1).all()
        assert (dynamic["uncensored_duration"] > 10.0).all()

    def test_duration_bin_within_window(self):
        dynamic = _dyn(_cohort(), **COMPETING)
        assert dynamic["duration_bin"].between(1, MAX_WINDOW).all()

    def test_rejects_nonpositive_window(self):
        with pytest.raises(ValueError, match="max_pred_window must be >= 1"):
            _dyn(_cohort(), max_pred_window=0, **PLATINUM_ONLY)
