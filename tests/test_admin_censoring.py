"""Step 0: administrative censoring (max_followup_days) in make_outcome_df /
build_landmark_merged.

Grounded directly in survival_common/cohort.py:236-250 (the censoring block)
and :245-257 (EITHER/t_either derivation, which runs after censoring so it
must reflect the clipped durations).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_common.cohort import build_landmark_merged, make_outcome_df

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_PREDICTION_INPUTS_DIR = REPO_ROOT / "COMPASS" / "data_preprocessing"
if str(BUILD_PREDICTION_INPUTS_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_PREDICTION_INPUTS_DIR))

from build_prediction_inputs import validate_max_followup_days  # noqa: E402

HORIZON = 3650.0


def _patient_row(
    mrn: int,
    *,
    platinum: int,
    t_platinum: float,
    death: int,
    t_death: float,
    t_last_contact: float,
) -> dict:
    return {
        "DFCI_MRN": mrn,
        "AGE_AT_TREATMENTSTART": 65.0,
        "FIRST_RECORD_DATE": "2015-01-01",
        "FIRST_TREATMENT": 1,
        "t_first_treatment": 0.0,
        "t_platinum": t_platinum,
        "PLATINUM": platinum,
        # make_outcome_df falls back t_death to t_last_contact when the
        # patient did not die (_derive_duration's fallback_duration_col), so
        # the fixture reproduces that directly rather than leaving it NaN.
        "t_death": t_death if death else t_last_contact,
        "DEATH": death,
        "t_last_contact": t_last_contact,
        # build_landmark_merged also calls build_feature_matrix on the same
        # df, which requires these columns to exist (even with no rows).
        "LAB_NAME": pd.NA,
        "LAB_VALUE": pd.NA,
        "t_lab": pd.NA,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _outcome(df: pd.DataFrame, max_followup_days: float | None) -> pd.DataFrame:
    return make_outcome_df(
        df,
        anchor_col=None,
        require_first_treatment=False,
        max_followup_days=max_followup_days,
    )


def test_event_past_horizon_is_censored():
    df = _make_df(
        [
            _patient_row(
                1, platinum=1, t_platinum=4000.0, death=0, t_death=np.nan,
                t_last_contact=4000.0,
            )
        ]
    )
    out = _outcome(df, HORIZON)
    row = out.loc[1]
    assert row["PLATINUM"] == 0
    assert row["t_platinum"] == HORIZON


def test_event_before_horizon_is_untouched():
    df = _make_df(
        [
            _patient_row(
                2, platinum=1, t_platinum=3000.0, death=0, t_death=np.nan,
                t_last_contact=3000.0,
            )
        ]
    )
    out = _outcome(df, HORIZON)
    row = out.loc[2]
    assert row["PLATINUM"] == 1
    assert row["t_platinum"] == 3000.0


def test_event_exactly_at_horizon_is_retained():
    # cohort.py:238 uses .gt(horizon) (strict >), so an event exactly at the
    # horizon must survive uncensored.
    df = _make_df(
        [
            _patient_row(
                3, platinum=1, t_platinum=HORIZON, death=0, t_death=np.nan,
                t_last_contact=HORIZON,
            )
        ]
    )
    out = _outcome(df, HORIZON)
    row = out.loc[3]
    assert row["PLATINUM"] == 1
    assert row["t_platinum"] == HORIZON


def test_death_past_horizon_is_censored_and_last_contact_clipped():
    df = _make_df(
        [
            _patient_row(
                4, platinum=0, t_platinum=np.nan, death=1, t_death=4200.0,
                t_last_contact=4200.0,
            )
        ]
    )
    out = _outcome(df, HORIZON)
    row = out.loc[4]
    assert row["DEATH"] == 0
    assert row["t_death"] == HORIZON
    assert row["t_last_contact"] == HORIZON


def test_either_and_t_either_reflect_censoring():
    # EITHER/t_either are derived after the clip (cohort.py:245-257), so a
    # platinum event pushed past the horizon must not count toward EITHER.
    df = _make_df(
        [
            _patient_row(
                5, platinum=1, t_platinum=4000.0, death=0, t_death=np.nan,
                t_last_contact=4000.0,
            )
        ]
    )
    out = _outcome(df, HORIZON)
    row = out.loc[5]
    assert row["EITHER"] == 0
    assert row["t_either"] == HORIZON


def test_none_reproduces_current_behavior_regression_guard():
    df = _make_df(
        [
            _patient_row(
                6, platinum=1, t_platinum=4000.0, death=0, t_death=np.nan,
                t_last_contact=4000.0,
            )
        ]
    )
    out = _outcome(df, None)
    row = out.loc[6]
    assert row["PLATINUM"] == 1
    assert row["t_platinum"] == 4000.0


def _with_pre_landmark_lab(row: dict) -> dict:
    row = dict(row)
    row["LAB_NAME"] = "PSA"
    row["LAB_VALUE"] = 1.0
    row["t_lab"] = -30.0
    return row


def test_build_landmark_merged_forwards_max_followup_days():
    df = _make_df(
        [
            _with_pre_landmark_lab(
                _patient_row(
                    7, platinum=1, t_platinum=4000.0, death=0, t_death=np.nan,
                    t_last_contact=4000.0,
                )
            )
        ]
    )
    _, _, merged = build_landmark_merged(
        df,
        landmark_offset_days=0,
        anchor_col=None,
        require_first_treatment=False,
        max_followup_days=HORIZON,
    )
    row = merged.loc[7]
    assert row["PLATINUM"] == 0
    assert row["t_platinum"] == HORIZON


def test_build_landmark_merged_default_is_uncensored():
    df = _make_df(
        [
            _with_pre_landmark_lab(
                _patient_row(
                    8, platinum=1, t_platinum=4000.0, death=0, t_death=np.nan,
                    t_last_contact=4000.0,
                )
            )
        ]
    )
    _, _, merged = build_landmark_merged(
        df,
        landmark_offset_days=0,
        anchor_col=None,
        require_first_treatment=False,
    )
    row = merged.loc[8]
    assert row["PLATINUM"] == 1
    assert row["t_platinum"] == 4000.0


def test_consistency_guard_raises_when_horizon_too_short():
    with pytest.raises(ValueError, match="shorter than"):
        validate_max_followup_days(
            1000.0, auc_max_time_units=260, time_unit_days=7
        )


def test_consistency_guard_passes_when_horizon_sufficient():
    # 260 * 7 = 1820 <= 3650 -> no raise.
    validate_max_followup_days(3650.0, auc_max_time_units=260, time_unit_days=7)


def test_consistency_guard_noop_when_disabled():
    validate_max_followup_days(None, auc_max_time_units=260, time_unit_days=7)
