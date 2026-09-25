"""within_stratum_table / plot_within_stratum recover a known within-stratum HR.

Constructs synthetic exponential survival times whose hazard is driven by the
(standardized) labs risk score with a known log-HR, independent of the
clinical stratifier, so the per-level and pooled Cox fits should both recover
that log-HR approximately. Also covers the underpowered path (a stratum with
fewer than 5 events reports status="underpowered" and never raises).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from COMPASS.survival_analysis import risk_score_stratified_figures as rsf  # noqa: E402

ID = "DFCI_MRN"
TRUE_LOG_HR = 0.7  # ~ HR = 2.0 per SD


def _synthetic_frame(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    # Clinical stratifier independent of risk: two balanced groups.
    gleason = rng.choice([6, 7, 9], size=n)

    baseline_hazard = 0.01
    hazard = baseline_hazard * np.exp(TRUE_LOG_HR * risk)
    event_time = rng.exponential(scale=1.0 / hazard)
    censor_time = rng.exponential(scale=1.0 / (baseline_hazard * 2))
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    return pd.DataFrame({
        ID: [str(i) for i in range(n)],
        "risk_score": risk,
        "duration_days": duration,
        "event": event,
        rsf.GLEASON_FEATURE: gleason,
    })


class TestWithinStratumRecoversKnownEffect:
    def test_pooled_hr_is_in_the_right_direction_and_roughly_correct(self):
        frame = _synthetic_frame(n=600, seed=1)
        cutpoint = float(frame["risk_score"].median())
        stratifiers = rsf.build_stratifiers(frame, cutpoint=cutpoint)
        clinical_stratifiers = [s for s in stratifiers if s.key == "gleason"]
        assert clinical_stratifiers

        table = rsf.within_stratum_table(
            frame, clinical_stratifiers, global_cutpoint=cutpoint
        )
        pooled = table.loc[table["level"] == "__pooled__"].iloc[0]
        assert pooled["status"] == "ok"
        true_hr = float(np.exp(TRUE_LOG_HR))
        # Generous tolerance: this is a finite synthetic sample, not an
        # asymptotic check -- we only need the recovered HR to land near the
        # true effect and clearly above 1 (right direction, right order of
        # magnitude).
        assert pooled["hr_per_sd"] > 1.2
        assert abs(pooled["hr_per_sd"] - true_hr) < 1.5

    def test_per_level_rows_all_report_elevated_risk(self):
        frame = _synthetic_frame(n=600, seed=2)
        cutpoint = float(frame["risk_score"].median())
        stratifiers = rsf.build_stratifiers(frame, cutpoint=cutpoint)
        clinical_stratifiers = [s for s in stratifiers if s.key == "gleason"]

        table = rsf.within_stratum_table(
            frame, clinical_stratifiers, global_cutpoint=cutpoint
        )
        per_level = table.loc[table["level"] != "__pooled__"]
        ok_rows = per_level.loc[per_level["status"] == "ok"]
        assert len(ok_rows) >= 2
        assert (ok_rows["hr_per_sd"] > 1.0).all()


class TestUnderpoweredPath:
    def test_stratum_with_fewer_than_five_events_is_underpowered(self):
        frame = _synthetic_frame(n=40, seed=3)
        # Force one Gleason level down to 1 event by censoring everyone else
        # in that level heavily.
        frame.loc[frame[rsf.GLEASON_FEATURE] == 9, "event"] = 0
        frame.loc[
            frame[rsf.GLEASON_FEATURE].eq(9) & (frame.index < frame.index[frame[rsf.GLEASON_FEATURE] == 9][0] + 1),
            "event",
        ] = 1
        cutpoint = float(frame["risk_score"].median())
        stratifiers = rsf.build_stratifiers(frame, cutpoint=cutpoint)
        clinical_stratifiers = [s for s in stratifiers if s.key == "gleason"]

        # Must not raise even though one level has near-zero events.
        table = rsf.within_stratum_table(
            frame, clinical_stratifiers, global_cutpoint=cutpoint
        )
        level_9 = table.loc[table["level"] == "Gleason 8-10"]
        assert not level_9.empty
        assert (level_9["status"] == "underpowered").all()
        assert level_9["hr_per_sd"].isna().all()

    def test_zero_event_stratum_never_raises(self):
        frame = _synthetic_frame(n=30, seed=4)
        frame.loc[frame[rsf.GLEASON_FEATURE] == 6, "event"] = 0
        cutpoint = float(frame["risk_score"].median())
        stratifiers = rsf.build_stratifiers(frame, cutpoint=cutpoint)
        clinical_stratifiers = [s for s in stratifiers if s.key == "gleason"]

        table = rsf.within_stratum_table(
            frame, clinical_stratifiers, global_cutpoint=cutpoint
        )
        low_level = table.loc[table["level"] == "Gleason <=6"]
        assert not low_level.empty
        assert (low_level["status"] == "underpowered").all()


class TestPlotWithinStratumDoesNotRaise:
    def test_plot_runs_and_returns_a_figure(self):
        frame = _synthetic_frame(n=200, seed=5)
        cutpoint = float(frame["risk_score"].median())
        stratifiers = rsf.build_stratifiers(frame, cutpoint=cutpoint)
        clinical_stratifiers = [s for s in stratifiers if s.key == "gleason"]

        fig = rsf.plot_within_stratum(
            frame, clinical_stratifiers, global_cutpoint=cutpoint,
            title="test", xlabel="days",
        )
        assert fig is not None

    def test_plot_with_no_clinical_stratifiers_does_not_raise(self):
        frame = _synthetic_frame(n=50, seed=6)
        cutpoint = float(frame["risk_score"].median())
        fig = rsf.plot_within_stratum(
            frame, [], global_cutpoint=cutpoint, title="empty", xlabel="days",
        )
        assert fig is not None
