"""Phase 5: incremental value of accruing lab history.

The load-bearing claims, each pinned here:

1. **The ablation compares like with like.** Both arms must be scored on the
   same patients and the same outcome, with only the input window differing.
   If the stale arm silently scored a different patient set or a different
   horizon, `auc_gain` would measure something other than the value of new
   labs -- and it would still look like a plausible number.

2. **Censored-before-horizon rows are excluded, not counted as non-events.**
   Counting them as non-events biases risk downward, and biases it *differently
   at different prediction times*, manufacturing exactly the trend these
   analyses exist to measure.

3. **The stale arm reads a horizon extended by delta.** Asking a 6-month
   question `delta` earlier is a `6 months + delta` question about the same
   calendar date. Getting this wrong makes the held-back arm look worse than it
   is, inflating the apparent gain.

4. **Underpowered prediction times are gated**, as in the by-time metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "COMPASS" / "survival_analysis" / "multivariate_longitudinal"))

pytest.importorskip("sklearn")
pytest.importorskip("lifelines")

import incremental_risk as ir  # noqa: E402

ID = "DFCI_MRN"
HORIZON_BIN = 26
DELTA_BINS = 13


def _rows(mrn, times, *, label, duration_at, risk_at, horizons=(13, 26, 39)):
    """Prediction rows for one patient across several prediction times."""
    out = []
    for t in times:
        row = {
            ID: mrn,
            "TIME": float(t),
            "label": label,
            "duration": float(duration_at(t)),
            "duration_bin": int(np.ceil(duration_at(t))),
            "event_1_risk_total": risk_at(t),
        }
        for h in horizons:
            row[f"event_1_risk_h{h}"] = risk_at(t)
        out.append(row)
    return out


@pytest.fixture
def pred() -> pd.DataFrame:
    """60 patients observed at t=0,13,26; half have the event.

    Durations are set so that, at every prediction time, BOTH cases and
    controls survive `_horizon_outcome`'s filter -- cases have the event inside
    the 26-bin horizon, controls are followed past it. A fixture where the
    controls are all censored early would leave one class and silently turn
    every AUC into NaN, which tests nothing.

    Risk is informative at t=26 and uninformative (0.5) at the earlier steps,
    so the ablation must show a positive gain from the newest labs.
    """
    rows = []
    for i in range(60):
        has_event = i % 2 == 0
        # Cases: event 20 bins after the last prediction time (inside the
        # horizon from every t). Controls: censored well past t=26 + 26.
        residual = (lambda t: 46.0 - t) if has_event else (lambda t: 80.0 - t)
        rows += _rows(
            1000 + i,
            (0, 13, 26),
            label=1 if has_event else 0,
            duration_at=residual,
            # Informative only at the last step.
            risk_at=lambda t, e=has_event: (
                (0.9 if e else 0.1) if t == 26 else 0.5
            ),
        )
    return pd.DataFrame(rows)


class TestHorizonOutcome:
    def test_event_inside_horizon_is_a_case(self):
        block = pd.DataFrame({"duration": [10.0], "label": [1]})
        label, keep = ir._horizon_outcome(block, event_idx=1, horizon_bin=26)
        assert label[0] == 1 and keep[0]

    def test_event_after_horizon_is_a_control(self):
        block = pd.DataFrame({"duration": [40.0], "label": [1]})
        label, keep = ir._horizon_outcome(block, event_idx=1, horizon_bin=26)
        assert label[0] == 0 and keep[0], "followed past the horizon => known control"

    def test_censored_before_horizon_is_excluded(self):
        """The core bias guard: unknown outcome must not become a non-event."""
        block = pd.DataFrame({"duration": [10.0], "label": [0]})
        label, keep = ir._horizon_outcome(block, event_idx=1, horizon_bin=26)
        assert not keep[0], "censored at 10 with a 26 horizon: outcome is unknown"

    def test_censored_after_horizon_is_a_control(self):
        block = pd.DataFrame({"duration": [30.0], "label": [0]})
        _label, keep = ir._horizon_outcome(block, event_idx=1, horizon_bin=26)
        assert keep[0]

    def test_competing_event_before_horizon_is_excluded(self):
        """A death at bin 10 leaves the platinum outcome at 26 unknown."""
        block = pd.DataFrame({"duration": [10.0], "label": [2]})
        _label, keep = ir._horizon_outcome(block, event_idx=1, horizon_bin=26)
        assert not keep[0]


class TestByLandmark:
    def test_one_row_per_grid_point(self, pred):
        out = ir.by_landmark(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            grid_bins=[0, 13, 26],
        )
        assert len(out) == 3

    def test_snaps_to_the_nearest_materialized_time(self, pred):
        """A grid in days need not land on an observation bin."""
        out = ir.by_landmark(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            grid_bins=[1],  # nearest available is 0
        )
        assert out.iloc[0]["prediction_time"] == 0.0
        assert out.iloc[0]["requested_prediction_day"] == 7.0

    def test_records_which_risk_column_was_used(self, pred):
        out = ir.by_landmark(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            grid_bins=[0],
        )
        assert out.iloc[0]["risk_column"] == "event_1_risk_h26"

    def test_falls_back_to_total_risk_and_says_so(self, pred):
        """If the horizon column is absent the fallback must be visible, or a
        time-constant total risk would masquerade as a fixed-horizon risk."""
        out = ir.by_landmark(
            pred, event_idx=1, event_name="platinum", horizon_bin=99,
            grid_bins=[0],
        )
        assert out.iloc[0]["risk_column"] == "event_1_risk_total"

    def test_no_duplicate_prediction_times(self, pred):
        """A grid finer than the observation spacing snaps several points to the
        same bin; the series must still have one row per time."""
        out = ir.by_landmark(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            grid_bins=[0, 1, 2, 13],
        )
        assert not out.duplicated(subset=["endpoint", "prediction_time"]).any()

    def test_underpowered_times_are_gated(self, pred):
        out = ir.by_landmark(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            grid_bins=[0], min_risk_set=1000,
        )
        assert out.iloc[0]["note"].startswith("underpowered")
        assert np.isnan(out.iloc[0]["auc"])


class TestAblationComparesLikeWithLike:
    def _run(self, pred, **kw):
        return ir.ablation(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            delta_bins=DELTA_BINS, id_col=ID, **kw
        )

    def test_the_newest_labs_show_a_positive_gain(self, pred):
        """Anti-vacuity: risk is informative only at t=26, so holding back the
        last 13 bins must measurably hurt."""
        out = self._run(pred, grid_bins=[26])
        row = out.loc[out["prediction_time"] == 26.0].iloc[0]
        assert row["note"] == ""
        assert row["auc_full_history"] > row["auc_held_back"]
        assert row["auc_gain"] > 0

    def test_both_arms_score_the_same_patients(self, pred):
        """n_scored is shared by construction; assert the pairing is exact."""
        out = self._run(pred, grid_bins=[26])
        row = out.loc[out["prediction_time"] == 26.0].iloc[0]
        assert row["n_paired"] == 60
        assert row["n_scored"] <= row["n_paired"]

    def test_stale_arm_reads_the_delta_extended_horizon(self, pred):
        """Asking a 26-bin question 13 bins earlier is a 39-bin question."""
        out = self._run(pred, grid_bins=[26])
        row = out.loc[out["prediction_time"] == 26.0].iloc[0]
        assert row["fresh_risk_column"] == "event_1_risk_h26"
        assert row["stale_risk_column"] == "event_1_risk_h39"

    def test_patients_without_a_step_at_t_minus_delta_are_excluded(self):
        """A patient missing the stale step must be dropped, never back-filled
        from some other step -- that would silently vary the input window."""
        rows = []
        # Patient A has both steps; patient B only the later one.
        rows += _rows(1, (13, 26), label=1, duration_at=lambda t: 40 - t,
                      risk_at=lambda t: 0.8)
        rows += _rows(2, (26,), label=0, duration_at=lambda t: 40 - t,
                      risk_at=lambda t: 0.2)
        pred = pd.DataFrame(rows)
        out = ir.ablation(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            delta_bins=DELTA_BINS, id_col=ID, grid_bins=[26],
            min_risk_set=1, min_events=0,
        )
        row = out.loc[out["prediction_time"] == 26.0].iloc[0]
        assert row["n_at_risk"] == 2
        assert row["n_paired"] == 1, "patient 2 has no step at t-delta"

    def test_no_stale_step_at_all_is_reported_not_crashed(self, pred):
        """At t=0 there is no t-delta; the row must say so."""
        out = self._run(pred, grid_bins=[0])
        row = out.loc[out["prediction_time"] == 0.0].iloc[0]
        assert "no patient has a step" in row["note"]

    def test_gain_signs_are_oriented_consistently(self, pred):
        """Positive gain = the newest labs helped, for BOTH metrics. Brier
        falls when it improves, so its gain must be stale - fresh."""
        out = self._run(pred, grid_bins=[26])
        row = out.loc[out["prediction_time"] == 26.0].iloc[0]
        assert row["auc_gain"] == pytest.approx(
            row["auc_full_history"] - row["auc_held_back"]
        )
        assert row["brier_gain"] == pytest.approx(
            row["brier_held_back"] - row["brier_full_history"]
        )
        # Informative risk at t=26 => Brier improves too.
        assert row["brier_gain"] > 0

    def test_underpowered_rows_are_gated(self, pred):
        out = self._run(pred, grid_bins=[26], min_risk_set=1000)
        row = out.loc[out["prediction_time"] == 26.0].iloc[0]
        assert row["note"].startswith("underpowered")
        assert np.isnan(row["auc_gain"])

    def test_identical_arms_give_zero_gain(self):
        """A sanity anchor: if risk does not change between the two steps, the
        measured gain must be exactly zero, not merely small."""
        rows = []
        for i in range(60):
            e = i % 2 == 0
            rows += _rows(
                1000 + i, (13, 26), label=1 if e else 0,
                # As in the `pred` fixture: cases event inside the horizon,
                # controls followed past it, so both classes survive scoring.
                duration_at=(lambda t: 46.0 - t) if e else (lambda t: 80.0 - t),
                risk_at=lambda t, e=e: 0.9 if e else 0.1,  # constant in t
            )
        pred = pd.DataFrame(rows)
        out = ir.ablation(
            pred, event_idx=1, event_name="platinum", horizon_bin=HORIZON_BIN,
            delta_bins=DELTA_BINS, id_col=ID, grid_bins=[26],
        )
        row = out.loc[out["prediction_time"] == 26.0].iloc[0]
        assert row["auc_gain"] == pytest.approx(0.0)


class TestCliContract:
    def test_landmark_frame_is_rejected(self, tmp_path):
        """A landmark prediction frame has no TIME column; the error must name
        the cause rather than failing later on a missing key."""
        p = tmp_path / "pred.csv"
        pd.DataFrame({ID: [1], "label": [1], "duration": [5.0]}).to_csv(p, index=False)
        args = ir.build_parser().parse_args(
            ["--output-dir", str(tmp_path), "--predictions", str(p), "--overwrite"]
        )
        with pytest.raises(ValueError, match="landmark prediction frame"):
            ir.main(args)

    def test_missing_predictions_names_the_fix(self, tmp_path):
        args = ir.build_parser().parse_args(
            ["--output-dir", str(tmp_path), "--overwrite"]
        )
        with pytest.raises(FileNotFoundError, match="--dynamic"):
            ir.main(args)

    def test_end_to_end_writes_both_analyses(self, tmp_path, pred):
        p = tmp_path / "dynamic_deephit_dyn_predictions_platinum.csv"
        pred.to_csv(p, index=False)
        args = ir.build_parser().parse_args([
            "--output-dir", str(tmp_path),
            "--horizon-days", "182", "--delta-days", "91",
            "--grid-days", "0", "91", "182",
            "--overwrite",
        ])
        ir.main(args)
        by_lm = pd.read_csv(tmp_path / "incremental_risk_by_landmark.csv")
        abl = pd.read_csv(tmp_path / "incremental_risk_ablation.csv")
        assert not by_lm.empty and not abl.empty
        # Identity columns front-loaded, so these join against the metrics CSVs.
        assert list(by_lm.columns[:5]) == [
            "model", "cohort", "endpoint", "landmark_days", "config"
        ]
        assert (tmp_path / "incremental_risk_manifest.json").exists()

    def test_skips_when_outputs_exist(self, tmp_path, pred, capsys):
        p = tmp_path / "dynamic_deephit_dyn_predictions_platinum.csv"
        pred.to_csv(p, index=False)
        (tmp_path / "incremental_risk_by_landmark.csv").write_text("x\n")
        (tmp_path / "incremental_risk_ablation.csv").write_text("x\n")
        args = ir.build_parser().parse_args(["--output-dir", str(tmp_path)])
        ir.main(args)
        assert "[skip]" in capsys.readouterr().out
