"""Property tests on survival_common.helper.compute_horizon_grid.

Pins the docstring's claims (helper.py:221) as executable assertions: output
is strictly positive, sorted, deduplicated, bounded by max_grid_points, and
clamped strictly inside admin_censor_days. None of these were previously
tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.helper import compute_horizon_grid


def _train_val_df(n_events: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    duration = rng.uniform(1, 500, size=n_events)
    event = np.ones(n_events, dtype=int)
    return pd.DataFrame({"DURATION": duration, "EVENT": event})


class TestBasicProperties:
    def test_output_is_positive_sorted_deduped(self):
        df = _train_val_df()
        horizons = compute_horizon_grid(df, duration_col="DURATION", event_col="EVENT")
        assert (horizons > 0).all()
        assert np.array_equal(horizons, np.sort(horizons))
        assert len(horizons) == len(np.unique(horizons))

    def test_bounded_by_max_grid_points(self):
        df = _train_val_df(n_events=2000)
        horizons = compute_horizon_grid(
            df, duration_col="DURATION", event_col="EVENT", max_grid_points=10
        )
        assert len(horizons) <= 10

    def test_clamped_strictly_inside_admin_censor_days(self):
        df = _train_val_df()
        time_unit_days = 7
        admin_censor_days = 260 * time_unit_days
        horizons = compute_horizon_grid(
            df,
            duration_col="DURATION",
            event_col="EVENT",
            time_unit_days=time_unit_days,
            admin_censor_days=admin_censor_days,
        )
        cap_units = np.ceil(admin_censor_days / time_unit_days)
        assert (horizons < cap_units).all()

    def test_returns_nonempty_for_sparse_events(self):
        # Only two distinct event times -- the tie-expansion branch.
        df = pd.DataFrame(
            {
                "DURATION": [10.0, 10.0, 10.0, 20.0],
                "EVENT": [1, 1, 1, 1],
            }
        )
        horizons = compute_horizon_grid(df, duration_col="DURATION", event_col="EVENT")
        assert len(horizons) >= 1
        assert (horizons > 0).all()

    def test_raises_when_no_events(self):
        df = pd.DataFrame({"DURATION": [10.0, 20.0], "EVENT": [0, 0]})
        with pytest.raises(ValueError, match="No observed events"):
            compute_horizon_grid(df, duration_col="DURATION", event_col="EVENT")

    def test_invalid_time_unit_days_raises(self):
        df = _train_val_df()
        with pytest.raises(ValueError, match="time_unit_days"):
            compute_horizon_grid(
                df, duration_col="DURATION", event_col="EVENT", time_unit_days=0
            )

    def test_invalid_max_grid_points_raises(self):
        df = _train_val_df()
        with pytest.raises(ValueError, match="max_grid_points"):
            compute_horizon_grid(
                df, duration_col="DURATION", event_col="EVENT", max_grid_points=1
            )

    def test_admin_censor_smaller_than_one_unit_raises(self):
        # Events at duration=0.5 days survive the admin-censor clip (cap=0.5)
        # but are smaller than one 7-day time unit, so no horizon fits
        # strictly inside the cap.
        df = pd.DataFrame({"DURATION": [0.5, 0.5, 0.5], "EVENT": [1, 1, 1]})
        with pytest.raises(ValueError, match="administrative censoring"):
            compute_horizon_grid(
                df,
                duration_col="DURATION",
                event_col="EVENT",
                time_unit_days=7,
                admin_censor_days=0.5,
            )

    def test_deterministic_for_fixed_input(self):
        df = _train_val_df()
        h1 = compute_horizon_grid(df, duration_col="DURATION", event_col="EVENT")
        h2 = compute_horizon_grid(df, duration_col="DURATION", event_col="EVENT")
        assert np.array_equal(h1, h2)
