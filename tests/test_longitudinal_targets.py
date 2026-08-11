"""Torch-free tests for survival_common/longitudinal_targets.py.

Pins the competing-risk label semantics before any model code exists, per the
plan's step-1 priority: highest risk to get wrong, lowest cost to verify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.longitudinal_targets import (
    LONGITUDINAL_CONFIGS,
    horizon_units_to_bins,
    manifest_horizons_for_config,
    patient_targets,
    resolve_config,
)


def _person_period_row(mrn: int, time: float, **overrides) -> dict:
    row = {
        "DFCI_MRN": mrn,
        "TIME": time,
        "PLATINUM": 0,
        "t_platinum": np.nan,
        "DEATH": 0,
        "t_death": np.nan,
    }
    row.update(overrides)
    return row


class TestResolveConfig:
    def test_platinum_config(self):
        manifest = {"event_cols": ["PLATINUM", "DEATH"]}
        cfg = resolve_config("platinum", manifest)
        assert cfg.event_cols == ["PLATINUM"]
        assert cfg.time_cols == ["t_platinum"]
        assert cfg.event_names == ["platinum"]
        assert cfg.n_events == 1

    def test_competing_config(self):
        manifest = {"event_cols": ["PLATINUM", "DEATH"]}
        cfg = resolve_config("competing", manifest)
        assert cfg.event_cols == ["PLATINUM", "DEATH"]
        assert cfg.n_events == 2

    def test_case_insensitive(self):
        manifest = {"event_cols": ["PLATINUM", "DEATH"]}
        cfg = resolve_config("PLATINUM", manifest)
        assert cfg.name == "platinum"

    def test_unknown_config_raises(self):
        manifest = {"event_cols": ["PLATINUM", "DEATH"]}
        with pytest.raises(ValueError, match="Unsupported --config"):
            resolve_config("death", manifest)

    def test_manifest_missing_event_col_raises(self):
        manifest = {"event_cols": ["PLATINUM"]}
        with pytest.raises(ValueError, match="DEATH"):
            resolve_config("competing", manifest)


class TestPatientTargets:
    def _run(self, rows, *, event_cols, time_cols, max_pred_window=260):
        df = pd.DataFrame(rows)
        return patient_targets(
            df,
            id_col="DFCI_MRN",
            time_col="TIME",
            event_cols=event_cols,
            time_cols=time_cols,
            max_pred_window=max_pred_window,
        )

    def test_platinum_config_death_only_patient_is_censored(self):
        # Cause-specific censoring: in the platinum config, a death-only
        # patient contributes no observed event -- label must be 0.
        rows = [
            _person_period_row(1, -30, PLATINUM=0, t_platinum=100, DEATH=1, t_death=50),
            _person_period_row(1, 0),
        ]
        targets = self._run(rows, event_cols=["PLATINUM"], time_cols=["t_platinum"])
        assert targets.loc[1, "label"] == 0

    def test_competing_config_same_patient_death_labeled(self):
        rows = [
            _person_period_row(1, -30, PLATINUM=0, t_platinum=100, DEATH=1, t_death=50),
            _person_period_row(1, 0),
        ]
        targets = self._run(
            rows, event_cols=["PLATINUM", "DEATH"], time_cols=["t_platinum", "t_death"]
        )
        assert targets.loc[1, "label"] == 2

    def test_platinum_first_labeled_one(self):
        rows = [
            _person_period_row(2, -30, PLATINUM=1, t_platinum=40, DEATH=1, t_death=80),
            _person_period_row(2, 0),
        ]
        targets = self._run(
            rows, event_cols=["PLATINUM", "DEATH"], time_cols=["t_platinum", "t_death"]
        )
        assert targets.loc[2, "label"] == 1
        assert targets.loc[2, "duration"] == 40.0

    def test_tie_breaks_toward_platinum(self):
        # argmin over [platinum_duration, death_duration]; a tie resolves to
        # index 0 (platinum) since np.argmin returns the first minimum.
        rows = [
            _person_period_row(3, -30, PLATINUM=1, t_platinum=50, DEATH=1, t_death=50),
            _person_period_row(3, 0),
        ]
        targets = self._run(
            rows, event_cols=["PLATINUM", "DEATH"], time_cols=["t_platinum", "t_death"]
        )
        assert targets.loc[3, "label"] == 1

    def test_event_past_window_is_censored_at_window(self):
        rows = [
            _person_period_row(4, -30, PLATINUM=1, t_platinum=500, DEATH=0, t_death=np.nan),
            _person_period_row(4, 0),
        ]
        targets = self._run(
            rows, event_cols=["PLATINUM"], time_cols=["t_platinum"], max_pred_window=260
        )
        row = targets.loc[4]
        assert row["label"] == 0
        assert row["duration"] == 260.0
        assert row["duration_bin"] == 260
        # Pre-window-censoring values are preserved for diagnostics.
        assert row["uncensored_label"] == 1
        assert row["uncensored_duration"] == 500.0

    def test_event_exactly_at_window_is_retained(self):
        # Deliberately contrasts with SurvLatent's strict-< at-horizon
        # censoring: here the window boundary is inclusive (duration >
        # max_pred_window triggers censoring, so == is retained).
        rows = [
            _person_period_row(5, -30, PLATINUM=1, t_platinum=260, DEATH=0, t_death=np.nan),
            _person_period_row(5, 0),
        ]
        targets = self._run(
            rows, event_cols=["PLATINUM"], time_cols=["t_platinum"], max_pred_window=260
        )
        row = targets.loc[5]
        assert row["label"] == 1
        assert row["duration"] == 260.0
        assert row["duration_bin"] == 260

    def test_no_valid_post_landmark_time_dropped(self):
        rows = [
            _person_period_row(6, -30, PLATINUM=0, t_platinum=np.nan, DEATH=0, t_death=np.nan),
            _person_period_row(6, 0),
        ]
        targets = self._run(rows, event_cols=["PLATINUM"], time_cols=["t_platinum"])
        assert 6 not in targets.index

    def test_duration_bin_always_at_least_one(self):
        rows = [
            _person_period_row(7, -30, PLATINUM=1, t_platinum=0.5, DEATH=0, t_death=np.nan),
            _person_period_row(7, 0),
        ]
        targets = self._run(rows, event_cols=["PLATINUM"], time_cols=["t_platinum"])
        assert targets.loc[7, "duration_bin"] >= 1


class TestManifestHorizonsForConfig:
    def _build_manifest(self):
        return {
            "auc_horizons_by_landmark": {
                "0": {"platinum": [30, 90, 180], "death": [20, 60, 150]},
            }
        }

    def test_platinum_only_uses_platinum_grid(self):
        horizons = manifest_horizons_for_config(
            self._build_manifest(), 0, ["platinum"], max_pred_window=260
        )
        assert list(horizons["platinum"]) == [30, 90, 180]

    def test_competing_both_causes_use_platinum_grid(self):
        horizons = manifest_horizons_for_config(
            self._build_manifest(), 0, ["platinum", "death"], max_pred_window=260
        )
        assert list(horizons["platinum"]) == [30, 90, 180]
        assert list(horizons["death"]) == [30, 90, 180]

    def test_horizons_clamped_to_max_pred_window(self):
        horizons = manifest_horizons_for_config(
            self._build_manifest(), 0, ["platinum"], max_pred_window=100
        )
        assert list(horizons["platinum"]) == [30, 90]

    def test_missing_landmark_raises(self):
        with pytest.raises(KeyError):
            manifest_horizons_for_config(
                self._build_manifest(), 999, ["platinum"], max_pred_window=260
            )


class TestHorizonUnitsToBins:
    def test_identity_when_units_match(self):
        result = horizon_units_to_bins(
            np.array([30, 90, 180]), manifest_time_unit_days=7, input_time_unit_days=7
        )
        assert list(result) == [30, 90, 180]

    def test_raises_on_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            horizon_units_to_bins(
                np.array([30, 90]), manifest_time_unit_days=7, input_time_unit_days=1
            )


def test_competing_registry_platinum_ordered_first():
    assert LONGITUDINAL_CONFIGS["competing"]["event_cols"][0] == "PLATINUM"
