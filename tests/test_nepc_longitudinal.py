"""Tests for the NEPC longitudinal configs and shared target semantics.

Companion to test_nepc_endpoint.py, which covers the Cox/XGBoost endpoint axis.
The two registries are separate by design --
cox_aggregated.ENDPOINTS selects a (duration, event) pair, LONGITUDINAL_CONFIGS
selects a cause of interest plus an optional competing cause -- so each
endpoint needs independent coverage on this side.

The invariants pinned here:
  1. label 1 is always the cause of interest and label 2 always death, for
     ALL competing configs. A reorder silently reinterprets every downstream
     risk column.
  2. each config scores on its own endpoint's AUC(t) horizon grid.
  3. the retired joint AVPC_NEPC endpoint cannot be selected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import compass_pipeline as cp
from survival_common.longitudinal_targets import (
    CONFIG_ENDPOINTS,
    LONGITUDINAL_CONFIGS,
    manifest_horizons_for_config,
    patient_targets,
    resolve_config,
)

FULL_MANIFEST = {"event_cols": ["PLATINUM", "DEATH", "NEPC"]}


def _row(mrn: int, time: float, **overrides) -> dict:
    row = {
        "DFCI_MRN": mrn,
        "TIME": time,
        "PLATINUM": 0,
        "t_platinum": np.nan,
        "DEATH": 0,
        "t_death": np.nan,
        "NEPC": 0,
        "t_nepc": np.nan,
    }
    row.update(overrides)
    return row


class TestNepcConfigResolution:
    def test_nepc_config(self):
        cfg = resolve_config("nepc", FULL_MANIFEST)
        assert cfg.event_cols == ["NEPC"]
        assert cfg.time_cols == ["t_nepc"]
        assert cfg.event_names == ["nepc"]
        assert cfg.n_events == 1

    def test_nepc_competing_config(self):
        cfg = resolve_config("nepc_competing", FULL_MANIFEST)
        assert cfg.event_cols == ["NEPC", "DEATH"]
        assert cfg.time_cols == ["t_nepc", "t_death"]
        assert cfg.event_names == ["nepc", "death"]
        assert cfg.n_events == 2

    def test_case_insensitive(self):
        assert resolve_config("NEPC_Competing", FULL_MANIFEST).name == "nepc_competing"

    def test_cohort_without_nepc_raises_actionable_error(self):
        """A platinum-only build must reject the nepc config, not KeyError later."""
        platinum_only = {"event_cols": ["PLATINUM", "DEATH"]}
        with pytest.raises(ValueError, match="NEPC"):
            resolve_config("nepc", platinum_only)

    def test_cause_of_interest_is_ordered_first(self):
        """Label 1 = cause of interest, label 2 = death, for all competing
        configs."""
        assert LONGITUDINAL_CONFIGS["competing"]["event_cols"] == ["PLATINUM", "DEATH"]
        assert LONGITUDINAL_CONFIGS["nepc_competing"]["event_cols"] == ["NEPC", "DEATH"]
        assert LONGITUDINAL_CONFIGS["avpc_competing"]["event_cols"] == ["AVPC", "DEATH"]

    def test_every_config_declares_a_horizon_endpoint(self):
        assert set(CONFIG_ENDPOINTS) == set(LONGITUDINAL_CONFIGS)


class TestRetiredJointConfig:
    def test_joint_config_is_not_registered(self):
        assert "avpc_nepc" not in LONGITUDINAL_CONFIGS
        assert "avpc_nepc_competing" not in LONGITUDINAL_CONFIGS
        with pytest.raises(ValueError, match="Unsupported --config"):
            resolve_config("avpc_nepc", FULL_MANIFEST)


class TestNepcTargets:
    def _run(self, rows, config, max_pred_window=260):
        cfg = resolve_config(config, FULL_MANIFEST)
        return patient_targets(
            pd.DataFrame(rows),
            id_col="DFCI_MRN",
            time_col="TIME",
            event_cols=cfg.event_cols,
            time_cols=cfg.time_cols,
            max_pred_window=max_pred_window,
        )

    def test_nepc_event_labeled_one(self):
        rows = [_row(1, 0.0, NEPC=1, t_nepc=50.0), _row(1, 10.0, NEPC=1, t_nepc=50.0)]
        out = self._run(rows, "nepc")
        assert out.loc[1, "label"] == 1
        # duration is post-landmark: landmark_time = max(TIME) = 10
        assert out.loc[1, "duration"] == pytest.approx(40.0)

    def test_censored_patient_labeled_zero(self):
        rows = [_row(2, 0.0, t_nepc=80.0), _row(2, 10.0, t_nepc=80.0)]
        out = self._run(rows, "nepc")
        assert out.loc[2, "label"] == 0

    def test_death_censors_nepc_in_cause_only_config(self):
        """DEATH is never read by the nepc config -- the patient is censored."""
        rows = [_row(3, 0.0, DEATH=1, t_death=30.0, t_nepc=90.0)]
        out = self._run(rows, "nepc")
        assert out.loc[3, "label"] == 0

    def test_nepc_competing_labels_death_two(self):
        rows = [_row(4, 0.0, DEATH=1, t_death=30.0, t_nepc=90.0)]
        out = self._run(rows, "nepc_competing")
        assert out.loc[4, "label"] == 2
        assert out.loc[4, "duration"] == pytest.approx(30.0)

    def test_nepc_competing_earliest_event_wins(self):
        """Both observed: labeled by the smaller post-landmark duration."""
        rows = [_row(5, 0.0, NEPC=1, t_nepc=40.0, DEATH=1, t_death=120.0)]
        out = self._run(rows, "nepc_competing")
        assert out.loc[5, "label"] == 1
        assert out.loc[5, "duration"] == pytest.approx(40.0)

    def test_event_beyond_window_is_censored_at_window(self):
        rows = [_row(6, 0.0, NEPC=1, t_nepc=500.0)]
        out = self._run(rows, "nepc", max_pred_window=260)
        assert out.loc[6, "label"] == 0
        assert out.loc[6, "duration"] == pytest.approx(260.0)
        # the pre-censoring pair is retained for diagnostics
        assert out.loc[6, "uncensored_label"] == 1
        assert out.loc[6, "uncensored_duration"] == pytest.approx(500.0)

    def test_prevalent_nepc_dropped(self):
        """A diagnosis at/before the landmark yields no valid target row."""
        rows = [_row(7, 0.0, NEPC=1, t_nepc=10.0), _row(7, 30.0, NEPC=1, t_nepc=10.0)]
        out = self._run(rows, "nepc")
        assert 7 not in out.index


class TestHorizonEndpointRouting:
    def _manifest(self):
        return {
            "auc_horizons_by_landmark": {
                "0": {
                    "platinum": [30, 90, 180],
                    "nepc": [25, 75, 150],
                    "death": [20, 60],
                }
            }
        }

    def test_nepc_config_reads_nepc_grid(self):
        horizons = manifest_horizons_for_config(
            self._manifest(), 0, ["nepc"], max_pred_window=260, endpoint="nepc"
        )
        assert list(horizons["nepc"]) == [25, 75, 150]

    def test_nepc_competing_scores_both_causes_on_nepc_grid(self):
        horizons = manifest_horizons_for_config(
            self._manifest(), 0, ["nepc", "death"], max_pred_window=260, endpoint="nepc"
        )
        assert list(horizons["nepc"]) == [25, 75, 150]
        assert list(horizons["death"]) == [25, 75, 150]

    def test_platinum_grid_still_the_default(self):
        """Omitting endpoint must reproduce the pre-change behaviour exactly."""
        horizons = manifest_horizons_for_config(
            self._manifest(), 0, ["platinum"], max_pred_window=260
        )
        assert list(horizons["platinum"]) == [30, 90, 180]

    def test_config_endpoint_map_routes_each_config(self):
        assert CONFIG_ENDPOINTS["nepc"] == "nepc"
        assert CONFIG_ENDPOINTS["nepc_competing"] == "nepc"
        assert CONFIG_ENDPOINTS["platinum"] == "platinum"
        assert CONFIG_ENDPOINTS["competing"] == "platinum"

    def test_missing_endpoint_grid_raises_actionable_error(self):
        platinum_only = {"auc_horizons_by_landmark": {"0": {"platinum": [30, 90]}}}
        with pytest.raises(KeyError, match="require-nepc"):
            manifest_horizons_for_config(
                platinum_only, 0, ["nepc"], max_pred_window=260, endpoint="nepc"
            )


class TestPlatinumConfigsUnchanged:
    """The nepc registration must not perturb the existing platinum arms."""

    def test_platinum_config_spec_identical(self):
        cfg = resolve_config("platinum", FULL_MANIFEST)
        assert cfg.event_cols == ["PLATINUM"]
        assert cfg.time_cols == ["t_platinum"]
        assert cfg.event_names == ["platinum"]

    def test_competing_config_spec_identical(self):
        cfg = resolve_config("competing", FULL_MANIFEST)
        assert cfg.event_cols == ["PLATINUM", "DEATH"]
        assert cfg.time_cols == ["t_platinum", "t_death"]
        assert cfg.event_names == ["platinum", "death"]

    def test_nepc_columns_present_do_not_change_platinum_targets(self):
        """Attaching NEPC columns leaves the platinum targets bit-identical."""
        base = [
            _row(1, 0.0, PLATINUM=1, t_platinum=60.0),
            _row(2, 0.0, DEATH=1, t_death=45.0),
            _row(3, 0.0, t_platinum=200.0),
        ]
        with_nepc = [dict(r) for r in base]
        with_nepc[0].update(NEPC=1, t_nepc=20.0)
        with_nepc[2].update(NEPC=1, t_nepc=15.0)

        cfg = resolve_config("competing", FULL_MANIFEST)
        kwargs = dict(
            id_col="DFCI_MRN",
            time_col="TIME",
            event_cols=cfg.event_cols,
            time_cols=cfg.time_cols,
            max_pred_window=260,
        )
        left = patient_targets(pd.DataFrame(base), **kwargs)
        right = patient_targets(pd.DataFrame(with_nepc), **kwargs)
        pd.testing.assert_frame_equal(left, right)


class TestTaskSpecWiring:
    """compass_pipeline's endpoint -> config routing and the survlatent gate."""

    def test_specs_follow_endpoint(self):
        platinum = [c for _, c, _ in cp.longitudinal_task_specs("platinum", include_survlatent=False)]
        nepc = [c for _, c, _ in cp.longitudinal_task_specs("nepc", include_survlatent=False)]
        assert platinum == ["platinum", "competing"]
        assert nepc == ["nepc", "nepc_competing"]

    def test_survlatent_off_by_default(self):
        """RUN_SURVLATENT gates it; 03b ships with it off."""
        assert cp.RUN_SURVLATENT is False
        models = {m for m, _, _ in cp.longitudinal_task_specs("platinum")}
        assert models == {"dynamic-deephit"}

    def test_survlatent_included_when_enabled(self):
        models = {m for m, _, _ in cp.longitudinal_task_specs("platinum", include_survlatent=True)}
        assert models == {"dynamic-deephit", "survlatent-ode"}

    def test_metrics_filenames_match_config(self):
        specs = {
            cfg: fname
            for _, cfg, fname in cp.longitudinal_task_specs("nepc", include_survlatent=False)
        }
        assert specs["nepc"] == "dynamic_deephit_metrics_nepc.csv"
        assert specs["nepc_competing"] == "dynamic_deephit_metrics_nepc_competing.csv"

    def test_every_pipeline_config_is_a_registered_config(self):
        """A spec naming a config the runner would reject is a wiring bug."""
        for endpoint in ("platinum", "nepc"):
            for _, cfg, _ in cp.longitudinal_task_specs(endpoint, include_survlatent=True):
                assert cfg in LONGITUDINAL_CONFIGS

    def test_unknown_endpoint_raises(self):
        with pytest.raises(ValueError, match="No longitudinal configs"):
            cp.longitudinal_task_specs("not_an_endpoint")
